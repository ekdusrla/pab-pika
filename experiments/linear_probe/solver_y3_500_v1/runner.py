"""Chunked, resumable fixed-protocol generation; each logical request is journaled."""
import argparse
import os
from pathlib import Path

from .evaluator import evaluator_info, validate_solver
from .audit import ARMS, DATA, OUT, PROBE, SEED, model_identity, probe_metadata
from ..common import (FEATURE, LEVEL_WORDS, allocate_words, decomp_messages, digest, file_hash,
                      original_definitions, project_credits, read, seal, verify)
from ..credit_a2000_tp4.teacher import payload_for
from ..followup_v1.client import validate
from ..learning import predict
from ..pilot import SOLVER
from experiment_state import atomic_save, lock, run_stage, stage, termination_handler
from run_solver_compare import messages_for

PROTOCOL = Path(__file__).with_name('protocol.json')


def code_hashes():
    from ..common import ROOT
    local = Path(__file__).parent
    paths = [p for p in local.iterdir() if p.suffix in ('.py', '.sh', '.sbatch')]
    for name in (
        'experiments/linear_probe/common.py', 'experiments/linear_probe/capture.py',
        'experiments/linear_probe/safety.py', 'experiments/linear_probe/learning.py',
        'experiments/linear_probe/credit_a2000_tp4/teacher.py',
        'experiments/linear_probe/followup_v1/client.py',
        'experiments/linear_probe/solver_y1_v1/gpu_mapping.py',
        'experiments/linear_probe/solver_y1_v1/identity.py',
        'experiments/llama_qwq/experiment_state.py', 'experiments/llama_qwq/run_planner.py',
        'experiments/llama_qwq/run_solver_compare.py', 'experiments/llama_qwq/math_answer_eval.py',
        'dataset/break_down_question.py', 'dataset/math_dataset.py',
    ):
        paths.append(ROOT / name)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def experiment(manifest):
    doc = verify(read(manifest))
    if doc.get('approved') is not True:
        raise ValueError('Independent 500-problem input has no approval. Candidate execution blocked.')
    conf = read(PROTOCOL)
    if doc.get('protocol_sha256') != file_hash(PROTOCOL):
        raise ValueError('Approved protocol hash mismatch')
    if doc.get('code_sha256') != code_hashes():
        raise ValueError('Approved implementation hash mismatch')
    rows = doc['rows']
    candidate_path = OUT / 'candidate_manifest.json'
    candidate = verify(read(candidate_path))
    if (file_hash(candidate_path) != doc.get('candidate_sha256')
            or rows != candidate['rows'] or doc.get('candidate_fingerprint') != candidate['fingerprint']):
        raise ValueError('Approved rows/order differ from the pinned candidate')
    if len(rows) != 500 or len({r['id'] for r in rows}) != 500:
        raise ValueError('Require unique 500-problem manifest')
    if tuple(doc['arms']) != ARMS or doc['seed'] != SEED or conf['planner_seed'] != SEED:
        raise ValueError('Arms or Planner seed changed')
    if any(row['chunk'] != i//50+1 for i, row in enumerate(rows)):
        raise ValueError('Chunk membership changed')
    for path, sha in doc['input_sha256'].items():
        if file_hash(path) != sha:
            raise ValueError('Protected source changed: ' + path)
    for model in doc['model_identity'].values():
        if model_identity(model['model'], model['revision']) != model:
            raise ValueError('Model identity changed')
    if probe_metadata()['fingerprint'] != doc['probe_fingerprint']:
        raise ValueError('Frozen Probe identity changed')
    return doc, conf


def position_table(rows):
    import numpy as np
    groups = {}
    for row in rows:
        credits = row['teacher']['credits']
        groups.setdefault(len(credits), []).append(credits)
    return {n: np.mean(c, axis=0).tolist() for n, c in groups.items()}


def allocations(row, capture, teacher_credits, positions):
    n = len(capture['steps'])
    raw = {'uniform': [1.] * n, 'position': positions.get(n, [1.] * n)}
    credits = {arm: project_credits(vector)['credits'] for arm, vector in raw.items()}
    credits.update(teacher=teacher_credits, ridge=capture.get('ridge', {}).get('credits'))
    total = LEVEL_WORDS[row['level']]
    return credits, {arm: allocate_words(vector, total, 'constant')
                     for arm, vector in credits.items() if vector is not None}


def solver_payload(row, steps, arm, budgets, seed):
    _, instruction, output = original_definitions()
    return dict(model=SOLVER, messages=messages_for({**row, 'steps': steps},
                'vanilla' if arm == 'vanilla' else 'weighted_pab', budgets, instruction, output),
                max_tokens=8192, temperature=.1, top_p=.9, seed=seed)


class Journal:
    def __init__(self, chunk_path, identity):
        self.path = chunk_path / (digest(identity) + '.json')
        self.identity = identity
        self.record = verify(read(self.path)) if self.path.exists() else {
            'identity': identity, 'request_id': digest(identity), 'stage': stage()}
        if self.record['identity'] != identity or self.record['request_id'] != digest(identity):
            raise ValueError('Checkpoint identity mismatch')

    def save(self):
        atomic_save(self.path, seal({k: v for k, v in self.record.items() if k != 'fingerprint'}))

    def api(self, payload, client, validator, pid=None):
        target = self.record['stage']
        if target.get('request', payload) != payload:
            raise ValueError('Request changed during resume')
        return run_stage(target, payload, client, self.save, validator, retries=2, server_pid=pid)


def journal(path, manifest, conf, row, arm, seed=SEED):
    return Journal(path, dict(experiment=manifest['fingerprint'], protocol=digest(conf),
                             problem=row['id'], chunk=row['chunk'], arm=arm, seed=seed))


def capture_problem(j, row, model, tokenizer, guard, layer):
    from ..safety import guarded_generate
    target = j.record['stage']
    if target['status'] == 'success' and target['result']['control'] is not None:
        return
    messages = decomp_messages(row)
    if target.get('request', messages) != messages:
        raise ValueError('Frozen decomposition prompt changed')
    target['request'] = messages
    for a in target['attempts']:
        if a['status'] == 'in_flight':
            a.update(status='interrupted', cost_unknown=True)
    for mode in ('control', 'capture'):
        if any(a['role'] == mode and a['status'] == 'success' for a in target['attempts']):
            continue
        attempt = dict(role=mode, status='in_flight', usage={})
        target['attempts'].append(attempt)
        target['status'] = 'running'
        j.save()
        outcome = guarded_generate(model, tokenizer, messages, guard, list(range(33)), seed=SEED,
                                   capture=mode == 'capture')
        result = outcome.pop('result')
        usage = (result['usage'] if result else dict(prompt_tokens=outcome['prompt_tokens'],
                 completion_tokens=outcome['completion_tokens'],
                 latency_seconds=outcome['diagnostics'].get('generation_seconds')))
        attempt.update(outcome, usage=usage)
        if result:
            features = result.pop('features', None)
            attempt['result'] = result
            if mode == 'capture':
                if result['feature_definition'] != FEATURE:
                    raise ValueError('Feature definition changed')
                # Preserve selected-layer audit features and reconstruct identical CPU input.
                attempt['selected_features'] = [{str(layer): f[str(layer)]} for f in features]
        target['status'] = 'response_received' if result else 'error'
        j.save()
        if outcome['stop_required']:
            raise RuntimeError('Capture safety stop. Inspect recorded attempt before resuming.')
    successful = {a['role']: a for a in target['attempts'] if a['status'] == 'success'}
    if 'capture' not in successful:
        return
    cap = successful['capture']
    result = cap['result']
    ctrl = successful.get('control', {}).get('result')
    target['result'] = {**result, 'control': ctrl,
                        'same_generated_ids': ctrl['generated_ids'] == result['generated_ids'] if ctrl else None}
    # Shared decomposition always comes from capture. Control is for timing/identity only.
    target['status'] = 'success'
    j.save()


def run(args):
    manifest, conf = experiment(args.manifest)
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURMD_NODENAME') != conf['node']:
        raise ValueError('Require authorized y3 allocation')
    rows = [row for row in manifest['rows'] if row['chunk'] == args.chunk]
    path = OUT / 'journal' / manifest['fingerprint'] / f'chunk-{args.chunk:02d}'
    data = verify(read(DATA))
    with lock(path / 'writer'), termination_handler():
        if args.phase == 'capture':
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from ..safety import MemoryGuard, require_device
            entries = [(r, journal(path, manifest, conf, r, 'capture')) for r in rows]
            if all(j.record['stage']['status'] == 'success' and j.record['stage']['result']['control'] is not None
                   for _, j in entries):
                return
            snapshot = manifest['model_identity']['planner']['snapshot']
            require_device(torch)
            guard = MemoryGuard(torch.cuda, limit_gib=21.)
            tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            frozen_capture = verify(read(OUT.parent / 'pilot_v1/evaluation_recovered.json'))
            if digest(tok.chat_template) != frozen_capture['sessions'][0]['chat_template_sha256']:
                raise ValueError('Planner tokenizer chat template changed')
            model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                        torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
            torch.cuda.synchronize()
            guard.check('model_load')
            layer = probe_metadata()['best_layer_idx']
            for row, j in entries:
                capture_problem(j, row, model, tok, guard, layer)
            return
        if args.phase == 'probe':
            import joblib
            layer = probe_metadata()['best_layer_idx']
            pipeline = joblib.load(PROBE / 'probe.joblib')
            for row in rows:
                state = journal(path, manifest, conf, row, 'capture').record['stage']
                target = journal(path, manifest, conf, row, 'probe')
                if state['status'] != 'success' or target.record['stage']['status'] == 'success':
                    continue
                features = next(a['selected_features'] for a in state['attempts']
                                if a['role'] == 'capture' and a['status'] == 'success')
                prediction = predict(pipeline, {'capture': {'features': features}}, layer)
                target.record['stage'].update(status='success', result=prediction,
                                              steps_fingerprint=digest(state['result']['steps']))
                target.save()
            return
        if args.phase == 'solver' and evaluator_info() != manifest['evaluator']:
            raise ValueError('Approved Solver evaluator differs from the installed runtime')
        from openai import OpenAI
        if not args.port or not args.pid:
            raise ValueError('Same-allocation server port and PID required')
        train = verify(read(PROBE.parent / 'train_labeled.json'))['records']
        pos = position_table(train)
        model_id = manifest['model_identity']['planner' if args.phase == 'teacher' else 'solver']['model']
        with OpenAI(base_url=f'http://127.0.0.1:{args.port}/v1', api_key='DUMMY', timeout=600, max_retries=0) as client:
            if model_id not in {m.id for m in client.models.list().data}:
                raise ValueError('Server model mismatch')
            for row in rows:
                cap_state = journal(path, manifest, conf, row, 'capture').record['stage']
                cap = cap_state.get('result', {}) if cap_state['status'] == 'success' else {}
                teacher = journal(path, manifest, conf, row, 'teacher_credit')
                if args.phase == 'teacher':
                    if cap:
                        payload = payload_for(row, {'capture': cap}, data)
                        teacher.api(payload, client, lambda r: validate(r, 'teacher', {'n_steps': len(cap['steps'])}), args.pid)
                    continue
                t = teacher.record['stage']
                truth = t['parsed']['credits'] if t['status'] == 'success' else None
                probe = journal(path, manifest, conf, row, 'probe').record['stage']
                if cap and probe['status'] == 'success':
                    if probe['steps_fingerprint'] != digest(cap['steps']):
                        raise ValueError('Probe decomposition changed')
                    cap = {**cap, 'ridge': probe['result']}
                credits, budgets = allocations(row, cap, truth, pos) if cap else ({}, {})
                offset = row['source_index'] % len(ARMS)
                for seed in conf['solver_seeds']:
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        j = journal(path, manifest, conf, row, arm, seed)
                        if arm != 'vanilla' and arm not in budgets:
                            if j.record['stage']['status'] in ('pending', 'blocked'):
                                j.record['stage']['status'] = 'blocked'
                                j.record['stage']['reason'] = 'missing_successful_decomposition_or_teacher'
                                j.save()
                            continue
                        metadata = dict(credits=credits.get(arm), budgets_words=budgets.get(arm, []),
                                        total_words=None if arm == 'vanilla' else LEVEL_WORDS[row['level']],
                                        steps_fingerprint=None if arm == 'vanilla' else digest(cap['steps']))
                        if j.record.get('allocation', metadata) != metadata:
                            raise ValueError('Frozen allocation changed during resume')
                        j.record['allocation'] = metadata
                        payload = solver_payload(row, cap.get('steps', []), arm, budgets.get(arm, []), seed)
                        j.api(payload, client, lambda r: validate_solver(r, row), args.pid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--chunk', type=int, choices=range(1, 11), required=True)
    parser.add_argument('--phase', choices=('capture', 'probe', 'teacher', 'solver'), required=True)
    parser.add_argument('--port', type=int)
    parser.add_argument('--pid', type=int)
    run(parser.parse_args())
