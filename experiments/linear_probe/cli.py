"""Explicit stages; --help/prepare never import torch or load any model."""
import argparse
import json
import sys
import time
from pathlib import Path

from .common import (MODEL, REVISION, FEATURE, SCHEMA, ROOT, LEGACY, read, output_path,
                     save_new, seal, verify, validate_data, prepare, read_manifest,
                     digest, file_hash, runtime, prompt_fingerprint, problem_hash, decomp_messages, assessment_messages,
                     parse_credits, credit_schema, credit_contract, allocate_words, LEVEL_WORDS)
from .learning import validate_capture, feature_rows, fit_probe, predict
from experiment_state import lock, atomic_save, stage, run_stage, attempt_cost


def configuration(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            if k not in ('resume', 'func', 'resume_after_safety_stop')}


def checkpoint(args, config, initial):
    path = output_path(args.output)
    if Path(str(path) + '.tmp').exists():
        raise ValueError('Uncommitted .tmp exists; inspect/preserve it before restarting. No automatic discard.')
    if path.exists():
        if not args.resume:
            raise FileExistsError('Existing output protected; explicit --resume required')
        document = verify(read(path))
        if document['config'] != config:
            raise ValueError('Resume configuration/source changed')
        return document
    if args.resume:
        raise FileNotFoundError('No checkpoint to resume')
    return {**initial, 'config': config}


def commit(args, doc):
    atomic_save(output_path(args.output), seal({k: v for k, v in doc.items() if k != 'fingerprint'}))


def prepare_command(args):
    doc = prepare(read(args.source), read_manifest(args.eval_manifest), args.validation_fraction, args.seed)
    doc['input_source_sha256'] = file_hash(args.source)
    save_new(args.output, seal({k: v for k, v in doc.items() if k != 'fingerprint'}))


def collect(args):
    from .safety import (MemoryGuard, require_device, guarded_generate, apply_attempt,
                         acknowledge_resume, failure_kind)
    from experiment_state import StopRequested, termination_handler
    limit=getattr(args,'memory_limit_gib',21.)
    data = validate_data(read(args.data))
    # All input validation and overwrite checks happen BEFORE model import/load.
    snapshot = args.model_path.resolve()
    if snapshot.name != REVISION:
        raise ValueError('Use the exact pinned local HF snapshot directory (revision basename)')
    architecture = read(snapshot / 'config.json')
    if architecture.get('hidden_size') != 4096 or architecture.get('num_hidden_layers') != 32:
        raise ValueError('Unexpected Llama architecture')
    layers = sorted(set(args.layers)) if args.layers else list(range(33))
    if any(i not in range(33) for i in layers):
        raise ValueError('Expected hidden state indices 0..32')
    config = {**configuration(args), 'data_fingerprint': data['fingerprint'],
              'model': MODEL, 'revision': REVISION, 'layers': layers,
              'model_config_sha256': file_hash(snapshot / 'config.json'),
              'weight_index_sha256': file_hash(snapshot / 'model.safetensors.index.json'),
              'decomp_prompt_fingerprint': prompt_fingerprint(),
              'capture_implementation_sha256': file_hash(Path(__file__).with_name('capture.py')),
              'tokenizer_files_sha256': {p.name: file_hash(p) for p in sorted(snapshot.glob('*token*')) if p.is_file()},
              'decoding': {'temperature': .1, 'top_p': .9, 'max_new_tokens': 1024,
                           'max_context': 8192, 'dtype': 'float16', 'attention': 'eager', 'seed': args.seed},
              'runtime': runtime(), 'feature_definition': FEATURE,
              'capture_safety': {'version':1,'reserved_peak_limit_gib':limit,
                  'interval_checks':True,'stop_on_oom_or_limit':True},
              'engine_differences': {'baseline': 'vllm 0.6.6 OpenAI API',
                 'current': 'transformers manual KV-cache, no guided decoding for decomposition',
                 'sampling': 'temperature/top_p matched; RNG/kernel implementations differ',
                 'hidden_states': 'HF embedding + post-layer outputs; last layer includes final model norm',
                 'timing': 'per-interval synchronization, no automatic warmup'}}
    with lock(output_path(args.output)):
        doc = checkpoint(args, config, {'schema_version': SCHEMA, 'kind': 'capture',
                          'data_fingerprint': data['fingerprint'], 'split': args.split, 'records': [],
                          'last_successful_problem_id':None,'safety_stop':None})
        acknowledge_resume(doc,args.resume,getattr(args,'resume_after_safety_stop',False))
        commit(args, doc)
        for row in doc['records']:
            if row['status'] == 'success':
                feature_rows(row)  # Fail before loading weights if stored feature bytes changed.
        if len(doc['records']) == len(data['splits'][args.split]) and all(r['status']=='success' for r in doc['records']):
            validate_capture(seal({k:v for k,v in doc.items() if k!='fingerprint'}), data, args.split)
            return
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        begin = time.perf_counter()
        session={'status':'device_preflight','device':'cuda:0','warmup':'none; first request is cold'}
        doc.setdefault('sessions',[]).append(session)
        commit(args, doc)
        guard=None
        try:
            session['gpu']=require_device(torch)  # Before any model weights are loaded.
            guard=MemoryGuard(torch.cuda,limit)
            guard.begin()
            session['status']='model_loading'
            commit(args,doc)
            with termination_handler():
                tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                            torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
                torch.cuda.synchronize()
                guard.check('model_load')
            session.update(status='loaded',model_load_seconds=time.perf_counter()-begin,
                memory=guard.snapshot(),actual_model_config=model.config.to_dict(),
                generation_config=model.generation_config.to_dict(),tokenizer_class=type(tok).__name__,
                chat_template_sha256=digest(tok.chat_template))
        except (Exception,StopRequested) as exc:
            session.update(status='error',model_load_seconds=time.perf_counter()-begin,
                           memory=guard.snapshot() if guard else None,
                           error={'type':type(exc).__name__,'message':str(exc)[:1000]})
            doc['safety_stop']={'reason':failure_kind(exc),'phase':'model_load','problem_id':None,
                'last_successful_problem_id':doc.get('last_successful_problem_id'),
                'error':session['error'],'memory':session['memory']}
            commit(args,doc)
            raise RuntimeError('Model loading failed; session diagnostics preserved') from None
        commit(args,doc)
        # SIGTERM inside generation/storage is caught and persisted before the job exits.
        with termination_handler():
            collect_rows(args,data,doc,model,tok,guard,layers)
    if any(r['status'] != 'success' for r in doc['records']):
        raise RuntimeError('Capture failures remain; inspect checkpoint before explicit resume')


def collect_rows(args,data,doc,model,tok,guard,layers):
        import numpy as np
        from .safety import guarded_generate,apply_attempt,failure_kind
        from experiment_state import StopRequested
        by_id = {r['id']: r for r in doc['records']}
        for source in data['splits'][args.split]:
            row = by_id.get(source['id'])
            if row and row['status'] == 'success':
                continue
            if row is None:
                row = {'id': source['id'], 'problem_hash': source['problem_hash'], 'attempts': []}
                doc['records'].append(row)
            # A prior in-flight attempt has unknown cost and is never erased.
            for previous in row['attempts']:
                if previous['status'] == 'running':
                    previous.update(status='interrupted', cost_unknown=True)
            attempt = {'status': 'running', 'session_index': len(doc['sessions'])-1}
            row['attempts'].append(attempt); row['status'] = 'running'
            commit(args, doc)
            begin = time.perf_counter()
            outcome=None;control=None
            previous_success_id=doc.get('last_successful_problem_id')
            try:
                messages = decomp_messages(source)
                if args.capture_control:
                    control=guarded_generate(model,tok,messages,guard,layers,seed=args.seed,capture=False)
                    attempt['capture_control_attempt']={k:v for k,v in control.items() if k!='result'}
                    if control['status']!='success':
                        outcome=control
                    commit(args,doc)
                if outcome is None:
                    outcome=guarded_generate(model,tok,messages,guard,layers,seed=args.seed)
                apply_attempt(doc,row,attempt,outcome)
                if outcome['status']!='success':
                    attempt['wall_seconds_including_storage']=time.perf_counter()-begin
                    commit(args,doc)
                    if outcome['stop_required']:
                        raise RuntimeError('Capture safety stop; checkpoint preserved')
                    print(source['id'],row['status'],flush=True)
                    continue
                cap=outcome['result']
                if control is not None:
                    ctrl=control['result'];same=ctrl['generated_ids']==cap['generated_ids']
                    cap['capture_control']={k:ctrl[k] for k in ('usage','metrics','generated_ids')}
                    cap['capture_control'].update(memory=control['memory'],same_generated_ids=same,
                        capture_minus_control_seconds=(cap['metrics']['generation_plus_feature_seconds']-
                            ctrl['metrics']['generation_plus_feature_seconds']) if same else None)
                    for key in ('peak_allocated_bytes','peak_reserved_bytes'):
                        outcome['memory'][key]=max(outcome['memory'][key],control['memory'][key])
                cap['metrics'].update(outcome['memory'])
                cap['metrics']['memory_limit_gib']=guard.limit_gib
                cap['metrics']['length_coverage']=outcome['length_coverage']
                features = cap.pop('features')
                target = output_path(Path(str(args.output) + '.features') /
                    f"{source['problem_hash']}.{len(row['attempts'])}.npz")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    np.savez_compressed(stream, **{str(layer): np.asarray([f[str(layer)] for f in features], dtype=np.float32)
                                                 for layer in layers})
                cap['features'] = {'path': str(target), 'sha256': file_hash(target), 'n_steps': len(features)}
                row.update(status='success', capture=cap, steps_fingerprint=digest(cap['steps']), messages=messages)
                row['memory']=outcome['memory']
                doc['last_successful_problem_id']=row['id']
                attempt.update(status='success', usage=cap['usage'], metrics=cap['metrics'],
                               capture_control=cap.get('capture_control'))
            except (Exception,StopRequested) as exc:
                if doc.get('safety_stop'):
                    commit(args,doc)
                    raise RuntimeError('Safety stop preserved; subsequent problems were not processed') from None
                doc['last_successful_problem_id']=previous_success_id
                row['status'] = 'error'
                failed={'status':failure_kind(exc),'error':{'type':type(exc).__name__,'message':str(exc)[:1000]},
                    'memory':guard.snapshot(),'diagnostics':outcome['diagnostics'] if outcome else {},
                    'completion_tokens':outcome['completion_tokens'] if outcome else 0,
                    'length_coverage':outcome['length_coverage'] if outcome else {},
                    'stop_required':failure_kind(exc) in ('memory_limit_exceeded','oom','interrupted','memory_measurement_failed')}
                failed['stop_required'] |= bool(failed['memory'].get('measurement_error'))
                apply_attempt(doc,row,attempt,failed)
                attempt['cost_unknown']=outcome is None
                attempt['wall_seconds_including_storage']=time.perf_counter()-begin
                commit(args,doc)
                if failed['stop_required']:
                    raise RuntimeError('Safety stop preserved; subsequent problems were not processed') from None
            attempt['wall_seconds_including_storage'] = time.perf_counter() - begin
            commit(args, doc)
            print(source['id'], row['status'], flush=True)


def label(args):
    data = validate_data(read(args.data)); cap = verify(read(args.capture))
    rows = validate_capture(cap, data, cap['split'])
    raw_teacher_config = read(args.teacher_config)
    launch_config = raw_teacher_config.get('config', raw_teacher_config)
    if launch_config.get('model') != MODEL or launch_config.get('revision') != REVISION:
        raise ValueError('Teacher must use pinned Llama model/revision')
    if launch_config.get('temperature') != .1 or launch_config.get('top_p') != .9:
        raise ValueError('Teacher sampling must match original .1/.9')
    if not 0 <= args.retries <= 5 or not 0 < args.timeout <= 3600:
        raise ValueError('Invalid bounded retry/timeout')
    from urllib.parse import urlparse
    url = urlparse(args.api_base)
    if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Invalid API URL')
    config = {**configuration(args), 'runtime': runtime(), 'capture_fingerprint': cap['fingerprint'],
              'data_fingerprint': data['fingerprint'], 'teacher_settings': teacher_signature(launch_config, args.seed),
              'teacher_launch_metadata': launch_config,
              'teacher_config_sha256': file_hash(args.teacher_config),
              'benchmark_policy': 'external train-independent benchmarks; differs from legacy evaluation examples'}
    from openai import OpenAI
    with lock(output_path(args.output)):
        doc = checkpoint(args, config, {'schema_version': SCHEMA, 'kind': 'labeled_capture',
            'data_fingerprint': data['fingerprint'], 'split': cap['split'],
            'capture_config': cap['config'], 'records': rows})
        def save():
            doc['teacher_all_attempt_cost'] = attempt_cost([r.get('teacher_stage', {}) for r in doc['records']])
            commit(args, doc)
        save()
        # Assert model identity advertised by server; revision is operator-attested config.
        with OpenAI(base_url=args.api_base, api_key='DUMMY', timeout=args.timeout, max_retries=0) as client:
            if MODEL not in {m.id for m in client.models.list().data}:
                raise ValueError('Teacher API does not advertise expected model')
            sources = {r['id']: r for r in data['splits'][cap['split']]}
            for row in doc['records']:
                target = row.setdefault('teacher_stage', stage())
                steps = row['capture']['steps']
                messages = assessment_messages(sources[row['id']], steps, data['benchmarks'])
                payload = {'model': MODEL, 'messages': messages, 'max_tokens': 2048,
                    'temperature': .1, 'top_p': .9, 'seed': args.seed,
                    'extra_body': {'guided_json': credit_schema(len(steps)), 'guided_decoding_backend': 'outlines'}}
                def validate(result):
                    if result['finish_reason'] != 'stop':
                        raise ValueError('Truncated teacher output')
                    return parse_credits(result['content'], len(steps))
                if run_stage(target, payload, client, save, validate, args.retries, 2):
                    row['teacher'] = {'credits': target['parsed'], 'steps_fingerprint': row['steps_fingerprint'],
                        'usage': target['result']['usage'], 'messages': messages}
                save()
    if any('teacher' not in r for r in doc['records']):
        raise RuntimeError('Invalid teacher labels remain; no uniform fallback was applied')


def teacher_rows(doc, data, split):
    rows = validate_capture(doc, data, split)
    for row in rows:
        if row.get('teacher', {}).get('steps_fingerprint') != row['steps_fingerprint']:
            raise ValueError('Missing or misaligned teacher label')
        target = row.get('teacher_stage', {})
        if target.get('status') != 'success' or target.get('parsed') != row['teacher']['credits']:
            raise ValueError('Teacher stage/projection mismatch')
    return rows


def teacher_signature(config, seed):
    """Compare experimental settings, not job IDs, log paths or package timestamps."""
    server = config.get('server_settings', {})
    return {key: config[key] for key in ('model', 'revision', 'temperature', 'top_p')} | {
        'seed': seed, 'max_tokens': 2048, 'guided_backend': 'outlines',
        'server': {key: server.get(key) for key in ('dtype', 'quantization', 'tensor_parallel_size', 'max_num_seqs', 'gpu_memory_utilization')},
        'max_model_len': config.get('max_model_len', server.get('max_model_len'))}


def capture_signature(doc):
    config = doc.get('capture_config', doc['config'])
    return {k: config[k] for k in ('model', 'revision', 'feature_definition', 'model_config_sha256',
        'weight_index_sha256', 'decomp_prompt_fingerprint', 'capture_implementation_sha256',
        'tokenizer_files_sha256', 'decoding')}


def train(args):
    data = validate_data(read(args.data))
    training, validation = verify(read(args.train)), verify(read(args.validation))
    tr = teacher_rows(training, data, 'train'); va = teacher_rows(validation, data, 'validation')
    if capture_signature(training) != capture_signature(validation):
        raise ValueError('Training/validation feature model mismatch')
    if training['config']['teacher_settings'] != validation['config']['teacher_settings']:
        raise ValueError('Training/validation teacher mismatch')
    target = output_path(args.output)
    # Artifact directory is immutable, not a mutable best_probe.joblib in the root.
    target.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    layers = sorted(int(k) for k in feature_rows(tr[0])[0])
    pipeline, selection = fit_probe(tr, va, layers, args.alpha)
    import joblib
    artifact = target / 'probe.joblib'
    joblib.dump(pipeline, artifact)
    metadata = seal({'schema_version': SCHEMA, 'kind': 'probe_artifact', **selection,
        'data_fingerprint': data['fingerprint'], 'feature_signature': capture_signature(training),
        'train_fingerprint': training['fingerprint'], 'validation_fingerprint': validation['fingerprint'],
        'train_problem_hashes': [r['problem_hash'] for r in tr],
        'validation_problem_hashes': [r['problem_hash'] for r in va],
        'artifact_sha256': file_hash(artifact), 'feature_dimension': pipeline.n_features_in_,
        'training_seconds': time.perf_counter()-begin, 'runtime': runtime(),
        'teacher_settings': training['config']['teacher_settings'],
        'projection_rule': 'clamp-zero-reserve-one-largest-remainder-v1'})
    save_new(target / 'metadata.json', metadata)


def evaluate(args):
    data = validate_data(read(args.data)); cap = verify(read(args.capture))
    if cap['split'] != 'evaluation':
        raise ValueError('Final evaluation must use the frozen evaluation split')
    rows = (teacher_rows(cap, data, 'evaluation') if cap.get('kind') == 'labeled_capture'
            else validate_capture(cap, data, 'evaluation'))
    metadata = verify(read(args.probe / 'metadata.json'))
    artifact = args.probe / 'probe.joblib'
    if metadata['artifact_sha256'] != file_hash(artifact) or metadata['data_fingerprint'] != data['fingerprint']:
        raise ValueError('Artifact/data changed')
    if metadata['feature_signature'] != capture_signature(cap):
        raise ValueError('Incompatible feature definition/model')
    if cap.get('kind') == 'labeled_capture' and metadata['teacher_settings'] != cap['config']['teacher_settings']:
        raise ValueError('Evaluation teacher differs from training teacher')
    forbidden = set(metadata['train_problem_hashes'] + metadata['validation_problem_hashes'])
    if any(r['problem_hash'] in forbidden for r in rows):
        raise ValueError('Probe training/validation overlap with evaluation')
    import joblib
    begin = time.perf_counter()
    pipeline = joblib.load(artifact)  # Load only a trusted locally generated artifact.
    load_seconds = time.perf_counter()-begin
    if pipeline.n_features_in_ != metadata['feature_dimension']:
        raise ValueError('Artifact dimension mismatch')
    reports = []
    for row in rows:
        projection = predict(pipeline, row, metadata['best_layer_idx'])
        item = {'id': row['id'], 'problem_hash': row['problem_hash'], 'steps': row['capture']['steps'],
                'steps_fingerprint': row['steps_fingerprint'], 'probe': projection,
                'decomposition_usage': row['capture']['usage'], 'capture_metrics': row['capture']['metrics'],
                'capture_control': row['capture'].get('capture_control')}
        probe_seconds = projection['inference_seconds'] + projection['projection_seconds']
        item['operational_estimate'] = {
            'probe_planner_generation_calls': 1,
            'probe_planner_seconds_excluding_startup_storage': row['capture']['metrics']['generation_plus_feature_seconds'] + probe_seconds,
            'probe_cpu_feature_read_seconds_offline_only': projection['feature_read_seconds'],
            'note': 'Generation includes capture instrumentation. Offline feature IO, control runs and teacher calls are separate; no speedup claim from this estimate alone.'}
        if row.get('teacher'):
            if row['teacher']['steps_fingerprint'] != row['steps_fingerprint']:
                raise ValueError('Teacher decomposition differs')
            truth = row['teacher']['credits']
            credit_contract(truth, len(row['capture']['steps']))
            difference = [a-b for a, b in zip(projection['credits'], truth)]
            budgets = {method: allocate_words(credits, LEVEL_WORDS[next(r['level'] for r in data['splits']['evaluation'] if r['id']==row['id'])], args.decay)
                       for method, credits in (('teacher', truth), ('probe', projection['credits']))}
            item.update(teacher=row['teacher'], teacher_all_attempt_cost=attempt_cost([row['teacher_stage']]),
                credit_mae=sum(abs(x) for x in difference)/len(difference),
                credit_mse=sum(x*x for x in difference)/len(difference),
                budgets=budgets, budget_l1=sum(abs(a-b) for a,b in zip(budgets['teacher'], budgets['probe'])))
            item['operational_estimate']['teacher_assessment_seconds'] = row['teacher']['usage']['latency_seconds']
            item['operational_estimate']['removed_assessment_minus_probe_cpu_seconds'] = row['teacher']['usage']['latency_seconds'] - probe_seconds
        reports.append(item)
    paired = [r for r in reports if 'teacher' in r]
    summary = {'problems': len(reports), 'paired_problems': len(paired),
        'credit_mae': sum(r['credit_mae'] for r in paired)/len(paired) if paired else None,
        'credit_rmse_problem_weighted': (sum(r['credit_mse'] for r in paired)/len(paired))**.5 if paired else None,
        'budget_l1_mean': sum(r['budget_l1'] for r in paired)/len(paired) if paired else None,
        'probe_inference_seconds': sum(r['probe']['inference_seconds'] for r in reports),
        'probe_projection_seconds': sum(r['probe']['projection_seconds'] for r in reports)}
    save_new(args.output, seal({'schema_version': SCHEMA, 'kind': 'evaluation',
        'data_fingerprint': data['fingerprint'], 'capture_fingerprint': cap['fingerprint'],
        'probe_fingerprint': metadata['fingerprint'], 'probe_load_seconds': load_seconds,
        'decay': args.decay, 'runtime': runtime(), 'summary': summary, 'records': reports}))


def export(args):
    """Schema consumed by unchanged run_solver_compare, NOT the legacy strict checker."""
    data = validate_data(read(args.data)); result = verify(read(args.evaluation))
    manifest = read_manifest(args.eval_manifest)
    if result['data_fingerprint'] != data['fingerprint'] or manifest['fingerprint'] != data['evaluation_manifest_fingerprint']:
        raise ValueError('Evaluation/manifest mismatch')
    mapping = {r['id']: r for r in result['records']}
    if len(mapping) != 500 or len(result['records']) != 500:
        raise ValueError('Solver export requires all 500 unique evaluation IDs')
    stats = []
    for source in manifest['records']:
        row = mapping[source['id']]
        if row['problem_hash'] != problem_hash(source['query']):
            raise ValueError('Evaluation identity mismatch')
        if args.method == 'teacher' and 'teacher' not in row:
            raise ValueError('Teacher export requires paired evaluation')
        credits = credit_contract(row[args.method]['credits'], len(row['steps']))
        usage = row['teacher']['usage'] if args.method == 'teacher' else {
            'prompt_tokens': 0, 'completion_tokens': 0,
            'latency_seconds': row['probe']['inference_seconds'] + row['probe']['projection_seconds']}
        stats.append({**source, 'status': 'success', 'steps': row['steps'], 'credits': credits,
            'decomposition': {**row['decomposition_usage'], 'latency_seconds': row['capture_metrics']['generation_plus_feature_seconds']},
            'assessment': usage,  # Accounting slot; method below identifies a CPU probe, not an LLM call.
            'credit_method': args.method, 'stages': {},
            'planner_latency_seconds': row['capture_metrics']['generation_plus_feature_seconds'] + usage['latency_seconds']})
    save_new(args.output, {'schema_version': 2, 'config': {'kind': 'probe_solver_adapter',
        'model': MODEL, 'revision': REVISION, 'manifest_fingerprint': manifest['fingerprint'],
        'evaluation_fingerprint': result['fingerprint'], 'credit_method': args.method,
        'decay': result['decay'],
        'accounting_note': 'assessment is an accounting slot; probe has zero LLM tokens. stages empty: attempt-cost summaries unavailable; use evaluation/capture/label ledgers.'},
        'stats': stats})


def compare_engine(args):
    """Run the original decomposition request on vLLM for saved HF inputs."""
    data = validate_data(read(args.data)); cap = verify(read(args.capture))
    validate_capture(cap, data, cap['split'])
    raw = read(args.server_config); settings = raw.get('config', raw)
    if settings.get('model') != MODEL or settings.get('revision') != REVISION:
        raise ValueError('Expected pinned 8B vLLM settings')
    config = {**configuration(args), 'capture_fingerprint': cap['fingerprint'],
              'server_settings_operator_attested': settings, 'runtime': runtime()}
    from openai import OpenAI
    from .common import parse_steps
    with lock(output_path(args.output)):
        doc = checkpoint(args, config, {'kind': 'engine_comparison', 'records': []})
        by_id = {r['id']: r for r in doc['records']}
        with OpenAI(base_url=args.api_base, api_key='DUMMY', timeout=600, max_retries=0) as client:
            if MODEL not in {m.id for m in client.models.list().data}:
                raise ValueError('Server model mismatch')
            for row in cap['records']:
                target = by_id.get(row['id'])
                if target is None:
                    target = {'id': row['id'], 'stage': stage()}; doc['records'].append(target)
                payload = {'model': MODEL, 'messages': row['messages'], 'max_tokens': 1024,
                    'temperature': .1, 'top_p': .9, 'seed': cap['config']['seed']}
                def valid(result):
                    if result['finish_reason'] != 'stop':
                        raise ValueError('Truncated decomposition')
                    return parse_steps(result['content'])
                if run_stage(target['stage'], payload, client, lambda: commit(args, doc), valid, 0):
                    result = target['stage']['result']
                    target.update(same_text=result['content']==row['capture']['text'],
                        same_steps=target['stage']['parsed']==row['capture']['steps'],
                        hf_usage=row['capture']['usage'], vllm_usage=result['usage'],
                        note='Engine+capture+transport timing difference; not hidden collection cost in isolation')
                commit(args, doc)
    if any(r['stage']['status'] != 'success' for r in doc['records']):
        raise RuntimeError('vLLM decomposition comparisons failed; inspect attempt ledger')


def solve(args):
    """Run unchanged Solver through a guarded new entry point, bypassing old checker."""
    plan = read(args.planner)
    if plan.get('config', {}).get('kind') != 'probe_solver_adapter':
        raise ValueError('Use a new experiment export as Solver input')
    if plan['config']['decay'] != args.decay:
        raise ValueError('Solver decay must match the frozen Credit/Budget comparison')
    import run_solver_compare as solver
    solver.planner_rows(plan, read_manifest(args.eval_manifest))
    argv = [sys.argv[0], '--manifest', str(args.eval_manifest), '--planner-input', str(args.planner),
        '--output', str(output_path(args.output)), '--model', 'Qwen/QwQ-32B',
        '--revision', '976055f8c83f394f35dbd3ab09a285a984907bd0',
        '--api-base', args.api_base, '--decay', args.decay, '--seed', str(args.seed),
        '--max-tokens', '8192', '--max-model-len', '12288', '--temperature', '.1', '--top-p', '.9',
        '--order', 'alternate', '--evaluator', 'original']
    if args.resume:
        argv.append('--resume')
    previous = sys.argv
    previous_hash = solver.file_hash
    def launch_hash(path):
        if Path(path).resolve() == (LEGACY / 'solver_compare_y4.sbatch').resolve():
            return file_hash(Path(__file__))
        return previous_hash(path)
    try:
        solver.file_hash = launch_hash  # Metadata adapter only, restored even on failure.
        sys.argv = argv
        result = solver.main()
    finally:
        sys.argv = previous
        solver.file_hash = previous_hash
    if result:
        raise RuntimeError(f'Solver exited with status {result}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    def base(name, func):
        p = commands.add_parser(name); p.set_defaults(func=func)
        p.add_argument('--data', type=Path, required=True) if name not in ('prepare','solve') else None
        p.add_argument('--output', type=Path, required=True)
        return p
    p = base('prepare', prepare_command)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--eval-manifest', type=Path, required=True)
    p.add_argument('--validation-fraction', type=float, default=.2); p.add_argument('--seed', type=int, default=42)
    p = base('collect', collect)
    p.add_argument('--split', choices=('train','validation','evaluation'), required=True)
    p.add_argument('--model-path', type=Path, required=True); p.add_argument('--layers', type=int, nargs='+')
    p.add_argument('--capture-control', action='store_true'); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--memory-limit-gib', type=float, default=21.)
    p.add_argument('--resume-after-safety-stop', action='store_true',
                   help='Requires --resume; acknowledge saved safety failure without relaxing its limit')
    p = base('label', label)
    p.add_argument('--capture', type=Path, required=True); p.add_argument('--teacher-config', type=Path, required=True)
    p.add_argument('--api-base', default='http://127.0.0.1:8000/v1'); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timeout', type=float, default=600); p.add_argument('--retries', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p = base('train', train)
    p.add_argument('--train', type=Path, required=True); p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=1.)
    p = base('evaluate', evaluate)
    p.add_argument('--capture', type=Path, required=True); p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--decay', choices=('constant','linear','polynomial','exponential','cosine'), default='constant')
    p = base('export', export)
    p.add_argument('--evaluation', type=Path, required=True); p.add_argument('--eval-manifest', type=Path, required=True)
    p.add_argument('--method', choices=('teacher','probe'), required=True)
    p = base('compare-engine', compare_engine)
    p.add_argument('--capture', type=Path, required=True); p.add_argument('--server-config', type=Path, required=True)
    p.add_argument('--api-base', default='http://127.0.0.1:8000/v1'); p.add_argument('--resume', action='store_true')
    p = base('solve', solve)
    p.add_argument('--planner', type=Path, required=True); p.add_argument('--eval-manifest', type=Path, required=True)
    p.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    p.add_argument('--decay', choices=('constant','linear','polynomial','exponential','cosine'), default='constant')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    output_path(args.output)
    args.func(args)


if __name__ == '__main__':
    main()
