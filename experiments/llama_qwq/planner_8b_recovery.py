"""Opt-in 8B recovery adapter. Shared 70B runner and historical configs stay intact."""
import os
import json
from pathlib import Path
import run_planner as p

CREDIT_INSTRUCTION = (
    'The previous credit assessment failed validation. Reassess ONLY the credits and '
    'difficulty for the SAME supplied sub-questions; do not change or add sub-questions. '
    'Each credit must be a positive integer. Their sum must be exactly 100. '
    'Check the sum before returning the original required JSON schema. '
    'Do not normalize an invalid result after generation.'
)
LEGACY_ADAPTER = '694895ba549d6050aca4615bf81daf624fad36af4cf2be465fe466971688af3a'
POSITIVE_CREDIT_INSTRUCTION = (
    'Repeated credit assessments assigned zero to a supplied step. This violates the '
    'mandatory scoring contract. Keep ALL existing sub-questions unchanged. Allocate '
    'an integer credit of AT LEAST 1 to EVERY step, including reference-information '
    'checks, trivial steps, and simple verification steps. A low-importance step may '
    'receive 1 credit; distribute the remaining credits among the important steps. '
    'Generate the allocation yourself, check every credit >= 1 and the exact sum = 100, '
    'then return the original required JSON schema with difficulty and reasons. '
    'Never give zero, omit a step, or change the decomposition.'
)


def repeated_zero(target, n):
    count = 0
    for attempt in target.get('attempts', []):
        if attempt.get('finish_reason') != 'stop' or attempt.get('status') != 'invalid_response':
            continue
        try:
            data = json.loads(attempt.get('content') or '')
            credits = [data[str(i)]['credit'] for i in range(1, n+1)]
        except (ValueError, KeyError, TypeError):
            continue
        if any(type(c) is int and c == 0 for c in credits):
            count += 1
    return count >= 2


def upgrade_metadata(document):
    meta = document['eight_b_recovery']
    current = p.file_hash(__file__)
    if meta['adapter_sha256'] == current:
        return
    if meta['adapter_sha256'] != LEGACY_ADAPTER:
        raise ValueError('Unaudited recovery adapter boundary')
    extension = document.get('positive_credit_recovery')
    if extension is None:
        extension = {'adapter_sha256': current, 'instruction': POSITIVE_CREDIT_INSTRUCTION,
            'legacy_metadata_fingerprint': p.digest(meta),
            'protected_successes': {r['id']: p.digest(r) for r in document['stats'] if r['status']=='success'},
            'baseline_stages': {r['id']: {k: {'count': len(t['attempts']), 'fingerprint': p.digest(t['attempts'])}
                for k,t in r['stages'].items()} for r in document['stats']},
            'baseline_attempt_cost': p.attempt_cost([t for r in document['stats'] for t in r['stages'].values()])}
        document['positive_credit_recovery'] = extension
    if extension['adapter_sha256'] != current or extension['instruction'] != POSITIVE_CREDIT_INSTRUCTION:
        raise ValueError('Positive-credit recovery code/policy changed')


BASELINE_SCRIPT = 'a4ed59e5c7f229162066d8b204e5e2692bccbf3331b5b573de8287f36ea37e26'
BASELINE_CODE = 'd49f00648880e2cd99b80f152ec13fc8927847369912c1db90e9737d4ec25428'


def credit_failure(row):
    s = row['stages']
    t = s['assessment']
    if s['decomposition']['status'] != 'success' or t['status'] != 'error':
        return False
    result = t.get('result') or {}
    if result.get('finish_reason') != 'stop':
        return False
    try:
        p.parse_credits(result.get('content') or '', len(s['decomposition']['parsed']))
    except (ValueError, TypeError):
        return True
    return False


def prepare(document, manifest, flags):
    meta = document.get('eight_b_recovery')
    policy = {'overlong': flags[0], 'credits': flags[1], 'decomposition_instruction': p.RECOVERY_INSTRUCTION,
              'credit_instruction': CREDIT_INSTRUCTION, 'version': 1}
    if meta is None:
        targets = {}
        for row in document['stats']:
            if row['status'] == 'success':
                continue
            if flags[0] and p.overlong_decomposition(row['stages']['decomposition']):
                targets[row['id']] = 'decomposition'
            elif flags[1] and credit_failure(row):
                targets[row['id']] = 'assessment'
        meta = {'policy': policy, 'adapter_sha256': p.file_hash(__file__),
                'manifest_fingerprint': manifest['fingerprint'], 'targets': targets,
                'protected_successes': {r['id']: p.digest(r) for r in document['stats'] if r['status'] == 'success'},
                'baseline_attempt_cost': p.attempt_cost([s for r in document['stats'] for s in r['stages'].values()]),
                'originals': {}}
        import copy
        meta['originals'] = {r['id']: copy.deepcopy(r) for r in document['stats'] if r['id'] in targets}
        document['eight_b_recovery'] = meta
    if meta['policy'] != policy or meta['manifest_fingerprint'] != manifest['fingerprint']:
        raise ValueError('8B recovery policy/code/manifest changed')
    upgrade_metadata(document)
    verify(document)
    return meta


def verify(document):
    meta = document.get('eight_b_recovery')
    if not meta:
        return
    rows = {r['id']: r for r in document['stats']}
    if len(rows) != 500 or len(document['stats']) != 500:
        raise ValueError('Duplicate or missing recovery IDs')
    for identifier, signature in meta['protected_successes'].items():
        if p.digest(rows[identifier]) != signature:
            raise ValueError('Successful record changed: ' + identifier)
    extension = document.get('positive_credit_recovery')
    if extension:
        if extension['legacy_metadata_fingerprint'] != p.digest(meta):
            raise ValueError('Legacy recovery metadata changed')
        for identifier, signature in extension['protected_successes'].items():
            if p.digest(rows[identifier]) != signature:
                raise ValueError('Previously recovered successful record changed: ' + identifier)
        for identifier, stages in extension['baseline_stages'].items():
            for name, baseline in stages.items():
                if p.digest(rows[identifier]['stages'][name]['attempts'][:baseline['count']]) != baseline['fingerprint']:
                    raise ValueError('Prior recovery attempt changed')
    for identifier, original in meta['originals'].items():
        for name, stage in original['stages'].items():
            current = rows[identifier]['stages'][name]
            if current['attempts'][:len(stage['attempts'])] != stage['attempts']:
                raise ValueError('Original attempts changed')
        if meta['targets'][identifier] == 'assessment' and rows[identifier]['stages']['decomposition'] != original['stages']['decomposition']:
            raise ValueError('Successful decomposition changed')


def install(runner):
    flags = (os.getenv('PLANNER_RECOVER_OVERLONG') == '1', os.getenv('PLANNER_RECOVER_CREDITS') == '1')
    if not any(flags):
        return
    original_refresh = runner.refresh_planner
    original_verify = runner.validate_recovery_history

    def validate_history(document):
        original_verify(document)
        verify(document)

    def refresh(document):
        original_refresh(document)
        meta = document.get('eight_b_recovery')
        if meta:
            stages = [{'attempts': [a for a in s['attempts'] if a.get('recovery_adapter') in (meta['adapter_sha256'], p.file_hash(__file__))]}
                      for r in document['stats'] for s in r['stages'].values()]
            document['summary']['eight_b_recovery'] = {'targets': meta['targets'],
                'additional_attempt_cost': p.attempt_cost(stages),
                'success': sum(r['status'] == 'success' for r in document['stats'] if r['id'] in meta['targets'])}

    def load(args, config, manifest):
        if not args.resume or args.limit != 500:
            raise ValueError('8B recovery requires RESUME=1 and LIMIT=500')
        if config['model'] != 'meta-llama/Llama-3.1-8B-Instruct':
            raise ValueError('8B recovery only')
        # Only the audited launcher boundary can be crossed. All other fields remain strict.
        try:
            doc = p.load_checkpoint(args.output, config, True, recover=False)
        except ValueError:
            baseline = {**config, 'launch_script_sha256': BASELINE_SCRIPT, 'code_sha256': BASELINE_CODE}
            doc = p.load_checkpoint(args.output, baseline, True, recover=False)
        if doc is None:
            raise ValueError('Recovery checkpoint missing')
        from run_solver_compare import planner_rows
        planner_rows(doc, manifest)
        prepare(doc, manifest, flags)
        return doc

    def run(args, manifest, document, client, save, tokenizer=None):
        meta = prepare(document, manifest, flags)
        prompts, _, _ = p.original_definitions()
        by_id = {r['id']: r for r in manifest['records']}
        benchmarks = '\n'.join(f"\nLevel {by_id[i]['level']} Example:\nQ: {by_id[i]['query']}\nA: {by_id[i]['solution']}\n" for i in manifest['benchmark_ids'])
        for row in document['stats']:
            if row['id'] not in meta['targets']:
                continue
            original = meta['originals'][row['id']]
            for name in ('decomposition', 'assessment'):
                target = row['stages'][name]
                if target['status'] == 'success':
                    continue
                if name == 'assessment' and row['stages']['decomposition']['status'] != 'success':
                    break
                values = dict(domain='math', problem=row['query'], level=row['level'], benchmarks=benchmarks,
                              steps=row['stages']['decomposition'].get('parsed', []))
                template = prompts['break_down' if name == 'decomposition' else 'evaluate_question_level_batch']
                messages = [{'role': k, 'content': template[k].format(**values)} for k in ('system','user')]
                strong = name == 'assessment' and repeated_zero(target, len(values['steps']))
                if name == 'decomposition' or meta['targets'][row['id']] == 'assessment' or strong:
                    messages += [{'role': 'assistant', 'content': target['result']['content'] if strong else original['stages'][name]['result']['content']},
                                 {'role': 'user', 'content': p.RECOVERY_INSTRUCTION if name == 'decomposition' else POSITIVE_CREDIT_INSTRUCTION if strong else CREDIT_INSTRUCTION}]
                cap = 1024 if name == 'decomposition' else 2048
                if tokenizer is not None:
                    count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                    if count + cap > args.max_model_len:
                        target.update(status='error', validation_error={'type':'ContextLimit','prompt_tokens':count,'max_tokens':cap})
                        save()
                        break
                payload = dict(model=args.model, messages=messages, max_tokens=cap, temperature=.1, top_p=.9, seed=args.seed)
                if name == 'assessment':
                    payload['extra_body'] = {'guided_json':p.credit_schema(len(values['steps'])), 'guided_decoding_backend':'outlines'}
                def validate(result):
                    if result['finish_reason'] != 'stop':
                        raise ValueError('Planner response truncated or abnormal finish')
                    return p.parse_steps(result['content'] or '') if name == 'decomposition' else p.parse_credits(result['content'] or '',len(values['steps']))
                attempt_start = len(target['attempts'])
                def save_call():
                    for attempt in target['attempts'][attempt_start:]:
                        attempt.update(recovery_adapter=p.file_hash(__file__), credit_strategy='repeated_zero_positive_minimum' if strong else 'standard', purpose='overlong_decomposition_recovery' if name=='decomposition' else 'credit_recovery' if meta['targets'][row['id']]=='assessment' else 'post_recovery_assessment', request=payload)
                    if target['status']=='success':
                        target.pop('validation_error',None)
                    save()
                p.run_stage(target,payload,client,save_call,validate,args.retries,args.retry_wait,args.server_pid)
            save()
            print(row['id'],row['status'],flush=True)
        return all(r['status']=='success' for r in document['stats'])

    runner.refresh_planner = refresh
    runner.validate_recovery_history = validate_history
    runner.load_planner_checkpoint = load
    runner.run_planner = run
