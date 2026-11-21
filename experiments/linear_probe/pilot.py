"""Five-question Pilot reports and QwQ runner, independent of the 500-question export."""
import argparse
import copy
from pathlib import Path

from .common import (read, verify, seal, save_new, digest, runtime, output_path, credit_contract,
                     allocate_words, LEVEL_WORDS, original_definitions)
from .pilot_data import validate_pilot
from .cli import checkpoint, commit
from experiment_state import stage, run_stage, lock, attempt_cost, termination_handler
from run_solver_compare import messages_for
from math_answer_eval import evaluate_answer, evaluator_info

SOLVER = 'Qwen/QwQ-32B'
SOLVER_REVISION = '976055f8c83f394f35dbd3ab09a285a984907bd0'


def report(data, evaluation):
    validate_pilot(data); verify(evaluation)
    if evaluation['data_fingerprint'] != data['fingerprint']:
        raise ValueError('Pilot evaluation/data mismatch')
    expected = {r['id'] for r in data['splits']['evaluation']}
    if len(evaluation['records']) != 5 or {r['id'] for r in evaluation['records']} != expected:
        raise ValueError('Require all five independent Pilot problems')
    result = copy.deepcopy(evaluation)
    signed, absolute, squared, budget_absolute = [], [], [], []
    for row in result['records']:
        n = len(row['steps'])
        teacher = credit_contract(row['teacher']['credits'], n)
        probe = credit_contract(row['probe']['credits'], n)
        row['step_differences'] = []
        for i, (t, p) in enumerate(zip(teacher, probe)):
            diff = p-t
            budget_diff = row['budgets']['probe'][i]-row['budgets']['teacher'][i]
            row['step_differences'].append({'step_index':i,'teacher_credit':t,'probe_credit':p,
                'probe_minus_teacher_credit':diff, 'raw_probe_credit':row['probe']['raw'][i],
                'teacher_budget_words':row['budgets']['teacher'][i],
                'probe_budget_words':row['budgets']['probe'][i],
                'probe_minus_teacher_budget_words':budget_diff})
            signed.append(diff); absolute.append(abs(diff)); squared.append(diff*diff)
            budget_absolute.append(abs(budget_diff))
    result['summary'].update(step_micro_mae=sum(absolute)/len(absolute),
        step_micro_rmse=(sum(squared)/len(squared))**.5,
        step_micro_bias=sum(signed)/len(signed), budget_step_micro_mae=sum(budget_absolute)/len(budget_absolute),
        pilot_scope='Five held-out external MATH train problems, not final MATH-500 evaluation')
    result.pop('fingerprint',None)
    return seal(result)


def prepare_solver(data, evaluation):
    comparison = report(data, evaluation)
    if comparison['decay'] != 'constant':
        raise ValueError('This Pilot preserves the original constant Budget schedule')
    sources = {r['id']:r for r in data['splits']['evaluation']}
    _, instruction, output_format = original_definitions()
    rows = []
    for index, row in enumerate(comparison['records']):
        source = sources[row['id']]
        if row['problem_hash'] != source['problem_hash'] or row['steps_fingerprint'] != digest(row['steps']):
            raise ValueError('Pilot source/decomposition mismatch')
        item = {**source,'steps':row['steps']}
        requests = {}
        for method in ('teacher','probe'):
            budgets = allocate_words(row[method]['credits'], LEVEL_WORDS[source['level']], comparison['decay'])
            if budgets != row['budgets'][method]:
                raise ValueError('Budget configuration mismatch')
            requests[method] = {'model':SOLVER,
                'messages':messages_for(item,'weighted_pab',budgets,instruction,output_format),
                'max_tokens':8192,'temperature':.1,'top_p':.9,'seed':42}
        rows.append({'id':row['id'],'answer':source['answer'],'level':source['level'],
            'problem_hash':source['problem_hash'],'steps_fingerprint':row['steps_fingerprint'],
            'steps':row['steps'], 'requests':requests,
            'order':['teacher','probe'] if index%2==0 else ['probe','teacher']})
    return seal({'kind':'linear_probe_pilot_solver_requests','data_fingerprint':data['fingerprint'],
        'evaluation_fingerprint':evaluation['fingerprint'],'rows':rows,
        'settings':{'model':SOLVER,'revision':SOLVER_REVISION,'dtype':'float16',
            'tensor_parallel_size':4,'max_model_len':12288,'max_tokens':8192,
            'temperature':.1,'top_p':.9,'seed':42,'decay':comparison['decay'],
            'instruction':instruction,'output_format':output_format},
        'note':'Exactly ten paired Solver requests; no extra Vanilla requests.'})


def check_server(config):
    settings = config.get('config',config)
    server = settings.get('server_settings',{})
    if settings.get('model') != SOLVER or settings.get('revision') != SOLVER_REVISION or \
       server.get('dtype') != 'float16' or server.get('tensor_parallel_size') != 4 or \
       server.get('quantization') is not None or server.get('max_model_len',settings.get('max_model_len')) != 12288:
        raise ValueError('Require unchanged QwQ-32B FP16 TP=4 context=12288 server')
    return settings


def solver_summary(doc):
    result = {'expected_problems':5,'expected_requests':10,'methods':{}}
    for method in ('teacher','probe'):
        stages = [r['methods'][method] for r in doc['records']]
        success = [s for s in stages if s['status']=='success']
        correct = sum(s.get('evaluation',{}).get('status')=='correct' for s in stages)
        def total(key):
            values = [s['result']['usage'].get(key) for s in success]
            known = [v for v in values if v is not None]
            return {'known_sum':sum(known),'missing_success_records':len(values)-len(known)}
        result['methods'][method]={'api_success':len(success),'correct':correct,'accuracy_all_5':correct/5,
            'normal_completion':sum(s['result']['finish_reason']=='stop' for s in success),
            'truncated':sum(s['result']['finish_reason']=='length' for s in success),
            'incorrect':sum(s.get('evaluation',{}).get('status')=='incorrect' for s in stages),
            'undetermined':sum(s.get('evaluation',{}).get('status') not in ('correct','incorrect') for s in stages),
            'completion_tokens':total('completion_tokens'),'prompt_tokens':total('prompt_tokens'),
            'latency_seconds':total('latency_seconds'),'all_attempt_cost':attempt_cost(stages)}
    paired = []
    for row in doc['records']:
        a,b = (row['methods'][m] for m in ('teacher','probe'))
        if a['status']==b['status']=='success' and 'evaluation' in a and 'evaluation' in b:
            paired.append({'id':row['id'],'teacher_evaluation':a['evaluation'],'probe_evaluation':b['evaluation'],
                'probe_minus_teacher':{k:(b['result']['usage'][k]-a['result']['usage'][k])
                    if a['result']['usage'].get(k) is not None and b['result']['usage'].get(k) is not None else None
                    for k in ('prompt_tokens','completion_tokens','latency_seconds')}})
    result['paired']=paired
    return result


def run_solver(a):
    requests = verify(read(a.requests))
    if requests.get('kind') != 'linear_probe_pilot_solver_requests' or len(requests['rows']) != 5 or len({r['id'] for r in requests['rows']}) != 5:
        raise ValueError('Require frozen five-problem Pilot requests')
    settings = check_server(read(a.server_config))
    # CPU tokenization and evaluator validation precede all model API requests.
    from transformers import AutoTokenizer
    if a.tokenizer_path.resolve().name != SOLVER_REVISION:
        raise ValueError('Use the pinned QwQ tokenizer snapshot')
    tok = AutoTokenizer.from_pretrained(a.tokenizer_path,local_files_only=True)
    for row in requests['rows']:
        for payload in row['requests'].values():
            if len(tok.apply_chat_template(payload['messages'],tokenize=True,add_generation_prompt=True))+8192 > 12288:
                raise ValueError('Solver context exceeded; do not silently reduce budgets')
    evaluator = evaluator_info('original')
    config = {'requests_fingerprint':requests['fingerprint'],'server_settings':settings,
              'runtime':runtime(),'api_base':a.api_base,'evaluator':evaluator,'retries':a.retries}
    from openai import OpenAI
    with lock(output_path(a.output)):
        doc=checkpoint(a,config,{'kind':'linear_probe_pilot_solver','records':[
            {'id':row['id'],'answer':row['answer'],'steps_fingerprint':row['steps_fingerprint'],
             'methods':{m:stage() for m in ('teacher','probe')}} for row in requests['rows']]})
        def save():
            doc['summary']=solver_summary(doc); commit(a,doc)
        save()
        with termination_handler(), OpenAI(base_url=a.api_base,api_key='DUMMY',timeout=600,max_retries=0) as client:
            if SOLVER not in {m.id for m in client.models.list().data}:
                raise ValueError('Server advertises another model')
            for row, request in zip(doc['records'],requests['rows']):
                for method in request['order']:
                    target=row['methods'][method]
                    if run_stage(target,request['requests'][method],client,save,lambda response:{},a.retries,2):
                        raw=target['result']
                        target['evaluation']=evaluate_answer(raw['content'],row['answer'],raw['finish_reason'],'success',mode='original')
                    save()
    if any(t['status']!='success' for r in doc['records'] for t in r['methods'].values()):
        raise RuntimeError('Pilot Solver requests failed; attempt ledger preserved')


def preflight_label(a):
    from transformers import AutoTokenizer
    from .common import assessment_messages, REVISION
    from .learning import validate_capture
    data=validate_pilot(read(a.data));cap=verify(read(a.capture))
    rows=validate_capture(cap,data,cap['split'])
    if a.tokenizer_path.resolve().name != REVISION:raise ValueError('Use pinned Llama tokenizer')
    tok=AutoTokenizer.from_pretrained(a.tokenizer_path,local_files_only=True)
    sources={r['id']:r for r in data['splits'][cap['split']]}
    checks=[]
    for row in rows:
        messages=assessment_messages(sources[row['id']],row['capture']['steps'],data['benchmarks'])
        count=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        checks.append({'id':row['id'],'prompt_tokens':count,'max_completion_tokens':2048,'fits':count+2048<=8192})
    save_new(a.output,seal({'kind':'pilot_teacher_context_preflight','data_fingerprint':data['fingerprint'],
        'capture_fingerprint':cap['fingerprint'],'rows':checks,'passed':all(r['fits'] for r in checks)}))
    if not all(r['fits'] for r in checks):raise ValueError('Teacher context exceeded; no truncation applied')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for name in ('compare','solver-prepare'):
        q=sub.add_parser(name); q.add_argument('--data',type=Path,required=True)
        q.add_argument('--evaluation',type=Path,required=True); q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('solver-run'); q.add_argument('--requests',type=Path,required=True)
    q.add_argument('--server-config',type=Path,required=True); q.add_argument('--tokenizer-path',type=Path,required=True)
    q.add_argument('--api-base',default='http://127.0.0.1:8000/v1'); q.add_argument('--output',type=Path,required=True)
    q.add_argument('--resume',action='store_true'); q.add_argument('--retries',type=int,default=2)
    q=sub.add_parser('preflight-label');q.add_argument('--data',type=Path,required=True)
    q.add_argument('--capture',type=Path,required=True);q.add_argument('--tokenizer-path',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='solver-run':
        if not 0 <= a.retries <= 5: raise ValueError('retries must be 0..5')
        run_solver(a)
    elif a.command=='preflight-label':
        preflight_label(a)
    else:
        data=validate_pilot(read(a.data)); evaluation=verify(read(a.evaluation))
        save_new(a.output,(report if a.command=='compare' else prepare_solver)(data,evaluation))


if __name__=='__main__':
    main()
