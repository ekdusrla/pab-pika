"""Sequential, checkpointed API calls; no Teacher calls in the Solver path."""
import argparse
import copy
import math
import os
import time
from urllib.parse import urlparse

from ..common import read,verify,seal,save_new,digest,parse_credits
from .plan import approve,require_compute,PROFILES,CACHE
from .hardware import path
from experiment_state import stage,run_stage,atomic_save,attempt_cost,termination_handler
from math_answer_eval import evaluate_answer


def planned(plan,role,session):
    if role=='teacher':
        rows={r['id']:r for r in plan['teacher_requests']}
        blocks=([{'cohort':'smoke','round':0,'seed':42,
                  'ids':[max(rows,key=lambda k:rows[k]['prompt_tokens'])]}]
                if session=='smoke' else plan['teacher_schedule'][session])
        result=[]
        for block in blocks:
            for identifier in block['ids']:
                source=rows[identifier];payload=copy.deepcopy(source['payload']);payload['seed']=block['seed']
                result.append({'id':identifier,'cohort':block['cohort'],'round':block['round'],
                    'seed':block['seed'],'steps_fingerprint':source['steps_fingerprint'],
                    'n_steps':source['n_steps'],'payload':payload,'stage':stage()})
        return result
    rows=plan['solver_requests']
    if session=='smoke':
        vanilla=max(rows,key=lambda r:r['prompt_tokens']['vanilla'])
        _,row,arm=max(((r['prompt_tokens'][a],r,a) for r in rows for a in r['order'] if a!='vanilla'),key=lambda x:x[0])
        pairs=[(vanilla,'vanilla'),(row,arm)]
    else:pairs=[(r,a) for r in rows for a in r['order']]
    return [{'id':r['id'],'arm':a,'answer':r['answer'],'steps_fingerprint':r['steps_fingerprint'],
             'credits':r['credits'].get(a),'budgets_words':r['budgets_words'][a],
             'total_words':None if a=='vanilla' else r['total_words'],
             'payload':r['requests'][a],'stage':stage()} for r,a in pairs]


def validate(result,role,row,smoke=False):
    reason=result['finish_reason'];usage=result['usage'];p=PROFILES[role]
    if reason not in ('stop','length') or ((role=='teacher' or smoke) and reason!='stop'):
        raise ValueError('Abnormal/truncated response (Smoke requires stop)')
    if not isinstance(result.get('content'),str) or not result['content'].strip():raise ValueError('Empty completion')
    for k in ('prompt_tokens','completion_tokens'):
        if type(usage.get(k)) is not int or usage[k]<0:raise ValueError('Missing token accounting')
    if usage['completion_tokens']>p['max_tokens'] or usage['prompt_tokens']+p['max_tokens']>p['context']:
        raise ValueError('Frozen token/context contract violated')
    t=usage.get('latency_seconds')
    if not isinstance(t,(int,float)) or not math.isfinite(t) or t<0:raise ValueError('Missing latency')
    if role=='teacher':return {'credits':parse_credits(result['content'],row['n_steps'])}
    return {'answer_evaluation':evaluate_answer(result['content'],row['answer'],reason,'success',mode='original'),
            'truncated':reason=='length'}


def run(a):
    require_compute();plan=approve(a.audit);profile=PROFILES[a.role]
    server=verify(read(path('server',a.role,a.session)))
    if server['audit_fingerprint']!=plan['fingerprint'] or server['profile']!=profile or server['cache_policy']!=CACHE:
        raise ValueError('Server profile differs')
    url=urlparse(a.api_base)
    if url.scheme!='http' or url.hostname!='127.0.0.1' or url.path!='/v1' or url.username or url.query or url.fragment:
        raise ValueError('Only same-allocation loopback server')
    output=path('responses',a.role,a.session)
    doc={'passed':False,'role':a.role,'session':a.session,'audit_fingerprint':plan['fingerprint'],
         'profile':profile,'cache_policy':CACHE,'server':server,'records':planned(plan,a.role,a.session),
         'retries':0 if a.role=='teacher' else 2,'sdk_retries':0,'timeout_seconds':600,
         'server_restarted_for_session':True,'resumed':False,'started_unix':time.time()}
    save_new(output,seal(doc))
    start=time.perf_counter()
    def save():
        doc['all_attempt_cost']=attempt_cost([r['stage'] for r in doc['records']])
        doc['wall_seconds']=time.perf_counter()-start;atomic_save(output,seal(doc))
    try:
        from openai import OpenAI
        with OpenAI(base_url=a.api_base,api_key='DUMMY',timeout=600,max_retries=0) as client,termination_handler():
            if profile['model'] not in {m.id for m in client.models.list().data}:raise ValueError('Wrong model')
            for index,row in enumerate(doc['records']):
                row['request_order']=index;row['request_fingerprint']=digest(row['payload'])
                row['cache_observation']='prefix cache disabled; grammar/CUDA/disk cache not cleared'
                if not run_stage(row['stage'],row['payload'],client,save,
                    lambda result:validate(result,a.role,row,a.session=='smoke'),
                    retries=doc['retries'],retry_wait=2,server_pid=a.server_pid):
                    # Continue semantic failures to preserve all denominators; no implicit repair.
                    row['failed']=True
            doc['passed']=all(r['stage']['status']=='success' for r in doc['records'])
    except BaseException as exc:
        doc['error']={'type':type(exc).__name__,'message':str(exc)[:2000]};save();raise
    finally:save()
    approve(a.audit)
    doc['worst_case_generation_memory_verified']=False
    doc['observed_completion_tokens']=[r['stage'].get('result',{}).get('usage',{}).get('completion_tokens') for r in doc['records']]
    save()
    if not doc['passed']:raise RuntimeError('API stage failures remain; preserve run and inspect')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--role',choices=PROFILES,required=True)
    p.add_argument('--session',required=True);p.add_argument('--api-base',required=True);p.add_argument('--server-pid',type=int,required=True)
    run(p.parse_args())
