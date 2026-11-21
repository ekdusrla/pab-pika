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
from ..followup_v1.client import planned as frozen_planned,validate as frozen_validate


def planned(plan,role,session):
    if role!='solver':raise ValueError('Solver-only namespace')
    if session!='smoke':return copy.deepcopy(frozen_planned(plan,'solver','main'))
    rows=frozen_planned(plan,'solver','smoke');result=[]
    # Two short infrastructure checks, then two unchanged full-cap qualification requests.
    for phase in ('short','qualification'):
        for source in rows:
            row=copy.deepcopy(source);row['phase']=phase
            if phase=='short':row['payload']['max_tokens']=128
            result.append(row)
    return result


def validate(result,role,row,smoke=False):
    parsed=frozen_validate(result,'solver',row,smoke and row.get('phase')=='qualification')
    if result['usage']['completion_tokens']>row['payload']['max_tokens']:raise ValueError('Request cap exceeded')
    if smoke and row.get('phase')=='qualification' and parsed['answer_evaluation']['status'] not in ('correct','incorrect'):
        raise ValueError('Full-cap qualification did not yield a parseable final answer')
    parsed['short_smoke_final_answer_not_required']=row.get('phase')=='short'
    return parsed


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
