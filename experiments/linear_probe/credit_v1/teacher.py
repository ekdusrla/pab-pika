"""Same PAB Credit requests against vLLM; never generates Decomposition."""
import argparse
import copy
import math
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from ..common import (MODEL,read,verify,seal,save_new,output_path,file_hash,digest,
                      assessment_messages,credit_schema,parse_credits)
from ..cli import checkpoint,teacher_signature,teacher_rows
from .audit import SOURCE,OUT,SERVER,approve,require_compute,capture_projection
from experiment_state import lock,atomic_save,stage,run_stage,attempt_cost,termination_handler


def payload_for(source,row,data):
    return {'model':MODEL,'messages':assessment_messages(source,row['capture']['steps'],data['benchmarks']),
        'max_tokens':2048,'temperature':.1,'top_p':.9,'seed':42,
        'extra_body':{'guided_json':credit_schema(len(row['capture']['steps'])),
                      'guided_decoding_backend':'outlines'}}


def expected_signature():return teacher_signature(SERVER,42)


def check_server_config(path):
    config=read(path);config=config.get('config',config)
    if teacher_signature(config,42)!=expected_signature():raise ValueError('Teacher server settings differ from frozen PAB')
    return config


def label_one(target,payload,n,client,save,server_pid):
    if target.get('attempts') and target['attempts'][-1]['status']=='invalid_response':
        raise ValueError('Invalid semantic response preserved: investigate before any retry')
    def validate(result):
        if result['finish_reason']!='stop':raise ValueError('Teacher completion truncated or abnormal')
        usage=result['usage']
        if any(type(usage.get(k)) is not int or usage[k]<0 for k in ('prompt_tokens','completion_tokens')):
            raise ValueError('Teacher token accounting missing')
        if usage['completion_tokens']>2048 or usage['prompt_tokens']+2048>8192:
            raise ValueError('Teacher exceeded the frozen token/context budget')
        if not isinstance(usage.get('latency_seconds'),(int,float)) or not math.isfinite(usage['latency_seconds']):
            raise ValueError('Teacher latency accounting missing')
        return parse_credits(result['content'],n)
    return run_stage(target,payload,client,save,validate,retries=2,retry_wait=2,server_pid=server_pid)


def frozen_probe(path,audit):
    if not verify(read(path/'status.json')).get('passed'):raise ValueError('Probe training did not complete')
    meta=verify(read(path/'metadata.json'))
    if meta.get('audit_fingerprint')!=audit['fingerprint'] or not meta.get('selection_frozen'):
        raise ValueError('Freeze Train/Validation model selection before Evaluation labels')
    if file_hash(path/'probe.joblib')!=meta['artifact_sha256']:raise ValueError('Probe changed')
    return meta


def run(a):
    require_compute();plan=approve(a.audit);config=check_server_config(a.server_config)
    url=urlparse(a.api_base)
    if url.scheme!='http' or url.hostname!='127.0.0.1' or url.username or url.password or url.path!='/v1':
        raise ValueError('Use the same-allocation loopback vLLM endpoint')
    if a.mode=='evaluation':frozen_probe(a.probe,plan)
    from openai import OpenAI
    data=verify(read(SOURCE/'data.json'))
    with OpenAI(base_url=a.api_base,api_key='DUMMY',timeout=600,max_retries=0) as client,termination_handler():
        if MODEL not in {m.id for m in client.models.list().data}:raise ValueError('Teacher model identity differs')
        if a.mode=='smoke':
            candidates=[(r['prompt_tokens'],s,r['id']) for s in ('train','validation') for r in plan['contexts'][s]]
            _,split,identifier=max(candidates,key=lambda x:(x[0],x[1],x[2]))
            cap=verify(read(plan['captures'][split]));row=next(r for r in cap['records'] if r['id']==identifier)
            source=next(r for r in data['splits'][split] if r['id']==identifier)
            path=output_path(OUT/f'smoke_{os.environ["SLURM_JOB_ID"]}.json')
            doc={'kind':'teacher_observed_case_smoke','passed':False,'audit_fingerprint':plan['fingerprint'],
                'teacher_settings':expected_signature(),'server_config':config,'split':split,'id':identifier,
                'steps_fingerprint':row['steps_fingerprint'],'teacher_stage':stage(),
                'hf_context_prompt_tokens':max(candidates,key=lambda x:(x[0],x[1],x[2]))[0],
                'worst_case_2048_tokens_verified':False}
            save_new(path,seal(doc))
            def save():atomic_save(path,seal(doc))
            ok=label_one(doc['teacher_stage'],payload_for(source,row,data),len(row['capture']['steps']),client,save,a.server_pid)
            doc['passed']=ok;doc['all_attempt_cost']=attempt_cost([doc['teacher_stage']]);save()
            if not ok:raise RuntimeError('Teacher Smoke failed; response/costs saved')
            print('TEACHER SMOKE RESPONSE PASS (observed case only; memory gate is separate)',flush=True)
            return
        splits=('evaluation',) if a.mode=='evaluation' else ('train','validation')
        for split in splits:
            cap=verify(read(plan['captures'][split]));sources={r['id']:r for r in data['splits'][split]}
            path=output_path(OUT/(split+'_labeled.json'))
            settings={'kind':'recovered_teacher_credit_v1','audit_fingerprint':plan['fingerprint'],
                'capture_fingerprint':cap['fingerprint'],'data_fingerprint':data['fingerprint'],
                'teacher_settings':expected_signature(),'seed':42,'retries':2,'timeout':600}
            with lock(path):
                doc=checkpoint(argparse.Namespace(output=path,resume=a.resume and path.exists()),settings,
                    {'kind':'labeled_capture','split':split,'data_fingerprint':data['fingerprint'],
                     'capture_config':cap['config'],'records':copy.deepcopy(cap['records']),'sessions':[]})
                doc['sessions'].append({'slurm_job_id':os.environ['SLURM_JOB_ID'],'server_config':config,
                    'actual_engine':'vLLM OpenAI API / guided_json / outlines',
                    'server_config_sha256':file_hash(a.server_config),'started_unix':time.time()})
                def save():
                    for r in doc['records']:
                        if capture_projection(r)!=plan['capture_projections'][split][r['id']]:
                            raise ValueError('Final Decomposition/features were modified')
                    doc['teacher_all_attempt_cost']=attempt_cost([r.get('teacher_stage',{}) for r in doc['records']])
                    atomic_save(path,seal({k:v for k,v in doc.items() if k!='fingerprint'}))
                save()
                for row in doc['records']:
                    target=row.setdefault('teacher_stage',stage());payload=payload_for(sources[row['id']],row,data)
                    context=next(r for r in plan['contexts'][split] if r['id']==row['id'])
                    if digest(payload['messages'])!=context['messages_sha256']:raise ValueError('Teacher prompt changed')
                    if not label_one(target,payload,len(row['capture']['steps']),client,save,a.server_pid):
                        save();raise RuntimeError('Invalid Teacher response; no fallback or subsequent split')
                    row['teacher']={'credits':target['parsed'],'steps_fingerprint':row['steps_fingerprint'],
                        'usage':target['result']['usage'],'messages':payload['messages'],
                        'tokenization_comparison':{'hf_context_tokens':context['prompt_tokens'],
                            'vllm_prompt_tokens':target['result']['usage']['prompt_tokens'],
                            'vllm_minus_hf':target['result']['usage']['prompt_tokens']-context['prompt_tokens']}}
                    save()
                teacher_rows(verify(read(path)),data,split)
                print('TEACHER '+split.upper()+' PASS: '+str(len(doc['records']))+' aligned labels',flush=True)
    approve(a.audit)  # Prove original inputs/NPZs were not changed.


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--server-config',type=Path,required=True)
    p.add_argument('--api-base',required=True);p.add_argument('--server-pid',type=int,required=True)
    p.add_argument('--mode',choices=('smoke','teacher','evaluation'),required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--probe',type=Path,default=OUT/'probe_v1')
    run(p.parse_args())
