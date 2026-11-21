"""One frozen overlong repair, preserving all original artifacts and histories."""
import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

from ..common import read,verify,seal,output_path,file_hash,digest,runtime
from ..cli import checkpoint
from ..pilot_smoke import allocation
from ..safety import (MemoryGuard,require_device,guarded_generate,apply_attempt,
                      acknowledge_resume,failure_kind)
from ..recovery_v1.audit import verify_protected
from ..recovery_v1.recover import messages_for_recovery
from .audit import RUN,policy,code_hashes,overlong,eos_ids,inspect_capture,inspect_inputs,audit,verify_feature_directory
from experiment_state import lock,atomic_save,StopRequested,termination_handler


def approval(path):
    doc=verify(read(path))
    if doc.get('kind')!='validation_evaluation_recovery_audit_v2' or not doc.get('passed'):
        raise ValueError('Successful V2 compute-node audit required')
    if doc['frozen_policy']!=policy() or doc['code_sha256']!=code_hashes():
        raise ValueError('Policy/code changed after audit; do not tune Evaluation post hoc')
    for p,h in doc['input_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Frozen source changed: '+p)
    for split,name in [('train','train_recovered.json'),('validation','validation_capture.json')]:
        verify_protected(verify(read(RUN/name)),doc[split]['protected'],True)
    return doc


def classification(row):
    if row['attempts'][0]['status']=='success':return 'initial_success'
    if row['status']=='success':return 'repaired_success'
    if row['status']=='running':return 'pending'
    if any(a.get('purpose') in ('overlong_recovery_v1','overlong_recovery_v2') for a in row['attempts']):return 'repair_failed'
    return 'initial_failure_unrepaired'


def cost_calls(row):
    """One entry per generation, without counting a failed control projection twice."""
    calls=[]
    for i,a in enumerate(row['attempts']):
        if 'calls' in a:
            calls.extend({'attempt_number':i+1,**call} for call in a['calls']);continue
        ctrl=a.get('capture_control_attempt')
        if ctrl:
            calls.append({'attempt_number':i+1,'role':'control','usage':{
                'prompt_tokens':ctrl.get('prompt_tokens'),'completion_tokens':ctrl.get('completion_tokens'),
                'latency_seconds':ctrl.get('diagnostics',{}).get('generation_seconds')},
                'wall_seconds':ctrl.get('generation_and_feature_wall_seconds'),'status':ctrl['status'],
                'extra_forward_passes':ctrl.get('extra_forward_passes',0)})
            if ctrl['status']!='success':continue
        calls.append({'attempt_number':i+1,'role':'capture','usage':a.get('usage'),
            'wall_seconds':a.get('wall_seconds_including_storage'),'status':a['status'],
            'cost_unknown':a.get('cost_unknown',False),'extra_forward_passes':a.get('observations',{}).get('extra_forward_passes',0)})
    return calls


def cost_summary(row):
    calls=cost_calls(row)
    return {'actual_generation_calls':len(calls),'control_calls':sum(c['role']=='control' for c in calls),
        'observed_prompt_tokens':sum((c.get('usage') or {}).get('prompt_tokens') or 0 for c in calls),
        'observed_completion_tokens':sum((c.get('usage') or {}).get('completion_tokens') or 0 for c in calls),
        'has_unknown_or_inflight_cost':any(c.get('cost_unknown',False) or c['status']=='running' or
            (c.get('usage') or {}).get('prompt_tokens') is None for c in calls)}


def repair_messages(doc,data,tok):
    by_id={r['id']:r for r in data['splits'][doc['split']]}
    return {r['id']:messages_for_recovery(by_id[r['id']],overlong(r,tok,eos_ids(doc)))
            for r in doc['records'] if r['status']!='success'}


def repair(a,split):
    frozen=approval(a.audit);source_path=RUN/(split+'_capture.json')
    output=output_path(RUN/(split+'_recovered.json'))
    source=verify(read(source_path));data=verify(read(RUN/'data.json'))
    if Path(str(source_path)+'.tmp').exists():raise ValueError('Source .tmp requires review')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(frozen['model_path'],local_files_only=True)
    source_check=inspect_capture(source,data,split,tok,complete=False)
    verify_feature_directory(source_path,source)
    messages=repair_messages(source,data,tok)
    proposed=copy.deepcopy(source);proposed.pop('fingerprint',None)
    proposed['config']={**proposed['config'],'output':str(output),'recovery_v2':{
        'audit_fingerprint':frozen['fingerprint'],'source_sha256':file_hash(source_path),'policy':policy(),
        'code_sha256':code_hashes()}}
    proposed['recovery_v2']={'source_path':str(source_path),'source_sha256':file_hash(source_path),
        'protected_successes':source_check['protected'],'accepted_repaired_rows':{},'status':'pending',
        'policy':policy(),'original_config':source['config']}
    with lock(source_path),lock(output),termination_handler():
        if file_hash(source_path)!=proposed['recovery_v2']['source_sha256']:raise ValueError('Source changed')
        args=argparse.Namespace(output=output,resume=a.resume and output.exists())
        doc=checkpoint(args,proposed['config'],proposed)
        acknowledge_resume(doc,args.resume,a.ack_safety_stop if args.resume else False)
        def save():
            if file_hash(source_path)!=doc['recovery_v2']['source_sha256']:raise ValueError('Source changed')
            verify_protected(doc,{**doc['recovery_v2']['protected_successes'],**doc['recovery_v2']['accepted_repaired_rows']})
            doc['recovery_v2']['problem_outcomes']=[{'id':r['id'],'category':classification(r),
                'status':r['status'],'cost_summary':cost_summary(r),'calls':cost_calls(r)} for r in doc['records']]
            atomic_save(output,seal({k:v for k,v in doc.items() if k!='fingerprint'}))
        pending=[r for r in doc['records'] if r['status']!='success']
        for row in pending:
            if row['id'] not in messages:raise ValueError('Unexpected pending row')
            for attempt in row['attempts']:
                if attempt.get('purpose')=='overlong_recovery_v2' and attempt['status'] not in (
                    'running','interrupted','oom','memory_limit_exceeded','memory_measurement_failed'):
                    raise ValueError('Completed invalid repair/storage failure: inspect; do not retry unchanged')
        save()
        if pending:
            import torch
            import numpy as np
            from transformers import AutoModelForCausalLM
            session={'purpose':'overlong_recovery_v2','allocation':allocation(),'status':'preflight'}
            doc.setdefault('sessions',[]).append(session);save();guard=None;started=time.perf_counter()
            try:
                session['gpu']=require_device(torch);session['runtime']=runtime()
                guard=MemoryGuard(torch.cuda,21.);guard.begin()
                model=AutoModelForCausalLM.from_pretrained(frozen['model_path'],local_files_only=True,
                    torch_dtype=torch.float16,attn_implementation='eager').to('cuda').eval()
                torch.cuda.synchronize();guard.check('model_load')
                session.update(status='loaded',model_load_seconds=time.perf_counter()-started,
                    memory=guard.snapshot(),generation_config=model.generation_config.to_dict(),actual_model_config=model.config.to_dict())
                save()
            except (Exception,StopRequested) as exc:
                session.update(status='error',error={'type':type(exc).__name__,'message':str(exc)[:1500]})
                doc['safety_stop']={'reason':failure_kind(exc),'phase':'model_load','error':session['error'],
                    'last_successful_problem_id':doc.get('last_successful_problem_id'),'memory':guard.snapshot() if guard else None}
                save();raise RuntimeError('Model loading failed; diagnostics saved') from None
            for row in pending:
                for old in row['attempts']:
                    if old['status']=='running':old.update(status='interrupted',cost_unknown=True)
                request=messages[row['id']]
                attempt={'status':'running','purpose':'overlong_recovery_v2','messages':request,
                    'request_fingerprint':digest(request),'session_index':len(doc['sessions'])-1,'calls':[],
                    'seed':42,'original_draft_sha256':digest(request[2]['content'])}
                row['attempts'].append(attempt);row['status']='running';save();begin=time.perf_counter()
                def generate(capture):
                    call={'role':'capture' if capture else 'control','status':'running',
                          'request_fingerprint':digest(request),'cost_unknown':True}
                    attempt['calls'].append(call);save()
                    outcome=guarded_generate(model,tok,request,guard,list(range(33)),seed=42,capture=capture)
                    call.update({k:v for k,v in outcome.items() if k!='result'})
                    call.update(cost_unknown=False,usage=outcome['result']['usage'] if outcome['result'] else {
                        'prompt_tokens':outcome['prompt_tokens'],'completion_tokens':outcome['completion_tokens'],
                        'latency_seconds':outcome['diagnostics'].get('generation_seconds')})
                    apply_attempt(doc,row,attempt,outcome);save()
                    return outcome
                control=generate(False) if split=='evaluation' else None
                outcome=control if control is not None and control['status']!='success' else generate(True)
                if outcome['status']!='success':
                    attempt['wall_seconds_including_storage']=time.perf_counter()-begin
                    doc['recovery_v2']['status']='stopped';save()
                    raise RuntimeError('One repair failed; no strategy changes or subsequent stage')
                cap=outcome['result']
                if control is not None:
                    ctrl=control['result'];same=ctrl['generated_ids']==cap['generated_ids']
                    cap['capture_control']={k:ctrl[k] for k in ('usage','metrics','generated_ids')}
                    cap['capture_control'].update(memory=control['memory'],same_generated_ids=same,
                        capture_minus_control_seconds=(cap['metrics']['generation_plus_feature_seconds']-ctrl['metrics']['generation_plus_feature_seconds']) if same else None)
                    for key in ('peak_allocated_bytes','peak_reserved_bytes'):
                        outcome['memory'][key]=max(outcome['memory'][key],control['memory'][key])
                features=cap.pop('features');previous=doc.get('last_successful_problem_id')
                target=output_path(Path(str(output)+'.features')/f"{row['problem_hash']}.{len(row['attempts'])}.npz")
                try:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('xb') as stream:
                        np.savez_compressed(stream,**{str(layer):np.asarray([f[str(layer)] for f in features],dtype=np.float32) for layer in range(33)})
                    cap['features']={'path':str(target),'sha256':file_hash(target),'n_steps':len(features)}
                    cap['metrics'].update(outcome['memory']);cap['metrics'].update(memory_limit_gib=21.,length_coverage=outcome['length_coverage'])
                    row.update(status='success',capture=cap,messages=request,steps_fingerprint=digest(cap['steps']),
                        recovery_context={'policy':policy()['name'],'original_draft_sha256':digest(request[2]['content'])})
                    attempt.update(status='success',usage=cap['usage'],metrics=cap['metrics'],wall_seconds_including_storage=time.perf_counter()-begin)
                    doc['last_successful_problem_id']=row['id']
                    doc['recovery_v2']['accepted_repaired_rows'][row['id']]={'row_sha256':digest(row),'npz_path':str(target),'npz_sha256':cap['features']['sha256']}
                    save()
                except (Exception,StopRequested) as exc:
                    row['status']='error';attempt.update(status='storage_failure',error={'type':type(exc).__name__,'message':str(exc)[:1500]})
                    doc['last_successful_problem_id']=previous;doc['recovery_v2']['accepted_repaired_rows'].pop(row['id'],None)
                    doc['recovery_v2']['status']='storage_failure';save()
                    raise RuntimeError('Storage failure; preserve and inspect partial files') from None
        inspect_capture(seal({k:v for k,v in doc.items() if k!='fingerprint'}),data,split,tok,messages)
        verify_protected(doc,{**doc['recovery_v2']['protected_successes'],**doc['recovery_v2']['accepted_repaired_rows']},True)
        verify_feature_directory(output,doc)
        doc['recovery_v2']['status']='success';save()
        print(split.upper()+' RECOVERY V2 PASS: all successes preserved',flush=True)


def evaluation(a):
    frozen=approval(a.audit)
    allocation()  # Recheck per-GPU policy before the native collection process.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(frozen['model_path'],local_files_only=True)
    data=verify(read(RUN/'data.json'));val_source=verify(read(RUN/'validation_capture.json'))
    val=verify(read(RUN/'validation_recovered.json'))
    inspect_capture(val,data,'validation',tok,repair_messages(val_source,data,tok))
    from ..common import save_new
    policy_file=RUN/'evaluation_policy_v2.json'
    declared={'kind':'predeclared_evaluation_recovery_v2','policy':policy(),
        'code_sha256':code_hashes(),'audit_fingerprint':frozen['fingerprint']}
    with lock(policy_file):
        if Path(str(policy_file)+'.tmp').exists():raise ValueError('Uncommitted Evaluation policy')
        if policy_file.exists():
            if verify(read(policy_file))!=seal(declared):raise ValueError('Evaluation policy is frozen')
        else:
            if (RUN/'evaluation_capture.json').exists():raise ValueError('Cannot declare policy after Evaluation')
            save_new(policy_file,seal(declared))
    target=RUN/'evaluation_capture.json';existing=verify(read(target)) if target.exists() else None
    if Path(str(target)+'.tmp').exists():raise ValueError('Evaluation .tmp requires review')
    if existing is not None and not a.resume:raise FileExistsError('Explicit --resume required')
    if existing and existing.get('safety_stop'):
        raise ValueError('Native Evaluation safety stop requires separate inspection; not auto-resumed')
    if existing and len(existing['records'])<5 and any(r['status']=='error' for r in existing['records']):
        raise ValueError('Partial Evaluation contains an error: review before continuing')
    if existing is None or len(existing['records'])<5:
        command=[sys.executable,'-B','-m','experiments.linear_probe.cli','collect','--data',str(RUN/'data.json'),
            '--split','evaluation','--model-path',frozen['model_path'],'--output',str(target),'--capture-control']
        if existing is not None:command.append('--resume')
        result=subprocess.run(command,check=False)
        if result.returncode not in (0,1):raise RuntimeError('Unexpected native collection exit')
    repair(a,'evaluation')


def final(a):
    frozen=approval(a.audit);checks=inspect_inputs();data=verify(read(RUN/'data.json'))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(frozen['model_path'],local_files_only=True)
    report={'kind':'capture_recovery_v2_final','passed':False,'policy':policy(),'audit_fingerprint':frozen['fingerprint'],
            'train':checks['train'],'splits':{},'rows':[]}
    output=output_path(RUN/f'final_v2_{os.environ["SLURM_JOB_ID"]}.json')
    if Path(str(output)+'.tmp').exists():raise FileExistsError('Uncommitted final audit')
    from ..common import save_new
    save_new(output,seal(report))
    try:
        for split in ('train','validation','evaluation'):
            path=RUN/(split+'_recovered.json');doc=verify(read(path))
            if split!='train':
                source=verify(read(RUN/(split+'_capture.json')));messages=repair_messages(source,data,tok)
                report['splits'][split]=inspect_capture(doc,data,split,tok,messages)
                if file_hash(RUN/(split+'_capture.json'))!=doc['recovery_v2']['source_sha256']:raise ValueError('Source changed')
                verify_protected(doc,{**doc['recovery_v2']['protected_successes'],**doc['recovery_v2']['accepted_repaired_rows']},True)
                verify_feature_directory(path,doc)
            report['rows'].extend({'id':r['id'],'split':split,'category':classification(r),'calls':cost_calls(r),
                'cost_summary':cost_summary(r),'steps_fingerprint':r.get('steps_fingerprint'),'status':r['status']} for r in doc['records'])
        report['passed']=True
    except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
    atomic_save(output,seal(report))
    if not report['passed']:raise RuntimeError('Final integrity audit failed')
    print('FINAL V2 PASS: Train 24 / Validation 6 / Evaluation 5, all feature integrity verified',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--phase',choices=('preflight','validation','evaluation','final'),required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--ack-safety-stop',action='store_true')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('SLURM compute node required')
    if a.phase=='preflight':
        frozen=approval(a.audit);fresh=audit(RUN/f'preflight_v2_{os.environ["SLURM_JOB_ID"]}.json')
        if any(fresh[k]!=frozen[k] for k in ('input_sha256','train','validation','frozen_policy','code_sha256')):raise ValueError('Preflight changed')
    elif a.phase=='validation':repair(a,'validation')
    elif a.phase=='evaluation':evaluation(a)
    else:final(a)


if __name__=='__main__':main()
