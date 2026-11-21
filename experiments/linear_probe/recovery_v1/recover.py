"""Explicit two-row overlong recovery into a NEW checkpoint; original stays immutable."""
import argparse
import copy
import os
import time
from pathlib import Path

from ..common import (read,verify,seal,output_path,file_hash,digest,decomp_messages,runtime)
from ..cli import checkpoint
from ..learning import validate_capture,feature_rows
from ..pilot_data import validate_pilot
from ..pilot_smoke import allocation
from ..safety import (MemoryGuard,require_device,guarded_generate,apply_attempt,acknowledge_resume,
                      failure_kind,validate_features)
from .audit import audit,verify_protected,code_hashes
from experiment_state import lock,atomic_save,termination_handler,StopRequested
from run_planner import RECOVERY_INSTRUCTION


def policy():
    return {'name':'failed_train_overlong_recovery_v1','instruction':RECOVERY_INSTRUCTION,
        'trigger':'Only the two audited normal-EOS sequential overlong lists (7 and 6 steps)',
        'original_system_user_prompt_unchanged':True,'additional_context':'failed assistant draft + repair user turn',
        'seed':42,'temperature':.1,'top_p':.9,'max_new_tokens':1024,'reserved_limit_gib':21.,
        'max_completed_invalid_recovery_attempts':1,
        'research_note':'Recovered rows use extra dialogue context. Report IDs and all extra generation costs; not a pure unchanged one-call decomposition for those rows.'}


def messages_for_recovery(source_row, failure):
    return decomp_messages(source_row)+[
        {'role':'assistant','content':failure['decoded_text']},
        {'role':'user','content':RECOVERY_INSTRUCTION}]


def prepare_document(source,approval,output):
    document=copy.deepcopy(source);document.pop('fingerprint',None)
    document['config']=copy.deepcopy(source['config'])
    document['config']['output']=str(output)
    document['config']['recovery_v1']={'source_checkpoint_sha256':approval['source_checkpoint_sha256'],
        'source_fingerprint':source['fingerprint'],'audit_fingerprint':approval['fingerprint'],
        'policy':policy(),'code_sha256':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    document['recovery_v1']={'original_config':copy.deepcopy(source['config']),
        'protected_successes':approval['protected_successes'],'accepted_repaired_rows':{},
        'source_path':approval['source_path'],'source_sha256':approval['source_checkpoint_sha256'],
        'immutable_source_snapshot':approval['immutable_source_snapshot'],'policy':policy(),
        'pending_ids':[r['id'] for r in approval['failures']],
        'all_original_failures_preserved':True,'status':'pending'}
    return document


def recovery_protection(doc,source_path,check_arrays=False):
    meta=doc['recovery_v1']
    if file_hash(source_path)!=meta['source_sha256']:raise ValueError('Original checkpoint changed')
    verify_protected(doc,{**meta['protected_successes'],**meta['accepted_repaired_rows']},check_arrays)


def run(a):
    if not a.recover_overlong:raise ValueError('Explicit --recover-overlong required; no blind retry')
    approval=verify(read(a.audit))
    if approval.get('kind')!='capture_recovery_audit_v1' or not approval.get('passed'):
        raise ValueError('Successful compute-node audit required before recovery')
    if approval.get('recovery_code_sha256')!=code_hashes():
        raise ValueError('Recovery code changed after CPU audit; rerun audit/tests first')
    if file_hash(a.source)!=approval['source_checkpoint_sha256']:raise ValueError('Source changed since audit')
    if verify(read(a.data))['fingerprint']!=approval['data_fingerprint']:raise ValueError('Data changed since audit')
    output=output_path(a.output)
    if output.resolve()==a.source.resolve():raise ValueError('Recovery output must NOT overwrite original')
    # Audit again on the allocated node before any model weights are loaded.
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise RuntimeError('Run recovery inside a compute-node SLURM allocation')
    fresh=audit(a.source,a.data,Path(str(output)+'.preflight_'+job+'.json'))
    if fresh['protected_successes']!=approval['protected_successes'] or fresh['failures']!=approval['failures']:
        raise ValueError('Protected features or decoded failures changed since approval')
    with lock(a.source),lock(output),termination_handler():
        source=verify(read(a.source));data=validate_pilot(read(a.data))
        if file_hash(a.source)!=approval['source_checkpoint_sha256']:raise ValueError('Source changed after preflight')
        proposed=prepare_document(source,approval,a.output)
        doc=checkpoint(a,proposed['config'],proposed)
        acknowledge_resume(doc,a.resume,a.resume_after_safety_stop)
        def save():
            recovery_protection(doc,a.source)
            atomic_save(output,seal({k:v for k,v in doc.items() if k!='fingerprint'}))
        save()
        rows={r['id']:r for r in doc['records']}
        for row in doc['records']:
            if row['status']=='success':feature_rows(row)
        pending=[identifier for identifier in doc['recovery_v1']['pending_ids'] if rows[identifier]['status']!='success']
        for identifier in pending:
            for attempt in rows[identifier]['attempts']:
                if attempt.get('purpose')=='overlong_recovery_v1' and attempt['status'] in ('parse_failure','generation_truncated','invalid_features','storage_failure'):
                    raise ValueError('A completed recovery was invalid; review its new output instead of repeating unchanged generation')
        if not pending:
            validate_capture(seal({k:v for k,v in doc.items() if k!='fingerprint'}),data,'train')
            recovery_protection(doc,a.source,True)
            return
        guard=None;session={'purpose':'overlong_recovery_v1','status':'preflight','audit_fingerprint':fresh['fingerprint']}
        doc.setdefault('sessions',[]).append(session);save()
        started=time.perf_counter()
        try:
            session['allocation']=allocation()
            import torch
            import numpy as np
            from transformers import AutoModelForCausalLM,AutoTokenizer
            session['gpu']=require_device(torch);session['runtime']=runtime()
            guard=MemoryGuard(torch.cuda,21.);guard.begin()
            model_path=Path(source['config']['model_path'])
            tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
            model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
                torch_dtype=torch.float16,attn_implementation='eager').to('cuda').eval()
            torch.cuda.synchronize();guard.check('model_load')
            session.update(status='loaded',model_load_seconds=time.perf_counter()-started,memory=guard.snapshot(),
                generation_config=model.generation_config.to_dict(),actual_model_config=model.config.to_dict())
            save()
        except (Exception,StopRequested) as exc:
            session.update(status='error',error={'type':type(exc).__name__,'message':str(exc)[:1500]},
                           memory=guard.snapshot() if guard else None)
            doc['safety_stop']={'reason':failure_kind(exc),'phase':'model_load','problem_id':None,
                'last_successful_problem_id':doc.get('last_successful_problem_id'),'error':session['error']}
            save();raise RuntimeError('Recovery model loading failed; original results untouched') from None
        source_rows={r['id']:r for r in data['splits']['train']}
        failures={r['id']:r for r in approval['failures']}
        for identifier in pending:
            row=rows[identifier]
            for attempt in row['attempts']:
                if attempt['status']=='running':attempt.update(status='interrupted',cost_unknown=True)
            messages=messages_for_recovery(source_rows[identifier],failures[identifier])
            attempt={'status':'running','purpose':'overlong_recovery_v1','session_index':len(doc['sessions'])-1,
                'messages':messages,'request_fingerprint':digest(messages),'seed':42,
                'original_draft_sha256':failures[identifier]['decoded_text_sha256']}
            row['attempts'].append(attempt);row['status']='running';save()
            begin=time.perf_counter();outcome=guarded_generate(model,tok,messages,guard,list(range(33)),seed=42)
            apply_attempt(doc,row,attempt,outcome)
            attempt['wall_seconds_before_storage']=time.perf_counter()-begin
            if outcome['status']!='success':
                doc['recovery_v1']['status']='stopped';save()
                raise RuntimeError('Recovery output failed validation; no subsequent stage started')
            cap=outcome['result'];features=cap.pop('features')
            previous_success_id=doc.get('last_successful_problem_id')
            target=output_path(Path(str(output)+'.features')/f"{row['problem_hash']}.{len(row['attempts'])}.npz")
            try:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:
                    np.savez_compressed(stream,**{str(layer):np.asarray([f[str(layer)] for f in features],dtype=np.float32)
                                                 for layer in range(33)})
                cap['features']={'path':str(target),'sha256':file_hash(target),'n_steps':len(features)}
                cap['metrics'].update(outcome['memory']);cap['metrics']['memory_limit_gib']=21.
                cap['metrics']['length_coverage']=outcome['length_coverage']
                row.update(status='success',capture=cap,messages=messages,steps_fingerprint=digest(cap['steps']),
                    recovery_context={'policy':policy()['name'],'original_draft_sha256':failures[identifier]['decoded_text_sha256']})
                attempt.update(usage=cap['usage'],metrics=cap['metrics'],wall_seconds_including_storage=time.perf_counter()-begin)
                doc['last_successful_problem_id']=identifier
                doc['recovery_v1']['accepted_repaired_rows'][identifier]={'row_sha256':digest(row),
                    'npz_path':str(target),'npz_sha256':cap['features']['sha256']}
                save()
            except (Exception,StopRequested) as exc:
                # Never rewrite an original NPZ or silently regenerate after storage failure.
                row['status']='error';attempt.update(status='storage_failure',error={'type':type(exc).__name__,'message':str(exc)[:1500]})
                doc['last_successful_problem_id']=previous_success_id
                doc['recovery_v1']['accepted_repaired_rows'].pop(identifier,None)
                doc['recovery_v1']['status']='storage_failure';save()
                raise RuntimeError('Recovery storage failure; inspect files before explicit continuation') from None
        doc['recovery_v1']['status']='success'
        validate_capture(seal({k:v for k,v in doc.items() if k!='fingerprint'}),data,'train')
        recovery_protection(doc,a.source,True);save()
        print('RECOVERY PASS: original checkpoint and 22 successful rows/NPZ preserved; recovered Train 24/24',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','data','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recover-overlong',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-after-safety-stop',action='store_true')
    run(p.parse_args(argv))


if __name__=='__main__':main()
