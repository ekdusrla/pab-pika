"""Compute-node-only integrity checks and predeclared Evaluation recovery policy."""
import argparse
import copy
import os
import re
from pathlib import Path

from ..common import (ROOT,MODEL,REVISION,FEATURE,read,verify,seal,save_new,output_path,
                      file_hash,digest,decomp_messages,parse_steps)
from ..capture import step_positions
from ..safety import validate_features
from ..learning import validate_capture
from ..pilot_data import validate_pilot
from ..recovery_v1.audit import inspect as inspect_original, protected_rows, verify_protected
from ..recovery_v1.recover import messages_for_recovery, recovery_protection
from experiment_state import atomic_save,lock
from run_planner import RECOVERY_INSTRUCTION

RUN=ROOT/'experiments/linear_probe/runs/pilot_v1'


def policy():
    return {'name':'overlong-only-one-repair-v2','instruction':RECOVERY_INSTRUCTION,
        'trigger':'normal EOS; sequential numbered list longer than 5; reproduced parser ValueError only',
        'max_repair_attempts':1,'temperature':.1,'top_p':.9,'seed':42,'max_new_tokens':1024,
        'dtype':'float16','attention':'eager','layers':list(range(33)), 'reserved_limit_gib':21.,
        'evaluation_frozen_before_first_generation':True,
        'on_other_failure':'stop; no prompt/seed/strategy tuning',
        'control':'same messages, seed and settings; control and capture outcomes/costs retained separately'}


def code_hashes():
    result={p.name:file_hash(p) for pattern in ('*.py','*.sbatch') for p in Path(__file__).parent.glob(pattern)}
    result['recovery_v1_dependencies']={p.name:file_hash(p) for p in Path(__file__).parent.with_name('recovery_v1').glob('*.py')}
    return result


def eos_ids(doc):
    eos=doc['sessions'][0]['generation_config']['eos_token_id']
    return {eos} if isinstance(eos,int) else set(eos)


def overlong(row,tok,eos):
    attempt=row['attempts'][-1];error=attempt.get('error') or {}
    ids=attempt.get('diagnostics',{}).get('generated_ids',[])
    if (attempt['status']!='parse_failure' or error.get('type')!='ValueError' or
        error.get('phase')!='step_alignment' or not ids or ids[-1] not in eos or len(ids)>=1024 or
        attempt['observations'].get('finish_reason')!='stop' or
        attempt['memory']['peak_reserved_bytes']>21*2**30 or
        len(ids)!=attempt['usage']['completion_tokens']):
        raise ValueError('Not an eligible normal-EOS overlong failure: '+row['id'])
    text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    numbers=[int(x) for x in re.findall(r'(?m)^\s*(\d+)\.\s+',text)]
    if len(numbers)<=5 or numbers!=list(range(1,len(numbers)+1)):
        raise ValueError('Not a sequential list longer than five')
    try:parse_steps(text)
    except ValueError as exc:
        if str(exc)!=error['message']:raise ValueError('Parser exception cannot be reproduced') from None
    else:raise ValueError('Original parser accepted the draft')
    return {'id':row['id'],'decoded_text':text,'decoded_text_sha256':digest(text),
            'step_count':len(numbers),'exception':error,'completion_tokens':len(ids)}


def inspect_capture(doc,data,split,tok,repair_messages=None,complete=True):
    """Seal, identities, all NPZs, prompts, features and exact token alignment."""
    import numpy as np
    verify(doc);repair_messages=repair_messages or {}
    expected={r['id']:r for r in data['splits'][split]}
    if doc['split']!=split or doc['data_fingerprint']!=data['fingerprint']:
        raise ValueError('Split/data mismatch')
    rows=doc['records']
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):raise ValueError('Require exact complete split IDs')
    if doc.get('safety_stop'):raise ValueError('Safety stop requires separate review')
    config=doc['config']
    if config['model']!=MODEL or config['revision']!=REVISION or config['layers']!=list(range(33)):
        raise ValueError('Model/layer mismatch')
    if config['decoding']!={'temperature':.1,'top_p':.9,'max_new_tokens':1024,'max_context':8192,
                           'dtype':'float16','attention':'eager','seed':42}:
        raise ValueError('Generation settings changed')
    for name,h in config['runtime']['source_sha256'].items():
        if file_hash(ROOT/'experiments/linear_probe'/name)!=h:raise ValueError('Core source changed')
    for name,h in config['runtime']['baseline_source_sha256'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('Baseline source changed')
    eos=eos_ids(doc);checks=[];failures=[]
    for row in rows:
        if row['problem_hash']!=expected[row['id']]['problem_hash']:raise ValueError('Problem hash mismatch')
        if row['status']!='success':
            if complete:raise ValueError('Incomplete split')
            failures.append(overlong(row,tok,eos));continue
        cap=row['capture'];meta=cap['features']
        if file_hash(meta['path'])!=meta['sha256']:raise ValueError('NPZ checksum mismatch: '+row['id'])
        with np.load(meta['path'],allow_pickle=False) as arrays:
            if set(arrays.files)!={str(i) for i in range(33)}:raise ValueError('NPZ layer mismatch')
            mats={k:arrays[k] for k in arrays.files}
        n=len(cap['steps'])
        if meta['n_steps']!=n or any(x.shape!=(n,4096) or x.dtype!=np.float32 for x in mats.values()):
            raise ValueError('NPZ shape/dtype mismatch')
        candidate=copy.copy(cap);candidate['features']=[{k:v[i] for k,v in mats.items()} for i in range(n)]
        validate_features(candidate,list(range(33)))
        if row['steps_fingerprint']!=digest(cap['steps']) or cap['feature_definition']!=FEATURE:
            raise ValueError('Step/feature fingerprint mismatch')
        ids=cap['generated_ids']
        if not ids or ids[-1] not in eos or cap['finish_reason']!='stop' or len(ids)!=cap['usage']['completion_tokens']:
            raise ValueError('Token history mismatch')
        text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        prefixes=[tok.decode(ids[:i+1],skip_special_tokens=True,clean_up_tokenization_spaces=False) for i in range(len(ids)-1)]
        if text!=cap['text'] or step_positions(text,prefixes)!=(cap['steps'],cap['feature_positions']):
            raise ValueError('Token alignment mismatch')
        messages=repair_messages.get(row['id'],decomp_messages(expected[row['id']]))
        if row['messages']!=messages or digest(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))!=cap['prompt_ids_sha256']:
            raise ValueError('Prompt mismatch')
        if row['memory']['peak_reserved_bytes']>21*2**30:raise ValueError('Memory limit exceeded')
        if split=='evaluation':
            ctrl=cap['capture_control']
            if ctrl['memory']['peak_reserved_bytes']>21*2**30:raise ValueError('Unsafe control')
            if ctrl['same_generated_ids']!=(ctrl['generated_ids']==ids):raise ValueError('Control comparison mismatch')
        checks.append({'id':row['id'],'shape':[n,33,4096],'finite':True,'steps_fingerprint':row['steps_fingerprint'],
                       'npz_sha256':meta['sha256'],'alignment_verified':True})
    if complete:validate_capture(doc,data,split)
    return {'success_count':len(checks),'failures':failures,'success_checks':checks,'protected':protected_rows(doc)}


def verify_feature_directory(path,doc):
    known={Path(x['npz_path']).resolve() for x in protected_rows(doc).values()}
    for f in Path(str(path)+'.features').glob('*'):
        if f.is_file() and f.resolve() not in known:raise ValueError('Orphan feature requires review: '+str(f))


def inspect_inputs():
    from transformers import AutoTokenizer
    policy_file=RUN/'evaluation_policy_v2.json'
    if (RUN/'evaluation_capture.json').exists() and not policy_file.exists():
        raise ValueError('Evaluation already exists without a predeclared V2 policy')
    if policy_file.exists():
        frozen=verify(read(policy_file))
        if frozen['policy']!=policy() or frozen['code_sha256']!=code_hashes():
            raise ValueError('Evaluation policy/code changed after its first execution')
    paths=[RUN/x for x in ('data.json','train_capture.json','train_recovered.json','validation_capture.json','audit_219630.json')]
    for path in paths:
        if Path(str(path)+'.tmp').exists():raise ValueError('Uncommitted checkpoint: '+str(path))
    hashes={str(p):file_hash(p) for p in paths}
    data=validate_pilot(read(RUN/'data.json'));original=verify(read(RUN/'train_capture.json'))
    model=Path(original['config']['model_path']);tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    old_audit=verify(read(RUN/'audit_219630.json'))
    if not old_audit['passed'] or old_audit['source_checkpoint_sha256']!=hashes[str(RUN/'train_capture.json')]:
        raise ValueError('Original audited source changed')
    original_check=inspect_original(original,data,tok,model)
    verify_protected(original,old_audit['protected_successes'],True)
    train=verify(read(RUN/'train_recovered.json'));recovery_protection(train,RUN/'train_capture.json',True)
    for name,h in train['config']['recovery_v1']['code_sha256'].items():
        if file_hash(Path(__file__).parent.with_name('recovery_v1')/name)!=h:raise ValueError('Original recovery code changed')
    repair={f['id']:messages_for_recovery(next(r for r in data['splits']['train'] if r['id']==f['id']),f)
            for f in original_check['failures']}
    train_check=inspect_capture(train,data,'train',tok,repair)
    verify_protected(train,old_audit['protected_successes'],True)
    validation=verify(read(RUN/'validation_capture.json'))
    val_check=inspect_capture(validation,data,'validation',tok,complete=False)
    if val_check['success_count']!=5 or len(val_check['failures'])!=1 or not val_check['failures'][0]['id'].endswith('/intermediate_algebra/train/1000') or val_check['failures'][0]['step_count']!=6:
        raise ValueError('Unexpected Validation checkpoint')
    for doc,path in ((original,RUN/'train_capture.json'),(train,RUN/'train_recovered.json'),(validation,RUN/'validation_capture.json')):
        verify_feature_directory(path,doc)
    if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Input changed during audit')
    return {'input_sha256':hashes,'data_fingerprint':data['fingerprint'],'model_path':str(model),
        'train':train_check,'validation':val_check,'original_22_reused':True,
        'frozen_policy':policy(),'code_sha256':code_hashes()}


def audit(output):
    output=output_path(output)
    with lock(output):
        if Path(str(output)+'.tmp').exists():raise FileExistsError('Uncommitted audit')
        report={'kind':'validation_evaluation_recovery_audit_v2','passed':False}
        save_new(output,seal(report))
        try:report.update(inspect_inputs());report['passed']=True
        except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        atomic_save(output,seal(report))
    if not report['passed']:raise RuntimeError('Audit failed; inspect '+str(output))
    print('AUDIT V2 PASS: Train 24, original 22 reused, Validation 5 NPZ verified; policy frozen',flush=True)
    return seal(report)


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Run tokenizer/NPZ audit on a SLURM compute node')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)


if __name__=='__main__':main()
