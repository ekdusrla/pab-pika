"""Compute-node audit of the frozen 22-success/2-overlong Train checkpoint."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from ..common import (ROOT,MODEL,REVISION,FEATURE,read,verify,seal,save_new,output_path,
                      file_hash,digest,decomp_messages,parse_steps)
from ..pilot_data import validate_pilot
from ..capture import step_positions
from ..safety import validate_features
from experiment_state import lock,atomic_save

TARGETS={'precalculus/train/511':7,'counting_and_probability/train/60':6}


def code_hashes():
    directory=Path(__file__).parent
    return {p.name:file_hash(p) for pattern in ('*.py','*.sbatch') for p in directory.glob(pattern)}


def target_suffix(identifier):
    return next((suffix for suffix in TARGETS if identifier.endswith('/'+suffix)),None)


def protected_rows(doc):
    return {row['id']:{'row_sha256':digest(row),'npz_path':row['capture']['features']['path'],
        'npz_sha256':row['capture']['features']['sha256']} for row in doc['records'] if row['status']=='success'}


def verify_protected(doc, protected, verify_npz=False):
    by_id={r['id']:r for r in doc['records']}
    for identifier, expected in protected.items():
        if identifier not in by_id or digest(by_id[identifier])!=expected['row_sha256']:
            raise ValueError('Protected successful row changed: '+identifier)
        if verify_npz and file_hash(expected['npz_path'])!=expected['npz_sha256']:
            raise ValueError('Protected NPZ changed: '+identifier)


def analyze_failure(row,tok,eos):
    suffix=target_suffix(row['id'])
    if suffix is None:raise ValueError('Unexpected failed ID')
    if not row.get('attempts'):raise ValueError('Missing failure history')
    attempt=row['attempts'][-1];trace=attempt.get('diagnostics',{});ids=trace.get('generated_ids',[])
    if attempt.get('status')!='parse_failure' or attempt.get('error',{}).get('type')!='ValueError':
        raise ValueError('Only recorded ValueError parse failures can use this adapter')
    if attempt['error'].get('phase')!='step_alignment' or not ids or ids[-1] not in eos:
        raise ValueError('Failure is not a normally terminated decomposition parse failure')
    if len(ids)!=attempt['usage']['completion_tokens'] or len(ids)>=1024:
        raise ValueError('Token history mismatch or truncated generation')
    if attempt['observations'].get('finish_reason')!='stop':raise ValueError('Abnormal termination')
    if attempt['memory']['peak_reserved_bytes']>21*2**30:raise ValueError('Unsafe memory history')
    text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    headers=list(re.finditer(r'(?m)^\s*(\d+)\.\s+(.+)',text))
    numbers=[int(m.group(1)) for m in headers]
    if numbers!=list(range(1,TARGETS[suffix]+1)):
        raise ValueError('Full output does not confirm the expected sequential overlong list')
    try:
        parse_steps(text)
    except ValueError as exc:
        if str(exc)!=attempt['error']['message']:
            raise ValueError('Decoded output does not reproduce the recorded parser error') from None
    else:raise ValueError('The old failure cannot be reproduced by the original parser')
    return {'id':row['id'],'attempt_count':len(row['attempts']),'failure_kind':'parse_failure',
        'exception':attempt['error'],'prompt_tokens':attempt['usage']['prompt_tokens'],
        'completion_tokens':len(ids),'finish_reason':'stop','numbered_headers':numbers,
        'n_numbered_steps':len(numbers),'decoded_text':text,'decoded_text_sha256':digest(text),
        'memory':attempt['memory'],'feature_validation_reached':False,
        'partial_output_available':True,'partial_npz_recorded':'capture' in row}


def inspect(source,data,tok,model_path):
    """All array/token decoding work belongs on a compute node."""
    import numpy as np
    verify(source);validate_pilot(data)
    if source.get('split')!='train' or source.get('data_fingerprint')!=data['fingerprint']:
        raise ValueError('Source checkpoint/data mismatch')
    expected={r['id']:r for r in data['splits']['train']}
    records=source['records']
    if len(records)!=24 or {r['id'] for r in records}!=set(expected):raise ValueError('Missing/duplicate Train IDs')
    success=[r for r in records if r['status']=='success'];failed=[r for r in records if r['status']=='error']
    if len(success)!=22 or len(failed)!=2 or {target_suffix(r['id']) for r in failed}!=set(TARGETS):
        raise ValueError('Require exactly the frozen 22 successful and two known failed rows')
    if source.get('safety_stop') is not None:raise ValueError('Source has a safety stop; recovery not authorized')
    config=source['config']
    if config['model']!=MODEL or config['revision']!=REVISION or model_path.resolve().name!=REVISION:
        raise ValueError('Pinned Llama model required')
    if config['decoding']!={'temperature':.1,'top_p':.9,'max_new_tokens':1024,'max_context':8192,
                           'dtype':'float16','attention':'eager','seed':42}:
        raise ValueError('Unexpected original generation settings')
    if config.get('capture_control') or config.get('layers')!=list(range(33)):
        raise ValueError('Expected original Train all-layer/no-control capture')
    for name, expected_hash in config['runtime']['source_sha256'].items():
        if file_hash(Path(__file__).resolve().parents[1]/name)!=expected_hash:
            raise ValueError('Original capture code changed: '+name)
    for name, expected_hash in config['runtime']['baseline_source_sha256'].items():
        if file_hash(ROOT/name)!=expected_hash:
            raise ValueError('Original baseline/prompt code changed: '+name)
    for name,key in [('config.json','model_config_sha256'),('model.safetensors.index.json','weight_index_sha256')]:
        if file_hash(model_path/name)!=config[key]:raise ValueError('Model metadata changed')
    for name,h in config['tokenizer_files_sha256'].items():
        if file_hash(model_path/name)!=h:raise ValueError('Tokenizer changed')
    eos=source['sessions'][0]['generation_config']['eos_token_id']
    eos={eos} if isinstance(eos,int) else set(eos)
    checks=[]
    for row in records:
        if row['problem_hash']!=expected[row['id']]['problem_hash']:raise ValueError('Problem identity changed')
    for row in success:
        cap=row['capture'];meta=cap['features'];path=Path(meta['path'])
        if not row.get('attempts') or row['attempts'][-1]['status']!='success' or row['attempts'][-1]['usage']!=cap['usage']:
            raise ValueError('Successful attempt/projection mismatch: '+row['id'])
        if row['attempts'][-1]['diagnostics']['generated_ids']!=cap['generated_ids']:
            raise ValueError('Successful token trace/projection mismatch: '+row['id'])
        if file_hash(path)!=meta['sha256']:raise ValueError('NPZ checksum mismatch: '+row['id'])
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files)!={str(i) for i in range(33)}:raise ValueError('NPZ layers mismatch')
            matrices={k:archive[k] for k in archive.files}
        n=len(cap['steps'])
        if meta['n_steps']!=n or any(a.shape!=(n,4096) or a.dtype!=np.float32 for a in matrices.values()):
            raise ValueError('NPZ shape/dtype mismatch: '+row['id'])
        candidate=copy.copy(cap)
        candidate['features']=[{k:a[i] for k,a in matrices.items()} for i in range(n)]
        validate_features(candidate,list(range(33)))
        if row['steps_fingerprint']!=digest(cap['steps']) or cap['feature_definition']!=FEATURE:
            raise ValueError('Steps/feature fingerprint mismatch')
        ids=cap['generated_ids']
        if not ids or ids[-1] not in eos or cap['finish_reason']!='stop' or len(ids)!=cap['usage']['completion_tokens']:
            raise ValueError('Successful token history mismatch')
        text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        prefixes=[tok.decode(ids[:i+1],skip_special_tokens=True,clean_up_tokenization_spaces=False) for i in range(len(ids)-1)]
        if text!=cap['text'] or step_positions(text,prefixes)!=(cap['steps'],cap['feature_positions']):
            raise ValueError('Stored text/token alignment mismatch')
        messages=decomp_messages(expected[row['id']])
        prompt_ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if row['messages']!=messages or digest(prompt_ids)!=cap['prompt_ids_sha256']:
            raise ValueError('Successful original prompt mismatch')
        if row['memory']['peak_reserved_bytes']>21*2**30:raise ValueError('Unsafe successful memory record')
        checks.append({'id':row['id'],'row_sha256':digest(row),'npz_path':str(path),
            'npz_sha256':meta['sha256'],'shape':[n,33,4096],'finite':True,
            'steps_fingerprint':row['steps_fingerprint'],'alignment_verified':True,
            'attempt_count':len(row['attempts']),'memory':row['memory']})
    failures=[analyze_failure(row,tok,eos) for row in failed]
    return {'success_count':22,'error_count':2,'success_checks':checks,'failures':failures,
        'protected_successes':protected_rows(source),'safety_stop':source.get('safety_stop'),
        'last_successful_problem_id':source.get('last_successful_problem_id'),
        'no_historical_overwrite_proof':'Current checksums and records verified; earlier overwrite history cannot be proven without an earlier external snapshot.'}


def audit(source_path,data_path,output):
    output=output_path(output)
    with lock(source_path),lock(output):
        for path in (output,Path(str(output)+'.tmp'),Path(str(output)+'.source_snapshot.json')):
            if path.exists():raise FileExistsError(path)
        result={'kind':'capture_recovery_audit_v1','passed':False,'source_path':str(source_path),
                'data_path':str(data_path),'recovery_code_sha256':code_hashes()}
        save_new(output,seal(result))
        try:
            raw=source_path.read_bytes();source=json.loads(raw);data=read(data_path)
            result.update(source_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                source_fingerprint=source['fingerprint'],data_fingerprint=data['fingerprint'])
            snapshot=output_path(Path(str(output)+'.source_snapshot.json'))
            with snapshot.open('xb') as stream:stream.write(raw)
            result['immutable_source_snapshot']=str(snapshot)
            if Path(str(source_path)+'.tmp').exists():raise ValueError('Uncommitted source checkpoint exists')
            from transformers import AutoTokenizer
            model_path=Path(source['config']['model_path'])
            tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
            result.update(inspect(source,data,tok,model_path))
            feature_dir=Path(str(source_path)+'.features')
            known={Path(c['npz_path']).resolve() for c in result['success_checks']}
            orphan=[str(p) for p in feature_dir.glob('*') if p.is_file() and p.resolve() not in known]
            result['unreferenced_feature_files']=orphan
            if orphan:raise ValueError('Unexpected partial/orphan feature files require inspection')
            if file_hash(source_path)!=result['source_checkpoint_sha256']:raise ValueError('Source changed during audit')
            result['passed']=True
        except Exception as exc:
            result['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        atomic_save(output,seal(result))
    if not result['passed']:raise RuntimeError('Audit failed; report saved: '+str(output))
    print('AUDIT PASS: 22 NPZ/row fingerprints verified; two normal-EOS overlong lists',flush=True)
    return seal(result)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);audit(a.source,a.data,a.output)


if __name__=='__main__':main()
