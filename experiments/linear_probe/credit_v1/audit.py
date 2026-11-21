"""CPU compute-node audit; never imports or loads model weights."""
import argparse
import os
from pathlib import Path

from ..common import (ROOT,MODEL,REVISION,read,verify,seal,save_new,output_path,file_hash,digest,
                      assessment_messages,credit_schema,original_definitions,problem_hash,read_manifest,runtime)
from ..recovery_v2.audit import inspect_inputs,inspect_capture
from ..recovery_v2.resume import repair_messages
from ..recovery_v1.audit import verify_protected
from experiment_state import atomic_save,lock

SOURCE=ROOT/'experiments/linear_probe/runs/pilot_v1'
OUT=ROOT/'experiments/linear_probe/runs/credit_v1'
ALPHAS=(.01,.1,1.,10.,100.,1000.)
CAPTURES={s:SOURCE/(s+'_recovered.json') for s in ('train','validation','evaluation')}
SERVER={'model':MODEL,'revision':REVISION,'temperature':.1,'top_p':.9,
        'server_settings':{'dtype':'float16','quantization':None,'tensor_parallel_size':1,
          'max_model_len':8192,'max_num_seqs':1,'gpu_memory_utilization':.85}}


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Run only on a SLURM compute node; no Python on master')


def code_hashes():
    base=Path(__file__).parent
    files=[p for pattern in ('*.py','*.sbatch') for p in base.glob(pattern)]
    files += list(base.parent.glob('*.py'))+[base.parent/'serve.sbatch']
    for name in ('recovery_v1','recovery_v2'):files+=list((base.parent/name).glob('*.py'))
    return {str(p.relative_to(ROOT)):file_hash(p) for p in files}


def capture_projection(row):
    return digest({k:row[k] for k in ('id','problem_hash','capture','steps_fingerprint','messages')})


def approve(path):
    report=verify(read(path))
    if report.get('kind')!='recovered_credit_audit_v1' or not report.get('passed'):
        raise ValueError('Successful new CPU credit audit required')
    if report['code_sha256']!=code_hashes():raise ValueError('Credit/core code changed after audit')
    for name,h in report['baseline_source_sha256'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('PAB source changed after audit')
    for p,h in report['protected_files'].items():
        if file_hash(p)!=h:raise ValueError('Frozen input changed: '+p)
    return report


def inspect():
    from transformers import AutoTokenizer
    original=inspect_inputs()  # Original 22, recovered Train, frozen policies/provenance.
    data=verify(read(SOURCE/'data.json'));final=verify(read(SOURCE/'final_v2_219684.json'))
    if not final.get('passed'):raise ValueError('Recovery V2 final PASS required')
    tok=AutoTokenizer.from_pretrained(original['model_path'],local_files_only=True)
    manifest=read_manifest(ROOT/'experiments/llama_qwq/math500_manifest.json')
    if manifest['fingerprint']!=data['evaluation_manifest_fingerprint']:raise ValueError('MATH-500 identity changed')
    forbidden={problem_hash(r['query']) for r in manifest['records']}
    all_rows=data['benchmarks']+[r for split in data['splits'].values() for r in split]
    if any(r['problem_hash'] in forbidden for r in all_rows):raise ValueError('MATH-500 normalized overlap')
    if len({r['problem_hash'] for r in all_rows})!=len(all_rows):raise ValueError('Cross-split/benchmark overlap')
    protected=dict(original['input_sha256']);contexts={};projections={}
    capture_config=verify(read(CAPTURES['train']))['config'];model_path=Path(original['model_path'])
    for name,h in {**capture_config['tokenizer_files_sha256'],
                   'config.json':capture_config['model_config_sha256'],
                   'model.safetensors.index.json':capture_config['weight_index_sha256']}.items():
        protected[str(model_path/name)]=h
    if any(not (model_path/name).is_file() for name in set(read(model_path/'model.safetensors.index.json')['weight_map'].values())):
        raise ValueError('Incomplete pinned model cache; no downloads permitted')
    template=original_definitions()[0]['evaluate_question_level_batch']
    benchmarks='\n'.join(f"\nLevel {r['level']} Example:\nQ: {r['query']}\nA: {r['solution']}\n" for r in data['benchmarks'])
    for path in (SOURCE/'final_v2_219684.json',SOURCE/'evaluation_policy_v2.json',
                 ROOT/'experiments/llama_qwq/math500_manifest.json'):
        protected[str(path)]=file_hash(path)
    for split,n in (('train',24),('validation',6),('evaluation',5)):
        path=CAPTURES[split];doc=verify(read(path));sources={r['id']:r for r in data['splits'][split]}
        if len(doc['records'])!=n:raise ValueError('Unexpected split count')
        check=final['train'] if split=='train' else final['splits'][split]
        verify_protected(doc,check['protected'],True)
        if split!='train':
            native_path=SOURCE/(split+'_capture.json');native=verify(read(native_path))
            inspect_capture(doc,data,split,tok,repair_messages(native,data,tok))
            protected[str(native_path)]=file_hash(native_path)
        protected[str(path)]=file_hash(path);contexts[split]=[];projections[split]={}
        for row in doc['records']:
            npz=row['capture']['features'];protected[npz['path']]=npz['sha256']
            steps=row['capture']['steps'];messages=assessment_messages(sources[row['id']],steps,data['benchmarks'])
            values={'domain':'math','problem':sources[row['id']]['query'],'steps':steps,'benchmarks':benchmarks}
            expected=[{'role':role,'content':template[role].format(**values)} for role in ('system','user')]
            if messages!=expected:raise ValueError('Assessment prompt differs from PAB')
            count=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
            if count+2048>8192:raise ValueError('Credit context exceeded: '+row['id'])
            contexts[split].append({'id':row['id'],'prompt_tokens':count,'max_completion_tokens':2048,
                'messages_sha256':digest(messages),'schema_sha256':digest(credit_schema(len(steps))),
                'steps_fingerprint':row['steps_fingerprint']})
            projections[split][row['id']]=capture_projection(row)
    return {'protected_files':protected,'code_sha256':code_hashes(),'model_path':original['model_path'],
        'runtime':{**runtime(),'engine':'CPU tokenizer/NPZ audit; no inference'},
        'engine_policy':{'capture':'HF manual KV-cache, unconstrained Decomposition',
                         'credit':'vLLM OpenAI API, guided_json/outlines; no HF Credit implementation'},
        'data_fingerprint':data['fingerprint'],'captures':{k:str(v) for k,v in CAPTURES.items()},
        'capture_projections':projections,'contexts':contexts,'teacher_settings':SERVER,
        'capture_categories':{s:{r['id']:r['category'] for r in final['rows'] if r['split']==s} for s in CAPTURES},
        'selection':{'layers':list(range(33)),'alphas':list(ALPHAS),
            'metric':'problem-mean projected integer Credit MAE','tie_break':'lowest layer, then lowest alpha',
            'fit_split':'train only','selection_split':'validation only','refit_on_validation':False},
        'leakage':{'math500_normalized_overlap':False,'all_problem_and_benchmark_hashes_disjoint':True,
            'semantic_duplicate_exclusion_guaranteed':False},
        'baseline_source_sha256':verify(read(CAPTURES['train']))['config']['runtime']['baseline_source_sha256']}


def run(output):
    require_compute();output=output_path(output)
    with lock(output):
        if Path(str(output)+'.tmp').exists():raise FileExistsError('Uncommitted audit')
        report={'kind':'recovered_credit_audit_v1','passed':False};save_new(output,seal(report))
        try:report.update(inspect());report['passed']=True
        except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        atomic_save(output,seal(report))
    if not report['passed']:raise RuntimeError('Credit audit failed; inspect '+str(output))
    print('CREDIT CPU AUDIT PASS: recovered 24/6/5, 33-layer NPZ, exact PAB assessment and context verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
