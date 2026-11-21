"""CPU-only provenance audit and immutable request plan; never regenerates predictions."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import importlib.metadata

from ..common import (ROOT,read,verify,seal,save_new,file_hash,digest,credit_contract,project_credits,
                      allocate_words,LEVEL_WORDS,original_definitions)
from ..credit_a2000_tp4.audit import approve as parent_approve,OUT as CREDIT,SOURCE,require_compute
from ..credit_a2000_tp4.gpu import check_complete
from ..credit_a2000_tp4.teacher import frozen_probe,payload_for
from ..credit_a2000_tp4.train import validate_labels
from ..pilot import SOLVER,SOLVER_REVISION
from experiment_state import atomic_save,lock
from run_solver_compare import messages_for
from math_answer_eval import evaluator_info,evaluate_answer

OUT=ROOT/'experiments/linear_probe/runs/followup_v1'
DIAGNOSTIC=ROOT/'experiments/linear_probe/runs/analysis_credit_tp4_220316/extended_220319.json'
LEGACY=ROOT/'experiments/llama_qwq/runs/math500_llama31_8b_v1/solver.json'
QWQ=Path('/data/dada0228/huggingface/hub/models--Qwen--QwQ-32B/snapshots')/SOLVER_REVISION
ARMS=('vanilla','teacher','uniform','position','ridge')
PROFILES={
 'teacher':{'model':'meta-llama/Llama-3.1-8B-Instruct','revision':'0e9e39f249a16976918f6564b8830bc894c89659',
  'gpu_name':'NVIDIA RTX A2000 12GB','min_total_gib':11,'max_total_gib':13,'min_free_gib':10.,'used_limit_gib':11.,
  'cpus':24,'ram_gib':48,'context':8192,'max_tokens':2048,'guided':'outlines','gpu_count':4,'dtype':'float16','tp':4},
 'solver':{'model':SOLVER,'revision':SOLVER_REVISION,'gpu_name':'NVIDIA GeForce RTX 3090',
  'min_total_gib':23,'max_total_gib':25,'min_free_gib':22.,'used_limit_gib':22.5,
  'cpus':48,'ram_gib':160,'context':12288,'max_tokens':8192,'guided':None,'gpu_count':4,'dtype':'float16','tp':4}}
CACHE={'prefix_caching':False,'disk_cuda_grammar_caches':'not cleared',
       'max_num_seqs':1,'concurrent_requests':1,'gpu_memory_utilization':.85,'executor':'mp'}


def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for pattern in ('*.py','*.sh','*.sbatch') for p in Path(__file__).parent.glob(pattern)}


def legacy_config():
    # Do not load/inspect the 65 MB MATH-500 Solver results: only its small leading config.
    with LEGACY.open(encoding='utf-8') as f:prefix=f.read(65536)
    match=re.search(r'(?m)^  "config":\s*',prefix)
    if not match:raise ValueError('Legacy config not in bounded header')
    return json.JSONDecoder().raw_decode(prefix[match.end():])[0]


def teacher_schedule(ids):
    blocks=[('same_seed',0,42),('different_seed',0,43),('different_seed',1,44),
            ('same_seed',1,42),('same_seed',2,42),('different_seed',2,45)]
    sessions={'warm':[{'cohort':c,'round':r,'seed':s,'ids':ids} for c,r,s in blocks]}
    for k in range(3):sessions['restart_'+str(k)]=[{'cohort':'restart_seed42','round':k,'seed':42,'ids':ids}]
    return sessions


def build_solver(data,cap,labels,diag,legacy):
    _,instruction,output_format=original_definitions()
    if instruction!=legacy['instruction'] or output_format!=legacy['output_format']:raise ValueError('Original Vanilla/PAB prompt changed')
    records=[];capmap={r['id']:r for r in cap['records']};labelmap={r['id']:r for r in labels['records']}
    methods=diag['splits']['evaluation']
    maps={k:{r['id']:r for r in methods[k]['records']} for k in ('probe','uniform','train_position_mean')}
    expected={r['id'] for r in data['splits']['evaluation']}
    for key,m in maps.items():
        if len(methods[key]['records'])!=5 or set(m)!=expected:raise ValueError('Frozen prediction coverage mismatch')
    for q,source in enumerate(data['splits']['evaluation']):
        identifier=source['id'];capture=capmap[identifier];label=labelmap[identifier];steps=capture['capture']['steps']
        credits={'teacher':label['teacher']['credits']}
        for arm,key in [('ridge','probe'),('uniform','uniform'),('position','train_position_mean')]:
            frozen=maps[key][identifier]
            if frozen['steps']!=steps or frozen['steps_fingerprint']!=capture['steps_fingerprint']:raise ValueError('Frozen steps mismatch')
            if frozen['teacher_credits']!=label['teacher']['credits']:raise ValueError('Frozen Teacher anchor changed')
            credits[arm]=frozen['probe']['credits']
            if project_credits(frozen['probe']['raw'])['credits']!=credits[arm]:raise ValueError('Frozen Credit projection changed')
        item={**source,'steps':steps};requests={};budgets={}
        for arm in ARMS:
            if arm=='vanilla':word_budget=[]
            else:
                credit_contract(credits[arm],len(steps));word_budget=allocate_words(credits[arm],LEVEL_WORDS[source['level']],'constant')
                if sum(word_budget)!=LEVEL_WORDS[source['level']]:raise ValueError('Total word budget changed')
                key={'ridge':'probe','position':'train_position_mean','uniform':'uniform'}.get(arm,'probe')
                frozen=maps[key][identifier]
                stored=frozen['budgets']['teacher' if arm=='teacher' else 'probe']
                if stored!=word_budget:raise ValueError('Frozen Budget cannot be reproduced')
            budgets[arm]=word_budget
            requests[arm]={'model':SOLVER,'messages':messages_for(item,'vanilla' if arm=='vanilla' else 'weighted_pab',word_budget,instruction,output_format),
                'max_tokens':8192,'temperature':.1,'top_p':.9,'seed':42}
        records.append({'id':identifier,'answer':source['answer'],'level':source['level'],'steps':steps,
            'steps_fingerprint':capture['steps_fingerprint'],'credits':credits,'budgets_words':budgets,
            'total_words':LEVEL_WORDS[source['level']],'requests':requests,'order':list(ARMS[q:]+ARMS[:q]),
            'historical_costs':{'capture_usage':capture['capture']['usage'],'capture_metrics':capture['capture']['metrics'],
                'capture_control':capture['capture'].get('capture_control'),'decomposition_attempts':capture['attempts'],
                'teacher_stage':label['teacher_stage'],'frozen_probe_timing':maps['probe'][identifier]['probe'],
                'uniform_timing':maps['uniform'][identifier]['probe'],'position_timing':maps['train_position_mean'][identifier]['probe'],
                'reused_not_zero':True,'note':'Offline source observations on different hardware, not a measured end-to-end run.'}})
    if len(records)!=5 or len({r['id'] for r in records})!=5:raise ValueError('Require exactly five problems')
    return records


def approve(path):
    doc=verify(read(path));parent=parent_approve(CREDIT/'audit_220310.json')
    if not doc.get('passed') or doc['kind']!='followup_cpu_audit' or doc['parent_fingerprint']!=parent['fingerprint']:
        raise ValueError('Follow-up audit PASS required')
    if code_hashes()!=doc['code_sha256'] or doc['profiles']!=PROFILES or doc['cache_policy']!=CACHE:
        raise ValueError('Frozen follow-up policy changed')
    for path,h in doc['protected_files'].items():
        if file_hash(path)!=h:raise ValueError('Source changed: '+path)
    if digest(legacy_config())!=doc['legacy_config_sha256']:raise ValueError('Stored Solver config changed')
    return doc


def run(output):
    require_compute();doc={'kind':'followup_cpu_audit','passed':False}
    with lock(output):
        if Path(str(output)+'.tmp').exists():raise FileExistsError('Uncommitted audit')
        save_new(output,seal(doc))
        try:
            parent=parent_approve(CREDIT/'audit_220310.json')
            check_complete(CREDIT/'complete_teacher_220313.json',parent,'teacher')
            check_complete(CREDIT/'complete_evaluation_220315.json',parent,'evaluation')
            meta=frozen_probe(CREDIT/'probe_v1',parent);diag=verify(read(DIAGNOSTIC))
            if not diag['passed'] or diag['probe_fingerprint']!=meta['fingerprint']:raise ValueError('Frozen CPU diagnostics required')
            for p,h in diag['input_sha256'].items():
                if file_hash(p)!=h:raise ValueError('Diagnostic source changed: '+p)
            data=verify(read(SOURCE/'data.json'));cap=verify(read(SOURCE/'evaluation_recovered.json'))
            labels=verify(read(CREDIT/'evaluation_labeled.json'));validate_labels(labels,data,'evaluation',parent)
            old=legacy_config()
            if importlib.metadata.version('vllm')!='0.6.6.post1':raise ValueError('Original vLLM version required')
            from vllm.engine.arg_utils import EngineArgs
            cache_default=EngineArgs(model=PROFILES['teacher']['model']).enable_prefix_caching
            if cache_default is not False:raise ValueError('Prefix cache default changed; do not silently change historical cache policy')
            for k,v in {'model':SOLVER,'revision':SOLVER_REVISION,'max_tokens':8192,'max_model_len':12288,'temperature':.1,'top_p':.9,'seed':42,'decay':'constant'}.items():
                if old[k]!=v:raise ValueError('Stored original Solver setting mismatch: '+k)
            if old['server_settings']!={'dtype':'float16','quantization':None,'tensor_parallel_size':4,'max_num_seqs':1,'gpu_memory_utilization':.85}:
                raise ValueError('Original Solver precision/TP changed')
            if old['level_words']!={str(k):v for k,v in LEVEL_WORDS.items()}:raise ValueError('Original total word budgets changed')
            if file_hash(ROOT/'experiments/llama_qwq/run_solver_compare.py')!=old['code_sha256']:raise ValueError('Original Vanilla/Solver helper changed')
            if file_hash(ROOT/'experiments/llama_qwq/experiment_state.py')!=old['state_code_sha256']:raise ValueError('Original request/cost helper changed')
            for name,version in old['generation_packages'].items():
                if importlib.metadata.version(name)!=version:raise ValueError('Original Solver runtime changed: '+name)
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(QWQ,local_files_only=True)
            requests=build_solver(data,cap,labels,diag,old)
            for row in requests:
                row['prompt_tokens']={}
                for arm,payload in row['requests'].items():
                    n=len(tok.apply_chat_template(payload['messages'],tokenize=True,add_generation_prompt=True))
                    if n+8192>12288:raise ValueError('Solver context exceeded; no silent truncation')
                    row['prompt_tokens'][arm]=n
            teacher_rows=[];source={r['id']:r for r in data['splits']['evaluation']}
            tctx={r['id']:r for r in parent['contexts']['evaluation']}
            for row in labels['records']:
                teacher_rows.append({'id':row['id'],'payload':payload_for(source[row['id']],row,data),
                    'original_teacher_credits':row['teacher']['credits'],'steps_fingerprint':row['steps_fingerprint'],
                    'n_steps':len(row['capture']['steps']),'total_words':LEVEL_WORDS[source[row['id']]['level']],
                    'frozen_predictions':{k:next(r['probe']['credits'] for r in diag['splits']['evaluation'][k]['records'] if r['id']==row['id'])
                                          for k in ('probe','uniform','train_position_mean')},
                    'prompt_tokens':tctx[row['id']]['prompt_tokens']})
            protected={str(p):file_hash(p) for p in [DIAGNOSTIC,CREDIT/'evaluation_labeled.json',CREDIT/'probe_v1/metadata.json',
                CREDIT/'probe_v1/probe.joblib',CREDIT/'complete_teacher_220313.json',CREDIT/'complete_evaluation_220315.json']}
            protected.update(diag['input_sha256'])
            for profile,path in [('solver',QWQ),('teacher',Path(parent['model_path']))]:
                index=read(path/'model.safetensors.index.json')
                if any(not (path/name).is_file() for name in set(index['weight_map'].values())):raise ValueError('Incomplete offline '+profile+' cache')
                for f in [path/'config.json',path/'model.safetensors.index.json',*path.glob('*token*')]:
                    if f.is_file():protected[str(f)]=file_hash(f)
                if (path/'generation_config.json').is_file():
                    protected[str(path/'generation_config.json')]=file_hash(path/'generation_config.json')
            evaluator=evaluator_info('original')
            if evaluate_answer('The answer is $\\boxed{2}$.','2','stop','success',mode='original')['status']!='correct':
                raise ValueError('Original answer evaluator not operational')
            doc.update(parent_fingerprint=parent['fingerprint'],code_sha256=code_hashes(),protected_files=protected,
                profiles=PROFILES,cache_policy=CACHE,solver_requests=requests,teacher_requests=teacher_rows,
                teacher_schedule=teacher_schedule([r['id'] for r in teacher_rows]),legacy_config_sha256=digest(old),
                inherited_leakage_checks=parent['leakage'],data_fingerprint=data['fingerprint'],
                legacy_solver_config=old,evaluator=evaluator,solver_weight_bytes=read(QWQ/'model.safetensors.index.json')['metadata']['total_size'],
                teacher_logical_requests=45,solver_logical_requests=25,passed=True)
            doc['runtime_defaults']={'vllm':importlib.metadata.version('vllm'),'enable_prefix_caching':cache_default,
                                     'VLLM_USE_V1':os.environ.get('VLLM_USE_V1')}
        except Exception as exc:doc['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        atomic_save(output,seal(doc))
    if not doc['passed']:raise RuntimeError('Follow-up CPU audit FAIL')
    print('FOLLOWUP CPU AUDIT PASS: frozen predictions, 45 stability and 25 original-QwQ requests',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
