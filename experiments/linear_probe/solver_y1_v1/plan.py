"""CPU-only supplemental audit of 220339 and frozen five-arm requests."""
import argparse
import copy
import os
from pathlib import Path
from ..common import ROOT,read,verify,seal,save_new,file_hash,digest
from ..followup_v1 import plan as original
from ..followup_v1.hardware import check_complete as old_complete
from ..stability_analysis_v1.analyze import integrity
from experiment_state import atomic_save
from math_answer_eval import evaluator_info,evaluate_answer

OUT=ROOT/'experiments/linear_probe/runs/solver_y1_v1'
PARENT=original.OUT/'audit_220327.json'
STABILITY=ROOT/'experiments/linear_probe/runs/stability_analysis_v1/analysis_220329_220339.json'
REPORT=STABILITY.parent/'REPORT_220329_220339.ko.md'
PROFILES={'solver':{**original.PROFILES['solver'],'cpus':16,'ram_gib':64}}
CACHE=copy.deepcopy(original.CACHE)
require_compute=original.require_compute
ARMS=original.ARMS
POLICY={'node':'moana-y1','cpus':16,'ram_gib':64,'short_tokens':128,
        'pilot_max_tokens':8192,'smoke_requests':4,'pilot_requests':25,
        'budget_compliance':'full-completion whitespace words vs total soft word budget; subtask uninstrumented'}


def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for pattern in ('*.py','*.sh','*.sbatch') for p in Path(__file__).parent.glob(pattern)}


def inspect_parent():
    parent=original.approve(PARENT);analysis=verify(read(STABILITY))
    if not analysis['passed'] or analysis['main_job']!='220329' or analysis['cpu_job']!='220339':raise ValueError('Require CPU analysis 220339 PASS')
    if analysis['audit_fingerprint']!=parent['fingerprint']:raise ValueError('Stability audit differs')
    for p,h in analysis['protected_input_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Stability protected input changed: '+p)
    for p,h in analysis['source_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Stability analysis source changed')
    if file_hash(REPORT)!=analysis['report_sha256']:raise ValueError('Korean report changed')
    gate=old_complete('teacher','main','220329',parent)
    if gate['fingerprint']!=analysis['completion_fingerprint']:raise ValueError('Teacher completion differs')
    _,checks=integrity(parent,gate,'220329')
    if checks!=analysis['request_checks']:raise ValueError('Rechecked request integrity differs')
    data=verify(read(original.SOURCE/'data.json'));cap=verify(read(original.SOURCE/'evaluation_recovered.json'))
    labels=verify(read(original.CREDIT/'evaluation_labeled.json'));diag=verify(read(original.DIAGNOSTIC))
    rebuilt=original.build_solver(data,cap,labels,diag,original.legacy_config())
    # Compare messages, frozen Credit and original Budget without any inference.
    expected=copy.deepcopy(parent['solver_requests'])
    for r in expected:r.pop('prompt_tokens')
    if rebuilt!=expected:raise ValueError('Frozen five-arm Solver request reconstruction differs')
    if len(rebuilt)!=5 or sum(len(r['order']) for r in rebuilt)!=25:raise ValueError('Require frozen 5x5 plan')
    return parent,analysis


def approve(path):
    doc=verify(read(path));parent,analysis=inspect_parent()
    if not doc.get('passed') or doc['parent_fingerprint']!=parent['fingerprint'] or doc['stability_fingerprint']!=analysis['fingerprint']:
        raise ValueError('New y1 CPU audit PASS required')
    if doc['code_sha256']!=code_hashes() or doc['profiles']!=PROFILES or doc['policy']!=POLICY or doc['cache_policy']!=CACHE:
        raise ValueError('y1 code/resource policy changed')
    for p,h in doc['protected_files'].items():
        if file_hash(p)!=h:raise ValueError('Protected y1 input changed')
    return doc


def run(output):
    require_compute();doc={'passed':False,'kind':'y1_solver_supplemental_cpu_audit'}
    save_new(output,seal(doc))
    try:
        parent,analysis=inspect_parent()
        from ..resources import validate_allocation
        from .analysis import budget_usage
        validate_allocation({'cpus':96,'gpus':8,'normal_gpus':8,'memory_mib':428737},4,16,64*1024)
        from .client import planned
        from .hardware import allocated_uuids
        inventory='\n'.join(f'{i}, GPU-{i}, NVIDIA GeForce RTX 3090, 24576, 24000' for i in range(4))
        assert allocated_uuids('0-3',inventory)=={f'GPU-{i}' for i in range(4)}
        assert allocated_uuids('0,1,2,3',inventory)=={f'GPU-{i}' for i in range(4)}
        smoke=planned(parent,'solver','smoke');main=planned(parent,'solver','main')
        assert len(smoke)==4 and len(main)==25
        assert [r['payload']['max_tokens'] for r in smoke]==[128,128,8192,8192]
        assert all(r['payload']['seed']==42 for r in smoke+main)
        assert budget_usage('one two','stop',2)['total_word_budget_compliant'] is True
        assert budget_usage('one two three','stop',2)['total_word_budget_compliant'] is False
        assert budget_usage('one','length',2)['total_word_budget_compliant'] is None
        assert evaluate_answer('The answer is $\\boxed{2}$.','2','stop','success',mode='original')['status']=='correct'
        protected={str(p):file_hash(p) for p in (PARENT,STABILITY,REPORT)}
        doc.update(passed=True,parent_fingerprint=parent['fingerprint'],stability_fingerprint=analysis['fingerprint'],
            protected_files=protected,profiles=PROFILES,policy=POLICY,cache_policy=CACHE,code_sha256=code_hashes(),
            solver_requests=parent['solver_requests'],legacy_solver_config=parent['legacy_solver_config'],evaluator=evaluator_info('original'),
            teacher_schedule={},teacher_logical_requests=0,solver_logical_requests=25,
            resource_change={'old_cpus':48,'old_ram_gib':160,'new_cpus':16,'new_ram_gib':64,
                             'host_memory_sufficiency':'BLOCKED until successful smoke under this allocation',
                             'latency_comparability':'new host CPU/RAM profile; record separately'},
            no_inference_or_refit=True,verified_teacher_requests=45)
    except Exception as exc:doc['passed']=False;doc['error']={'type':type(exc).__name__,'message':str(exc)}
    atomic_save(output,seal(doc))
    if not doc['passed']:raise RuntimeError('y1 supplemental CPU audit FAIL')
    print('Y1 SOLVER CPU AUDIT PASS: 220339, 45 Teacher records and frozen 25 Solver requests verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
