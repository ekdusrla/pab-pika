"""Compute-node-only descriptive analysis; no model fits or GPU/Teacher calls."""
import argparse
import itertools
import math
from statistics import mean
from ..common import read,verify,seal,save_new,allocate_words
from ..credit_a2000_tp4.train import spearman
from .plan import OUT,approve,require_compute,ARMS
from .hardware import check_complete
from experiment_state import atomic_save,attempt_cost


def compare(a,b,words):
    if len(a)!=len(b) or not a:raise ValueError('Aligned nonempty vectors required')
    d=[abs(x-y) for x,y in zip(a,b)]
    ba=allocate_words(a,words,'constant');bb=allocate_words(b,words,'constant')
    bd=[abs(x-y) for x,y in zip(ba,bb)]
    ranks=[(a[i]>a[j])-(a[i]<a[j]) == (b[i]>b[j])-(b[i]<b[j])
           for i,j in itertools.combinations(range(len(a)),2)]
    return {'credit_mae_pp':mean(d),'credit_rmse_pp':math.sqrt(mean([x*x for x in d])),
            'credit_max_pp':max(d),'spearman':spearman(a,b),
            'pair_order_agreement_including_ties':mean(ranks),
            'budget_mae_words':mean(bd),'budget_max_words':max(bd),
            'budget_a_words':ba,'budget_b_words':bb}


def aggregate(rows):
    result={}
    for k in ('credit_mae_pp','credit_rmse_pp','credit_max_pp','spearman',
              'pair_order_agreement_including_ties','budget_mae_words','budget_max_words'):
        values=[r[k] for r in rows if r[k] is not None]
        result[k]={'mean':mean(values) if values else None,'defined':len(values),'undefined':len(rows)-len(values)}
    return result


def online_estimate(h,arm,solver_cost):
    """Historical online reconstruction, not an observed online end-to-end measurement."""
    if arm=='vanilla':planner={'prompt_tokens':0,'completion_tokens':0,'latency_seconds':0}
    else:
        control=h.get('capture_control') or {}
        planner=dict(h['capture_usage'] if arm=='ridge' else control.get('usage',{}))
        metrics=h['capture_metrics'] if arm=='ridge' else control.get('metrics',{})
        # Replace generation-only latency with the recorded combined wall interval.
        planner['latency_seconds']=metrics.get('generation_plus_feature_seconds')
        if arm=='ridge' and planner['latency_seconds'] is not None:
            extra=metrics.get('feature_validation_seconds')
            planner['latency_seconds']=planner['latency_seconds']+extra if extra is not None else None
        if arm=='teacher':
            teacher_cost=attempt_cost([h['teacher_stage']])
            for key in ('prompt_tokens','completion_tokens','latency_seconds'):
                v=teacher_cost[key]
                planner[key]=(planner.get(key)+v['known_sum'] if planner.get(key) is not None and v['missing_attempts']==0 else None)
        elif arm in ('ridge','uniform','position'):
            p=h[{'ridge':'frozen_probe_timing','uniform':'uniform_timing','position':'position_timing'}[arm]]
            components=[p.get(k) for k in ('inference_seconds','projection_seconds')]
            if arm=='ridge':components.append(p.get('feature_read_seconds'))
            t=planner.get('latency_seconds')
            planner['latency_seconds']=t+sum(components) if t is not None and all(v is not None for v in components) else None
    result={}
    for key in ('prompt_tokens','completion_tokens','latency_seconds'):
        v=solver_cost[key];p=planner.get(key)
        result[key]=p+v['known_sum'] if p is not None and v['missing_attempts']==0 else None
    return {'estimated_online':result,'historical_planner_component':planner,
        'includes_model_startup':False,'measured_online_end_to_end':False,
        'note':'Mixed-hardware, final successful Eval decomposition path; retained source attempts disclose any extra generation costs.'}


def teacher(plan,sessions):
    sources={r['id']:r for r in plan['teacher_requests']}
    records=[r for s in sessions for r in s['records']]
    cohorts={};anchors=[];predictors=[]
    for cohort in ('same_seed','different_seed','restart_seed42'):
        rows=[];problems=[]
        for identifier,source in sources.items():
            runs=[r for r in records if r['id']==identifier and r['cohort']==cohort]
            if len(runs)!=3:raise ValueError('Three repeats per problem/cohort required')
            credits=[r['stage']['parsed']['credits'] for r in runs]
            comparisons=[]
            for i,j in itertools.combinations(range(3),2):
                metric=compare(credits[i],credits[j],source['total_words'])
                comparisons.append({'rounds':[runs[i]['round'],runs[j]['round']],**metric});rows.append(metric)
            problems.append({'id':identifier,'credits':credits,'pairwise':comparisons,
                'max_step_range_pp':max(max(c[k] for c in credits)-min(c[k] for c in credits) for k in range(source['n_steps'])),
                'exactly_identical_all_three':credits[0]==credits[1]==credits[2]})
        cohorts[cohort]={'problems':problems,'pairwise_macro':aggregate(rows),
            'independent_problem_units':5,'dependent_pair_comparisons':15}
    for r in records:
        source=sources[r['id']];actual=r['stage']['parsed']['credits']
        base={'id':r['id'],'cohort':r['cohort'],'round':r['round'],'seed':r['seed']}
        anchors.append({**base,**compare(source['original_teacher_credits'],actual,source['total_words'])})
        for name,c in source['frozen_predictions'].items():
            predictors.append({**base,'predictor':name,**compare(c,actual,source['total_words'])})
    return {'cohorts':cohorts,'original_teacher_anchor':anchors,'frozen_predictors_against_repeats':predictors,
        'predictor_macro':{name:aggregate([r for r in predictors if r['predictor']==name])
                           for name in ('probe','uniform','train_position_mean')},
        'all_attempt_cost':attempt_cost([r['stage'] for r in records]),
        'interpretation':'Descriptive paired repeats on five previously examined problems. Dependent pairs are not new samples. '
                         'Teacher variability is NOT an irreducible lower bound for predictor error. Same-seed runs can vary under TP4.'}


def solver(plan,sessions):
    records=[r for s in sessions for r in s['records']]
    if len(records)!=25 or len({(r['id'],r['arm']) for r in records})!=25:raise ValueError('Require all 25 logical requests')
    arms={};sources={r['id']:r for r in plan['solver_requests']}
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm];details=[]
        for row in rows:
            target=row['stage'];result=target.get('result',{});parsed=target.get('parsed',{})
            correct=parsed.get('answer_evaluation',{}).get('status')=='correct'
            details.append({'id':row['id'],'stage_status':target['status'],'correct':correct,
                'answer_evaluation':parsed.get('answer_evaluation'),'finish_reason':result.get('finish_reason'),
                'credits':row['credits'],'total_words':row['total_words'],'budgets_words':row['budgets_words'],
                'successful_response_usage':result.get('usage'),'all_attempt_cost':attempt_cost([target]),
                'reconstructed_cost':online_estimate(sources[row['id']]['historical_costs'],arm,attempt_cost([target])),
                'subtask_generated_tokens':None,'subtask_tokens_note':'One joint completion; per-subtask actual tokens not instrumented.'})
        arms[arm]={'correct':sum(r['correct'] for r in details),'denominator':5,
            'accuracy':sum(r['correct'] for r in details)/5,'records':details,
            'all_attempt_cost':attempt_cost([r['stage'] for r in rows])}
        arms[arm]['estimated_online_totals']={k:(sum(values) if all(v is not None for v in values) else None)
            for k in ('prompt_tokens','completion_tokens','latency_seconds')
            for values in [[r['reconstructed_cost']['estimated_online'][k] for r in details]]}
    costs=[]
    for identifier,source in sources.items():
        h=source['historical_costs'];m=h['capture_metrics'];p=h['frozen_probe_timing']
        costs.append({'id':identifier,'source_observations':h,
            'hidden_collection_instrumented_seconds':{k:m.get(k) for k in
                ('hidden_retention_seconds','alignment_seconds','feature_transfer_seconds','feature_validation_seconds')},
            'probe_inference_seconds':p.get('inference_seconds'),'feature_read_seconds':p.get('feature_read_seconds'),
            'credit_projection_seconds':p.get('projection_seconds'),
            'teacher_assessment_all_attempt_cost':attempt_cost([h['teacher_stage']]),
            'decomposition_usage':h['capture_usage'],'capture_control':h['capture_control'],
            'hidden_collection_total_causal_overhead_seconds':None,
            'causal_note':'Instrumented component times and capture/control wall difference do not isolate causal overhead; timings overlap.',
            'online_cost_recipe':{
                'vanilla':'Solver only',
                'teacher':'control decomposition + stored Teacher all attempts + Solver',
                'uniform':'control decomposition + frozen Uniform timing + Solver',
                'position':'control decomposition + frozen Position timing + Solver',
                'ridge':'capture decomposition with hidden states + frozen feature read/predict/projection + Solver'},
            'measured_end_to_end_seconds':None,'reconstructed_online_cost_is_estimate':True,
            'hardware_note':'HF 3090 capture, A2000 TP4 Teacher, CPU Probe and 3090 TP4 Solver are different environments.'})
    paired=[{'id':identifier,'correct':{arm:next(r['correct'] for r in arms[arm]['records'] if r['id']==identifier)
                                      for arm in ARMS}} for identifier in sources]
    return {'arms':arms,'paired_correctness':paired,'historical_component_cost_ledger':costs,
        'current_run':{'planner_calls':0,'planner_artifacts_reused':True,
                       'all_solver_attempt_cost':attempt_cost([r['stage'] for r in records])},
        'reused_historical_cost_is_zero':False,'actual_full_online_end_to_end_measured':False,
        'interpretation':'Exploratory five-question paired Pilot. Never select a winner here and reuse these five as independent confirmation.'}


def run(a):
    require_compute();output=OUT/f'analysis_{a.role}_{a.main_job}_{__import__("os").environ["SLURM_JOB_ID"]}.json'
    doc={'passed':False,'role':a.role,'main_job':a.main_job};save_new(output,seal(doc))
    try:
        plan=approve(a.audit);gate=check_complete(a.role,'main',a.main_job,plan)
        sessions=[verify(read(p)) for p in gate['protected_outputs'] if '/responses_' in p]
        for s in sessions:
            if not s['passed'] or s['audit_fingerprint']!=plan['fingerprint']:raise ValueError('Incomplete session')
        result=teacher(plan,sessions) if a.role=='teacher' else solver(plan,sessions)
        memory=[verify(read(p)) for p in gate['protected_outputs'] if '/memory_' in p]
        result['allocation_gpu_seconds_estimate']=4*sum(r['wall_seconds'] for r in memory)
        result['allocation_gpu_seconds_scope']='Four assigned GPUs x monitored session wall; includes loading/idling, excludes preflight/cleanup. Full allocation cost needs sacct ElapsedRaw x4.'
        doc.update(passed=True,audit_fingerprint=plan['fingerprint'],profile=plan['profiles'][a.role],
                   main_completion_fingerprint=gate['fingerprint'],results=result)
        approve(a.audit)
    except Exception as exc:doc['passed']=False;doc['error']={'type':type(exc).__name__,'message':str(exc)}
    atomic_save(output,seal(doc))
    if not doc['passed']:raise RuntimeError('CPU analysis failed')
    print('FOLLOWUP '+a.role.upper()+' CPU ANALYSIS PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('teacher','solver'),required=True)
    p.add_argument('--audit',required=True);p.add_argument('--main-job',required=True);run(p.parse_args())
