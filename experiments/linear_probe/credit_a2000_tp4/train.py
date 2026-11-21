"""CPU-only 33-layer x fixed-alpha search; Validation selects, never fits."""
import argparse
import copy
import math
import os
import time
import traceback
from pathlib import Path

from ..common import (read,verify,seal,save_new,output_path,file_hash,digest,project_credits,runtime,
                      allocate_words,LEVEL_WORDS)
from ..learning import feature_rows,matrix,predict
from ..cli import teacher_rows,capture_signature
from .audit import SOURCE,OUT,approve,require_compute,capture_projection
from .teacher import expected_signature,payload_for,frozen_probe
from .gpu import check_complete
from experiment_state import atomic_save


def ranks(values):
    result=[0.]*len(values);order=sorted(range(len(values)),key=lambda i:values[i]);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):result[order[k]]=(i+j+1)/2
        i=j
    return result


def spearman(a,b):
    x,y=ranks(a),ranks(b);mx=sum(x)/len(x);my=sum(y)/len(y)
    denominator=math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
    return sum((u-mx)*(v-my) for u,v in zip(x,y))/denominator if denominator else None


def validate_labels(doc,data,split,plan):
    rows=teacher_rows(doc,data,split)
    if doc['config']['audit_fingerprint']!=plan['fingerprint'] or doc['config']['teacher_settings']!=expected_signature():
        raise ValueError('Teacher audit/settings mismatch')
    sources={r['id']:r for r in data['splits'][split]}
    for row in rows:
        if capture_projection(row)!=plan['capture_projections'][split][row['id']]:raise ValueError('Features/steps changed')
        expected=payload_for(sources[row['id']],row,data)
        if row['teacher_stage']['request']!=expected or row['teacher']['messages']!=expected['messages']:
            raise ValueError('Teacher did not assess exactly the frozen PAB request')
    return rows


def cached(rows):
    output=[]
    for row in rows:
        item=copy.copy(row);item['capture']=copy.copy(row['capture'])
        item['capture']['features']=feature_rows(row)  # Private RAM cache; never written back.
        output.append(item)
    return output


def fit_grid(train,validation,layers,alphas):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    if not layers or not alphas or any(not math.isfinite(a) or a<=0 for a in alphas):raise ValueError('Invalid fixed grid')
    best=None;scores=[]
    for layer in sorted(layers):
        X,y=matrix(train,layer);V,_=matrix(validation,layer)
        for alpha in sorted(alphas):
            model=Pipeline([('scaler',StandardScaler()),('regressor',Ridge(alpha=alpha))])
            model.fit(X,y)  # ONLY Train arrays enter scaler.fit / Ridge.fit.
            raw=model.predict(V);offset=0;errors=[]
            for row in validation:
                n=len(row['capture']['steps']);credits=project_credits(raw[offset:offset+n])['credits'];offset+=n
                errors.append(float(np.mean(np.abs(np.asarray(credits)-row['teacher']['credits']))))
            score=float(np.mean(errors));key=(score,layer,alpha)
            scores.append({'layer':layer,'alpha':alpha,'validation_problem_mean_credit_mae':score})
            if best is None or key<best[0]:best=(key,model)
    return best[1],{'best_layer_idx':best[0][1],'alpha':best[0][2],
        'selection_metric':'validation problem-mean projected-credit MAE','selection_scores':scores,
        'validation_used_for_fit':False,'tie_break':'lowest layer, then lowest alpha'}


def comparison(rows,pipeline,layer,data,split,categories=None):
    levels={r['id']:r['level'] for r in data['splits'][split]};records=[];all_diff=[];all_budget=[]
    for row in rows:
        prediction=predict(pipeline,row,layer);truth=row['teacher']['credits'];credits=prediction['credits']
        diff=[p-t for p,t in zip(credits,truth)];all_diff+=diff
        budgets={name:allocate_words(values,LEVEL_WORDS[levels[row['id']]],'constant')
                 for name,values in [('teacher',truth),('probe',credits)]}
        bd=[p-t for p,t in zip(budgets['probe'],budgets['teacher'])];all_budget+=bd
        records.append({'id':row['id'],'steps_fingerprint':row['steps_fingerprint'],'steps':row['capture']['steps'],
            'decomposition_category':(categories or {}).get(row['id'],'unspecified'),
            'teacher_credits':truth,'probe':prediction,'teacher_ranks':ranks(truth),'probe_ranks':ranks(credits),
            'spearman_projected':spearman(truth,credits),'spearman_raw':spearman(truth,prediction['raw']),
            'credit_mae':sum(abs(d) for d in diff)/len(diff),'credit_rmse':math.sqrt(sum(d*d for d in diff)/len(diff)),
            'budgets':budgets,'budget_l1':sum(abs(x) for x in bd),
            'steps_comparison':[{'index':i,'teacher_credit':t,'probe_credit':p,'credit_difference':d,
                'teacher_budget_words':bt,'probe_budget_words':bp,'budget_difference':db}
                for i,(t,p,d,bt,bp,db) in enumerate(zip(truth,credits,diff,budgets['teacher'],budgets['probe'],bd))]})
    correlations=[r['spearman_projected'] for r in records if r['spearman_projected'] is not None]
    return {'records':records,'summary':{'problems':len(records),'steps':len(all_diff),
        'problem_macro_credit_mae':sum(r['credit_mae'] for r in records)/len(records),
        'problem_weighted_rmse':math.sqrt(sum(r['credit_rmse']**2 for r in records)/len(records)),
        'step_micro_mae':sum(abs(d) for d in all_diff)/len(all_diff),
        'step_micro_rmse':math.sqrt(sum(d*d for d in all_diff)/len(all_diff)),
        'mean_spearman_defined_only':sum(correlations)/len(correlations) if correlations else None,
        'spearman_undefined_problems':len(records)-len(correlations),
        'budget_step_micro_mae':sum(abs(d) for d in all_budget)/len(all_budget)},
        'rules':{'credit_projection':'existing clamp-zero-reserve-one-largest-remainder-v1',
            'teacher_target':'positive integer percentage points, sum=100; no dataset-wide target scaling',
            'budget':'unchanged PAB allocate_words; constant schedule; leftovers favor largest weights',
            'level_words':LEVEL_WORDS,'rank_ties':'average ranks; constant-vector Spearman is null'}}


def train(a,plan):
    gate=check_complete(OUT/f'complete_teacher_{a.teacher_job}.json',plan,'teacher')
    data=verify(read(SOURCE/'data.json'))
    trdoc=verify(read(OUT/'train_labeled.json'));vadoc=verify(read(OUT/'validation_labeled.json'))
    tr=validate_labels(trdoc,data,'train',plan);va=validate_labels(vadoc,data,'validation',plan)
    if capture_signature(trdoc)!=capture_signature(vadoc):raise ValueError('Train/Validation feature signature differs')
    target=output_path(a.probe);target.mkdir(parents=True,exist_ok=False)
    state={'passed':False,'phase':'fit','teacher_gate_fingerprint':gate['fingerprint']};save_new(target/'status.json',seal(state))
    begin=time.perf_counter()
    try:
        training=cached(tr);validation=cached(va)
        pipeline,selection=fit_grid(training,validation,plan['selection']['layers'],plan['selection']['alphas'])
        report=comparison(validation,pipeline,selection['best_layer_idx'],data,'validation',plan['capture_categories']['validation'])
        import joblib
        joblib.dump(pipeline,target/'probe.joblib')
        meta={'kind':'credit_probe_artifact_v1','selection_frozen':True,'audit_fingerprint':plan['fingerprint'],
            **selection,'data_fingerprint':data['fingerprint'],'teacher_settings':expected_signature(),
            'feature_signature':capture_signature(trdoc),'feature_dimension':pipeline.n_features_in_,
            'artifact_sha256':file_hash(target/'probe.joblib'),'train_fingerprint':trdoc['fingerprint'],
            'validation_fingerprint':vadoc['fingerprint'],'train_problem_hashes':[r['problem_hash'] for r in tr],
            'validation_problem_hashes':[r['problem_hash'] for r in va],
            'train_problem_count':24,'validation_problem_count':6,'layers':list(range(33)),
            'runtime':{**runtime(),'engine':'CPU StandardScaler+Ridge'},
            'alphas':plan['selection']['alphas'],'training_seconds':time.perf_counter()-begin,
            'refit_on_validation':False,'evaluation_labels_accessed':False,
            'note':'Validation scores are model-selection results, not independent performance estimates.'}
        save_new(target/'validation_report.json',seal(report));save_new(target/'metadata.json',seal(meta))
        approve(a.audit);state.update(passed=True,phase='complete')
    except Exception as exc:
        state['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        raise
    finally:atomic_save(target/'status.json',seal(state))
    print('PROBE TRAIN PASS: Train-only fit; layer+alpha selected on six Validation problems',flush=True)


def evaluate(a,plan):
    check_complete(OUT/f'complete_evaluation_{a.evaluation_job}.json',plan,'evaluation')
    meta=frozen_probe(a.probe,plan)
    if not verify(read(a.probe/'status.json'))['passed']:raise ValueError('Training failed')
    data=verify(read(SOURCE/'data.json'));labels=verify(read(OUT/'evaluation_labeled.json'))
    rows=validate_labels(labels,data,'evaluation',plan)
    forbidden=set(meta['train_problem_hashes']+meta['validation_problem_hashes'])
    if any(r['problem_hash'] in forbidden for r in rows):raise ValueError('Evaluation overlap')
    import joblib
    pipeline=joblib.load(a.probe/'probe.joblib')
    report=comparison(rows,pipeline,meta['best_layer_idx'],data,'evaluation',plan['capture_categories']['evaluation'])
    report.update(passed=True,audit_fingerprint=plan['fingerprint'],probe_fingerprint=meta['fingerprint'],
                  split='evaluation',teacher_all_attempt_cost=labels['teacher_all_attempt_cost'],
                  recovery_cost_ledger=str(SOURCE/'final_v2_219684.json'))
    save_new(OUT/f'evaluation_report_{os.environ["SLURM_JOB_ID"]}.json',seal(report))
    approve(a.audit);print('PROBE EVALUATION PASS: five independent problems; no model selection',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--mode',choices=('train','evaluate'),default='train');p.add_argument('--teacher-job')
    p.add_argument('--evaluation-job');p.add_argument('--probe',type=Path,default=OUT/'probe_v1')
    a=p.parse_args();require_compute();plan=approve(a.audit)
    if a.mode=='train':train(a,plan)
    else:evaluate(a,plan)
