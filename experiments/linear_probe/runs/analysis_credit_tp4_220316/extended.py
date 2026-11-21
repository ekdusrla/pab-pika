"""Optional compute-node-only read-only revalidation/CPU diagnostics; never refits Ridge."""
import argparse
import collections
import copy
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from experiments.linear_probe.common import read,verify,seal,save_new,file_hash
from experiments.linear_probe.credit_a2000_tp4.audit import SOURCE,OUT,approve,require_compute
from experiments.linear_probe.credit_a2000_tp4.gpu import check_complete
from experiments.linear_probe.credit_a2000_tp4.teacher import frozen_probe
from experiments.linear_probe.credit_a2000_tp4.train import validate_labels,comparison,spearman
from experiment_state import atomic_save,lock


class Uniform:
    def predict(self,X):
        import numpy as np
        return np.ones(len(X),dtype=float)


class TrainPositionMean:
    def __init__(self,rows):
        import numpy as np
        groups=collections.defaultdict(list)
        for row in rows:groups[len(row['capture']['steps'])].append(row['teacher']['credits'])
        self.means={n:np.mean(values,axis=0) for n,values in groups.items()}
        self.counts={n:len(values) for n,values in groups.items()}
    def predict(self,X):
        import numpy as np
        return self.means.get(len(X),np.ones(len(X))).copy()


def additional(report):
    rows=report['records'];diffs=[abs(s['credit_difference']) for r in rows for s in r['steps_comparison']]
    budget=[abs(s['budget_difference']) for r in rows for s in r['steps_comparison']]
    correlations=[spearman(r['budgets']['teacher'],r['budgets']['probe']) for r in rows]
    defined=[v for v in correlations if v is not None]
    totals=[sum(r['budgets']['teacher']) for r in rows]
    return {'credit_max_abs_error_pp':max(diffs),'budget_max_abs_error_words':max(budget),
        'budget_problem_mean_l1_words':sum(r['budget_l1'] for r in rows)/len(rows),
        'budget_problem_macro_step_mae_words':sum(r['budget_l1']/len(r['steps']) for r in rows)/len(rows),
        'budget_spearman_defined_mean':sum(defined)/len(defined) if defined else None,
        'budget_spearman_undefined_problems':len(rows)-len(defined),
        'reallocated_word_mass':sum(r['budget_l1'] for r in rows)/2,
        'reallocated_fraction_macro':sum(r['budget_l1']/(2*t) for r,t in zip(rows,totals))/len(rows),
        'reallocated_fraction_total_weighted':sum(r['budget_l1'] for r in rows)/(2*sum(totals)),
        'last_step_is_predicted_max_problems':sum(r['probe']['credits'][-1]==max(r['probe']['credits']) for r in rows),
        'teacher_last_step_is_max_problems':sum(r['teacher_credits'][-1]==max(r['teacher_credits']) for r in rows),
        'budget_unit':'words requested in prompt, not actual generated tokens',
        'spearman_note':'Integer uniform projection can introduce artificial rank ties/breaks; inspect raw ranks too.'}


def resample_difference(a,b):
    import numpy as np
    differences=np.array([x['credit_mae']-y['credit_mae'] for x,y in zip(a['records'],b['records'])])
    rng=np.random.default_rng(20261116);n=len(differences)
    means=differences[rng.integers(0,n,size=(2000,n))].mean(axis=1)
    return {'probe_minus_baseline_problem_macro_mae':float(differences.mean()),
        'probe_wins':int((differences<0).sum()),'ties':int((differences==0).sum()),'probe_losses':int((differences>0).sum()),
        'paired_problem_bootstrap_95_percentile_interval':[float(x) for x in np.quantile(means,[.025,.975])],
        'bootstrap_samples':2000,'seed':20261116,'unit':'whole problem, no step-level resampling',
        'interpretation':'Exploratory/post-hoc; five problems cannot support a strong generalization claim.'}


def run():
    require_compute()
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=1:raise RuntimeError('Request one CPU')
    dest=Path(__file__).parent/f'extended_{os.environ["SLURM_JOB_ID"]}.json'
    report={'kind':'posthoc_cpu_analysis','passed':False,'no_ridge_refit':True,'no_gpu_or_llm_calls':True}
    with lock(dest):
        if Path(str(dest)+'.tmp').exists():raise FileExistsError('Uncommitted analysis output')
        save_new(dest,seal(report))
        try:
            audit=approve(OUT/'audit_220310.json')
            check_complete(OUT/'complete_teacher_220313.json',audit,'teacher')
            check_complete(OUT/'complete_evaluation_220315.json',audit,'evaluation')
            metadata=frozen_probe(OUT/'probe_v1',audit)
            data=verify(read(SOURCE/'data.json'))
            saved=verify(read(OUT/'evaluation_report_220316.json'))
            if saved['probe_fingerprint']!=metadata['fingerprint'] or saved['audit_fingerprint']!=audit['fingerprint']:
                raise ValueError('Evaluation metadata mismatch')
            docs={s:verify(read(OUT/(s+'_labeled.json'))) for s in ('train','validation','evaluation')}
            if docs['train']['fingerprint']!=metadata['train_fingerprint'] or docs['validation']['fingerprint']!=metadata['validation_fingerprint']:
                raise ValueError('Training label fingerprints differ')
            rows={s:validate_labels(docs[s],data,s,audit) for s in docs}
            if any(len(rows[s])!=n for s,n in [('train',24),('validation',6),('evaluation',5)]):raise ValueError('Split count mismatch')
            ids=[{r['id'] for r in rows[s]} for s in rows]
            hashes=[{r['problem_hash'] for r in rows[s]} for s in rows]
            if any(ids[i]&ids[j] or hashes[i]&hashes[j] for i in range(3) for j in range(i)):
                raise ValueError('Cross-split overlap')
            import joblib
            import numpy as np
            model=joblib.load(OUT/'probe_v1/probe.joblib')  # Already checksum-verified locally trusted artifact.
            position=TrainPositionMean(rows['train']);layer=metadata['best_layer_idx']
            report.update(audit_fingerprint=audit['fingerprint'],probe_fingerprint=metadata['fingerprint'],
                source_report_fingerprint=saved['fingerprint'],selected_layer=layer,alpha=metadata['alpha'],
                position_baseline_train_counts=position.counts,
                position_baseline_definition='Mean Teacher Credit per (n_steps,index), Train only; unseen n uses uniform; same Credit projection/Budget.',
                input_sha256={str(p):file_hash(p) for p in [OUT/'evaluation_report_220316.json',OUT/'probe_v1/metadata.json',OUT/'probe_v1/probe.joblib']},
                splits={})
            for split in rows:
                results={}
                for name,pipeline in [('probe',model),('uniform',Uniform()),('train_position_mean',position)]:
                    result=comparison(rows[split],pipeline,layer,data,split,audit['capture_categories'][split])
                    result['additional']=additional(result);results[name]=result
                report['splits'][split]=results
            actual=report['splits']['evaluation']['probe']
            for old,new in zip(saved['records'],actual['records']):
                if old['id']!=new['id'] or old['probe']['credits']!=new['probe']['credits'] or old['budgets']!=new['budgets']:
                    raise ValueError('Saved frozen evaluation cannot be reproduced')
            for split,path in [('validation',OUT/'probe_v1/validation_report.json'),('evaluation',OUT/'evaluation_report_220316.json')]:
                original=verify(read(path));fresh=report['splits'][split]['probe']
                for key,value in original['summary'].items():
                    computed=fresh['summary'][key]
                    if value is None:
                        if computed is not None:raise ValueError('Summary null mismatch')
                    elif not np.isclose(value,computed,rtol=1e-10,atol=1e-10):raise ValueError('Summary mismatch: '+key)
            report['evaluation_paired_comparisons']={name:resample_difference(actual,report['splits']['evaluation'][name])
                                                    for name in ('uniform','train_position_mean')}
            report['teacher_costs']={s:docs[s]['teacher_all_attempt_cost'] for s in docs}
            approve(OUT/'audit_220310.json');report['passed']=True
        except Exception as exc:
            report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        atomic_save(dest,seal(report))
    if not report['passed']:raise RuntimeError('Post-hoc CPU analysis failed; new diagnostic file saved')
    print('EXTENDED CPU ANALYSIS PASS: original artifacts unchanged; diagnostics saved separately',flush=True)


if __name__=='__main__':run()
