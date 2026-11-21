"""Compute-node-only integrity checks, repeated-label metrics and Korean report."""
import argparse
import itertools
import os
from pathlib import Path
from statistics import mean

from ..common import read,verify,seal,save_new,file_hash,digest
from ..followup_v1.plan import approve,require_compute,OUT as SOURCE,PROFILES,CACHE
from ..followup_v1.hardware import check_complete
from ..followup_v1.client import planned,validate
from ..followup_v1.analysis import teacher,compare,aggregate
from experiment_state import atomic_save

OUT=SOURCE.parent/'stability_analysis_v1'
COHORTS=('same_seed','different_seed','restart_seed42')


def integrity(plan,gate,job):
    sessions=[];checks=[]
    for name in plan['teacher_schedule']:
        response=SOURCE/f'responses_teacher_{job}_{name}.json'
        actual=verify(read(response));expected=planned(plan,'teacher',name)
        if str(response) not in gate['protected_outputs']:raise ValueError('Unprotected response')
        if not actual['passed'] or actual['session']!=name or actual['role']!='teacher':raise ValueError('Wrong/failed session')
        if actual['audit_fingerprint']!=plan['fingerprint'] or actual['profile']!=PROFILES['teacher'] or actual['cache_policy']!=CACHE:
            raise ValueError('Session configuration differs')
        if actual['retries']!=0 or actual['sdk_retries']!=0:raise ValueError('Stability retry policy differs')
        if len(actual['records'])!=len(expected):raise ValueError('Request count mismatch')
        for index,(row,want) in enumerate(zip(actual['records'],expected)):
            for key in ('id','cohort','round','seed','steps_fingerprint','n_steps','payload'):
                if row[key]!=want[key]:raise ValueError('Request plan mismatch: '+key)
            stage=row['stage']
            if row['request_order']!=index or row['request_fingerprint']!=digest(want['payload']):raise ValueError('Order/request fingerprint mismatch')
            if stage['status']!='success' or stage['request']!=want['payload']:raise ValueError('Failed/wrong request')
            attempts=stage['attempts']
            if len(attempts)!=1 or attempts[0]['status']!='success':raise ValueError('Unexpected retry/failed attempt')
            parsed=validate(stage['result'],'teacher',row)
            if parsed!=stage['parsed']:raise ValueError('Stored credits do not match reparsed raw response')
            if attempts[0]['request_fingerprint']!=digest(want['payload']):raise ValueError('Attempt fingerprint mismatch')
            if any(attempts[0].get(k)!=stage['result'].get(k) for k in ('content','finish_reason','usage','raw_response')):
                raise ValueError('Attempt/result disagreement')
            checks.append({'session':name,'order':index,'id':row['id'],'cohort':row['cohort'],
                'round':row['round'],'seed':row['seed'],'steps_fingerprint':row['steps_fingerprint'],
                'passed':True,'finish_reason':stage['result']['finish_reason'],'usage':stage['result']['usage']})
        for kind in ('launch','memory','server'):
            p=SOURCE/f'{kind}_teacher_{job}_{name}.json'
            if str(p) not in gate['protected_outputs']:raise ValueError('Unprotected session metadata')
            d=verify(read(p))
            if kind in ('launch','memory') and not d['passed']:raise ValueError('GPU guard failed')
        sessions.append(actual)
    if len(checks)!=45:raise ValueError('Require all 45 checked requests')
    return sessions,checks


def enrich(plan,result):
    for cohort in COHORTS:
        c=result['cohorts'][cohort];exact=0
        for row in c['problems']:
            pairs=list(itertools.combinations(row['credits'],2))
            row['exact_pair_count']=sum(a==b for a,b in pairs)
            row['pair_count']=3;row['exact_pair_fraction']=row['exact_pair_count']/3
            row['pairwise_problem_mean']=aggregate(row['pairwise'])
            exact+=row['exact_pair_count']
        c['exact_pair_count']=exact;c['pair_count']=15;c['exact_pair_fraction']=exact/15
        c['all_three_identical_problem_count']=sum(r['exactly_identical_all_three'] for r in c['problems'])
        c['all_three_identical_problem_fraction']=c['all_three_identical_problem_count']/5
        c['original_anchor_macro']=aggregate([r for r in result['original_teacher_anchor'] if r['cohort']==cohort])
        c['frozen_predictor_macro']={key:aggregate([r for r in result['frozen_predictors_against_repeats']
            if r['cohort']==cohort and r['predictor']==key]) for key in ('probe','uniform','train_position_mean')}
    anchored={}
    for key in ('probe','uniform','train_position_mean'):
        rows=[{'id':s['id'],'n_steps':s['n_steps'],
               **compare(s['frozen_predictions'][key],s['original_teacher_credits'],s['total_words'])}
              for s in plan['teacher_requests']]
        anchored[key]={'problems':rows,'macro':aggregate(rows),
            'step_micro_credit_mae_pp':sum(r['credit_mae_pp']*r['n_steps'] for r in rows)/sum(r['n_steps'] for r in rows),
            'step_micro_budget_mae_words':sum(r['budget_mae_words']*r['n_steps'] for r in rows)/sum(r['n_steps'] for r in rows)}
    result['frozen_predictors_against_original_teacher']=anchored
    result['ridge_minus_position_original_credit_mae_pp']=(anchored['probe']['macro']['credit_mae_pp']['mean']-
                                                          anchored['train_position_mean']['macro']['credit_mae_pp']['mean'])
    return result


def fmt(v):return '미정의' if v is None else f'{v:.4f}'


def markdown(doc):
    r=doc['results'];lines=[f'# Teacher Stability 분석: Main {doc["main_job"]}',
        '',f'CPU 분석 Job {doc["cpu_job"]}: **PASS**. 기존 파일은 읽기 전용으로 검사했으며 GPU/LLM 호출·학습·예측 재실행은 없다.',
        '',f'- 원본 완료 gate fingerprint: `{doc["completion_fingerprint"]}`',
        f'- 감사 fingerprint: `{doc["audit_fingerprint"]}`',
        '- 45개 요청의 순서·seed·payload·steps fingerprint·원본 응답 재파싱·usage·단일 성공 시도를 검사했다.',
        '- 보호된 응답/launch/memory/server SHA와 부모 감사의 기존 데이터·NPZ·모델 SHA를 다시 확인했다.',
        '', '## 반복 내 안정성', '',
        '| 조건 | Credit MAE %p | Spearman | Budget MAE words | 완전 일치 쌍 | 3회 모두 동일한 문제 |',
        '|---|---:|---:|---:|---:|---:|']
    for cohort in COHORTS:
        c=r['cohorts'][cohort];m=c['pairwise_macro']
        lines.append(f'| {cohort} | {fmt(m["credit_mae_pp"]["mean"])} | {fmt(m["spearman"]["mean"])} '
            f'| {fmt(m["budget_mae_words"]["mean"])} | {c["exact_pair_count"]}/15 ({100*c["exact_pair_fraction"]:.1f}%) '
            f'| {c["all_three_identical_problem_count"]}/5 |')
    lines+=['','Credit/Budget MAE는 문제별3쌍의 step 평균을 다시 평균한 macro 값이다. 각 조건의15쌍은 독립15문제가 아니라 동일5문제의 종속 비교다.',
        'Spearman은 tied ranks를 사용하며 상수 벡터는 미정의(null)다. JSON에 defined/undefined 개수를 공개한다.',
        '', '### 문제별 반복 비교', '',
        '| 조건 | 문제 ID | Credit MAE %p | Spearman | Budget MAE words | 최대 step 편차 %p | 일치 쌍 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for cohort in COHORTS:
        for p in r['cohorts'][cohort]['problems']:
            m=p['pairwise_problem_mean']
            lines.append(f'| {cohort} | {p["id"]} | {fmt(m["credit_mae_pp"]["mean"])} | {fmt(m["spearman"]["mean"])} '
                f'| {fmt(m["budget_mae_words"]["mean"])} | {p["max_step_range_pp"]} | {p["exact_pair_count"]}/3 |')
    lines+=['','## 기존 Teacher 라벨 기준 frozen 예측', '',
        '| 방법 | Credit macro MAE %p | Spearman | Budget macro MAE words | Budget step-micro MAE words |',
        '|---|---:|---:|---:|---:|']
    for key,p in r['frozen_predictors_against_original_teacher'].items():
        m=p['macro'];lines.append(f'| {key} | {fmt(m["credit_mae_pp"]["mean"])} | {fmt(m["spearman"]["mean"])} '
            f'| {fmt(m["budget_mae_words"]["mean"])} | {fmt(p["step_micro_budget_mae_words"])} |')
    lines+=['','### 기존 라벨 대비 반복 Teacher / 반복 라벨 대비 frozen 예측', '',
        '| 조건 | 반복 Teacher→기존 Teacher Credit MAE %p | Ridge→반복 Teacher MAE %p | Position→반복 Teacher MAE %p |',
        '|---|---:|---:|---:|']
    for cohort in COHORTS:
        c=r['cohorts'][cohort];p=c['frozen_predictor_macro']
        lines.append(f'| {cohort} | {fmt(c["original_anchor_macro"]["credit_mae_pp"]["mean"])} '
            f'| {fmt(p["probe"]["credit_mae_pp"]["mean"])} | {fmt(p["train_position_mean"]["credit_mae_pp"]["mean"])} |')
    exact=[r['cohorts'][k]['exact_pair_count']==15 for k in COHORTS]
    delta=r['ridge_minus_position_original_credit_mae_pp']
    lines+=['','## 판단', '',
        ('- **Teacher:** 이5문제의 세 조건 모두 Credit 벡터가 반복 내 완전 일치했다. 관측 범위에서 안정적이지만 다른 문제/seed/환경의 보장은 아니다.'
         if all(exact) else '- **Teacher:** 일부 반복에 Credit 변동이 관측됐다. 동일 seed·다른 seed·재시작을 위 표로 구분하며 사전 허용 오차 없이 일괄 안정/불안정 판정을 붙이지 않는다.'),
        f'- **Ridge 대 Position:** 기존 라벨 기준 macro MAE 차이(Ridge−Position)는 {delta:.4f}%p다. '
        +('이 표본에서 Ridge가 더 나은 근거는 없다.' if delta>=0 else '이 표본에서는 Ridge의 점 추정치가 낮지만 유의한 우위는 입증되지 않았다.'),
        '- 반복 Teacher 오차는 다른 두 확률 변수의 차이이고, frozen 예측 오차는 특정 예측과 라벨의 차이다. Teacher 변동을 Probe 오차의 단순 하한으로 보지 않는다.',
        '- 동일 seed에서도 TP4 수치 연산/서버 이력 영향이 가능하다. 서로 다른 seed43/44/45는 세 샘플뿐이며 seed 전반의 분산 추정은 제한적이다.',
        '- 서버 재시작은 프로세스 재시작이며 disk CUDA/Outlines cache를 지우지 않았다. 순서 및 warm-up 효과를 완전히 분리한 실험은 아니다.',
        '- **Solver Pilot: CONDITIONAL GO.** Credit 차이가 실제 정답률/토큰/latency에 미치는 영향을 paired5군으로 탐색할 가치가 있다. 이는 Ridge 우위 또는 PAB 대체의 확증 실험이 아니다.',
        '- 모델·Credit·평가5문제는 그대로 고정하고 Teacher/Uniform/Position/Ridge와 Vanilla 모두 포함한다. GPU 자원 및 Solver Smoke 통과가 별도 조건이다.',
        '- 24/6 학습·선택 및198후보 탐색의 과적합 위험은 남는다. 이5문제로 방법을 고르면 이후 독립 확인에는 새 문제를 사용한다.',
        '', '## 근거와 계산', '',
        '- 입력 SHA는 JSON의 protected_input_sha256, 원본 요청별 검사와 usage는 request_checks, 모든 pairwise 벡터/지표는 results에 보존했다.',
        '- Credit MAE=mean(abs(a_i−b_i)), 단위 %p. Budget은 기존 PAB allocate_words(..., constant)로 변환 후 같은 MAE를 계산한다.',
        '- 완전 일치는 정수 Credit 벡터 전체의 동등성이다. 생성 텍스트 전체 일치율이나 의미적 정답 일치율을 뜻하지 않는다.',
        '- 같은3회가 모두 일치해도 기존 평가 라벨과 다를 수 있으므로 원본 라벨 anchor 비교를 별도로 보고했다.',
        '- 처리/무결성 PASS는 통계적 우위 또는 실제 수학 정답률 PASS를 뜻하지 않는다.', '']
    return '\n'.join(lines)


def run(a):
    require_compute();job=os.environ['SLURM_JOB_ID'];output=OUT/f'analysis_{a.main_job}_{job}.json'
    report=OUT/f'REPORT_{a.main_job}_{job}.ko.md'
    doc={'passed':False,'main_job':a.main_job,'cpu_job':job,'no_inference_or_refit':True}
    save_new(output,seal(doc))
    try:
        # Synthetic metric sanity check, never involving research data inference.
        assert compare([40,60],[40,60],200)['credit_mae_pp']==0
        assert compare([40,60],[60,40],200)['credit_mae_pp']==20
        assert compare([50,50],[40,60],200)['spearman'] is None
        plan=approve(a.audit);gate=check_complete('teacher','main',a.main_job,plan)
        sessions,checks=integrity(plan,gate,a.main_job)
        result=enrich(plan,teacher(plan,sessions))
        hashes={**gate['protected_outputs'],**plan['protected_files'],str(Path(a.audit).resolve()):file_hash(a.audit)}
        approve(a.audit);check_complete('teacher','main',a.main_job,plan)
        for p,h in hashes.items():
            if file_hash(p)!=h:raise ValueError('Input changed during analysis')
        doc.update(passed=True,audit_fingerprint=plan['fingerprint'],completion_fingerprint=gate['fingerprint'],
            protected_input_sha256=hashes,request_checks=checks,results=result,
            source_sha256={str(p):file_hash(p) for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.sbatch')})
        report.parent.mkdir(parents=True,exist_ok=True)
        with report.open('x',encoding='utf-8') as stream:stream.write(markdown(doc))
        doc['report_path']=str(report);doc['report_sha256']=file_hash(report)
    except Exception as exc:
        doc['passed']=False;doc['error']={'type':type(exc).__name__,'message':str(exc)}
    atomic_save(output,seal(doc))
    if not doc['passed']:raise RuntimeError('Stability CPU analysis FAIL; diagnostics preserved')
    print('TEACHER STABILITY CPU ANALYSIS PASS: 45 requests verified; '+str(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--main-job',required=True)
    run(p.parse_args())
