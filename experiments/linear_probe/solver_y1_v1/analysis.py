"""CPU summaries; original answer evaluator, cost ledger and explicit word proxy."""
import argparse
import os
from statistics import mean
from ..common import read,verify,seal,save_new,file_hash
from ..followup_v1.analysis import solver as original_solver
from .plan import OUT,approve,require_compute,ARMS
from .hardware import check_complete
from experiment_state import atomic_save


def budget_usage(text,finish_reason,budget):
    words=len(text.split())
    complete=finish_reason=='stop'
    return {'generated_whitespace_words':words,'allocated_total_words':budget,
        'word_budget_ratio':words/budget if budget else None,
        'total_word_budget_compliant':words<=budget if budget and complete else None,
        'complete_response':complete,'subtask_word_budget_compliance':None,
        'subtask_measurement':'BLOCKED: original joint prompt does not guarantee identifiable step spans',
        'scope':'Whole completion including thinking/final text, same whitespace count as legacy; aggregate proxy only'}


def run(a):
    require_compute();job=os.environ['SLURM_JOB_ID'];output=OUT/f'analysis_{a.main_job}_{job}.json'
    report=OUT/f'REPORT_{a.main_job}_{job}.ko.md'
    doc={'passed':False,'main_job':a.main_job,'cpu_job':job};save_new(output,seal(doc))
    try:
        plan=approve(a.audit);gate=check_complete('solver','main',a.main_job,plan)
        session=verify(read(OUT/f'responses_solver_{a.main_job}_main.json'))
        from .client import planned
        expected=planned(plan,'solver','main')
        if len(session['records'])!=25:raise ValueError('Require 25 requests')
        for row,want in zip(session['records'],expected):
            for k in ('id','arm','payload','steps_fingerprint','credits','budgets_words','total_words'):
                if row[k]!=want[k]:raise ValueError('Frozen Solver request mismatch: '+k)
        result=original_solver(plan,[session])
        for arm in ARMS:
            summary=result['arms'][arm];observations=[]
            for row in session['records']:
                if row['arm']!=arm:continue
                r=row['stage'].get('result',{})
                observations.append({'id':row['id'],**budget_usage(r.get('content') or '',r.get('finish_reason'),row['total_words'])})
            known=[r['total_word_budget_compliant'] for r in observations if r['total_word_budget_compliant'] is not None]
            summary['word_budget_observations']=observations
            summary['word_budget_compliance']={'compliant_count':sum(known),'measured_count':len(known),
                'all_problem_count':5,'unknown_or_not_applicable':5-len(known),'rate_on_measured':mean(known) if known else None}
        memory=verify(read(OUT/f'memory_solver_{a.main_job}_main.json'))
        result['sampled_gpu_memory']=memory
        result['startup']=verify(read(OUT/f'ready_solver_{a.main_job}_main.json'))
        approve(a.audit)
        doc.update(passed=True,audit_fingerprint=plan['fingerprint'],profile=plan['profiles']['solver'],
            results=result,input_sha256=gate['protected_outputs'])
        lines=['# y1 frozen 5문제 Solver Pilot', '',
            f'CPU 분석 {job}: PASS. 원본 Teacher/Probe/steps는 재생성하지 않았다.', '',
            '| 군 | 정답/5 | Solver input tokens | Solver output tokens | Solver total tokens | API latency 합계(s) | word 준수/측정 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            s=result['arms'][arm];c=s['all_attempt_cost'];w=s['word_budget_compliance']
            lines.append(f'| {arm} | {s["correct"]}/5 | {c["prompt_tokens"]["known_sum"]} | {c["completion_tokens"]["known_sum"]} '
                f'| {c["total_tokens"]["known_sum"]} | {c["latency_seconds"]["known_sum"]} | {w["compliant_count"]}/{w["measured_count"]} |')
        lines+=['','모든 군은 동일5문제이며 PAB4군 steps와 총 word budget이 같다. 입력/출력/지연시간 합계는 재시도를 포함한다.',
            '누락 비용은 JSON missing_attempts로 별도 확인한다. 정답률 분모는 항상5, answer undetermined 및 절단 상태도 상세 JSON에 보존한다.',
            'word 준수는 전체 completion whitespace words와 총 soft word budget의 비교 proxy다. 토큰 상한 준수율이나 subtask별 준수율이 아니다.',
            'finish_reason=length이면 준수 여부를 null로 두고 측정 수를 공개한다. Vanilla는 word budget이 없어 해당 없음이다.',
            '세부 Budget·원본 생성·Teacher·Hidden State·frozen 예측 비용·재구성 online 총토큰/시간은 JSON results에 포함된다.',
            '재사용 Planner 비용을0으로 해석하지 않는다. 혼합 하드웨어로 재구성한 online estimate는 실측 end-to-end가 아니다.',
            'CPU16/RAM64GiB 프로필의 y1 결과다. 과거48CPU/160GiB 환경과 latency를 직접 동일시하지 않는다.',
            '5문제는 탐색적 비교다. 여기서 방법을 선택하면 별도 독립 문제로 확인해야 한다.','']
        with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
        doc['report_path']=str(report);doc['report_sha256']=file_hash(report)
    except Exception as exc:doc['passed']=False;doc['error']={'type':type(exc).__name__,'message':str(exc)}
    atomic_save(output,seal(doc))
    if not doc['passed']:raise RuntimeError('y1 Solver CPU analysis FAIL')
    print('Y1 SOLVER CPU ANALYSIS PASS: '+str(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--main-job',required=True);run(p.parse_args())
