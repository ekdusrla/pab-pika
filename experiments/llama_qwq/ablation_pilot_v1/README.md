# MATH-500 최소 Ablation Pilot v1

기존 `runs/math500_llama31_8b_v1/{planner,solver}.json`은 읽기 전용이다.
모든 신규 산출물은 `experiments/llama_qwq/runs/math500_llama31_8b_ablation_pilot_v1/`에 저장한다.
준비·감사·check·analyze·tests는 CPU만 사용하며 모델 가중치 로딩이나 API 호출을 하지 않는다.
`run`과 sbatch 실행만 실제 Solver 호출을 허용한다. 이번 준비 작업에서는 제출하지 않는다.

## 고정 조건

- Level 1~5에서 9/18/21/25/27개, seed `20261109`.
- 각 Level의 ID를 정수 오름차순 정렬한 뒤 하나의 `random.Random(seed)`로 Level 순서대로 `sample`한다.
- 선택 코드는 ID·Level만 읽는다. 정답·토큰·credit·실패 여부를 사용하지 않는다.
- Qwen/QwQ-32B revision `976055f8c83f394f35dbd3ab09a285a984907bd0`.
- 기존 tokenizer snapshot/chat template, temperature 0.1, top-p 0.9, request/server seed 42,
  max output 8192, context 12288, FP16, TP=4, max_num_seqs=1, memory utilization 0.85.
- 원본 evaluator 및 패키지 fingerprint를 고정한다. 원래 답 형식도 유지한다.
- 신설 두 조건의 호출 순서는 Pilot 인덱스에 따라 교대한다.

## Prompt와 재사용

Plan-only는 저장된 Weighted 요청의 decomposition 블록에서 각 단계 뒤 budget 문장만 제거한다.
Equal은 같은 블록의 budget 숫자만 바꾼다. 전체 B는 유지하고 `[1]*단계수`를 기존 constant
배분 함수에 넣어 정수 예산을 만든다. 잔여 배분과 tie ordering도 기존 NumPy 구현을 사용한다.

기존 Vanilla·Weighted 각 100응답을 복사하되 source hash·응답·평가·호출 이력을 보존한다.
Equal은 messages뿐 아니라 전체 API payload가 Weighted와 같을 때만 응답을 재사용한다.
현재 Pilot은 Equal 10개 재사용, 신규 Plan-only 100 + Equal 90 = 190 조건이다.
이는 정상 실행의 예정 호출 수다. 실패·retry 발생 시 API 시도 수가 늘어난다.

## 평가 감사

`audit_baseline.json`은 기존 500문제 × 2방법을 CPU로 재채점하고 저장 판정과 대조한다.
공식 판정·출력·정답은 바꾸지 않는다. 불일치가 있으면 prepare를 중단한다.
진단은 마지막 balanced boxed 원문, 기존 extractor 출력, 검토 플래그만 저장한다.
중복 중괄호·집합·구간·tuple은 자동 정규화하거나 정답으로 바꾸지 않는다.
`manual_review`는 비워둔다. 수동 판정은 별도 파일로 기록하며 공식 결과에 합치지 않는다.
진단 플래그는 수학적 오답의 확정 분류가 아니다.
`audit_current.json`도 네 조건에 같은 진단 함수를 적용한다.

## 산출물

- `selection.json`: ID·Level 목록, seed·선정 규칙.
- `requests.json`: 네 조건의 전체 payload·fingerprint, 단계·budget, Equal 재사용 여부.
- `protocol.json`: 입력/코드/환경/tokenizer 해시, 평가기, 고정 실행·비용 규칙.
- `audit_baseline.json`: 1,000개 공식 판정 재검증과 변환 없는 진단.
- `checkpoint.json`: 원본 재사용 provenance와 신규 단계별 상태·시도·raw 응답·usage·평가.
- `metrics.json`: 네 조건의 전체 분모 정확도, 출력 cap 비율, 단계별/전체 비용, 정답 전환,
  paired 토큰 차이. 미완료 상태의 정확도는 완결 실험 결과로 해석하면 안 된다.
- `per_problem.csv`: 문제별 공식 판정·최종/전체시도 비용·paired 차이·재사용 출처.
- `audit_current.json`: 생성 완료된 네 조건의 공식 판정과 진단.
- SLURM 실행 후 `client_JOBID.log`, `server_JOBID.log`.

immutable 파일은 `prepare`에서 새로 만들고 재작성하지 않는다.
checkpoint와 파생 metrics/audit_current/CSV는 신규 디렉터리 안에서만 갱신한다.
기존 baseline·config·checkpoint는 변경하지 않는다.

## 비용 정의

주 지표는 실패·복구 포함 pipeline completion의 문제당 평균이다.
Vanilla=S, Plan-only/Equal=D+S, Weighted=D+A+S.
각 조건의 `final_pipeline`과 `all_attempt_pipeline`을 분리하고 입력+출력도 계산한다.
재사용 Planner/응답의 표준 pipeline 비용과 이번 신규 GPU 실행비(`new_execution_only`)는 별개다.
Equal 재사용 10개는 새 GPU 비용 0이지만 표준 pipeline에서는 D+S로 센다.
실패한 API 시도의 미상 토큰은 0으로 채우지 않는다. `missing_calls`와 known lower bound를 표시하고
완전한 mean은 null로 둔다. paired delta도 미상이 있는 문제는 제외하고 paired_n을 명시한다.
시간은 별도 호출의 latency 합이며 큐 대기·모델 로딩을 포함하는 end-to-end 시간이 아니다.

## CPU 준비 및 검증

```bash
cd /data/dada0228/plan-budget/P-and-B
/data/dada0228/conda_envs/vllm_server/bin/python -B experiments/llama_qwq/ablation_pilot_v1/test_pilot.py
# 최초 한 번만: 기존 신규 run 디렉터리가 있으면 prepare는 덮어쓰지 않고 거부한다.
/data/dada0228/conda_envs/vllm_server/bin/python -B experiments/llama_qwq/ablation_pilot_v1/pilot.py prepare
/data/dada0228/conda_envs/vllm_server/bin/python -B experiments/llama_qwq/ablation_pilot_v1/pilot.py check
/data/dada0228/conda_envs/vllm_server/bin/python -B experiments/llama_qwq/ablation_pilot_v1/pilot.py analyze
```

## 사용자가 제출할 명령

```bash
cd /data/dada0228/plan-budget/P-and-B
sbatch --export=ALL,RESUME=0 experiments/llama_qwq/ablation_pilot_v1/solver_pilot_y4.sbatch
# 실행 이력이 생긴 뒤 재개:
sbatch --export=ALL,RESUME=1 experiments/llama_qwq/ablation_pilot_v1/solver_pilot_y4.sbatch
```

자원은 기존 moana-y4, batch_ce_ugrad/ugrad_ce, RTX 3090 4장, CPU 8, RAM 48G다.
time limit 24시간은 상한이며 완료 시 서버를 종료한다. 기존 약 72~83초/호출을 적용하면
190회는 약 3.8~4.4시간, 15~18 GPU·h이며 로딩·대기·retry와 새 출력 길이에 따라 달라진다.

## Resume·중복 방지

- 실행 전체 job lock과 checkpoint writer lock으로 같은 run의 동시 실행을 막는다.
- 입력·코드·환경·선정·요청·재사용 응답의 변화는 API 전 검증에서 거부한다.
- 기존 success는 재호출하지 않는다. response_received는 저장 응답만 검증해 복구한다.
- success 응답의 채점이 undetermined여도 생성 재시도하지 않는다.
- SIGTERM/INT는 시도 상태·미상 비용을 checkpoint에 보존한다.
- 네트워크 transient는 기존 bounded retry(추가 최대 2회)를 사용한다. 원격 수락 이후
  연결이 끊긴 경우의 exactly-once는 API만으로 보장할 수 없으며 모든 시도를 보존한다.
- 재개 시 in_flight/interrupted/cost_unknown 이력이 있으면 기본적으로 GPU 시작 전 거부한다.
  사용자가 checkpoint·서버 로그를 검토하고 재시도 비용을 수용할 때만 아래를 사용한다.

```bash
sbatch --export=ALL,RESUME=1,RETRY_UNKNOWN=1 experiments/llama_qwq/ablation_pilot_v1/solver_pilot_y4.sbatch
```

- uncommitted `checkpoint.json.tmp`가 있으면 옛 checkpoint를 조용히 재실행하지 않고 중단한다.
  이 경우 CPU에서 두 파일과 응답 이력을 먼저 검토해야 한다.
- 완료된 run을 다시 제출하면 CPU check 후 서버를 띄우지 않고 종료한다.
- 코드·해시가 달라진 프로토콜은 기존 run에 이어 실행하지 말고 새 버전 디렉터리로 준비해야 한다.
