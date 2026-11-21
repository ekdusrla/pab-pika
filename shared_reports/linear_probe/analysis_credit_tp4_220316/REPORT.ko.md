# A2000 TP4 Teacher / Ridge Probe Pilot 최종 분석

## 결론

**실행 파이프라인은 PASS, 두 번째 Planner 호출 대체 가설은 아직 미입증이다.**
최종 선택은 HF hidden-state 인덱스 **18**, Ridge alpha **100**이다.
독립 Pilot 5문제의 projected Credit MAE는 **5.6133 percentage points(%p)**,
step-micro Budget MAE는 **19.4 words**, 평균 Credit Spearman은 **0.3643**이다.
같은 정수 Credit 계약의 균등 Baseline MAE는 소규모 벡터 수기 계산으로 **6.1667%p**다.
개선 폭은 **0.5533%p**이며, 문제별로 3승 2패다. 이를 견고한 대체 성능의 증거로 보기 어렵다.

**다음 단계 판정: CONDITIONAL GO.** 먼저 CPU 진단으로 위치-only Baseline과 Train
오차·재현성을 확인하고, 현재 모델을 고정한 탐색적 소규모 Solver 비교로 진행할 수 있다.
전체 PAB 대체, 통계적 우월성 또는 최종 MATH-500 실험으로의 확대를 승인하는 GO는 아니다.

## 분석 범위와 근거 표기

- **[E] 기존 결과:** 저장된 보고서·로그·SLURM 조회에서 확인한 값.
- **[D] 소규모 유도값:** 아래에 제시한 5문제/20단계 벡터의 수기 산술. 서버에서 Python을 실행하지 않았다.
- **[P] 추가 계산 대기:** 기존 보고서에 없고, 별도 CPU 분석 스크립트로 계산 가능하지만 이번에 실행하지 않은 값.
- **[H] 해석/가설:** 관측과 일치하나 대조 실험으로 검증하지 않은 주장.

원본 디렉터리의 Teacher 라벨·Probe·NPZ·평가 보고서·코드는 수정하지 않았다.
새 파일은 runs/analysis_credit_tp4_220316 아래에만 작성했다. Python/GPU 추론·테스트와
추가 SLURM 제출은 수행하지 않았다. 감사가 실행 시 검증한 checksum을 이번에 다시 계산한 것은 아니다.

### 근거 파일

모든 원본 결과의 공통 경로는 `experiments/linear_probe/runs/credit_a2000_tp4/`다.

| 표기 | 파일 / 의미 |
|---|---|
| S1 | 이 디렉터리의 SLURM_STATUS.txt: 읽기 전용 sacct 조회 기록 |
| S2 | audit_220310.json/log: 보완 감사, 신규 5개 테스트 PASS; 기존 220305 감사 재사용 |
| S3 | complete_teacher_220313.json, gpu_220313.log: Train/Validation Teacher 완료·출력 SHA |
| S4 | probe_v1/status.json, metadata.json, probe_220314.log: 학습 완료·198후보 선택·artifact SHA |
| S5 | probe_v1/validation_report.json: Validation 지표·문제별 벡터 |
| S6 | complete_evaluation_220315.json, evaluation_labeled.json: Evaluation Teacher 완료 |
| S7 | evaluation_report_220316.json, probe_220316.log: 독립 Pilot 지표·5문제 전체 벡터 |
| S8 | memory_220313.json, memory_220315.json, server_* JSON/log: GPU·서버 실행 증거 |
| S9 | train_labeled.json, validation_labeled.json: 최종 라벨·fingerprint·전체 시도 비용 |
| S10 | experiments/linear_probe/credit_a2000_tp4/{audit,teacher,train,gpu}.py: 검사와 지표의 구현 |
| S11 | experiments/llama_qwq/run_solver_compare.py: allocate_words 및 messages_for |
| S12 | experiments/linear_probe/runs/pilot_v1/final_v2_219684.json: 수집·보정 비용 원장 |

## 1. 완료·무결성

| Job | 역할 | SLURM | 시간 | 노드 | 실제 할당 |
|---|---|---|---|---|---|
| 220313 | Teacher Train/Validation | COMPLETED / 0:0 | 4분08초 | u2 | GPU4, CPU24, RAM48GiB |
| 220314 | Ridge 선택·학습 | COMPLETED / 0:0 | 10초 | y2 | **CPU2**, RAM8GiB, GPU 없음 |
| 220315 | Evaluation Teacher | COMPLETED / 0:0 | 1분49초 | u2 | GPU4, CPU24, RAM48GiB |
| 220316 | Probe Evaluation | COMPLETED / 0:0 | 5초 | y2 | **CPU2**, RAM8GiB, GPU 없음 |

[E, S1] CPU 작업은 설계/스크립트의 CPU1과 달리 실제 CPU2를 할당받았다. 이는 자원·비용
보고의 차이이며, 데이터 누수나 평가 실패 증거는 아니다. 학습 스크립트는 BLAS/OMP1을
설정하지만 실제 스케줄링 비용은 CPU2로 보고해야 한다. 향후 CPU-only 요청은 CPU1로 유지한다.
AllocTRES와 요청 CPU는 구분한다. 이 차이가 core 단위 할당 반올림인지 제출 옵션 때문인지는
이번 자료만으로 단정하지 않는다.

S3와 S6의 passed는 true, S4 status도 passed=true, S7도 passed=true다. 로그의
TEACHER LABELS FINAL PASS / TEACHER EVALUATION FINAL PASS / PROBE TRAIN PASS /
PROBE EVALUATION PASS와 일치한다.

### fingerprint 연결 [E]

| 대상 | 식별값 |
|---|---|
| 공통 TP4 감사 | d9f35ccfaa9b1957bc76b0bd7c89b272611033a4f3be233e220a8e3e43feb75e |
| Teacher Train/Validation gate | b60e9503bef0c9e0918e71b8a56d820934d771bdd5da52766ec6dc4710acef24 |
| Train 라벨 | dbf1e2e582b6bee24439964d0c79d11f89e6785c5e06b984f378e5617acd910d |
| Validation 라벨 | a60cc49b9f9cdfa4bfeaafa964fd87fe4fe8c9f7d67139561f6225cbb256ac8b |
| Probe metadata | 46c049a59c3fbf5046007e7b0c5e9c487ffabb4ad2cba7641a30502b347263e7 |
| Evaluation 라벨 | f0c431bcb67b607da45deabbc72dfa7ddd3f8547f6704d415b3f5c9c2a9c6739 |
| Evaluation 보고서 | fc6e6a7257621110b34dbf5813a2cf7546d64acdd9dd3cb0ba556c64746816b1 |

S4의 train/validation fingerprint는 S9 말미와 같고, teacher_gate_fingerprint는 S3와
같다. S7의 probe_fingerprint는 S4 metadata fingerprint와 같다. 두 Teacher 완료 gate와
Probe metadata/report의 감사 식별자도 같다. 원본 joblib SHA 검증, 라벨 SHA 검증,
steps/feature projection 검사와 입력 파일 검증이 S10 코드의 실행 경로에 들어 있다.
S7 자체에는 Evaluation 라벨 fingerprint/gate가 별도 필드로 직접 포함되지 않으므로
독립 재감사 시 S6와 실행 코드의 연결도 함께 확인해야 한다. 별도 CPU 스크립트가 이를 확인한다.

### 분할·누수·평가 처리

- Train24 / Validation6 / Evaluation5 문제 단위 분리가 검사되었다. Sub-question을
  랜덤 분할하지 않는다. StandardScaler와 Ridge fit은 Train만 사용한다.
- Validation은 layer/alpha 선택에 사용되며, 재학습에는 합치지 않는다. Evaluation 라벨은
  selection_frozen 확인 후 생성되었다. metadata에는 refit_on_validation=false,
  evaluation_labels_accessed=false가 기록되어 있다. 이는 코드·저장된 lineage에 관한 확인이다.
- 기존 데이터 감사가 MATH-500 normalized-hash 제외와 benchmark/split disjointness를
  확인했다. 의미상 유사 문제·공통 풀이 패턴까지 배제했다는 보장은 없다.
- Teacher35개에 각각 1회 시도, 누락 usage0: 라벨 누락·API 재시도 정황은 없다.
- Evaluation은5문제/20단계가 모두 포함되어 있고, Spearman undefined0이다.
  Probe의 clamp0, uniform_fallback=false가 5문제 모두 기록되어 있다.
- Evaluation5는 모두 initial_success다. Train은 보정2개, Validation은 보정1개를
  포함한다. 보정은 추가 대화 context를 사용했으므로 층별 Feature 분포의 잠재적 차이다.
- 두 GPU job memory gate는 PASS이고, 최대 관측 장치 사용량은 각 job에서
  10,825,498,624 bytes(약10.08GiB), 기준11GiB 이하다. polling 기반으로 순간적인
  allocator peak를 증명하지는 않는다. 지정한 ERROR/OOM/Traceback 검색과 완료 로그에서
  새 OOM/engine failure는 발견하지 못했다.

**판정:** 저장된 실행 증거 기준 파이프라인 PASS. 현재 원본 전체의 재해시/재추론은 미실행;
원하면 준비된 CPU-only 분석에서 재검증할 수 있다. Trainer/JSON에 기록된 규칙 이상의
실제 데이터 처리 행위를 이 보고서가 독립적으로 관측한 것은 아니다.

## 2. Probe 성능

[E, S4] hidden-state index18, alpha100,4096차원. 인덱스0은 embedding이므로18은
18번째 Transformer block 출력에 해당한다. 33개 층 x6개 alpha=198후보를 비교했다.
타깃은 Teacher의 양의 정수 percentage points(sum100); 출력은 기존 양수·합계100
정수 투영을 거친 Credit이다. 아래 MAE/RMSE는 raw Ridge 값의 지표가 아니다.

| 지표 | Train | Validation(6문제/24단계) | Evaluation(5문제/20단계) | 근거 |
|---|---|---:|---:|---|
| 문제 macro Credit MAE, %p | **미저장 [P]** | 4.8500 | 5.6133 | S5/S7 summary |
| 문제 가중 RMSE, %p | 미저장 [P] | 5.9226 | 6.8523 | sqrt(mean(문제별 MSE)) |
| step micro MAE, %p | 미저장 [P] | 4.9167 | 5.4000 | 모든 step을 같은 가중치 |
| step micro RMSE, %p | 미저장 [P] | 5.9931 | 6.6030 | 모든 step squared error |
| 평균 Credit Spearman | 미저장 [P] | 0.6097 | 0.3643 | tie-average ranks |
| Budget step micro MAE, words | 미저장 [P] | 20.4167 | 19.4000 | S5/S7 summary |

선택된 모델의 Validation→Evaluation MAE 차이는 **+0.7633%p [D]**다. 순위 상관도는
감소했다. 반면 word 예산 MAE가 감소한 것은 난이도별 총예산과 단계 수 구성이 달라서일
수 있다. 예산 크기를 정규화하지 않은 word 지표만으로 일반화가 좋아졌다고 판단하면 안 된다.
Train 오차는 보고서에 없으므로 0 또는 낮은 값이라고 추정하지 않는다.

### 문제별 Credit 및 균등 Baseline

균등 Baseline은 n단계에 project_credits([1]*n)을 적용한다. 따라서 n3은[34,33,33],
n4는[25,25,25,25], n5는[20,20,20,20,20]이다. Teacher를 보고 만든 예측이 아니라
합계100 계약만으로 정해지는 feature-free Baseline이다. 표의 Uniform MAE는 수기 유도값[D]이다.

| 문제 ID(접미사) | Teacher | Probe | Probe MAE [E] | Uniform MAE [D] | Probe 판정 |
|---|---|---|---:|---:|---|
| counting_and_probability/train/407 | [20,40,40] | [24,27,49] | 8.6667 | 9.3333 | 개선 |
| precalculus/train/352 | [15,20,25,40] | [21,23,18,38] | 4.5000 | 7.5000 | 개선 |
| intermediate_algebra/train/765 | [20,30,30,20] | [20,22,23,35] | 7.5000 | 5.0000 | 악화 |
| algebra/train/1144 | [20,30,20,30] | [20,24,25,31] | 3.0000 | 5.0000 | 개선 |
| prealgebra/train/584 | [15,20,15,20,30] | [18,13,18,16,35] | 4.4000 | 4.0000 | 악화 |
| **문제 macro** | | | **5.6133** | **6.1667** | **3승2패** |
| **step micro** | | | **5.4000** | **5.9000** | |

계산식은 MAE_q=sum_i|p_i-t_i|/n_q, macro=mean_q MAE_q다. Uniform의 문제별
절대오차 합은28,30,20,20,20으로 micro=(28+30+20+20+20)/20=5.9다.
Macro 개선0.5533%p는 상대 약9%에 불과하고 표본5개이므로 유의성 주장은 할 수 없다.

**[H] 위치 prior 가설:** Probe는5/5에서 마지막 step에 가장 높은 Credit을 준다. Teacher는
4/5에서 마지막 step이 최고 또는 공동 최고다. intermediate_algebra/765는 마지막에
15%p를 과다 배분했다. 의미 있는 Hidden State 정보보다 단계 위치 prior를 학습했을
가능성을 배제하지 못한다. Train의 (단계 수,위치)별 Teacher 평균 Baseline이 필요하다.
이 Baseline·Train MAE·bootstrap 수치는 [P]로 남겨두고 CPU 스크립트를 준비했다.

## 3. Budget 영향: 토큰이 아닌 words

[E, S11] PAB는 LEVEL_WORDS={1:200,2:250,3:350,4:450,5:600}에 Credit와 schedule을
곱해 정규화하고 floor 후 큰 weight부터 잔여를 더한다. 현재 constant schedule이다.
Solver 메시지는 `using up to {budget} words`를 요청하며 다른 풀이법도 허용한다.
이는 Subtask의 실제 생성 token 수나 엄격한 token cap이 아니다. 실제 token 소비는
QwQ 실행 전에는 알 수 없다. 따라서 현재 결과로 token 절약 또는 정확도 보존을 주장할 수 없다.

| 문제 | Teacher words [E] | Probe words [E] | Budget L1 [E] | step MAE [D] | 최대 절대차 [D] |
|---|---|---|---:|---:|---:|
| counting/407 | [40,80,80] | [48,54,98] | 52 | 17.3333 | 26 |
| precalculus/352 | [37,50,62,101] | [52,57,45,96] | 44 | 11.0000 | 17 |
| intermediate/765 | [70,105,105,70] | [70,77,80,123] | 106 | 26.5000 | **53** |
| algebra/1144 | [90,135,90,135] | [90,108,112,140] | 54 | 13.5000 | 27 |
| prealgebra/584 | [90,120,90,120,180] | [108,78,108,96,210] | 132 | 26.4000 | 42 |

- [E/D] 전체 Budget 절대차388 words /20단계=**19.4 words**, 평균 문제 L1=**77.6 words**.
- [D] 두 예산 합이 같으므로 실제 재배치 word mass는 L1/2다. 전체194 words,
  평균 문제당38.8 words가 다른 step으로 이동했다. 총예산1850 words 대비 약10.49%다.
- [D] 최대 step 차이53 words는 intermediate/765의 마지막 step70→123이다.
  원래 해당 step 예산의 약75.7% 증가이므로 평균 MAE만으로 무시할 수 없다.
- [E] Credit 순위 상관은 문제별0.8660,0.4,0,0.4472,0.1081이다.
  [D] 이5문제에서는 Budget 벡터도 각 Credit 벡터와 같은 순위/tie 패턴을 보인다.
  독립적인 Budget Spearman 필드는 미저장[P]; CPU 스크립트에서 명시적으로 재계산한다.
- [H] 반올림을 무시하면 delta_budget_i ≈ T_q * delta_credit_i /100이다.
  높은 난이도의 큰 T에서는 같은 Credit 오차가 더 큰 word 차이가 된다. 이 배분 차이가
  실제 Solver 정확도·reasoning 길이·token 소비를 바꿨는지는 아직 전혀 검증되지 않았다.

## 4. 비용과 대체 가설

| Teacher 범위 | 호출 | 입력 tokens | 출력 tokens | API latency 합(초) | 근거 |
|---|---:|---:|---:|---:|---|
| Train | 24 | 42,535 | 5,238 | 142.506669 | S9 |
| Validation | 6 | 10,808 | 1,455 | 32.601217 | S9 |
| Evaluation | 5 | 8,974 | 1,130 | 43.828367 | S6/S7 |
| **합계 [D]** | **35** | **62,317** | **7,823** | **218.936253** | 부분합 |

라벨 생성 합계70,140 tokens이며 Smoke 호출·서버 시작·종료 비용은 별도다. 이35회는
Probe 학습/평가용 offline Teacher 비용이므로 그대로 온라인 절감액이라고 합산하면 안 된다.
Evaluation Teacher 평균 API 지연은 약8.77초, Probe CPU predict는 저장된 값에서
문제당 약0.88–1.24ms이며 정수 투영은 약0.05–0.06ms다. NPZ read는 약3.6–4.7ms로
offline IO이며 배포시 in-memory Feature 처리와 구분한다. 대규모 speedup 배수는 계산하지 않는다.

[D, S1] GPU job 할당시간 기준 비용은 Teacher30에서248초*4=992 A2000-GPU-seconds,
Evaluation5에서109초*4=436, 합계1428 A2000-GPU-seconds다. 이는 사용률100% 시간이나
GPU kernel 실행시간이 아니며 Smoke·CPU 작업·이전 capture/복구 비용을 제외한다.
GPU 기종별 시간을 분리하고 가격이 없으므로 금액은 추정하지 않는다.

Ridge fit 자체는5.731초로 기록되어 있다. 아직 학습비용을 포함한 손익분기점, Online
Planner 전체 지연, 실제 generated tokens 및 Solver 비용은 없다. 기존capture는3090/HF,
Teacher는A2000x4/vLLM, Probe는CPU이므로 이 부분시간만으로 원본 PAB 전체 가속을 주장할
수 없다. PAB 원본 prompt 템플릿/함수는 유지됐지만 Teacher benchmark는 누수 방지를 위해
별도 외부 train 문제를 사용하므로 legacy evaluation-derived few-shot과 내용까지 같지는 않다.

## 5. 가설의 상태와 한계

### 확인된 사실

- 동일 최종 steps/Feature에서 Ridge가 계약에 맞는 Credit를 생성하는 기술적 경로는 동작한다.
- held-out5에서 균등 Credit보다 aggregate MAE가 조금 낮지만 모든 문제에서 우세하지 않다.
- 순위 일치도는 낮고 특정 step 예산을 크게 바꾸는 사례가 있다.
- Teacher 라벨은 mathematical ground-truth가 아니라 단일 Llama judgment다.

### 추론 또는 미검증

- 고차원4096 Feature에 문제24개, 최대120개의 상호 독립적이지 않은 step 타깃이다.
  alpha100 정규화가 위험을 줄여도198후보 중 Validation6으로 고르는 선택 편향은 남는다.
- Validation 지표는 선택에 사용되어 낙관적이다. 5문제 Evaluation도 넓은 일반화 추정에
  너무 작다. 표준오차·유의성·안정적 신뢰구간은 현재 보고서에 없다.
- temperature .1/top_p .9/seed42의 한 라벨만 있다. 동일 seed 재현성과 여러 seed의
  Teacher-vs-Teacher MAE, TP1-vs-TP4 차이, model/backend 변화에 대한 안정성은 미검증이다.
- Teacher 자체 변동성이5%p인지1%p인지 몰라 Probe5.61%p 오차의 의미를 교정할 수 없다.
- TP4는 reduction/kernel 및 hardware 경로가 TP1과 달라 같은 seed라도 동일 Credit를
  보장하지 않는다. 현재 세 split은 같은TP4서명이어서 내부 일관성은 갖췄다.
- 높은 last-step prior, 문제 난이도 구성, 회복 대화 Feature 분포, semantic 중복 및 Teacher
  subjective grading을 분리하지 못했다. Solver 성능은 아직 없다.

## 6. 추가 CPU 진단: 실행하지 않고 준비한 항목

이 디렉터리의 extended.py와 analysis_cpu.sbatch는 원본을 읽기만 하며 별도
extended_JOB.json을 독점 생성한다. 원본 joblib를 체크섬 검증 후 CPU predict만 수행하고
Ridge를 재학습하지 않는다. 실행 시 계산 가능한 항목:

1. 현재 source/NPZ/Teacher gate/joblib fingerprint 재검증과 저장된 Eval 예측·지표 재현.
2. 선택된 모델의 Train/Validation/Evaluation 전체 지표. Train은 in-sample 진단으로 표시.
3. Uniform 및 Train-only (n_steps,index) 평균 Baseline. 보지 못한 n은 균등 처리.
4. Budget max/정규화 재배치율/Budget Spearman, last-step 통계.
5. 문제 단위 paired bootstrap(2000회,seed20261116)과 승패. step 단위 독립 샘플로
   간주하지 않는다. 5문제 bootstrap도 확정적 추론이 아닌 탐색적 진단이다.
   고정된 선택 모델에 조건부인 구간이며, 학습/198후보 선택 전체의 불확실성을 포함하지 않는다.

선택적 **사용자 실행용** 명령(이번에 제출하지 않음):

```bash
sbatch --gres=none --cpus-per-task=1 \
  experiments/linear_probe/runs/analysis_credit_tp4_220316/analysis_cpu.sbatch
```

CPU1/RAM8GiB/20분이며 GPU 호출은 없다. 이 Python 파일의 실행/문법 검사는 이번에
수행하지 않았다. 현 보고서의 [P] 수치를 채웠다고 주장하지 않는다. 이번 Evaluation을 보고
추가한 Baseline은 post-hoc임을 표시하고, 결과가 나쁘다고 같은5문제로 다시 층/alpha를 튜닝하지 않는다.

## 7. 다음 실험 우선순위

### P0: Solver 전에 CPU 진단 완료

Train 오차, 위치-only Baseline, frozen 결과 재현을 먼저 확인한다. Ridge가 위치-only에
못 미치면 Hidden State 복잡성과 capture 비용을 정당화할 증거가 부족하다. Uniform도
독립 ablation으로 유지한다. Probe/평가 코드는 고정하고 새로운 GPU 실험은 자동 실행하지 않는다.

### P1: Teacher 잡음/실행 재현성

사전 고정한 Train/Validation subset에서 동일seed 반복과 소수 다른seed 반복을 별도
분석한다. prompt/model/schema는 바꾸지 않고 Teacher-vs-Teacher MAE/순위/예산 변동을
측정한다. Evaluation 성능을 보고 Teacher 프롬프트를 바꾼 뒤 같은5문제를 독립 결과로
다시 제시하지 않는다. 이 단계는 향후 사용자가 승인·제출할 GPU 작업이다.

### P2: 현재5문제의 고정된 탐색적 Solver Pilot

권장4군: **Vanilla / Teacher-PAB / Uniform-PAB / Probe-PAB**. 한seed면20 Solver 요청이다.
세 PAB군은 같은 final Decomposition, 난이도별 총word예산, Solver prompt, max_tokens,
sampling 및 QwQ 모델/FP16/TP4를 공유하고 Credit만 바꾼다. Vanilla는 원본 별도 프롬프트를
유지하고 Planner가 없는 차이를 명시한다. 요청 순서는 문제별 counterbalance/고정seed로
섞고 warmup/startup/cache 정책을 기록한다. 기존5문제의 역할은 탐색적이므로 성능을 본 뒤
구조를 바꾸면 새 독립 문제로 재평가해야 한다.

**자원 주의:** 원래 QwQ-32B FP16은 가중치만 대략64GB(약60GiB)다. A2000 12GiB x4의
총48GiB보다 커서 현재 Teacher 할당을 그대로 Solver에 쓸 수 없다. 기존3090x4/TP4 등
원래 Solver 조건에 맞는 자원을 기다리고, 비교 편의를 위해 몰래 양자화/모델/TP를 바꾸지 않는다.

| 측정 | 정의 |
|---|---|
| 정확도 | 원래 answer evaluator의 correct/incorrect/undetermined; truncation·failure 모두 분모에 포함 |
| tokens | Decomposition/repair/Credit/Solver의 실제 prompt/completion, 실패 재시도, 중복 context 모두 단계별 합산 |
| 지연 | cold startup, warm online latency, stage별 latency, IO/전송, Probe predict/투영을 분리; p50/p95는 충분한 수에서 |
| GPU 비용 | GPU 기종·개수 x 할당시간 및 가능하면 활성시간/메모리; A2000와3090를 한 비용으로 섞지 않음 |
| 배분 영향 | Teacher/Uniform/Probe의 word차이와 실제 Solver 출력 token/정확도 변화의 paired 관계 |

두번째 호출 대체 효과를 분리하려면 동일 HF Decomposition 경로에서 Teacher Credit와
Probe를 비교하는 paired 실험이 우선이다. 원본 vLLM Decomposition PAB와 HF Probe-PAB의
전체 비교는 backend 차이까지 포함하는 별도 end-to-end 결과로 보고한다. Hidden State
capture의 control 대비 overhead, 올바른 token alignment, 추가 forward0 기록과 회복
호출 비용을 포함한다. 선택층만 유지하는 최적화의 비용 절감은 아직 측정되지 않았다.

### P3: 더 큰 새 독립 평가와 모델 선택 안정성

Problem-group 교차검증/반복 분할, 학습량 증가, 층/alpha 선택 안정성, 다른 subject/level을
검증한다. 이미 본5문제는 튜닝 후 독립검증으로 재사용하지 않는다. MATH-500은 최종
lockbox로 유지한다. Solver5문제는 한 문제당20%p 정확도가 바뀌므로 통계적 우월성의
근거로 쓰지 않는다.

## 최종 판단

**CONDITIONAL GO — CPU Baseline/재현성 진단을 먼저 하고, 고정된 모델의 소규모 Solver
Pilot으로 제한하여 진행.** 기술적 실행·라벨 정합성은 PASS이고 균등 대비 소폭의 유리한
신호는 있지만, 위치-only 대조·Teacher 잡음·Solver 정확도·공정한 end-to-end 비용 검증이
남았다. 현재 결과만으로 두 번째 Planner 호출을 일반적으로 대체할 수 있다고 결론낼 수 없다.
