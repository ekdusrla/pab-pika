# Teacher Stability 및 5-arm QwQ Solver Pilot 준비

2026-11-17. **판정: CONDITIONAL GO.** 새 CPU 감사와 각 GPU Smoke는 아직 실행하지 않았다.
현재 자원으로 Teacher 경로를 준비할 수 있다. Solver는 동일 노드 RTX3090 4장이 확보될 때까지 실행 차단이다.
이 문서는 실행 준비 결과이며 새 실험 결과나 Python 테스트 PASS 보고서가 아니다.

## 보존 범위와 입력

새 소스는 `experiments/linear_probe/followup_v1/`, 새 산출물은
`experiments/linear_probe/runs/followup_v1/`에만 저장한다. 기존 코드·라벨·NPZ·모델을 수정하지 않는다.
원래 감사 220305/220310 및 학습을 재실행하지 않는다. 새 감사는 그 PASS와 보호된 파일 SHA를 확인하고
이번 요청 계획·원본 Solver 설정·동결된 예측·새 코드에 대한 추가 검사를 한다.

| 입력 | 용도 |
|---|---|
| `runs/pilot_v1/data.json` | Train24/Validation6/Evaluation5 및 별도 benchmark 분할 |
| `runs/pilot_v1/{train,validation,evaluation}_recovered.json` | 원본 최종 steps·fingerprint·NPZ·생성 비용 |
| `runs/pilot_v1/final_v2_219684.json` | 기존 35개 feature 감사 |
| `runs/credit_a2000_tp4/audit_220310.json` | TP4 정책 및 기존 감사 연결 |
| `runs/credit_a2000_tp4/evaluation_labeled.json` | 변경하지 않는 기존 Teacher 기준 라벨 |
| `runs/credit_a2000_tp4/probe_v1/` | frozen Layer18/alpha100 확인만; 새 학습·예측 없음 |
| `runs/analysis_credit_tp4_220316/extended_220319.json` | 기존 Ridge/Uniform/위치 평균의 raw·정수 Credit·Budget·시간 그대로 사용 |
| `experiments/llama_qwq/runs/math500_llama31_8b_v1/solver.json` | 원본 QwQ/Vanilla 설정; 작은 선두 config만 읽음 |

위 `runs/`는 `experiments/linear_probe/runs/` 기준이다. 부모 감사가 데이터/NPZ SHA 및 분할 검사를 보호한다.
MATH-500 normalized-hash overlap 없음과 Train/Validation/Evaluation/benchmark 분리는 기존 감사에서 계승한다.
의미상 유사 문제까지 완전히 배제했다는 보장은 없다. 이번 실험에서 fit이나 레이어 선택을 하지 않는다.
새 감사에서 기존 진단 보고서 input SHA도 확인하고, 각 예측의 문제 ID·최종 steps·Teacher 라벨·fingerprint·Budget을 대조한다.

## 고정 설정

| 항목 | Teacher | Solver |
|---|---|---|
| 모델 | meta-llama/Llama-3.1-8B-Instruct | **Qwen/QwQ-32B** |
| revision | `0e9e39f249a16976918f6564b8830bc894c89659` | `976055f8c83f394f35dbd3ab09a285a984907bd0` |
| 정밀도/TP | FP16 / TP4 / 단일 노드 mp | FP16 / TP4 / 단일 노드 mp |
| temperature / top_p | 0.1 / 0.9 | 0.1 / 0.9 |
| seed | 서버42; 요청 seed는 아래 cohort별 명시 | 서버·모든 요청42 |
| 생성 상한 / context | 2048 / 8192 | 8192 / 12288 |
| 엔진 | vLLM 0.6.6.post1 | vLLM 0.6.6.post1 |
| guided JSON | 원래 PAB schema + Outlines | 없음; 원래 QwQ 자유 생성 |
| max_num_seqs / GPU utilization | 1 / 0.85 | 1 / 0.85 |

Teacher는 기존 `payload_for()`를 재사용하므로 Credit 프롬프트·few-shot·guided JSON을 재작성하지 않는다.
Solver는 `run_solver_compare.messages_for()`의 원래 Vanilla 및 weighted_pab 메시지를 재사용한다.
PAB4군의 총 budget은 level별 200/250/350/450/600 **words**이고 steps가 같다.
기존 `project_credits()`와 `allocate_words(...,'constant')`로 frozen 정수 Credit·Budget이 정확히 재현되는지만 검사한다.
새 배분식을 쓰거나 fractional remainder와 원본의 weights 우선 잔여 배분을 혼동하지 않는다.
words는 프롬프트의 soft 지시이며 생성 tokens 상한과 다르다. Vanilla에는 subtask word budget을 넣지 않는다.

원본 설정의 모델·revision·sampling·context·서버 설정·패키지 버전·프롬프트·Solver helper SHA를 CPU 감사에서 검사한다.
새 경로는 executor mp를 명시한다. 원래 Solver 기록에는 executor 이름이 없으므로 동일 노드 TP4의 암묵적 선택을 명시화한 것이다.
25군 요청 순서는 원본 2군 alternate 대신 아래 사전 고정 순서다. 분석 시 이 차이를 보고한다.
vLLM engine 기본 prefix cache가 false임을 CPU 및 서버 시작 전에 확인하고, 다르면 실행을 차단한다.
기존 서버의 실제 캐시 상태를 소급해 완전히 입증한 것은 아니다.

## A. Teacher Stability: 45개 요청

동일한 Evaluation5와 동일한 steps를 사용한다. 예측과 기존 라벨은 갱신하지 않는다.

| cohort | 반복 | 요청 seed | 서버 |
|---|---:|---|---|
| same_seed | 3×5 | 42,42,42 | 하나의 warm 서버 공유 |
| different_seed | 3×5 | 43,44,45 | 위 warm 서버 공유 |
| restart_seed42 | 3×5 | 42,42,42 | 반복마다 새 서버 |

warm block 순서는 same0 → different0 → different1 → same1 → same2 → different2로 고정한다.
매 block 문제 순서는 동일한 frozen ID 순서다. 나머지 세 서버는 warm 종료 후 순차로 시작한다.
cohort·round·요청 순번·요청 seed·전체 payload·서버 argv·캐시 정책·원본 응답·usage·latency·시도 이력을 저장한다.
prefix caching은 비활성 기본값을 검증한다. disk CUDA/Outlines grammar cache는 지우지 않는다.
프로세스 재시작은 완전한 cold machine 실험을 뜻하지 않는다. 초기 warm-up과 disk cache 영향은 남는다.
동일 seed에서도 TP4 수치 연산/요청 이력의 재현성은 실측 대상이며 보장하지 않는다.

SDK 및 명시적 Teacher 재시도는 **0회**다. API 오류를 다른 샘플로 몰래 교체하지 않는다.
의미상 잘못된 라벨은 보존하고 후속 요청을 수집하되 전체 gate는 FAIL이다. 서버 사망 시 즉시 중단한다.
실패한 warm session을 새 서버에 이어 붙이지 않는다. 새 Job 출력에서 전체 반복 계획을 다시 실행하고 이전 실패 이력을 유지한다.

CPU 분석은 cohort별 문제당 3쌍의 Credit MAE/RMSE, Spearman, tie 포함 pair-order agreement,
Budget MAE/최대 오차, 문제별 최대 step range, exact-repeat 일치를 계산한다.
상수 벡터의 Spearman은 null이며 제외 개수를 공개한다.
15쌍은 독립 15문제가 아니라 **독립 단위 5문제의 종속 비교**다. 문제별 값을 함께 공개한다.
기존 Teacher 라벨과의 비교 및 frozen Ridge/Position/Uniform의 반복 Teacher 대비 오차도 보존한다.
Teacher 반복 변동을 predictor의 성능 하한으로 단정하지 않는다. seed와 restart 실험은 paired 기술통계다.

## B. Solver: 25개 요청

Vanilla / Teacher / Uniform / Position / Ridge 순서를 문제 index별로 한 칸씩 회전한다.
5문제 전체에서 각 군이 각 순서 위치에 한 번씩 배치된다. 실행 전에 계획이 감사 JSON에 고정된다.
Teacher는 저장 라벨만 사용한다. Uniform/Position/Ridge는 extended_220319의 frozen 예측만 읽는다.
이번 Solver 실행에 Teacher 호출·Decomposition 재생성·Probe 재학습·새 예측은 없다.
동일 seed를 모든 요청에 명시하고 직렬 요청한다. SDK retry0, 원본과 같은 transient retry2/대기2초다.
semantic 실패는 재생성하지 않고 실패 이력을 유지한다. 모든 실패·추가 시도는 별도로 집계한다.

원본 math_answer_eval(mode=original)로 최종 정답을 평가한다. 정답 수 분모는 항상5다.
판정 불가/실패도 분모에서 빼지 않는다. 본 실행의 finish_reason=length는 원본처럼 보존·평가하고 절단을 명시한다.
Smoke는 stop만 허용하여 긴 생성 절단이 나타나면 본 실행 전에 검토하게 한다.
문제별 총/단계별 budget, 실제 API prompt/completion tokens, 모든 시도 latency, 최종 정답 상태를 저장한다.
하나의 joint completion이므로 실제 **subtask별** 생성 tokens는 계측하지 않는다(null). 전체 tokens를 임의로 나누지 않는다.

원본 HF capture/control·Teacher 단계·Probe 예측/feature read/정수화 비용을 비용 ledger에 유지한다.
실제 이번 실행의 Planner API 호출0은 **산출물 재사용**이라는 뜻이며 배포 시 Planner 비용0이라는 뜻이 아니다.
별도 reconstructed_online 값은 control Decomposition 또는 hidden capture + 해당 Credit 비용 + 실측 Solver를 합친 추정이다.
성공한 Evaluation decomposition 경로의 추정이며 원본 attempts도 함께 보존한다.
겹칠 수 있는 hidden component 시간을 무작정 합쳐 causal overhead라고 부르지 않는다.
Teacher는 A2000 TP4, capture는3090 HF, Probe는CPU, Solver는3090 TP4여서 혼합 환경 추정이다.
모델 시작 비용은 이 online 합계에 포함하지 않는다. 실측 full end-to-end 비용은 null이다.
GPU 비용은 별도로 4×monitor wall seconds를 산출한다(모델 로딩·대기 포함, preflight/cleanup 제외).
전체 할당 비용은 sacct ElapsedRaw×4를 확인한다. 금액 환산은 과금 단가가 없으므로 제공하지 않는다.

5문제는 기존에 이미 분석한 탐색적 Pilot다. 여기서 유리한 군을 선택했다면 같은5개를 독립 확인 평가로 재사용하지 않는다.

## 자원 확인 결과와 요청 정책

11월17일 `scontrol show node ... -o`, `sinfo -N -p batch_ce_ugrad` 경량 조회를 수행했다.
GPU 모델의 현재 물리 상태는 master에서 nvidia-smi/CUDA로 검사하지 않았다.

| 노드 | GRES 구성/할당 | CPU / RealMemory MiB | 판단 |
|---|---|---|---|
| u2 | gpu:normal:8 / 0 | 48 /122174 | Teacher 우선 후보; 과거 launch_220313에서 A2000 12GB×4 확인 |
| u3,u4 | 각8 /0 | 각48 /122174 | IDLE이나 실제 모델/VRAM은 이번 작업에서 미확인; 자동 확장하지 않음 |
| y2 | 8 /8 | 96 /428743 | TP4 Solver 예약 후보, 현재 즉시 실행 불가 |
| y5 | 6 /4 | 96 /428743 | 2장만 남고 과거 GRES/VRAM 이력 있어 이번 경로 제외 |
| y7 | 3 /3 | 96 /428743 | 총 GPU부터4장 미만 |
| y1,y3,y4,y6 | DRAIN/DRAINING | 각96 /약428743 | 후보에서 제외 |
| r2,r3 | 총4, DOWN/DRAIN | 각32 /244730 | 제외; r1/r4/r5는 총3장 |

Slurm GRES는 generic normal이므로 모델명을 증명하지 않는다. 잘못된 카드 배정을 막기 위해 u2/y2를 명시하고
할당 후 실제 모델·용량·UUID를 검사한다. 제출 직전에 상태를 다시 조회해야 한다.
SLURM의 FreeMem은 Linux MemAvailable과 같지 않아 단독으로 가용 RAM을 단정하지 않는다.

| 작업 | GPU | CPU | RAM | 제한 / 계획 시간(미실측) |
|---|---|---:|---:|---|
| CPU 감사 | 없음 | 1 |8GiB|30분 / 약1–10분 |
| Teacher Smoke | A2000 12GB×4 |24|48GiB|2시간 / 약5–20분 |
| Teacher main | 동일4장 |24|48GiB|2시간 / 약10–45분; 서버4회 로딩 영향 |
| Solver Smoke |3090 24GB×4|48|160GiB|2시간 / 약5–30분; 최대2×8192생성 |
| Solver main |동일4장|48|160GiB|2시간 / 약20–120분 이상 가능; 25×8192상한 |
| 각 CPU 분석 |없음|1|8GiB|20분 / 대체로1–5분 |

시간은 예약 계획용 추정이다. Solver smoke의 실제 tokens/s·startup·두 응답 시간을 보고 25개가2시간에 들어갈지 재판단한다.
u2 정책: CPU/GPU=6,95% headroom RAM floor=14GiB/GPU. 요청은6CPU/12GiB perGPU.
y2 정책: CPU/GPU=12, headroom RAM floor=49GiB/GPU. 요청은12CPU/40GiB perGPU.
런타임에 실제 노드 CPUTot·RealMemory·GRES로 정책을 다시 검사한다. CPU 전용 task는1개만 요청한다.

QwQ pinned snapshot weight index는 **65,527,752,704 bytes(약61GiB)**,14개 shard다.
A2000×4의 약48GiB 합계는 **가중치만으로도 부족**하여 FP16 Solver 경로를 제공하지 않는다.
3090×4는 가중치 하한상 가능하지만 KV/CUDA graph/NCCL 포함 실측 Smoke가 필요하다.
Teacher TP4는 기존 성공 이력이 있지만 이번 런타임을 새 PASS로 간주하지 않는다.
GPU당 Teacher free≥10GiB, Solver free≥22GiB를 시작 전에 요구한다.
샘플링된 device 사용량 중단 기준은 각각11GiB/22.5GiB다. 이는 전체 카드 사용량이며 정확한 PyTorch allocator peak는 아니다.
vLLM GPU utilization .85 및 과학 설정을 바꾸지 않는 별도 안전 guard다.
GPU UUID·SLURM 매핑·CUDA_VISIBLE_DEVICES·프로세스 메모리·CUDA 초기화·작은 할당을 JSON에 기록한다.
CUDA_VISIBLE_DEVICES나 실제 GPU 번호를 바꾸거나 타인의 프로세스를 종료하지 않는다.
새 실험 내부 flock은4GPU 작업의 동시 실행을 막는다. 사용자 GPU 상한5장 조건에서는 두4GPU 작업을 순차 제출한다.
전체 user GPU 사용량까지 보장하는 lock은 아니므로 제출 전 본인의 다른 작업도 확인한다.

## 실행 및 판정 순서

**아래 명령은 사용자 실행용이며 Codex가 제출하지 않았다.** 프로젝트 루트에서 실행한다.
출력 디렉터리는 이미 준비했다. 각 `<...>`를 실제 새 Job ID로 바꾼다.

```bash
cd /data/dada0228/plan-budget/P-and-B
sinfo -N -p batch_ce_ugrad -o '%N %t %G %C %m'
scontrol show node moana-u2,moana-y2 -o
squeue -u dada0228

# 1. 새 계획/코드 감사: GPU 없음, CPU1. 이전 감사 전체는 재실행하지 않음.
sbatch experiments/linear_probe/followup_v1/audit.sbatch

# 2. audit_<AUDIT_JOB>.json passed=true 확인 후 Teacher Smoke
sbatch --export=ALL,AUDIT_JOB=<AUDIT_JOB>,MODE=smoke experiments/linear_probe/followup_v1/teacher.sbatch

# 3. complete_teacher_smoke_<TEACHER_SMOKE_JOB>.json passed=true 확인 후
sbatch --export=ALL,AUDIT_JOB=<AUDIT_JOB>,MODE=main,SMOKE_JOB=<TEACHER_SMOKE_JOB> experiments/linear_probe/followup_v1/teacher.sbatch

# 4. Teacher main 완료 후 CPU 분석
sbatch --export=ALL,AUDIT_JOB=<AUDIT_JOB>,ROLE=teacher,MAIN_JOB=<TEACHER_MAIN_JOB> experiments/linear_probe/followup_v1/analysis.sbatch

# 5. 동일 노드 3090×4 확보 및 Teacher 종료 확인 후 Solver Smoke
sbatch --export=ALL,AUDIT_JOB=<AUDIT_JOB>,MODE=smoke experiments/linear_probe/followup_v1/solver.sbatch

# 6. Smoke PASS와 2시간 내 처리 가능성 검토 후 Solver25요청
sbatch --export=ALL,AUDIT_JOB=<AUDIT_JOB>,MODE=main,SMOKE_JOB=<SOLVER_SMOKE_JOB> experiments/linear_probe/followup_v1/solver.sbatch

# 7. Solver main 완료 후 CPU 분석
sbatch --export=ALL,AUDIT_JOB=<AUDIT_JOB>,ROLE=solver,MAIN_JOB=<SOLVER_MAIN_JOB> experiments/linear_probe/followup_v1/analysis.sbatch
```

CPU 감사 자체가 fingerprint/policy의 변경을 검사하므로, 감사 뒤 소스를 변경했다면 새 감사가 필요하다.
새 CPU 감사 PASS를 기존220310으로 대체할 수 없다. 본 실험은 같은 역할의 새 Smoke gate와 audit fingerprint를 요구한다.

| 단계 | PASS 조건 | 실패 시 |
|---|---|---|
| CPU 감사 | 새8개 합성 테스트·Python AST/heredoc 문법·shell 문법 통과, 모든 checksum·frozen예측·25요청context·원본 evaluator·런타임검사 통과 | audit JSON 또는 로그 확인; GPU 제출 금지 |
| GPU Smoke | Teacher1 또는 Solver2 요청 정상stop/usage/의미 계약,4GPU guard·sample memory 통과, 완료 JSON passed=true | 원인 조사; 짧은 응답 PASS는2048/8192 worst-case 검증이 아님 |
| Teacher main |45개 모두 유효,4세션모두 guard 통과, 원본파일 불변 | complete passed=false, 기존 raw 응답/시도 보존 |
| Solver main |25개 모두 정상 수집·판정, 모든guard 통과; 정답률 자체에는 임의 PASS 임계값 없음 | 완료 실패/누락이 있으면 성공 gate 없음; 실패도분모5 원칙 |
| CPU 분석 |해당 main gate·출력SHA·audit 재검증, 전체 요청 coverage/문제별 계산 완료 | analysis passed=false |

모든 출력은 Job ID별 배타적 생성이다. 같은 Job의 기존 파일을 덮어쓰지 않는다.
본 경로에 자동 resume는 없다. 실패·시간초과의 체크포인트를 보존하고 원인 분석 후 새 Job으로 명시적으로 다시 수행한다.
부분 warm 반복을 다른 서버 세션과 섞지 않는다. 재실행 비용은 이전 실패 Job도 포함해 별도 보고해야 한다.
SIGTERM/오류는 가능한 범위에서 complete passed=false를 남긴다. SIGKILL/노드 장애면 완료 JSON이 없을 수 있으며 PASS가 아니다.

## 산출물과 완료 확인

모두 `runs/followup_v1/` 안에 생성한다.

- `audit_<JOB>.log/json`: CPU 감사. Python 컴파일/테스트는 이 계산 노드에서만 실행한다.
- `teacher_<JOB>.log`, `solver_<JOB>.log`: SLURM 로그.
- `launch_<role>_<JOB>_<session>.json`: 실제4GPU UUID·매핑·CUDA/VRAM·자원 정책 진단.
- `server_<role>_<JOB>_<session>.json/log`: 실행 argv·런타임·캐시·로딩로그.
- `memory_<role>_<JOB>_<session>.json`: 약0.2초+조회시간 간격, perGPU sample peak/guard.
- `responses_<role>_<JOB>_<session>.json`: raw 출력·사용량·실패/재시도·순서·fingerprint.
- `complete_<role>_<smoke|main>_<JOB>.json`: passed 및 모든 결과 파일 SHA.
- `analysis_<role>_<MAIN_JOB>_<CPU_JOB>.json`: 비용·오차·정답률·한계가 포함된 CPU 분석.

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,ElapsedRaw,AllocTRES -P
tail -n 60 experiments/linear_probe/runs/followup_v1/teacher_<JOB_ID>.log
tail -n 60 experiments/linear_probe/runs/followup_v1/solver_<JOB_ID>.log
# 작은 gate만 조회; responses 전체 출력이나 master Python 분석은 금지
head -n 35 experiments/linear_probe/runs/followup_v1/complete_teacher_main_<JOB_ID>.json
```

## 이번에 실제 수행한 검증과 남은 차단 요인

수행: 특정 원본 파일/선두 config 읽기, SLURM 경량 자원 조회, 새 shell 파일5개의 `bash -n` 문법 검사(정상 종료), 수동 코드 검토.
미수행: Python AST/8개 단위 테스트/새 CPU 감사/새 GPU Smoke/실험/SLURM 제출. 실행하지 않은 항목을 PASS로 표시하지 않는다.
기존 PASS를 새 실행의 PASS로 전용하지 않는다.

차단 요인은 (1) 새 CPU 감사 미실행, (2) 각 GPU Smoke 미실행, (3) 현재3090×4 가용성 없음,
(4) 긴 Solver 출력25개가2시간에 끝날지 미확인이다. 자원 확보와 Smoke 실측으로 해소해야 한다.
198후보 선택/Train24의 과적합 가능성, 위치 Baseline보다 나쁜 기존 Eval 평균 오차,5문제의 통계적 한계는
실행상의 PASS와 별개다. 이 Pilot의 성공만으로 Credit 대체 또는 Solver 우위를 확정하지 않는다.
