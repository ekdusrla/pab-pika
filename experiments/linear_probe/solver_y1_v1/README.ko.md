# y1 QwQ-32B Solver Smoke / frozen5 Pilot 준비

최신221039 실패는 `runs/solver_y1_v1/DIAGNOSIS_221039.ko.md`를 참조한다.
새 CPU 감사 후 y1의 CPU-only TASK=metadata로 실제 help-query-gpu와 GRES File 증거를 먼저 확인한다.
metadata가 BLOCKED면 GPU를 다시 제출하지 않는다. 코드 변경 후 감사221037은 재사용하지 않는다.

Job221033 preflight 인덱스 namespace 오류와 수정·회귀 검사·재제출 절차는
`runs/solver_y1_v1/DIAGNOSIS_221033.ko.md`를 참조한다(프로젝트 experiments/linear_probe 기준).
수정 후에는 source SHA가 바뀌므로 audit221030을 고치거나 재사용하지 않고 새 CPU 감사가 필요하다.

2026-11-18. 기존 실행 코드는 보존하고 이 디렉터리에만 별도 경로를 추가했다.
새 출력은 `experiments/linear_probe/runs/solver_y1_v1/`로 분리한다.
현재 판정은 **CONDITIONAL GO(실행 경로 준비)**, 실제 Pilot 승인은 새 CPU 감사 및 Smoke 전까지 **BLOCKED**다.
GPU 실행·추론·SLURM 제출·master Python 실행은 하지 않았다.

## 확인한 근거와 보호 범위

- `runs/stability_analysis_v1/REPORT_220329_220339.ko.md`: CPU220339 PASS, Teacher45요청 검증.
- 같은 디렉터리 `analysis_220329_220339.json`: passed=true, 원본 감사220327 및 Teacher Main220329 연결.
- 보고서 SHA를 가볍게 재확인했다: `746e1232e1b6bf57a41ec356c8ba52fdaa3d15b0bbb847f08dae12409815897f`.
  저장된 CPU220339의 report_sha256와 일치한다.
- 기존 Teacher 동일seed/서버재시작 MAE0, 다른seed MAE2.6556%p; Ridge가 Position보다 낫다는 근거는 없다.
- Teacher 라벨, NPZ, Ridge 모델, frozen 예측, 기존 감사 및 followup_v1 실행 코드는 수정하지 않았다.

새 CPU 감사는 기존 감사 전체/학습을 재실행하지 않는다. 다음 읽기 전용 검사를 수행한다.

1. 기존 followup audit220327 승인 체인, 보호 데이터/NPZ/원본 코드 SHA 검증.
2. CPU220339 분석 JSON fingerprint·보고서 SHA·분석 코드 SHA·보호 파일 SHA 재검증.
3. Teacher220329의45개 요청을 원래 계획대로 재파싱/검증하고 CPU220339 request_checks와 비교.
4. 기존5문제·Teacher·Uniform·Position·Ridge에서 원래 Solver payload/steps/budget을 재구성해 audit220327의25요청과 정확히 비교.
   Credit 예측·Decomposition·Teacher를 재실행하지 않는다.
5. 새 코드 및 자원 정책,4개 Smoke/25개Pilot 요청 수,seed와 상한,word 계수 sanity check,원본 evaluator 확인.

전체 SHA 재계산은 master에서 하지 않았다. **현재 시점 전체 산출물 재검증 PASS는 새 CPU 감사까지 BLOCKED**다.
MATH-500 및 Train24/Validation6/Eval5 분할·NPZ 검증은 기존 감사 체인을 계승한다.
의미상 중복까지 완전히 배제했다는 보장은 없다. Evaluation은 모델 선택에 다시 사용하지 않는다.

## y1 자원과 변경 사항

`scontrol show node moana-y1 -o` 실제 조회:

| 항목 | 값 |
|---|---|
| State / Partition | MIXED / batch_ce_ugrad 포함 |
| GRES / 할당 | gpu:normal:8 /4개 할당 |
| CPU 총/할당 |96 /26 |
| RealMemory / AllocMem |428737MiB /98504MiB |
| FreeMem |99084MiB (MemAvailable이 아니므로 단독 가용성 판단 금지) |

이 조회 시점 스케줄러 기준으로4GPU가 남는다. 제출 직전에 다시 확인해야 한다.
GRES가 generic normal이므로 현재 실제 카드 모델·UUID·유휴 VRAM은 **BLOCKED(할당 후 확인)**다.
y1을 명시하고 실제4장이 모두 NVIDIA GeForce RTX3090,23–25GiB급인지 검사한다.

| 설정 | 기존 followup_v1 | 새 solver_y1_v1 |
|---|---|---|
| 노드 | y2 | **y1** |
| GPU / TP |3090×4 /4 | 동일 |
| CPU / RAM |48 /160GiB | **16 /64GiB** |
| 모델 | Qwen/QwQ-32B | 동일 |
| revision |976055f8c83f394f35dbd3ab09a285a984907bd0 | 동일 |
| dtype / 양자화 |FP16 /없음 | 동일 |
| temperature/top_p/seed |0.1/0.9/42 | 동일 |
| max_tokens/context |8192/12288 | 동일(Pilot/qualification) |
| vLLM |0.6.6.post1,mp,seq1,util0.85 | 동일 |

CPU16은 GPU당4CPU, RAM64는 GPU당16GiB다. 노드 몫 CPU12/GPU,95% 메모리 여유를 둔 상한49GiB/GPU 이하라 정책에 맞는다.
이는 최대 몫 이내라는 뜻이며 CPU/RAM이 실행에 충분하다는 증명은 아니다.
기존 guard는48CPU/160GiB를 정확히 요구하므로 sbatch 옵션만 바꾸면 실패한다.
새 프로필을 별도 감사하고 기존 guard는 보존했다. CPU 감소가 지연시간/모델 로딩에 영향을 줄 수 있으므로 환경 차이를 기록한다.

QwQ pinned weight index 총량은65,527,752,704bytes(약61GiB),14개 shard다.
3090×4의 총 VRAM은 가중치 하한상 가능하지만 각 GPU의 KV/cache/NCCL/CUDA graph 여유는 실측이 필요하다.
**RAM64GiB의 실제 host peak 및 모델 로딩 성공 여부는 BLOCKED**다. 가중치 크기만으로 host RAM 충분/부족을 단정하지 않는다.
Smoke가 OOM이면 로그와 sacct MaxRSS/State를 보존하고 자원 프로필을 재검토한다. 임의로 dtype/모델/TP를 바꾸지 않는다.
RAM 변경 시 새 프로필·감사·Smoke가 필요하다. 호스트 OOM/SIGKILL이면 진단 JSON 없이 종료될 수도 있으며 PASS가 아니다.

## Smoke: 짧은 생성 + 원래 상한 qualification

한 서버에서 **4요청**을 순차 실행한다. 선택 문제는 기존 계획의 longest Vanilla / longest PAB다.

| phase | 요청 | 설정 | PASS 조건 |
|---|---:|---|---|
| short |2 |max_tokens128; 나머지 원래 메시지/모델/seed |비어 있지 않은 생성,usage·latency 정상; length 허용 |
| qualification |2 |max_tokens8192, Pilot과 동일 |finish_reason=stop, 원본 evaluator에서 correct 또는 incorrect로 파싱 가능 |

short의128토큰은 인프라 검사용 변경이며 Pilot 결과에 섞지 않는다.
짧게 절단된 QwQ thinking에서 최종 정답이 없을 수 있어 short만으로 최종 답안 파싱 PASS를 주장하지 않는다.
qualification에서 실제 생성된 최종 답안을 추출/평가한다. 오답도 실행·파싱 Smoke로는 PASS다.
상한8192를 설정했더라도 실제 completion이 짧으면 최악8192 메모리를 검증한 것이 아니다.
JSON에 observed tokens,finish_reason,latency와 worst_case_generation_memory_verified=false를 보존한다.

사전 guard는 SLURM 매핑·CUDA_VISIBLE_DEVICES·4개 고유 UUID·모델/VRAM·CUDA 초기화/작은 할당·환경 버전·CPU/RAM 정책을 기록한다.
GPU당 free≥22GiB가 아니면 로딩 전에 중단한다. 약0.2초+조회시간 간격의 sampled device memory가22.5GiB 초과면 자체 서버만 중단한다.
이는 nvidia-smi 전체 device 사용량의 표본 peak이며 정확한 allocator peak가 아니다.
ready JSON에는 server 시작→API ready 시간을 저장한다. 이 값은 로딩·초기화·polling을 포함하며 순수 weight 로딩 시간과 같지 않다.
타인의 프로세스를 종료하거나 GPU 매핑을 변경하지 않는다.

## frozen5 Pilot

기존 계획의 **Vanilla / Teacher / Uniform / Position / Ridge,5군×5문제=25요청**을 유지한다.
요청하신 Teacher/Ridge/Position 비교에 더해 기존 control2군을 보존해 이전 설계와 일치시킨다.
PAB4군은 같은 최종 steps·fingerprint·총 word budget이며 Credit 배분만 다르다.
Teacher는 저장 라벨, Ridge/Position/Uniform은 저장 frozen 예측을 사용한다. 추가 Planner 호출은 없다.
원본 프롬프트·정수화·allocate_words(...,constant)·QwQ evaluator를 그대로 재사용한다.

- 정확도: 원래 evaluator 판정, 각 군 분모5. undetermined/절단/실패 상태를 별도 보존한다.
- 사용량: Solver prompt/completion/total tokens, API latency, 모든 시도 비용, 예산 word 및 단계별 배분량.
- 전체 비용: 기존 Planner/Teacher/hidden capture/frozen Probe 관측 비용을 보존하고 reconstructed online 합계를 별도 보고한다.
  현재 Solver 실행에서 Planner 호출0은 저장 산출물 재사용이지 배포 비용0이 아니다. 재구성값은 혼합 하드웨어 추정이며 실측 end-to-end는 아니다.
- 예산 준수율: **전체 completion whitespace words / 총 soft word budget의 aggregate proxy**.
  원래 baseline과 같은 len(text.split())이며 thinking/최종 답안을 포함한다. stop일 때만 준수 여부를 정하고 length는 null/coverage에 포함한다.
  Vanilla에는 word budget이 없어 해당 없음이다. subgroup별 정확도와 word 준수율은 JSON/한국어 표로 생성한다.
- **단계별 실제 word/token 준수율은 BLOCKED**: 원래 joint 답변에 명확한 step span 계약이 없다.
  임의로 문장을 step에 매칭하거나 프롬프트를 바꾸지 않는다. 따라서 total-word proxy로 단계별 준수까지 입증할 수 없다.
- seed42와 고정5군 회전 순서, SDK retry0/원래 transient retry2 유지. 실패/재시도 비용 별도 저장.

신규 결과는 Job ID별 배타적 생성이다. 자동 resume는 없다. 시간초과/실패 원인을 확인한 뒤 새 Job을 명시적으로 제출한다.
새 namespace의 flock이 동시 실행을 막지만, 순차 중복 제출이나 다른 실험의 GPU 사용까지 차단하지 않는다.
사용자 GPU 상한을 고려해 squeue로 다른 본인 작업을 확인하고 순차 실행한다.
25개×8192 생성이2시간 내 가능한지는 **BLOCKED**이며 Smoke의 실제 속도/길이로 판단해야 한다.

## 제출 명령

모든 명령은 사용자 실행용이다. Codex는 제출하지 않았다. `<...>`는 실제 새 Job ID로 바꾼다.

```bash
cd /data/dada0228/plan-budget/P-and-B
scontrol show node moana-y1 -o
squeue -u dada0228

# 1. CPU1/RAM8GiB/30분. 새 자원 프로필과 기존220339 보호 입력 검사
sbatch --export=ALL,TASK=audit experiments/linear_probe/solver_y1_v1/cpu.sbatch

# 2. audit_<Y1_AUDIT_JOB>.json passed=true 후 Smoke
sbatch --export=ALL,AUDIT_JOB=<Y1_AUDIT_JOB>,MODE=smoke \
  experiments/linear_probe/solver_y1_v1/gpu.sbatch

# 3. complete_solver_smoke_<Y1_SMOKE_JOB>.json passed=true,
#    실제 RAM/VRAM/파싱/latency/2시간 소요 검토 후 Pilot
sbatch --export=ALL,AUDIT_JOB=<Y1_AUDIT_JOB>,MODE=main,SMOKE_JOB=<Y1_SMOKE_JOB> \
  experiments/linear_probe/solver_y1_v1/gpu.sbatch

# 4. Pilot 완료 후 CPU 집계/한국어 보고서
sbatch --export=ALL,TASK=analysis,AUDIT_JOB=<Y1_AUDIT_JOB>,MAIN_JOB=<Y1_PILOT_JOB> \
  experiments/linear_probe/solver_y1_v1/cpu.sbatch
```

원래220327 또는 Teacher Smoke220328를 새 Solver 감사/Smoke ID로 대신 쓰면 안 된다.
새 소스/자원 정책을 변경하면 새 CPU 감사가 필요하다.

## 완료 확인과 현재 검증 상태

모두 `runs/solver_y1_v1/`에 생성한다.

- `audit_<JOB>.json`, `cpu_<JOB>.log`: 보완 CPU 감사/분석.
- `gpu_<JOB>.log`, `server_solver_<JOB>_<session>.log/json`: 실행 및 로딩 설정/로그.
- `launch_solver_<JOB>_<session>.json`, `memory_solver_<JOB>_<session>.json`, `ready_solver_<JOB>_<session>.json`.
- `responses_solver_<JOB>_<session>.json`: short/qualification 구분 또는 Pilot25개, 원본 응답·usage·latency.
- `complete_solver_smoke_<JOB>.json`:4요청 및 guard 통과 시 passed=true.
- `complete_solver_main_<JOB>.json`:25요청 수집 및 guard 통과 시 passed=true. 정답률 임의 기준은 없다.
- `analysis_<PILOT_JOB>_<CPU_JOB>.json`, `REPORT_<PILOT_JOB>_<CPU_JOB>.ko.md`.

```bash
sacct -j <JOB> --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P
tail -n 60 experiments/linear_probe/runs/solver_y1_v1/gpu_<JOB>.log
head -n 40 experiments/linear_probe/runs/solver_y1_v1/complete_solver_smoke_<JOB>.json
```

이번에 실제 수행한 것은 원본의 가벼운 조회·보고서 SHA 비교·y1 자원 조회·신규 shell3개 bash -n·수동 정적 검토다.
Python AST/heredoc 문법 검사와 synthetic sanity check는 CPU 작업으로 준비했으며 **미실행**이다.
CPU 보완 감사·GPU Smoke·Pilot·실제 budget 준수 분석도 미실행이다. 실행 전 PASS로 표시하지 않는다.

따라서 **경로 준비 CONDITIONAL GO**, **실제 Pilot 실행 BLOCKED(새 CPU 감사/Smoke 필요)**다.
단계별 예산 준수율은 별도 계측 연구가 필요하며 이번 경로에서는 BLOCKED로 유지한다.
