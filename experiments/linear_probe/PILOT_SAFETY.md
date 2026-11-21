# 실제 Pilot 조건의 메모리 Smoke Test와 수집 보호

초기 구현 시 Codex는 Python/CPU 테스트, 실제 모델 로딩·GPU 실행 및 SLURM 제출을 수행하지 않았다. 이후 사용자가 실행한 219621·219622의 검증 결과는 마지막 절에 기록했다.
학교 서버 정책에 따라 테스트와 토큰화·모델 실행은 계산 노드 작업에서만 수행한다. 기존 Baseline·Teacher·Solver 함수와 이전 결과 파일은 변경하지 않는다.

## Smoke Test

`pilot_smoke.py`는 실제 `pilot_v1/data.json`의 train 24·validation 6·evaluation 5문제를 원본 Decomposition 프롬프트로 CPU 토큰화한다. benchmark는 수집 대상이 아니므로 최장 입력 선택에서 제외한다.
토큰 수가 가장 큰 문제를 선택하며 같은 길이는 ID 순으로 결정한다. 현재 `context_plan.json`의 최대 939토큰과 재계산 결과를 대조하고, manifest·tokenizer fingerprint가 다르면 중단한다.

- 모델: 고정 revision의 Llama-3.1-8B-Instruct, FP16, eager, batch=1, 전체 hidden state 인덱스 0~32.
- 생성 설정: temperature=.1, top_p=.9, seed=42, `max_new_tokens=1024`, context 상한 8192.
- SLURM allocation의 노드 CPU/GPU·MEM/GPU 정책, CPU 12개·RAM 40GiB, visible GPU 1개와 RTX 3090 모델명을 **가중치 로딩 전에** 검사한다.
- 원본 프롬프트를 수정하거나 EOS를 억제해 생성을 억지로 늘리지 않는다.
- 성공한 feature는 별도 NPZ로 저장하고 JSON에는 path/hash/shape와 단계·token position을 남긴다. JSON에 거대한 feature 배열을 넣지 않는다.
- 모델 로딩·생성·GPU hidden 보관·경계 매핑·CPU 전송 시간을 기록한다. hidden 보관 시간은 생성 시간 안에 포함되므로 둘을 다시 합산하지 않는다.
- 전체 실행 시간에는 토큰화·모델 로딩·feature 저장도 포함된다.

PASS는 EOS 정상 종료, 2~5 단계 파싱, 단계/feature 수 및 token 위치 대응, `[단계 수,33,4096]` feature shape·유한값, 추가 forward 0회, peak reserved <=21GiB를 모두 만족한 **관찰된 실행**에만 부여한다.
JSON의 `passed`, `safe_observed_case`, `parse_success`, `features_valid`, `peak_reserved_bytes`를 확인한다. job도 COMPLETED/0:0이어야 한다.

`max_new_tokens=1024`는 생성 상한이다. 실제 출력이 150토큰이면 150토큰 궤적만 측정한 것이다.
`length_coverage.observed_completion_tokens`, `near_cap_90_percent`를 기록하고 `worst_case_1024_tokens_verified=false`로 명시한다. 90% 근처까지 생성했더라도 일반적인 최악 조건을 증명한 것으로 해석하지 않는다.

실패는 `memory_limit_exceeded`, `oom`, `generation_truncated`, `parse_failure`, `invalid_features`, `context_exceeded`, `memory_measurement_failed`, `interrupted`, `io_error`, `runtime_error`로 구분한다.
모델 로딩·사전 검사 실패도 새 JSON에 저장한다. 일부 구간 시간은 실패 시 측정되지 않아 null일 수 있으며 이를 0초로 해석하지 않는다.
SIGTERM은 가능한 시점에 잡아 진단을 저장한다. SIGKILL·노드 장애·디스크 쓰기 실패처럼 기록 자체가 불가능한 상황은 초기 running JSON과 SLURM 로그를 확인해야 한다.

## GPU 메모리 guard

공유 `safety.py`를 Smoke와 실제 `cli collect`가 사용한다.
각 문제와 모델 로딩에서 peak 통계를 reset하고, 현재 allocated/reserved와 peak allocated/reserved를 구분한다. allocator 캐시를 임의로 비우지는 않는다.
forward·hidden clone·sampling·feature 전송 구간이 끝날 때 peak reserved를 검사한다. **21GiB 초과 시 다음 토큰/문제로 진행하지 않는다.**
단일 forward 내부의 큰 할당은 완료 구간 검사 전에 OOM을 낼 수 있다. OOM도 별도로 기록하고 중단하며, 이 guard가 OOM 자체를 예방한다고 주장하지 않는다.
메모리 측정 실패도 안전한 상태로 간주하지 않고 중단한다. 한도를 21GiB보다 높이는 입력은 거부한다.

실제 수집의 각 `records[].memory`와 `attempts[].memory`에 문제별 수치를 기록한다. 실패한 문제의 원인·부분 token trace·마지막 정상 문제 ID는 `safety_stop`과 attempt에 남긴다.
완료한 문제의 JSON 및 NPZ는 보존한다. 새 문제 전에 checkpoint를 저장하므로 강제 중단된 running attempt도 재개 시 interrupted/cost_unknown으로 보존된다.
평가 split의 선택적 capture off/on 대조 생성도 각각 guard를 적용하며 별도 메모리·비용을 기록한다.

## 재개 규칙

- 일반적인 미완료 실행: 동일 code/config/data로 `--resume`.
- 저장된 안전 중단이 있으면 자동 재개를 거부한다. 원인 확인 후 `--resume --resume-after-safety-stop`을 함께 사용해야 한다.
- 이 확인 옵션은 메모리 한도를 완화하지 않는다. 성공한 문제는 건너뛰고 실패한 문제에 새 attempt를 추가하며, 이전 안전 중단은 `safety_stop_history`에 보존한다.
- code·환경·모델·프롬프트·한도 등 config가 바뀌면 기존 strict checkpoint 검사가 재개를 거부한다. 변경 전 수집 결과를 새 안전 정책으로 수행한 것처럼 재해석하지 않는다. 필요한 경우 새 run/output을 사용한다.
- 모델 로딩 실패도 session과 safety_stop에 남으며 이미 완료한 문제를 지우지 않는다.
- 기존 `.tmp` 파일은 자동 폐기하지 않는다. 디스크 상태와 저장 내용을 먼저 점검한다.

향후 실제 수집 시 기본 한도는 21GiB다. 현재 단계에서는 수집을 실행하지 않는다.

## 계산 노드 CPU 테스트: 사용자가 제출

```bash
cd /data/dada0228/plan-budget/P-and-B
TEST_JOB=$(sbatch --parsable --gres=none experiments/linear_probe/safety_cpu_tests.sbatch)
echo "$TEST_JOB"
```

CPU 2개, RAM 4GiB, 10분 제한이며 GPU는 사용하지 않는다. AST/셸 문법 검사와 기존 테스트 및 신규 `test_safety`를 실행한다.
신규 테스트는 가짜 CUDA counter·작은 CPU causal model·합성 feature만 사용한다. 실제 Llama 로딩, 연구 데이터 Probe fit, 모델 API 호출은 하지 않는다.
21GiB 경계, OOM/절단/파싱 구분, 짧은 EOS 출력의 coverage, 잘못된 GPU 거부, 실패 JSON/덮어쓰기 보호, 완료 결과 보존·후속 문제 중단·명시적 재개를 검증한다.
결과 로그: `experiments/linear_probe/runs/safety_tests_JOBID.log`. 사용자가 실행한 219621에서 42개가 통과했다. 이후 추가된 수집 스크립트 재개 플래그의 셸 문법은 실제 계산 노드 작업 시작 시 검사하며, 마스터에서 테스트하지 않는다.

## Pilot Smoke 제출 명령: 사용자가 직접 실행

제출 직전에 `sinfo`/`squeue`로 상태를 재확인한다. 조회 시 y4는 DRAIN, y7은 MIXED이며 GPU 3개 중 2개가 할당돼 있었다.
현재 확인된 두 후보에서 CPU 12개·RAM 40GiB는 노드별 정책 상한 이내다. 다른 GPU 노드를 제외하고 실제 GPU 모델도 작업 내부에서 재확인한다.
새 노드가 partition에 추가되는 등 구성이 바뀌면 제외 목록도 다시 확인해야 한다.

```bash
sbatch \
  --parsable \
  --account=ugrad_ce \
  --qos=qos_dada0228_2026_2 \
  --partition=batch_ce_ugrad \
  --nodes=1 --ntasks=1 \
  --gres=gpu:normal:1 \
  --cpus-per-task=12 --mem=40G \
  --time=02:00:00 \
  --exclude='moana-r[1-5],moana-u[1-4,8],moana-y[1-3,5-6]' \
  --dependency=afterok:"$TEST_JOB" \
  experiments/linear_probe/pilot_smoke.sbatch
```

`TEST_JOB`은 위 계산 노드 테스트의 실제 Job ID다. 실패한 4K Job 219610에 afterok 의존성을 걸지 않는다.
예상 Smoke 실행 시간은 초기 로딩·I/O를 포함해 약 1~5분이며 실측 전 추정이다. 제한은 2시간이다.
로그: `experiments/linear_probe/runs/pilot_smoke_JOBID.log`.
JSON: `experiments/linear_probe/runs/pilot_v1/smoke_JOBID.json`.
성공 feature: 같은 JSON 경로 뒤에 `.features.npz`가 붙는다.

이 job은 최장 문제 하나만 검사한다. 전체 수집, Teacher, Probe 학습, Solver는 호출하지 않는다.

## PASS 이후 수집 준비 조건

1. 계산 노드 CPU 테스트 통과와 Smoke의 SLURM COMPLETED/0:0.
2. 실제 manifest/tokenizer/프롬프트가 맞고 prompt 최대 939가 재현됨.
3. `passed=true`, `safe_observed_case=true`, 파싱·feature 검증 통과, peak reserved <=21GiB.
4. 실제 completion 길이를 확인하고 짧으면 1024토큰 최악 조건은 미검증으로 남김.
5. 후속 수집에서도 동일한 문제별 guard·안전 중단·기록 보존을 사용함. 전체 생성 길이 범위에 대한 사전 보장이 필요하면 별도 길이 스트레스 검증이 추가로 필요함.
6. 이후 사용자가 별도로 수집을 요청·제출함. 현재 Smoke job이 후속 단계를 자동으로 시작하지 않음.

## 219622 검증 후 실제 수집 준비

계산 노드 테스트 219621은 42개 테스트 통과, COMPLETED/0:0이었다. Smoke 219622도 moana-y7에서 COMPLETED/0:0이었다.
939 prompt / 114 completion, feature shape [3,33,4096], 유한값 검사·파싱 성공, 추가 forward 0회, allocated 약15.69GiB / reserved 약15.89GiB를 기록했다.
이는 관찰 조건 PASS이며 1024토큰 최악 조건은 미검증이다. 실제 수집에서도 guard를 유지하고 문제마다 실제 peak를 확인해야 한다.

실제 수집 진입점은 `pilot_gpu.sbatch`의 STAGE=capture → `cli collect` → `safety.guarded_generate` → `capture.generate`다.
기본 수집은 24/6/5의 35문제이며 평가 5문제의 off/on 대조를 포함하면 생성은 총40회다. 대조 생성도 각각 guard를 적용한다.
각 split은 순서대로 별도 프로세스에서 모델을 로딩한다. Teacher·학습·Solver는 시작하지 않는다.

`pilot_gpu.sbatch`에 수집 전용 재개 환경변수를 추가했다. Python 안전성 코드는 변경하지 않았으며 스크립트 문법 검사는 계산 노드 작업 시작 시 수행한다.

- 첫 제출: `CAPTURE_RESUME=0`, `ACK_SAFETY_STOP=0`.
- 일반 실패 재개: `CAPTURE_RESUME=1`, `ACK_SAFETY_STOP=0`.
- 메모리/OOM 등 저장된 안전 중단 후: 원인 확인 후 `CAPTURE_RESUME=1`, `ACK_SAFETY_STOP=1`.
- 존재하는 checkpoint에만 --resume을 전달한다. 완료 split은 검증 후 건너뛰고, 아직 시작하지 않은 split은 새로 생성한다.
- 코드/config/data/대조 설정/경로를 그대로 유지한다. .tmp는 자동 폐기하지 않는다. 확인 옵션은 21GiB 한도를 완화하지 않는다.

수집 출력은 `train_capture.json`, `validation_capture.json`, `evaluation_capture.json` 및 각 `.json.features/` 아래 NPZ다.
job 로그는 `runs/pilot_JOBID.log`, 노드 정책 기록은 `pilot_v1/resources_JOBID.json`이다.
안전 중단 시 `safety_stop`, `last_successful_problem_id`, 문제별 `attempts`·`memory`를 계산 노드에서 확인하고, 기존 결과를 삭제하거나 덮어쓰지 않는다.
