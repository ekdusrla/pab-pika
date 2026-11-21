# 입력 길이 확장 및 24/6/5 Linear Probe Pilot

기존 Baseline, 전체 500개 평가 경로 및 기존 결과를 보존한다. 이 작업에서는 GPU 모델 로딩/실험/서버 시작/SLURM 제출을 하지 않았다.
Job 219600의 PASS 수치는 사용자가 공유한 기록으로 보관했으며 재실행하지 않았다.

## 준비된 데이터

`runs/pilot_v1/data.json`은 실제 공개 데이터로 만든 immutable manifest다.

| 항목 | 값 |
|---|---|
| Dataset | `EleutherAI/hendrycks_math` |
| Revision | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| Source split | 7개 분야의 **train만** 다운로드 |
| Source 크기 | 7,500문제 |
| Seed | 20261113 |
| Train / Validation / 독립 Pilot 평가 | 24 / 6 / 5 |
| Teacher benchmark | 별도의 외부 train 5개, native level 1~5 각각 1개 |
| 최종 평가 | MATH-500 전체 보존, Pilot에도 사용하지 않음 |

로컬 프로젝트와 확인한 HF 캐시에는 외부 수학 train 데이터가 없었다. GSM8K는 쉬운 산술 검증에 적합하지만 native PAB 1~5 난이도가 없어 추가 매핑이 필요하다.
이번 Pilot은 최종 평가 분포에 더 가까우며 native level/solution이 있는 MATH train을 선택했다. 참고: [MATH 데이터](https://huggingface.co/datasets/EleutherAI/hendrycks_math), [GSM8K 원본](https://github.com/openai/grade-school-math).
HF source card는 MIT license를 명시하며 `source_download/README.md`에 원문을 보존했다.

문항 ID는 `dataset@revision/subject/train/row_index`다. 원본 HF 파일에 문항 ID 컬럼이 없으므로 고정 revision의 파일·행 번호와 문제 hash를 함께 사용한다.
`source.json`은 모든 다운로드 파일의 SHA-256과 URL을 기록한다. 총 다운로드는 train parquet 7개와 README이며 test는 받지 않았다.

7,500개 중 native `Level ?` 2개, 정규화 중복 2개, 마지막 boxed 정답을 추출할 수 없는 4개를 제외하여 7,492개가 후보로 남았다. 전체 train과 MATH-500 사이의 정규화 일치는 0개였다.
선택은 native 난이도별 SHA256(seed,문제 hash) 순위와 고정 역할 순서로 한다. Teacher/Probe/Solver 결과는 선택에 사용하지 않는다.
모든 단계는 원문 문제의 split을 상속하므로 sub-question이 split 사이에 섞이지 않는다.

중복 검사는 다음 두 단계를 사용한다.

1. NFKC + casefold + 공백 제거 hash로 MATH-500 및 외부 내부 중복을 제거한다.
2. 선택 후보를 MATH-500과 먼저 선택된 외부 문항에 비교한다. 일부 LaTeX 표기를 통일한 문자열 일치 또는 문자 5-gram Jaccard >= .90이면 제외한다.

선택된 40개는 두 검사에서 모두 중복 0개다. 선택 문항과 MATH-500 사이 최대 문자 5-gram Jaccard는 약 0.37255다.
이 수치는 의미상 중복 부재의 증명이 아니다. manifest의 `nearest_final_neighbours`를 이용해 가까운 원문 쌍을 수동 검토하고, 숫자/변수 치환을 정규화한 수식 구조 비교나 문장 embedding 후보 검색을 추가하는 것이 좋다.
Pilot 평가 5개의 자동 boxed 정답도 원문 풀이와 수동 대조해야 한다.

`data_audit.json`은 원본 파일 hash 검증과 선택 재현 결과를 기록한다. `context_plan.json`의 CPU 토큰화 결과는 Decomposition 최대 939 tokens, Teacher 보수적 입력+출력 예약 최대 4,990 tokens다.
Teacher의 **실제 생성 steps**를 사용한 정확한 길이는 서버 시작 전에 다시 검사한다.

CPU에서 데이터를 처음부터 재현하려면 새 출력 디렉터리를 사용한다. 이미 준비된 파일을 다시 덮어쓰지 않는다.

```bash
export HF_PY=/data/dada0228/conda_envs/vllm_server/bin/python
export PROBE_PY=/data/dada0228/conda_envs/shortcut-xai/bin/python
export RUN=experiments/linear_probe/runs/pilot_v1
export LENGTH_RUN=experiments/linear_probe/runs/length_v1
export EVAL=experiments/llama_qwq/math500_manifest.json
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 준비된 source/data를 검증만 하고 별도의 새 audit 파일에 쓴다.
CUDA_VISIBLE_DEVICES='' "$HF_PY" -B -m experiments.linear_probe.pilot_data audit \
  --source "$RUN/source_download" --data "$RUN/data.json" \
  --eval-manifest "$EVAL" --output "$RUN/data_audit_recheck.json"

# 새 run을 만들 경우에만: 네트워크를 사용하는 CPU 다운로드 명령
# "$HF_PY" -B -m experiments.linear_probe.pilot_data fetch --output NEW_RUN/source
# "$HF_PY" -B -m experiments.linear_probe.pilot_data select --source NEW_RUN/source \
#   --eval-manifest "$EVAL" --output NEW_RUN/data.json --seed 20261113
```

## 노드별 자원 정책과 제출 방법

CPU 상한은 `CPUTot / 노드 GPU 수`, MEM 상한은 `RealMemory / 노드 GPU 수 × .95`다.
CPU는 정수 내림, RAM은 GiB 정수 내림한다. GPU 4개 작업에는 GPU당 값의 4배를 요청한다.
예시 노드 CPU=96, GPU=8, MEM=418GiB이면 GPU당 CPU=12, RAM=49GiB다. 이는 예시이며 실제 값은 제출 직전 갱신한다.

```bash
# 사용자가 제출 직전에 실행: 현재 노드/큐 정보 확인, 노드별 리소스 snapshot 저장
sinfo -N -l
squeue -u "$USER"
scontrol show nodes -o > /tmp/probe_nodes.txt
```

`submit_pilot`은 batch_ce_ugrad partition에서 해당 수의 gpu:normal을 제공할 수 있는 후보 노드들의 공통 최소 CPU/GPU·MEM/GPU 값을 계산한다.
사용자가 더 낮은 값을 지정할 수 있지만 계산한 상한을 넘으면 거부한다. 특정 노드를 고정하지 않는다.
RTX 3090을 의미하는 실제 cluster feature가 있으면 `--constraint 실제_feature`를 전달한다. 이 값은 snapshot 검사와 sbatch 모두에 적용된다. feature 이름을 임의로 추정하지 않는다.
이런 feature가 없으면 gpu:normal 요청 후 실제 할당 GPU가 모두 RTX 3090인지 작업 내부에서 검사한다. 다른 GPU면 모델 로딩 전에 중단한다.

**아래 명령들은 sbatch 문자열을 출력할 뿐 제출하지 않는다. 출력된 문자열을 사용자가 확인 후 직접 실행한다.**
CPU/RAM은 기본적으로 fresh snapshot에서 자동 산정된다. RAM이 작으면 CPU model loading의 host memory도 부족할 수 있으므로 32~48GiB 이상 후보를 권장한다. 이를 위해 정책 상한을 초과해서는 안 된다.
GPU 작업의 walltime은 모두 2시간이다. 실제 할당 노드의 `scontrol show node` 정보로 정책을 다시 검사하고 `resources_JOBID.json`에 저장한다.

## 1K → 2K → 4K 길이 검증

준비된 `length_v1/plan.json`은 원본 system/user 템플릿을 유지하고 합성 문제의 식별자 목록 길이만 조정했다.

| Target | 실제 prompt tokens | 생성 상한 |
|---|---:|---:|
| 1K | 1,026 | 512 |
| 2K | 2,046 | 512 |
| 4K | 4,096 | 512 |

FP16, eager attention, 전체 33개 레이어, batch=1, seed=42, .1/.9를 유지한다.
각 조건은 별도 GPU job/process로 실행하며 `length_TARGET_JOBID.json`에 기록한다. 모델 로딩 peak와 생성 peak의 allocated/reserved를 구분한다.
실제 completion 수, 단계/feature, 파싱 여부, forward 수, 토큰 trace, 구간 시간과 전체 wall time을 기록한다.
실패 상태는 `oom`, `context_exceeded`, `generation_truncated`, `parse_failure`, `runtime_error`, `resource_busy`로 나눈다.
SIGKILL/노드 장애처럼 Python finally를 실행할 수 없는 강제 종료는 JSON이 남지 않을 수 있어 SLURM 로그/exit state도 확인한다.

```bash
"$HF_PY" -B -m experiments.linear_probe.submit_pilot length --run "$LENGTH_RUN" \
  --nodes-file /tmp/probe_nodes.txt --target 1024

# 위 출력 sbatch를 사용자가 제출한 Job ID를 L1에 넣는다.
export L1=실제_1K_Job_ID
# 새 노드 snapshot을 확보한 뒤:
"$HF_PY" -B -m experiments.linear_probe.submit_pilot length --run "$LENGTH_RUN" \
  --nodes-file /tmp/probe_nodes.txt --target 2048 --dependency "$L1" \
  --previous "$LENGTH_RUN/length_1024_${L1}.json"

export L2=실제_2K_Job_ID
"$HF_PY" -B -m experiments.linear_probe.submit_pilot length --run "$LENGTH_RUN" \
  --nodes-file /tmp/probe_nodes.txt --target 4096 --dependency "$L2" \
  --previous "$LENGTH_RUN/length_2048_${L2}.json"
```

이전 조건의 plan fingerprint·target·성공 상태를 대조한다. 이전 조건이 실패했거나 reserved peak가 21GiB를 넘으면 다음 조건으로 진행하지 않는다.
모델 로딩 전 free GPU memory가 22GiB 미만일 때도 중단한다.
4K 생성·파싱이 성공해도 reserved peak가 21GiB를 넘으면 성공 데이터를 저장하고 job은 nonzero로 종료해 추가 확장을 막는다. SDPA·양자화로 자동 전환하지 않는다.
반복 식별자로 만든 합성 입력 검사이므로 같은 길이의 실제 수학 문제에 대한 의미 처리·정답 품질을 보장하지 않는다.

## Probe Pilot 실행 순서

1. fresh snapshot으로 capture 제출 명령을 만들고 사용자가 제출한다.

```bash
"$HF_PY" -B -m experiments.linear_probe.submit_pilot capture --run "$RUN" \
  --nodes-file /tmp/probe_nodes.txt
```

같은 job에서 train/validation/evaluation을 순서대로 수집한다. 각 CLI 프로세스 종료 시 모델을 해제한다.
평가 5문제는 기본적으로 capture off/on 대조 생성도 수행한다. 운영 경로에 해당하는 35회와 대조 5회를 합쳐 **실험 생성은 총 40회**다. 대조 비용을 운영 호출 1회에 숨기지 않는다.
대조 생성을 빼려면 제출 전에 `export CAPTURE_CONTROL=0`으로 설정하고 새 run을 사용한다.

2. capture 성공 후 Teacher 제출 명령을 만들고 사용자가 제출한다.

```bash
"$HF_PY" -B -m experiments.linear_probe.submit_pilot teacher --run "$RUN" \
  --nodes-file /tmp/probe_nodes.txt --dependency 실제_capture_Job_ID
```

Teacher job은 먼저 CPU tokenizer로 실제 모든 steps의 context를 검사하고, 같은 allocation에서 vLLM Llama 서버와 API 클라이언트를 실행한다.
기존 Decomposition을 재생성하지 않는다. 외부 benchmark만 사용하며 기존 prompt/schema/parser를 재사용한다.
무효 Credit을 균등 라벨로 바꾸지 않는다. 35문제 전체의 라벨 성공이 학습 조건이다.
서버 시작·ready 확인·자식 프로세스 종료를 job 안에서 관리한다. API port는 job ID로 구분하고 시작 전에 빈 포트인지 확인한다.

3. CPU에서 학습한다. 실제 연구 데이터 학습은 아직 실행하지 않았다.

```bash
CUDA_VISIBLE_DEVICES='' "$PROBE_PY" -B -m experiments.linear_probe.cli train \
  --data "$RUN/data.json" --train "$RUN/train_labeled.json" \
  --validation "$RUN/validation_labeled.json" --output "$RUN/probe"
```

train 24문제로만 StandardScaler+Ridge(alpha=1)를 fit하고 validation 6문제의 문제 평균 Credit MAE로 레이어를 선택한다.
validation이나 Pilot 평가로 다시 fit하지 않는다. 4096차원에 비해 매우 작은 학습셋이므로 기능·비용 검증이 목적이며 일반화 성능의 결론을 내릴 수는 없다.

4. CPU에서 독립 Pilot 평가 5문제를 비교한다.

```bash
CUDA_VISIBLE_DEVICES='' "$PROBE_PY" -B -m experiments.linear_probe.cli evaluate \
  --data "$RUN/data.json" --capture "$RUN/evaluation_labeled.json" \
  --probe "$RUN/probe" --decay constant --output "$RUN/evaluation_core.json"
CUDA_VISIBLE_DEVICES='' "$PROBE_PY" -B -m experiments.linear_probe.pilot compare \
  --data "$RUN/data.json" --evaluation "$RUN/evaluation_core.json" --output "$RUN/comparison.json"
```

문제 평균 MAE/RMSE와 전체 step을 동일 가중치로 계산한 micro MAE/RMSE/bias, step별 Teacher/Probe/raw Credit과 부호 있는 차이, word Budget과 부호 있는 차이를 기록한다.
Teacher 성공·전체 시도 비용, hidden 보관·전송·경계 매핑, capture off/on 차이, Probe feature 읽기·예측·정수화 시간은 원래 evaluation/capture/label 기록에 분리한다.
`capture_control` 차이는 token열이 같을 때만 유효하다. GPU 동기화·실행 순서·cold start의 영향을 받는 단발 수치로 속도 향상을 단정하지 않는다.

5. CPU에서 5문제×2방식 Solver request를 고정한다.

```bash
"$PROBE_PY" -B -m experiments.linear_probe.pilot solver-prepare \
  --data "$RUN/data.json" --evaluation "$RUN/comparison.json" --output "$RUN/solver_requests.json"
```

6. 사용자가 GPU 4장·TP=4 Solver job을 제출한다.

```bash
"$HF_PY" -B -m experiments.linear_probe.submit_pilot solver --run "$RUN" \
  --nodes-file /tmp/probe_nodes.txt
```

QwQ-32B의 기존 revision/FP16/TP=4, context=12288, 생성 상한 8192, .1/.9/seed42, 기존 PAB prompt·난이도별 word budget을 유지한다.
Teacher/Probe는 같은 steps를 사용하고 Budget만 달라진다. 5문제×2방식의 요청 10개만 실행하며 추가 Vanilla는 실행하지 않는다.
방식별 실행 순서는 문제마다 교대한다. 정답률 분모는 각 방식 5문제이며 truncation/undetermined/실패를 숨기지 않고 token/latency 차이와 전체 시도 비용을 저장한다.
기존 500문제 export/solve/checker를 변경하지 않고 Pilot 전용 runner로 제약을 해결한다.

같은 출력이 이미 있으면 중단한다. `solver-run`과 기존 CLI의 `--resume`은 동일 request/config에 한정한다. 새 job에서 API endpoint나 server metadata가 바뀌면 자동 재개하지 않는다.
실패 시 JSON/로그와 원래 config를 보존하고 동일 설정으로 명시적으로 재개하거나 새 run을 사용한다. 과거 실패 비용을 버리고 성공 사례만 보고하지 않는다.

## 병렬화와 필요 자원

길이 ramp는 반드시 직렬 실행한다. capture/teacher는 `--split train` 등으로 별도 allocation에 나눠 병렬화할 수 있다.
각 GPU job은 GPU 1장을 정식으로 요청한다. Teacher는 대응 capture의 `afterok`를 지정한다. 다중 의존성은 `--dependency ID1:ID2:ID3`을 지원한다.
같은 split은 중복 제출하지 않는다. CPU 학습은 train/validation 라벨 완료 후, 비교는 평가 라벨 완료 후, Solver는 request 고정 후에 진행한다.
HF와 서버를 같은 GPU에 동시에 올리지 않는다. server와 API client는 같은 allocation 안에서 관리하고, 다른 job과의 GPU 사용은 SLURM 할당에 따른다.

| 단계 | GPU | CPU/RAM | 초기 예상 실행 시간 | 제한 |
|---|---:|---|---|---|
| 길이 검증 1조건 | RTX3090×1 | fresh 노드 비율 적용 | 2~10분 | 2시간 |
| HF 수집 35문제+대조 5회 | RTX3090×1 | 동일 | 5~30분 | 2시간 |
| Teacher 35문제 | RTX3090×1 | 동일 | 10~60분 | 2시간 |
| Ridge/비교/request 작성 | GPU 불필요 | CPU 1~4, RAM 8~16GiB 예상 | 1~10분 | CPU 환경 |
| QwQ Solver 요청 10개 | RTX3090×4, TP=4 | GPU당 비율×4 | 10~90분 | 2시간 |

시간은 실측 전 예약 기준이며 모델 cache I/O·생성 길이·JSON 실패/재시도에 따라 달라진다. 큐 대기는 포함하지 않는다.
자원이 부족하면 실패 기록을 확인하고 다음 실험 조건을 별도로 고정한다. 기존 설정을 자동으로 바꾸지 않는다.

## CPU 검증과 남은 과제

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$PROBE_PY" -B -m unittest -v experiments.linear_probe.test_probe experiments.linear_probe.test_pilot
```

실제 GPU 없이 guard·실패 분류·24/6/5 분리·정규화 중복·예측 계약·Budget·기존 prompt·모의 Teacher 요청 35개→작은 합성 Ridge→5문제 비교→모의 Solver 요청 10개와 실패 재개를 검증한다.
실제 연구 데이터의 Probe 학습이나 모델 API 요청은 실행하지 않는다.
남은 과제는 실제 길이 ramp, 의미 중복·정답의 수동 확인, Teacher 유효 라벨 비율, Ridge 실데이터 성능, Solver 정답률·시간 차이 실측, 반복 시간 측정이다.
