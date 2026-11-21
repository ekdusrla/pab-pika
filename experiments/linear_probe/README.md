# Linear Probe Credit Assessment 실험

기존 PAB 소스/결과와 공유받은 `probe_pipeline/`을 수정하지 않는 별도 실행 경로다.
이 디렉터리의 Python 모듈은 프로젝트 루트에서 `python -m`으로 실행한다.
모든 출력은 `experiments/linear_probe/runs/` 아래로 제한하며 기존 출력은 덮어쓰지 않는다.
어떤 명령도 SLURM 제출이나 서버 시작을 자동 수행하지 않는다.

## 구현 범위와 비교 조건

```text
외부 학습 후보 + 외부 benchmark 5개 + 기존 MATH-500 manifest
  prepare → 문제 단위 train / validation / evaluation 고정
  collect → HF Llama Decomposition + 생성 중 hidden state → JSON + NPZ
  label   → 저장된 동일 steps를 기존 vLLM Credit Assessment에 전달
  train   → StandardScaler + Ridge(alpha=1), validation으로 레이어 선택
  evaluate → 동일 steps의 Teacher Credit / Probe Credit / Budget / 시간 비교
  export → Teacher 또는 Probe Planner JSON
  solve  → 기존 QwQ-32B Solver와 기존 Budget 함수 재사용
```

- Decomposition은 `original_definitions()`로 추출한 기존 프롬프트를 그대로 사용한다. 마커를 추가하지 않는다.
- Llama 모델/revision은 기존 8B 실험과 동일하다. FP16, temperature=.1, top_p=.9, seed=42, 생성 상한 1024, context 상한 8192다.
- HF 수동 KV-cache decoding으로 생성에 이미 사용한 hidden state를 보관한다. embedding 인덱스 0과 Transformer 출력 1~32가 후보이며 32는 HF 최종 norm을 거친 출력이다.
- 생성 후 기존 numbered-step 파서의 각 단계 끝과 token decode prefix를 대응시킨다. feature는 해당 단계 안에 완전히 들어오는 마지막 생성 토큰을 **입력으로 처리한 뒤의** hidden state다. 그 토큰을 예측했던 이전 위치가 아니다.
- 모델 generation config의 모든 EOS ID를 처리한다. 정상 EOS 종료에서는 마지막 내용 토큰도 이미 처리되므로 추가 forward는 0이다. 절단·파싱 실패·불안정한 경계는 수용하지 않는다.
- 단계 feature는 FP32 NPZ, 토큰별 임시 feature는 모델 dtype GPU tensor로 보관한다. 평가에서는 학습 후 선택한 레이어만 `--layers N`으로 저장할 수 있다. `output_hidden_states=True` 자체는 전체 hidden tuple을 반환한다.
- Probe는 각 단계 hidden vector → 실수 Credit 하나를 예측한다. 음수는 0으로 clamp하고 단계당 1점을 예약한 뒤 나머지 점수를 largest remainder 방식으로 배분한다. 모두 0 이하일 때만 결정적 균등 배분을 사용한다. NaN/Inf는 실패시킨다. raw/normalized/integer 값과 보정 여부를 모두 기록한다.
- train에만 scaler와 Ridge를 fit한다. validation의 문제별 Credit MAE로 레이어를 고른다. validation으로 재학습하지 않으며 test는 선택에 사용하지 않는다. alpha=1은 기본 고정값이다. 평가 결과를 보고 레이어/alpha/보정 규칙을 다시 고르면 별도 실험이어야 한다.

## 데이터 준비: 먼저 외부 파일 필요

이 구현은 외부 데이터를 다운로드하거나 학습 데이터를 임의 선택하지 않는다.
MATH-500을 제외한 학습 후보를 별도 JSON으로 준비해야 한다. 예를 들어 적법하게 확보한 MATH train에서 MATH-500 및 중복을 제거한 문제를 사용할 수 있다.
아래는 **입력 형식 설명**이며 실제 연구 데이터가 아니다.

```json
{
  "provenance": {"dataset": "external source name", "split": "train", "revision": "pinned revision"},
  "records": [
    {"id": "source/train/001", "problem": "문제 원문", "level": 3}
  ],
  "benchmarks": [
    {"id": "source/benchmark/001", "problem": "별도의 benchmark 문제", "level": 1, "solution": "정답 풀이"}
  ]
}
```

실제 파일은 후보가 2개 이상이고 benchmark가 난이도 1~5를 모두 포함해야 한다.
benchmark는 후보 train/validation 및 평가와 중복되면 안 된다. 데이터 양은 CPU 테스트의 최소 조건보다 충분히 크게 확보해야 한다.

문제의 NFKC 정규화 + casefold + 공백 제거 hash로 평가셋과의 중복 및 전체 내부 중복을 거부한다. 문항 ID도 후보/benchmark 사이에 중복될 수 없다.
문제 hash와 seed로 분할하므로 같은 문제의 모든 단계는 한 split에만 들어간다.
이 검사는 의미적으로 같은 문제의 재서술·수식 표기 변형까지 탐지하지 않는다. 출처의 원본 ID/데이터 계보와 의미 중복은 데이터 준비자가 추가 점검해야 한다.

기존 Baseline은 평가 문제 5개의 정답 풀이를 benchmark로 사용한다. 신규 실험은 평가 외부 benchmark를 사용하므로 **기존 과거 결과와 Teacher 조건이 완전히 같지는 않다.**
공정한 Credit 대체 비교는 신규 실험 안에서 같은 분해·같은 benchmark를 공유한 Teacher와 Probe 사이에서 수행한다. 기존 결과는 보존한다.

## 실행 순서

아래는 향후 사용자 실행용 명령이다. 구현 중 실제 수집·라벨링·실험 학습·추론은 실행하지 않았다.
`$PY`는 별도로 준비한 HF/Probe 환경의 Python이다. 기존 vLLM 서버 환경과 분리한다.
환경 설치는 자동 수행하지 않는다. CPU 테스트에는 기존 `shortcut-xai` 환경을 사용했다.

```bash
cd /data/dada0228/plan-budget/P-and-B
export PY=/path/to/probe-env/bin/python
export RUN=experiments/linear_probe/runs/probe_v1
export EVAL=experiments/llama_qwq/math500_manifest.json
export LLAMA=/data/dada0228/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

# 1. CPU: 평가셋과 분리된 외부 후보/benchmark를 고정한다.
"$PY" -B -m experiments.linear_probe.cli prepare \
  --source /path/to/external_training_source.json --eval-manifest "$EVAL" \
  --output "$RUN/data.json"

# 2. GPU: HF Llama 하나만 로드한다. 각 명령 종료 시 모델이 해제된다.
"$PY" -B -m experiments.linear_probe.cli collect --data "$RUN/data.json" \
  --split train --model-path "$LLAMA" --output "$RUN/train_capture.json"
"$PY" -B -m experiments.linear_probe.cli collect --data "$RUN/data.json" \
  --split validation --model-path "$LLAMA" --output "$RUN/validation_capture.json"
```

3. 사용자가 별도 작업에서 기존과 같은 Llama vLLM 서버를 시작한다. 모델과 tokenizer revision을 위 revision에 고정하고 FP16, TP=1, max_model_len=8192, max_num_seqs=1, outlines backend를 사용한다.
server-only 제출 템플릿 `serve.sbatch`를 추가했다. 제출은 사용자가 직접 수행한다. prepare 후 runs 디렉터리가 존재하는 상태에서 다음과 같이 제출할 수 있다. 계정/partition/QoS는 기존 설정에서 가져왔으므로 실행 시 사용 가능한 값인지 확인한다.

```bash
sbatch --export=ALL,ROLE=teacher experiments/linear_probe/serve.sbatch
```

이 템플릿은 서버만 실행하고 Baseline 클라이언트를 실행하지 않는다. API 클라이언트는 loopback 서버가 실행되는 **동일 할당 노드**에서 실행해야 한다. 다른 노드의 localhost로 접속할 수는 없다.

```bash
# 새 서버가 저장한 실제 launch metadata 파일로 경로를 지정한다.
export TEACHER_CONFIG=experiments/linear_probe/runs/server_JOBID.json
# 기존 planner.json의 config를 읽기만 하는 방식도 지원한다.
# 해당 파일 자체가 현재 서버의 설정을 증명하지는 않는다.
export API=http://127.0.0.1:8000/v1

"$PY" -B -m experiments.linear_probe.cli label --data "$RUN/data.json" \
  --capture "$RUN/train_capture.json" --teacher-config "$TEACHER_CONFIG" \
  --api-base "$API" --output "$RUN/train_labeled.json"
"$PY" -B -m experiments.linear_probe.cli label --data "$RUN/data.json" \
  --capture "$RUN/validation_capture.json" --teacher-config "$TEACHER_CONFIG" \
  --api-base "$API" --output "$RUN/validation_labeled.json"

# 4. CPU: 실제 Probe 학습. 실험용 학습은 사용자가 이후 실행한다.
"$PY" -B -m experiments.linear_probe.cli train --data "$RUN/data.json" \
  --train "$RUN/train_labeled.json" --validation "$RUN/validation_labeled.json" \
  --output "$RUN/probe"
```

Teacher는 기존 prompt/schema/parser와 .1/.9/2048 생성 설정을 재사용한다. 잘못된 Credit을 균등 라벨로 바꾸지 않는다.
실패한 라벨이 있으면 학습을 중단한다. `--resume`으로 명시적 재시도할 수 있으며 이전 실패/비용을 보존한다. 기존 자동 recovery adapter는 이 경로에 적용하지 않는다.
OpenAI `/models`로 모델 이름은 확인하지만 원격 revision/launch 인자는 API로 검증할 수 없으므로 서버 로그와 설정을 사용자가 대조해야 한다.

```bash
# 5. GPU: 학습 후 vLLM 서버를 종료하고 HF 평가 Decomposition을 수집한다.
# N은 probe/metadata.json의 best_layer_idx 정수로 바꾼다.
"$PY" -B -m experiments.linear_probe.cli collect --data "$RUN/data.json" \
  --split evaluation --model-path "$LLAMA" --layers N \
  --capture-control --output "$RUN/evaluation_capture.json"
```

`--capture-control`은 문제마다 먼저 hidden 수집을 끈 대조 생성을 수행한다. 따라서 계측 실험의 생성 호출은 문제당 2회다.
수집 전용 운영 경로를 측정하려면 이 옵션을 빼서 **새 경로**에 수집한다. 기존 수집 결과의 config를 바꿔 resume하지 않는다.

6. HF 작업 종료 후 사용자가 Llama vLLM 서버를 다시 시작한다. 동일 steps에 Teacher Credit을 얻고 엔진 차이를 확인한다.
새 서버 job의 metadata 경로로 `TEACHER_CONFIG`를 갱신한다. train/validation/evaluation 사이에는 job ID가 아닌 모델·추론 관련 설정을 비교한다. 각 job의 전체 metadata는 별도로 보존한다.

```bash
"$PY" -B -m experiments.linear_probe.cli label --data "$RUN/data.json" \
  --capture "$RUN/evaluation_capture.json" --teacher-config "$TEACHER_CONFIG" \
  --api-base "$API" --output "$RUN/evaluation_labeled.json"
"$PY" -B -m experiments.linear_probe.cli compare-engine --data "$RUN/data.json" \
  --capture "$RUN/evaluation_capture.json" --server-config "$TEACHER_CONFIG" \
  --api-base "$API" --output "$RUN/engine_comparison.json"

# 7. CPU: 같은 분해에 대한 Credit/Budget 비교.
"$PY" -B -m experiments.linear_probe.cli evaluate --data "$RUN/data.json" \
  --capture "$RUN/evaluation_labeled.json" --probe "$RUN/probe" \
  --decay constant --output "$RUN/evaluation.json"

# Teacher 없는 Probe 전용 평가도 가능: --capture에 evaluation_capture.json 사용.
# 이때 paired Credit 지표는 null이며 Teacher Solver export는 불가하다.

# 8. CPU: 기존 Solver가 읽을 Planner 입력을 Teacher/Probe 각각 만든다.
"$PY" -B -m experiments.linear_probe.cli export --data "$RUN/data.json" \
  --evaluation "$RUN/evaluation.json" --eval-manifest "$EVAL" \
  --method teacher --output "$RUN/teacher_planner.json"
"$PY" -B -m experiments.linear_probe.cli export --data "$RUN/data.json" \
  --evaluation "$RUN/evaluation.json" --eval-manifest "$EVAL" \
  --method probe --output "$RUN/probe_planner.json"
```

9. 사용자가 Llama 서버를 종료하고 기존 설정의 QwQ-32B vLLM 서버를 시작한다. 모델 revision은 `976055f8c83f394f35dbd3ab09a285a984907bd0`, FP16, TP=4, context=12288이다.

```bash
sbatch --gres=gpu:normal:4 --export=ALL,ROLE=solver experiments/linear_probe/serve.sbatch
```

```bash
"$PY" -B -m experiments.linear_probe.cli solve --planner "$RUN/teacher_planner.json" \
  --eval-manifest "$EVAL" --api-base "$API" --decay constant --output "$RUN/teacher_solver.json"
"$PY" -B -m experiments.linear_probe.cli solve --planner "$RUN/probe_planner.json" \
  --eval-manifest "$EVAL" --api-base "$API" --decay constant --output "$RUN/probe_solver.json"
```

Solver 명령은 기존 `run_solver_compare`를 호출한다. `solve` 실행 환경에는 기존 Solver의 평가 의존성도 있어야 한다.
원래 8B sbatch의 사전 검사는 실제 Assessment 원문을 요구하므로 Probe 입력에는 사용하지 않는다.
새 진입점에서 전용 export를 검사한다. 공유 Solver의 launch hash만 실제 새 CLI hash로 메모리 안에서 바꿔 기록하고 호출 종료 후 복구한다. 파일은 수정하지 않는다.
각 실행은 기존 runner 동작대로 vanilla와 weighted_pab를 모두 실행하므로 두 Solver 작업의 vanilla 비용을 Probe 대체 효과에 합산하지 않는다.
Solver decay는 `evaluation.json`과 동일하게 유지한다.

`collect`, `label`, `compare-engine`, `solve`만 `--resume`을 지원한다. 소스/config/fingerprint가 달라지면 거부한다.
미완료 `.tmp`가 있으면 자동 폐기하지 않고 중단하므로 파일을 보존하고 상태를 확인해야 한다.
학습 artifact와 일회성 보고서는 덮어쓰지 않으며 실패 후 재실행에는 새 출력 경로를 사용한다.

## 비용과 결과 해석

| 기록 | 의미 |
|---|---|
| `usage.prompt_tokens/completion_tokens` | 실제 생성 입력/출력 수, EOS 포함 출력 토큰 |
| `metrics.forward_calls` | prefill + autoregressive decode forward 수. 생성 호출 수와 다름 |
| `extra_forward_passes` | feature 때문에 추가한 forward 수: 정상 수집 0 |
| `hidden_retention_seconds` | 토큰별 선택 레이어 tensor clone/보관 시간 |
| `alignment_seconds` | 생성 후 step/token 경계 매핑 CPU 시간 |
| `feature_transfer_seconds` | 경계 feature의 FP32 변환·GPU→CPU·리스트 변환 시간 |
| `generation_seconds` | forward + sampling + GPU 보관 + Python loop 및 계측 시간 |
| `generation_plus_feature_seconds` | 위 생성 + 매핑 + feature 전송. NPZ 저장/모델 로딩 제외 |
| `hidden_state_forward_overhead_seconds` | 단일 run으로 분리 불가하므로 null |
| `capture_control.capture_minus_control_seconds` | 같은 seed의 수집 off/on 생성 토큰이 동일할 때만 계산. 음수도 보존 |
| `probe.inference_seconds` | scaler + Ridge CPU predict 시간 |
| `probe.feature_read_seconds` | offline NPZ hash 검증·읽기·배열 준비; predict와 별도 |
| `probe.projection_seconds` | Credit 계약 변환 시간 |
| `teacher.usage` | 기존 Credit API 호출의 토큰·응답 시간 |
| `teacher_all_attempt_cost` | 실패·재시도를 포함한 Teacher 비용. 미확인 usage는 missing으로 남김 |
| `sessions` | 모델 로딩 시간, 실제 GPU/config/tokenizer 및 세션별 cold-start 조건 |
| `attempts` | 수집 실패/중단/재시도 이력과 저장 포함 wall time. 실패 비용 미상은 명시 |

Probe 운영 경로는 Decomposition 생성 1회 + Probe CPU 계산이며 Credit 생성 토큰은 0이다.
라벨링, 엔진 비교, 대조 생성은 실험을 위한 별도 비용이다. 이들의 호출을 운영 경로 1회에 숨기지 않는다.
Teacher/Probe 비용 비교는 `evaluation.json`의 상세 기록을 사용한다. 모델 로딩, 디스크 I/O, 워밍업을 포함할지 연구 보고에서 명시해야 한다.

기존 Solver와의 호환 export에서 `assessment`는 비용 계산용 슬롯이다. `credit_method=probe`이면 LLM 응답을 의미하지 않는다.
export에는 가짜 Teacher JSON을 넣지 않으며 `stages={}`이다. 따라서 기존 Solver의 `all_attempt_cost`는 Planner 비용을 집계하지 못한다.
실제 Planner 전체 시도 비용은 capture/label/evaluation 원장으로 집계해야 한다. Solver 성공 호출의 토큰/시간과 최종 정답 평가는 기존 runner가 그대로 기록한다.

`compare-engine`은 같은 입력 메시지를 vLLM Decomposition에 보내 text/steps 일치 여부, 입력/출력 토큰, 응답 시간을 기록한다.
HF↔vLLM 시간 차이는 hidden 수집뿐 아니라 엔진·sampling RNG·kernel·API transport 차이를 포함한다.
정확한 기존 vLLM 대비 총 개선은 이 별도 비교와 Solver 결과를 포함해 평가해야 하며 현재 구현만으로 속도 향상을 주장할 수 없다.

대조 생성은 항상 off→on 순서이고 워밍업을 자동 수행하지 않는다. 순서·cold start·GPU clock·동기화 영향이 있다.
논문용 시간 측정은 별도 출력 경로로 반복하고 cold 요청을 따로 집계해야 한다. 매 GPU 구간 synchronize하는 계측 자체도 오버헤드가 있다.
선택 레이어만 수집하는 최종 경로와 모든 레이어를 수집한 학습 경로의 비용을 구분한다.

## GPU/CPU 자원: 실측 전 예상

- HF 수집: FP16 8B weights 약 16GB + KV cache + prefill hidden tuple/logits + 임시 token feature. GPU 1개를 사용한다. 짧은 입력에서 24GB가 가능할 수 있으나 **수용 여부 미검증**이며 eager attention의 긴 prefill에는 40~48GB 단일 GPU를 권장한다. CPU RAM 48GB 이상을 권장한다.
- 모든 레이어 token feature 보관은 최대 1023×33×4096×2 bytes ≈264MiB이다. 실제 prefill/attention 메모리는 별도이며 더 클 수 있다. 저장 feature는 문제당 2~5×33×4096×4 bytes, 압축 전 최대 약 2.58MiB다.
- Teacher/엔진 비교: 기존 Llama vLLM 설정의 단일 24GB GPU 후보. 기존 설정을 사용하되 실제 prompt 길이별 fit은 검증해야 한다.
- Ridge 학습·예측: GPU 불필요. 레이어별 X와 Ridge 계산을 위해 CPU RAM 16~32GB 이상에서 시작하되 후보 문제 수에 따라 산정한다. CPU 예측은 BLAS thread 수에 영향을 받으므로 환경변수를 고정한다.
- Solver: 기존 QwQ-32B FP16 TP=4, 예를 들어 24GB GPU 4개. 기존 설정/환경을 유지한다.
- 단계별 순차 실행이므로 HF 모델과 Teacher 모델을 동시에 올릴 필요는 없다. 서버 시작·종료와 SLURM 제출은 사용자가 수행한다.

## CPU 검증

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data/dada0228/conda_envs/shortcut-xai/bin/python -B -m unittest -v experiments.linear_probe.test_probe
```

테스트는 작은 CPU 가짜 causal model과 합성 feature/label만 사용한다. 실제 Llama/QwQ 로딩, 외부 데이터 다운로드, 연구 데이터 학습은 하지 않는다.
Credit 계약/극단값, 기존 프롬프트 일치, split 누수/변조, EOS/토큰 위치, 대조 호출 분리, scaler의 train-only fit, NPZ/artifact 무결성, checkpoint 보호, 기존 Solver 입력 연결을 확인한다.

남은 검증은 실제 GPU의 HF 버전/KV-cache 호환성·메모리 fit, 서버의 outlines/JSON 동작, 실제 데이터 품질과 의미 중복, 레이어 선택·예측 성능, QwQ 정답률 및 반복 시간 측정이다.
joblib artifact는 신뢰할 수 있는 자체 생성 파일만 로드한다. metadata hash는 변조 탐지용이지 외부 파일의 안전성을 보증하지 않는다.
