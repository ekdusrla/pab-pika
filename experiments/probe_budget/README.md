# Probe 기반 토큰 예산 분배 (probe_budget)

Plan-and-Budget(PaB)의 planning은 Llama-3.1-8B를 두 번 호출합니다.
1. **call 1**: 문제를 하위 문제로 나누고 각 하위 문제에 hint를 붙입니다(분해).
2. **call 2**: 각 하위 문제의 난이도(level)와 credit을 매깁니다.

이 실험은 **call 2를 hidden-state probe로 대체**합니다. call 1 동안 각 하위 문제 끝 위치의 hidden state를 저장하고, 작은 Ridge 회귀로 난이도를 예측합니다. 그러면 planning에 필요한 Llama 호출이 한 번으로 줄어듭니다. 추론 모델은 QwQ-32B이고, 모든 실행은 같은 GPU(RTX A5000)에서 했습니다.

| 방법 | planning | 예산 분배 |
|---|---|---|
| vanilla | 없음 | 없음 (QwQ가 바로 풂) |
| PaB | Llama call 1 + call 2 | call 2 credit × 다항 감쇠 |
| probe | Llama call 1 + probe | probe 예측 난이도 × 다항 감쇠 |
| joint (ablation) | 분해와 level/credit을 한 번의 Llama 호출로 | 그 credit × 다항 감쇠 |

세 방법 모두 예산 분배 코드는 공용 `inference/_5_planned_local_weighted_model.py`(`--decay polynomial`)를 그대로 씁니다. 방법 간 차이는 plan json에 들어가는 credit 값뿐입니다.

## 파이프라인

모든 작업은 **저장소 루트에서** 제출합니다. 클러스터에 맞는 설정(conda env 이름, 모델 경로, TMPDIR)은 `slurm/env.sh`에 모여 있고, `export`로 덮어쓸 수 있습니다.

```bash
mkdir -p logs
S=experiments/probe_budget/slurm
sbatch $S/prepare_math_train.sbatch            # 1. MATH train 2000문제 샘플 (커밋돼 있으므로 생략 가능)
sbatch $S/gen_train_data.sbatch                # 2. Llama call1/call2 + hidden state -> data_v2/gen/ (8 shard)
sbatch $S/train_probe.sbatch                   # 3. probe 학습·교차검증 -> probe_artifacts/, data_v2/probe_report/
sbatch --export=ALL,STAGE=v2 $S/plan.sbatch    # 4. MATH-500 planning (PaB + probe)
sbatch --export=ALL,STAGE=v2 $S/plan_merge.sbatch
sbatch --export=ALL,STAGE=v3,DATASET=math500 $S/plan.sbatch   #    joint (MATH-500)
sbatch --export=ALL,STAGE=v3,DATASET=travel  $S/plan.sbatch   #    TravelPlanner: PaB + probe + joint
#    (각각 끝나면 같은 STAGE/DATASET으로 plan_merge.sbatch)
sbatch --export=ALL,DATASET=math,RUN=probe $S/reason_qwq.sbatch   # 5. QwQ 추론 (RUN=vanilla|pab|probe|joint)
```

- 1~4단계를 건너뛰려면 `PLAN_DIR=experiments/probe_budget/plans/math500`으로 커밋된 MATH-500 plan을 쓸 수 있습니다. probe도 `probe_artifacts/`에 커밋돼 있습니다.
- 5단계 결과는 `results/`에 생깁니다. MATH-500 요약표는 `python experiments/probe_budget/analyze_v2.py`로 만듭니다(compute node에서 실행).
- TravelPlanner 평가에는 공식 sandbox DB가 `dataset/TravelPlanner/database/`에 있어야 합니다(gitignore 대상이라 따로 받습니다).
- 환경: planning/probe는 torch + transformers + scikit-learn + datasets env, QwQ 서버는 vLLM 0.5.4 env를 따로 썼습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `prepare_math_train.py` | MATH train에서 MATH-500과 겹치는 문제를 빼고 level별 400문제 샘플 (seed 42) |
| `hidden_state_capture.py`, `hidden_state_capture_v2.py` | `<<<STEP>>>` marker 분해 프롬프트(v1 파일에서 가져옴), marker 토큰 위치의 hidden state 저장 |
| `generate_training_data_v2.py` | 학습 데이터 생성: call 1 hidden state + 공식 PaB call 2의 level 라벨 |
| `train_probe_mlp.py` | Ridge/MLP × 특징 F1~F4, 문제 단위 GroupKFold-5, 내부 분할로 layer·α 선택 |
| `probe_inference_v2.py` | 저장된 probe로 단계별 난이도 예측 |
| `plan_math500.py` | MATH-500 planning: call 1 → probe, call 2 (PaB) → `plan_v2_{pab,probe}.json` |
| `plan_v3.py` | joint 단일 호출(MATH-500), TravelPlanner의 PaB/probe/joint planning |
| `analyze_v2.py` | 정확도, 입출력 토큰, FLOPs 추정, 지연 시간 요약 |
| `smoke_report.py`, `smoke_v3_cpu.py` | 소규모 점검 |

## Probe

- **학습 데이터**: MATH train 1988문제, 7791개 단계. 라벨은 공식 PaB call 2가 매긴 단계별 level(1~5)입니다.
- **채택한 설정**: Ridge, 특징 F3 = [hᵏ ; h_end], layer 12, α = 1e4. 예측값은 0.1 아래로 내려가지 않게 자릅니다.
- **교차검증 (문제 단위 5-fold, out-of-fold)**:

| 방법 | R² | 문제 내 Spearman | 가장 어려운 단계 일치 |
|---|---|---|---|
| 위치 평균 (baseline) | 0.218 | 0.508 | 0.586 |
| Ridge F3 (채택) | 0.559 | 0.680 | 0.782 |

전체 비교(MLP, F1~F4)는 `data_v2/probe_report/report.md`에 있습니다.

## MATH-500 결과 (QwQ-32B, 문제당 평균)

| | vanilla | PaB | probe |
|---|---|---|---|
| 정확도 (%) | 86.0 | 87.8 | 87.8 |
| planning 입력 / 출력 토큰 | – | 2124 / 504 | 729 / 254 |
| 추론 입력 / 출력 토큰 | 198 / 3389 | 562 / 2302 | 563 / 2376 |
| planning TFLOPs | – | 39.9 | 14.9 |
| 전체 TFLOPs | 235.8 | 227.8 | 207.7 |
| planning 시간 (s) | – | 15.9 | 7.8 |
| 전체 시간 (s) | 377.3 | 279.2 | 288.8 |

- PaB와 probe의 정확도는 같습니다. 한쪽만 맞힌 문제가 14:14로 갈립니다.
- probe는 planning 토큰과 시간을 약 절반으로 줄였습니다. 다만 추론 시간이 조금 늘어 전체 시간은 PaB와 비슷합니다.
- FLOPs는 2NT + L·d_attn·T²로 추정한 값입니다(측정값이 아닙니다).
- joint와 TravelPlanner 결과는 실행 중이며, 나오면 추가합니다.

## 주의

- planning 시간은 shard마다 GPU 1장으로 문제를 하나씩 순차 처리해서 잰 값입니다. 추론 시간은 vLLM에 동시 요청 32개를 보낸 상태에서 잰 요청별 시간입니다.
- TravelPlanner의 `run_inf` 결과 헤더 score는 자리표시값(-1)입니다. 실제 점수는 `run_eval`이 만드는 `*_posteval.json`에 있습니다.
  - `pab_score`: 저장소의 `eval_score`
  - `final_pass`: 표준 TravelPlanner 기준
- TravelPlanner에서 plan을 JSON으로 바꾸는 parser는 원 논문의 GPT-4가 아니라 QwQ입니다.
- vLLM 0.5.4 TP=4에서 엔진이 멈추는 경우가 한 번 있었습니다. `reason_qwq.sbatch`의 watchdog이 이를 감지하면 job을 실패로 끝냅니다. 같은 명령으로 다시 제출하면 끝난 문제는 건너뜁니다.
