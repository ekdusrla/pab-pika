# 독립 500문제 Solver 검증 준비

**현재 상태는 데이터 승인 대기다.** 기존 MATH-500의 문제 500개를 SHA-256으로 고정했지만, 프로젝트에서 한 번도 평가하지 않은 문제는 **0개**였다. 기존 독립 검증 계획의 미사용 조건을 적용해 실행용 승인을 만들지 않았다. 대체 데이터는 추가하지 않았다. [후보 Manifest](candidate_manifest.json), [SHA-256](candidate_manifest.sha256)

## 데이터 독립성

| 검사 | 결과 |
|---|---|
| 기존 MATH-500 | 500개, 고유 ID와 정규화 해시 각각 500개 |
| 난이도 1/2/3/4/5 | 43 / 90 / 105 / 128 / 134 |
| 기존 Frozen Pilot 5문제와 교차 중복 | 0 |
| Probe 학습 24문제와 교차 중복 | 0 |
| 층·alpha 선택에 사용된 검증 6문제와 교차 중복 | 0 |
| 현 Pilot Planner 예시 5문제와 교차 중복 | 0 |
| MATH-500 내부 정규화·유사 중복 | 0 |
| 기존 프로젝트 평가 결과 | Llama-3.1-8B 실험 500개, Original 실험 500개, 기존 Ablation 100개 |

정규화 해시(NFKC/casefold/공백 제거), LaTeX relax 일치와 문자 5-gram Jaccard ≥0.90를 사용했다. Probe 학습/검증 라벨의 실제 fingerprint와 artifact의 학습/검증 문제 해시도 교차 확인했다. MATH-500과 학습 24문제의 최대 Jaccard는 0.373, MATH-500 내부 최고는 0.705였다. 의미적으로 같은 문제나 기반 모델 사전학습 겹침까지 판별할 수는 없다.

과거 Planner 두 실행은 MATH-500 내부의 다섯 문제(472, 36, 2, 272, 251)를 Credit prompt 예시로도 사용했다. 새 설정은 Frozen Pilot의 외부 예시 5개를 그대로 사용한다. 현재 Ridge 훈련·선택과 겹친 흔적은 없지만, MATH-500 전체를 이 프로젝트에서 미사용인 confirmatory holdout으로 지정할 수 없다. 사용자가 독립성 범위를 명시적으로 **Probe fit/selection 및 Frozen Pilot에 대한 독립성**으로 정하면, 이전 프로젝트 노출을 밝혀 같은 500개를 기존 벤치마크 재평가로 사용할 수 있다. 엄격한 프로젝트 미사용 500개를 유지하려면 사용자 지정 단일 출처가 필요하다. 둘 모두 확정하기 전 후보의 `approved=false` 상태를 유지한다.

원본 정답 필드는 500개 모두 존재한다. Frozen evaluator에 정답 문자열을 boxed 응답으로 되넣은 간이 회귀에서 493개를 맞혔고, 6개가 `incorrect`, 1개가 `undetermined`였다. 해당 ID는 102, 97, 96, 257, 284, 422, 467이다. 이 점검은 표현식, ±, 단위, 텍스트 정답의 채점 취약성을 드러낸다. 재평가를 승인할 경우 baseline 호환 점수를 유지하면서 7건의 원인 및 보조 판정 방식을 결과를 보기 전에 고정해야 한다. 원본 정답·평가기 설정은 변경하지 않았다.

## 준비한 실행 경로와 재사용

- 기존 HF decomposition·hidden 추출 및 메모리 guard, 18번 층의 마지막 fully-contained token feature, 기존 scaler+Ridge(`alpha=100`), 정수 Credit projection과 기존 `allocate_words(constant)`, Teacher guided JSON payload, QwQ prompt와 원본 정답 평가기를 재사용한다. CPU에서 frozen Probe를 로드해 기존 5개 예측과 25개 Solver payload가 동일하게 생성되는 것을 확인했다.
- 모델 revision·snapshot·tokenizer/config SHA-256, weight index와 shard 식별자·크기, 원본 데이터 및 중복 검사의 입력 SHA-256을 후보 Manifest에 담았다. weight shard 전체를 다시 해시하는 대신 pinned HF snapshot/가중치 인덱스와 blob identity를 대조했다. Protocol JSON은 기본 Solver seed 42, 온도 0.1, top-p 0.9, TP4 FP16, 4 GPU, 1 동시 요청을 기록한다. 추후 seed 추가는 새 승인 버전에서 고정하고 요청 ID에 포함한다.
- 순서는 기존 MATH-500 Manifest의 원본 순서다. 50문제씩 10 chunk, 문제별 arm 순환, 고유 journal ID(Manifest·Protocol·문제·chunk·arm·seed)로 저장한다. 네 PAB 군은 같은 capture 분해와 같은 총 word Budget을 사용한다. no-hidden control은 비교 측정용이다. control 결과가 다르면 해당 no-hidden 온라인 합산을 비워 놓는다. y3 Teacher는 같은 Llama checkpoint와 TP4를 RTX 3090에서 실행하게 되어 기존 A2000과 물리 환경 차이가 남는다.
- 원본 API 응답·시도·누락 비용을 독립 JSON으로 먼저 저장하며, 성공 요청은 Resume에서 건너뛴다. 기록된 raw 응답의 검증만 미완료면 채점부터 이어간다. 마지막 in-flight 시도는 중단·비용 미상으로 보존하고 재시도한다. 응답을 받았더라도 안전 저장 전에 프로세스가 강제 종료되면 서버측 생성 완료를 증명할 수 없어 재실행될 수 있다. Writer/batch lock 및 원자적 commit을 사용하고 TERM을 자식 단계로 전달한다. 분해/Teacher 실패 시 의존 요청을 `blocked`로 보존하며 분모에 남긴다. 결과를 바꾸는 자동 분해 수리는 사용하지 않는다.
- 분석은 완료된 chunk의 합과 독립 500문제 분모를 요구하도록 준비했다. 정확도 차이와 보수적 paired 95% 구간, 실패/절단, 토큰 차이의 평균·중앙값·분위수·승패·한 문제 제외 민감도, Credit/Budget의 난이도별 요약, staged 비용을 JSON/CSV로 내보낸다. 불완전한 비용은 알려진 수와 누락된 수를 구분한다. `length`의 전체 word 준수와 구간을 식별할 수 없는 subtask 준수는 미판정이다. [실험 설정](../../solver_y3_500_v1/protocol.json), [실행 코드](../../solver_y3_500_v1/runner.py), [분석 코드](../../solver_y3_500_v1/analysis.py)

## 가설과 측정 범위

H1은 Ridge−Teacher의 정확도 차이와, Teacher 두 번째 호출을 포함한 전체 입·출력/총토큰 및 비용의 paired 차이를 본다. 이전 계획의 **−3%p는 제안된 허용 하락 폭**이다. 승인된 Margin이 없어 설정은 null이며 비열등성 자동 판정을 하지 않는다. 500개에서는 discordance에 따라 이전 750개 설계보다 검정력이 낮을 수 있다.

H2는 Ridge−Uniform 및 Ridge−Position을 각각 비교한다. 세 군이 모두 Teacher 호출을 생략하므로 Planner 호출 수 감소 자체를 Ridge 고유의 이점으로 계산하지 않는다. 정확도와 시스템/Solver 효율성, capture에 추가로 드는 구간을 각각 판단한다. 세부 지표는 보조 비교로 해석하고 단일 Solver seed 결과를 decoding 변동성 전체로 일반화하지 않는다.

공통 분해와 순차 모델 교체를 사용하는 현재 단일 노드 구성에서는 stage wall/API를 저장할 수 있다. **실제 per-request end-to-end latency는 계측하지 않으며 `null`**로 남긴다. 구성요소 합은 staged 재구성이다. 실제 운영 latency 검증에는 Planner·Teacher·Solver를 운영할 토폴로지와 요청 수신부터 반환까지 연결되는 trace가 추가로 필요하다. 4×3090에서 Teacher와 QwQ TP4를 동시에 올릴 수 있는지는 검증되지 않았다. 현재 서버는 `max_num_seqs=1`이고 병렬 요청을 사용하지 않는다.

## CPU 검증과 제출

CPU 테스트는 26건(신규 준비 6건과 기존 GPU identity/mapping 모의 테스트 20건)을 통과했다. 후보 SHA, tokenizer 실제 로드, 패키지 및 평가기 동일성, frozen Probe 추론, payload/예산, 중단 후 resume, 재요청 방지, JSON/CSV와 집계, 셸 문법을 확인했다. GPU 실험·SLURM 제출·기존 파일 수정·재학습은 수행하지 않았다. [CPU 검증 로그](cpu_validation.log)

y3에 **GPU 4개, CPU 16개, RAM 64 GiB**, `6-00:00:00`을 요청한다. 예측은 **46.77 wall h**, 4개 예약 기준 **187.1 GPU·h**다. y3 할당 후 실제 GPU와 CUDA/TP4 동작, VRAM, Smoke qualification은 미검증 상태다. [자원 보고서](RESOURCES.ko.md)

아래는 **독립성 범위·데이터·채점·Margin 결정과 GPU qualification을 마친 후** 사용할 명령이다. 승인 Manifest는 아직 존재하지 않는다. 승인본에는 승인 입력 SHA, Protocol SHA 및 코드 SHA를 기록해야 하며 후보 파일을 인자로 넘기면 preflight에서 중단된다.

```bash
cd [local-path]
scontrol show node moana-y3
squeue -w moana-y3
sbatch experiments/linear_probe/solver_y3_500_v1/gpu.sbatch \
  experiments/linear_probe/runs/solver_y3_500_v1/approved_manifest.json
```

인자를 생략한 chunk 범위는 1–10이다. 제한이나 중단 이후에는 **같은 승인 Manifest** 뒤에 필요한 번호를 붙여 다시 제출한다. 예를 들어 `.../approved_manifest.json 7 8 9 10`은 7–10번 chunk를 대상으로 한다. 단순 완료 여부가 불확실한 경우 동일한 전체 명령으로 기존 성공 요청을 건너뛸 수 있다. [SLURM 스크립트](../../solver_y3_500_v1/gpu.sbatch), [CPU 사전 검증](../../solver_y3_500_v1/cpu_preflight.sh)

완료 후 프로젝트 루트에서 집계한다. 결과의 분모가 누락되지 않도록 실행 전혀 안 된 요청은 오류로 처리하고, 기록된 최종 실패·의존성 차단은 정답 분모에 포함한다. 성공 요청은 raw 응답에서 기존 evaluator 결과를 다시 생성해 journal과 대조한다.

```bash
CUDA_VISIBLE_DEVICES='' [local-path] -B \
  -m experiments.linear_probe.solver_y3_500_v1.analysis \
  --manifest experiments/linear_probe/runs/solver_y3_500_v1/approved_manifest.json
```
