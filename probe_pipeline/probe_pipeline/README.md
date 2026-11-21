# Probe 학습 파이프라인 — 파일별 역할

## 파일 목록과 역할

| 파일 | 역할 | 모델을 직접 로드하는가? |
|---|---|---|
| `hidden_state_capture.py` | **핵심 라이브러리.** 마커(`<<<STEP>>>`) 기반 수동 디코딩 루프. sub-question 경계마다 전체 레이어의 hidden state를 "그 자리에서" 캡처. | **아니요** — model/tok을 인자로 받기만 함 |
| `generate_training_data.py` | 모델을 로드하고, MATH-500 500문제에 대해 STEP1(분해+hidden state)과 STEP2(credit 평가, 라벨 생성용)를 돌려서 `probe_training_data.pt`를 만듦 | 예 |
| `train_probe.py` | `probe_training_data.pt`를 읽어서 레이어별 교차검증으로 가장 좋은 레이어를 찾고, 그 레이어의 hidden state → credit 회귀 모델을 학습해 `probe_artifacts/`에 저장 | 아니요 (학습만, GPU 거의 안 씀) |
| `probe_inference_demo.py` | 학습된 probe를 가지고 **새 문제 하나**를 처리하는 전체 흐름 예시. STEP1 호출 1번 + probe 추론만 있고 STEP2가 없음 | 예 |

## 왜 `hidden_state_capture.py`는 혼자 못 돌아가는가

이 파일은 "어떻게 생성하면서 hidden state를 뽑는가"라는 **로직만** 담고 있습니다. 실제로 모델을 메모리에 올리는 건 그걸 가져다 쓰는 쪽(`generate_training_data.py` 또는 `probe_inference_demo.py`)의 책임입니다. 이렇게 나눠둔 이유는, **학습 데이터를 만들 때와 실제 추론할 때가 "모델을 로드하고 몇 번 반복하는지"는 다르지만, "sub-question 경계에서 hidden state를 어떻게 뽑는지"는 완전히 같아야** 하기 때문입니다. 로직을 한 곳에만 두면 둘이 서로 다르게 동작할 위험이 없어집니다.

## 두 단계의 LLM 호출 횟수 비교

```
학습 데이터 생성 (generate_training_data.py) — 딱 한 번만 하는 일회성 작업
  STEP 1: LLaMA 호출 1번 (분해 + hidden state 동시 획득, 추가 호출 없음)
  STEP 2: LLaMA 호출 1번 (credit 평가 — 이게 probe가 배울 정답표가 됨)
  → 문제당 LLaMA 호출 2번 (원본 PAB와 동일. 라벨을 만들려면 불가피함)

실제 추론 (probe_inference_demo.py) — probe 학습 완료 후, 매번 반복되는 구조
  STEP 1: LLaMA 호출 1번 (분해 + hidden state 동시 획득)
  Probe:  행렬곱 한 번 (생성 없음, GPU라 하더라도 사실상 무비용)
  → 문제당 LLaMA 호출 1번
```

## 실행 순서

```bash
python generate_training_data.py   # probe_training_data.pt 생성
python train_probe.py              # probe_artifacts/best_probe.joblib 생성
python probe_inference_demo.py     # 학습된 probe로 새 문제 1개 테스트
```
