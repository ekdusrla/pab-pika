<div align="center">

# 🧠 Plan-and-Budget

## Effective and Efficient Test-Time Scaling on Large Language Model Reasoning

### Accepted at ICLR 2026 🎉

[![ICLR](https://img.shields.io/badge/ICLR%20-2026-blue)](https://openreview.net/forum?id=ctspw4CqbS)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2505.16122&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2505.16122)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<strong>Junhong Lin\*¹</strong>, <strong>Xinyue Zeng\*²</strong>, <strong>Jie Zhu²</strong>, <strong>Song Wang²</strong>, <strong>Julian Shun¹</strong>, <strong>Jun Wu⁴</strong>, <strong>Dawei Zhou²</strong><br> ¹ MIT CSAIL, ² Virginia Tech, ³ University of Virginia, ⁴ Michigan State University

(\*Equal Contribution)

</div>


<p align="center">
<img src="imgs/overview.png" width="100%" height="auto"/>
</p>

## PAB-Pika 팀 실험

이 저장소는 [원본 Plan-and-Budget](https://github.com/junhongmit/P-and-B)의 추론 및 평가 코드를 기반으로, MATH-500에서 토큰 예산 배분과 linear probe가 정확도·토큰 사용량·지연 시간에 미치는 영향을 비교합니다. 원본 연구의 저자, 논문 인용과 MIT 라이선스는 아래에 보존했습니다.

| 경로 | 내용 |
|---|---|
| `inference/`, `utils/local_llm.py` | PAB 예산 상한·확장·압축·검증·단계별 배분과 로컬 Transformers 추론 |
| `scripts/` | Qwen 및 Llama의 소규모 비교, MATH-500 비교, Slurm 환경 점검 |
| [`probe_pipeline/`](probe_pipeline/probe_pipeline/README.md) | Llama의 하위 문제 경계 hidden state를 저장하고 Ridge probe를 학습하는 초기 파이프라인 |
| [`experiments/linear_probe/`](experiments/linear_probe/README.md) | 데이터 분할, hidden state 수집, teacher 라벨, probe 학습·검증, solver 비교 |
| `experiments/llama_qwq/` | Llama planner와 QwQ solver의 분리 실행, 재개와 최종 분석, [8B ablation pilot](experiments/llama_qwq/ablation_pilot_v1/README.md) |
| [`experiments/probe_budget/`](experiments/probe_budget/README.md) | PaB call 2(난이도·credit)를 hidden-state probe로 대체: probe 학습, MATH-500·TravelPlanner planning, QwQ-32B 추론·평가 |

### 팀 환경 설정과 실행

저장소 루트에서 실행합니다. 기본 추론은 Python 3.12, linear probe 검증은 Python 3.11 환경을 사용했습니다. vLLM을 쓸 경우 모델 서버 환경과 probe 학습 환경을 분리합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install datasets numpy openai python-dotenv regex sympy
cp -n .env_template .env
mkdir -p results
```

로컬 `.env`의 `API_BASE`, `API_KEY`, `MODEL_NAME`을 연결할 서버에 맞게 설정합니다. 로컬 Transformers 추론은 `torch`와 `transformers`를 설치하고 `LOCAL_LLM=1`, `LOCAL_MODEL_ID`, `LOCAL_MAX_NEW_TOKENS`를 설정합니다. 공통 모듈이 로드할 때 API 클라이언트도 생성하므로 로컬 실행 시에도 `API_KEY=dummy`를 설정합니다. 모델에 따라 Hugging Face 접근 권한과 GPU 메모리가 필요합니다.

공유 스냅샷의 데이터는 별도로 준비합니다. 원본 저장소의 `2ca2d90`에서 데이터 절에 나온 디렉터리의 JSON을 같은 위치에 복원합니다. MATH-500 분해 JSON은 `MATH_DECOMPOSED_PATH`로도 지정할 수 있습니다. 일부 소규모 비교 스크립트는 `dataset/MATH-500/decomposed_questions_10.json`을 사용하므로 실행 전 필요한 샘플 파일을 확인합니다. `run.run_inf`는 시작 시 ROUGE 모듈을 로드하므로 Hugging Face 접속 또는 준비된 캐시가 필요합니다.

```bash
python -m run.run_inf --dataset math --model planned_local_weighted \
  --decay polynomial --num-workers 1 --queue-size 1 --postfix team
python experiments/llama_qwq/experiment_state.py \
  --dataset dataset/MATH-500/decomposed_questions.json --output experiments/llama_qwq/math500_manifest.json
python experiments/llama_qwq/run_planner.py --help
python experiments/llama_qwq/run_solver_compare.py --help
```

별도의 probe 환경에서 다음 명령으로 인터페이스와 CPU 테스트를 확인할 수 있습니다. 통합 테스트 한 건은 위 명령으로 생성하는 `experiments/llama_qwq/math500_manifest.json`을 사용합니다.

```bash
python -m pip install -r experiments/linear_probe/requirements.txt
python -m experiments.linear_probe.cli --help
python -m unittest experiments.linear_probe.test_probe experiments.linear_probe.test_safety
```

초기 `probe_pipeline`은 추가로 `datasets`가 필요합니다. `probe_pipeline/probe_pipeline/`에서 `LLAMA_PATH`를 사용 가능한 모델 ID 또는 로컬 경로로 지정하고 해당 디렉터리의 README 순서대로 실행합니다. Slurm 파일과 일부 Python 실행기·프로토콜에는 실행 당시 클러스터 경로와 자원 조건이 들어 있습니다. 실행 전 프로젝트·Conda·캐시 경로와 파티션, 노드, GPU, 서버 포트를 현재 환경에 맞게 확인합니다. 일부 프로토콜은 원본 체크포인트와 해시를 검증하므로 기존 설정을 바꿀 경우 새 실행용 프로토콜과 산출물 경로를 사용합니다.

### 결과 파일

- [`shared_reports/llama_qwq/`](shared_reports/llama_qwq)에서 MATH-500 최종 분석 Markdown, 선별된 JSON 지표, 문항별 CSV와 8B ablation 지표를 볼 수 있습니다.
- [`shared_reports/linear_probe/`](shared_reports/linear_probe)에서 credit TP4, solver Y1/Y3, teacher 안정성 분석과 선별된 지표를 볼 수 있습니다.
- [`shared_reports/budget_comparisons/`](shared_reports/budget_comparisons)에 배분 방식별 비교 요약 JSON을 모았습니다.

공유 사본은 문항 본문·정답·예측 텍스트의 CSV 열과 일부 내부 실행 메타데이터를 제외합니다. Markdown의 로컬 경로는 가렸으며, 보고서가 언급하는 원본 로그·체크포인트·아티팩트는 해당 실행 환경에서 관리합니다. 신규 원시 출력은 `results/`와 각 실험의 `runs/` 아래에 생성됩니다.

# 📌 Overview

**Plan-and-Budget** is a training-free test-time reasoning framework
that improves both **reasoning accuracy** and **efficiency** in large
language models (LLMs).

Modern reasoning LLMs often suffer from:

-   🔄 **Overthinking** --- excessive and redundant reasoning on simple
    queries\
-   ⚡ **Underthinking** --- premature termination on complex tasks

We introduce:

-   **BAM (Budget Allocation Model)** --- a theoretical framework for
    adaptive token allocation\
-   **Plan-and-Budget** --- a practical inference-time implementation
    using structured decomposition and local budget scheduling\
-   **E³ (Efficiency-aware Effectiveness Evaluation)** --- a principled
    metric balancing accuracy and compute

This repository contains the full inference and evaluation pipeline to
reproduce our results.

------------------------------------------------------------------------

# ⚙️ Environment Setup

``` bash
conda create -n plan_budget python=3.12 -y
conda activate plan_budget
pip install vllm
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔐 Environment Variables

``` bash
cp .env_template .env
```

Edit `.env`:

-   `API_BASE` (e.g., `http://localhost:7878/v1`)
-   `API_KEY` (use `"DUMMY"` for local vLLM)

You can specify a different config at runtime:

``` bash
ENV_FILE=path/to/your/.env python ...
```

------------------------------------------------------------------------

# 📂 Datasets

Download TravelPlanner database:

https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link

Unzip into:

dataset/TravelPlanner/

Pre-decomposed datasets:

-   dataset/MATH-500\
-   dataset/NaturalInstruction-Sampled-500\
-   dataset/TravelPlanner

To re-run decomposition:

``` bash
python -m dataset.break_down_question \
  --num-workers 32 \
  --queue-size 32 \
  --dataset DATASET_NAME
```

DATASET_NAME ∈ {math, instruction, travelplanner}

------------------------------------------------------------------------

# 🚀 Reproducing Experimental Results

Example (MATH-500):

``` bash
python -m run.run_inf --num-workers 32 --dataset math --model vanilla
python -m run.run_inf --num-workers 32 --dataset math --model planned
python -m run.run_inf --num-workers 32 --dataset math --model global_budget
python -m run.run_inf --num-workers 32 --dataset math --model planned_global
```

Plan-and-Budget (Local Allocation):

``` bash
python -m run.run_inf --dataset math --model planned_local_uniform

python -m run.run_inf --dataset math \
  --model planned_local_weighted \
  --decay polynomial --postfix polynomial
```

`--postfix` only affects log naming.

------------------------------------------------------------------------

# 📊 Evaluation

For MATH-500 and NaturalInstructions: - Results computed automatically.

For TravelPlanner: - Requires structured JSON evaluation via a secondary
LLM.

``` bash
ENV_FILE=.env.eval python -m run.run_eval \
  --dataset travelplanner \
  --model MODEL_NAME \
  --postfix POSTFIX
```

Ensure `.env.eval` specifies a model supporting JSON output.

------------------------------------------------------------------------

# 📚 Citation

``` bibtex
@inproceedings{lin2026plan,
  title={Plan-and-Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning},
  author={Lin, Junhong and Zeng, Xinyue and Zhu, Jie and Wang, Song and Shun, Julian and Wu, Jun and Zhou, Dawei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

------------------------------------------------------------------------

⭐ If you find this repository useful, please consider starring it!
