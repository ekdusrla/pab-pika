# 구현 및 CPU 검증 기록

검증일: 2026-11-13. 신규 파일은 모두 `experiments/linear_probe/` 아래에 추가했다.
기존 PAB 소스, 기존 결과, 공유받은 Probe 파일은 수정하지 않았다.

## 수행한 검증

- `test_probe.py`의 CPU unittest 17개 통과.
- 작은 가짜 CPU causal model로 EOS 처리, 단계 경계의 처리 후 hidden state, 추가 feature forward 0회, 대조 생성 별도 계측을 검증했다.
- 작은 합성 데이터의 Ridge fit으로 scaler train-only 학습과 레이어 선택, 정수 Credit 계약을 검증했다. 실제 연구 데이터 학습은 하지 않았다.
- 500개 **합성 feature**를 기존 평가 manifest의 ID에 대응시켜 Teacher/Probe 평가·export·기존 Solver의 `planner_rows/attach_plans` 호환성을 검증했다. 실제 문제를 모델에 입력하거나 답변을 생성하지 않았다.
- 누수/중복/변조/파일 무결성/출력 경로 및 symlink 이탈/재개 config 불일치를 거부하는 테스트를 수행했다.
- Python 모듈 6개와 `serve.sbatch`의 내장 Python에 AST/compile 정적 검사 통과.
- `bash -n experiments/linear_probe/serve.sbatch` 통과. 스크립트 실행/제출은 하지 않았다.
- CLI 8개 하위 명령의 `--help` 통과. CLI import가 torch/transformers를 로드하지 않는 것도 별도 프로세스로 확인했다.
- 작업 전에 기록한 기존 `.py/.json/.sbatch/.md` 229개 파일의 SHA-256 재비교: 변경 0개. Git에 이미 존재하던 사용자 변경은 그대로 보존했다.

테스트 환경은 `/data/dada0228/conda_envs/shortcut-xai/bin/python`이다.
`CUDA_VISIBLE_DEVICES=''`, offline 환경, BLAS/OMP thread=1, bytecode 쓰기 비활성 상태에서 실행했다.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data/dada0228/conda_envs/shortcut-xai/bin/python -B -m unittest -v experiments.linear_probe.test_probe
```

## 실행하지 않은 작업과 남은 검증

실제 Llama/QwQ 가중치 로딩, GPU 수집·학습·추론, 실제 데이터 Probe 학습, API 모델 호출, 데이터 다운로드, 서버 시작, SLURM 제출은 수행하지 않았다.

외부 학습 후보/benchmark 파일은 아직 제공되지 않았으므로 연구용 학습 artifact는 없다.
GPU 메모리 fit, 실제 Transformers cache 동작, vLLM guided JSON, 실제 Teacher 실패율, Probe 정확도, QwQ 정답률과 시간 개선 여부는 검증 전이다.
과거 Baseline의 평가셋 benchmark 노출을 재현하지 않기 위해 신규 비교는 외부 benchmark를 사용한다. 따라서 동일 steps에서 신규 Teacher와 Probe를 비교해야 한다.
실행 순서·설정·GPU 예상 자원·측정 한계는 [README.md](README.md)에 정리했다.
