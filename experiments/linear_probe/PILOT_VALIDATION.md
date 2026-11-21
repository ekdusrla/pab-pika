# Pilot 준비 검증 기록

검증일: 2026-11-13. GPU 모델 로딩, 실제 GPU 실험, 실제 데이터 Probe 학습, 모델 API 요청, SLURM 제출은 수행하지 않았다.
공개 외부 데이터 다운로드, CPU 데이터 감사·토큰화와 합성 단위·통합 테스트만 수행했다.

## 결과

- 기존 17개와 신규 12개를 합친 CPU unittest **29개 통과**.
- 가짜 CPU causal model로 실패 시 실제 생성 token trace와 완료 forward 수가 보존되는지 검증.
- 길이 조건의 이전 상태·plan fingerprint·reserved memory guard 및 실패 분류 검증.
- 24/6/5 문제 단위 분리, 고정 선택 재현, 최종 평가 hash 중복 차단, 외부 benchmark 독립성 검증.
- 모의 Teacher API 요청 35개 → 작은 합성 feature의 StandardScaler+Ridge fit → 독립 5문제 Credit/Budget 비교 → Solver request 생성 통합 테스트 통과.
- 별도 모의 Solver API 테스트에서 요청 10개 중 1개 실패 후 해당 1개만 재시도하여 총 11회 호출, 기존 성공·실패 기록 보존 확인.
- 기존 Solver prompt/Budget/생성 설정 일치, CPU·RAM의 노드 GPU 비율 및 5% 메모리 여유 정책 검증.
- Python AST/compile, sbatch 내장 Python, `bash -n` 정적 검사 통과.
- 실제 Solver 원본 evaluator CPU preflight 통과. fingerprint: `48f99eebcd002f2a2bba7e64114a749b0b8bb74429be51dad7080fb4cc7b80c6`.

## 실제 데이터 감사

원본은 `EleutherAI/hendrycks_math`, revision `21a5633873b6a120296cce3e2df9d5550074f4a3`의 train 7,500개다.
원본 파일 SHA-256 및 선택 재현 검사를 통과했다. 난이도 미상 2개·내부 정규화 중복 2개·boxed 정답 미추출 4개를 제외했다.
고정 seed=20261113으로 train 24개·validation 6개·외부 Pilot 평가 5개·외부 benchmark 5개를 선택했다.
MATH-500과 선택된 40개의 정규화 hash 일치 및 추가 문자열/5-gram 유사도 .90 이상 일치는 0개다. 의미 중복과 정답의 수동 검토는 남아 있다.

Data manifest fingerprint:
`19afe609722b62d4a516f8cf8ad4db9b8e7a11b52e5bacc79238d8a76eddc868`

CPU 토큰화 결과:

- 합성 길이 조건의 실제 prompt tokens: **1026, 2046, 4096**.
- Pilot Decomposition prompt 최대: **939 tokens**.
- Teacher 빈 steps prompt + 단계 직렬화 보수적 예약 1088 + 출력 2048의 최대: **4990 tokens**.
- 실제 생성 steps에 대한 정확한 context 검사는 향후 Teacher job에서 다시 수행한다.

## 기존 파일 보존

작업 전에 기록한 기존 `.py/.json/.md/.sbatch/.sh/.txt` 241개 중 변경은 `capture.py` 한 개뿐이다.
변경 내용은 선택적 `diagnostics` 인자로 실패 시 입력·생성 token 수와 완료 forward 수를 관측하는 기능이다. 기존 모델 로딩·dtype·attention·feature 정의는 바꾸지 않았다.
나머지 240개 파일의 SHA-256은 유지됐다. 기존 Baseline, 전체 평가 runner, 이전 결과 및 공유 Probe 구현은 보존했다.

실행 명령, 제출 직전 노드 정보 조회, 정책 기반 sbatch 명령 생성, 예상 자원과 남은 검증은 [PILOT.md](PILOT.md)에 정리했다.
