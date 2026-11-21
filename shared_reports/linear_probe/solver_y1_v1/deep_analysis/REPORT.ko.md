# Frozen Solver Pilot 심층 분석

Ridge는 이 5문제에서 Teacher PAB와 같은 **5/5** 정답을 얻으면서, Solver에서 **1,086토큰(8.54%)과 API 시간 합계 31.56초(10.60%)**를 줄였다. 다만 Teacher보다 토큰을 적게 쓴 문제는 **2/5**였고, 문제별 총토큰 차이의 중앙값은 **Ridge가 113토큰 더 사용**이었다. Planner를 포함한 구성요소 합산에서 Ridge는 Teacher 대비 **41.06% 절감**, Vanilla 대비 **19.99% 증가**했다. 실제 온라인 대체 여부는 독립 문제에서 정답률과 end-to-end 시간을 함께 확인한 뒤 판단해야 한다.

## 원본과 검증 범위

- 본 실행 GPU 221051은 QwQ-32B FP16, 4× RTX 3090 TP4, CPU 16개/RAM 64 GiB, 동시 요청 1, prefix cache 비활성이다. 5문제에 동일 arm 집합과 순환 순서를 사용했다. 모든 본 실행 요청에서 seed 42, temperature 0.1, top-p 0.9, 최대 8,192 생성 토큰을 사용했다. [원본 응답](../responses_solver_221051_main.json), [서버 메타데이터](../server_solver_221051_main.json)
- CPU 221040은 frozen 계획과 과거 Teacher 반복 45건의 감사 결과를 남겼고 CPU 221793은 본 실행 집계를 남겼다. 이번 CPU 재계산에서는 완료 gate와 보호 파일 SHA, 저장 evaluator의 코드·패키지 지문, raw API usage와 시도 로그, 정답 파싱, frozen 요청 순서·payload, 원본 분해에서 파싱한 steps, 원본 Teacher JSON에서 파싱한 Credit, Ridge/Position/Uniform의 정수 투영과 Budget을 검증했다. 정확도·전체 word proxy·비용은 기존 집계와 일치했다. [감사 로그](../cpu_221040.log), [기존 보고서](../REPORT_221051_221793.ko.md), [재계산 로그](validation.log)
- Smoke 221043은 L2 Vanilla/Teacher 두 군에 각각 128 및 8,192 상한을 적용했다. 전자는 둘 다 `length`이고 후자는 둘 다 `stop` 정답이다. 긴 두 건의 출력 토큰과 word count는 본 실행과 각각 동일했다. 별도 서버 실행에서 L2 두 arm의 제한된 재현 사례다. Smoke 4건의 8,433토큰과 API 194.94초는 본 실행 집계에서 제외했다. 두 실행 모두 서버 POST 수가 시도 로그 수와 맞고, 재시도와 usage 누락이 없다. [Smoke 원본](../responses_solver_221043_smoke.json), [문제별 CSV](problem_comparison.csv)
- 원본 Llama 생성은 `evaluation_capture.json`과 `evaluation_recovered.json`에서 5행 모두 동일했고 수리한 행은 없었다. 문제마다 capture와 no-hidden control 각각 한 번이 성공했으며 생성 토큰 ID가 일치했다. Teacher 라벨의 두 번째 호출도 문제마다 한 번 성공했다. 기존 훈련·검증·Teacher 안정성 측정은 실험 준비 비용이다. 온라인 요청 하나당 반복되는 비용과 구별한다. [원본 분해](../../pilot_v1/evaluation_capture.json), [복구 기록](../../pilot_v1/evaluation_recovered.json), [Teacher 라벨 및 요청](../../credit_a2000_tp4/evaluation_labeled.json), [frozen 예측](../../analysis_credit_tp4_220316/extended_220319.json)

## Solver 실측

| 방법 | 정답 | 입력 | 출력 | 총토큰 | API 합계 초 | 전체 word proxy 준수 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 5/5 | 1,191 | 12,193 | 13,384 | 367.59 | 해당 없음 |
| Teacher PAB | 5/5 | 2,886 | 9,826 | 12,712 | 297.67 | 1/5 |
| Uniform | 5/5 | 2,887 | 9,371 | 12,258 | 284.90 | 1/5 |
| Position-only | 5/5 | 2,885 | 10,280 | 13,165 | 311.39 | 0/5 |
| Ridge | 5/5 | 2,885 | 8,741 | 11,626 | 266.11 | 0/5 |

본 실행은 모두 `stop`, 각 1시도이며 입력·출력·총토큰·latency 누락이 각각 0건이다. CSV에 요청별 정답 및 추출값, 정답표, 순서, 재시도, 누락 필드, whitespace count를 실었다. 25건 전체의 합계는 63,145토큰, API 1,527.67초, 클라이언트 세션 wall 1,533.20초다. 서버 시작부터 ready까지 165.39초를 별도로 관측했다. 모니터링 wall 1,716.74초에 4 GPU를 곱하면 **1.91 GPU·h**의 모니터링 구간 할당량이다. 전체 SLURM 할당 경과, 전력, 통화 비용을 얻으려면 해당 Job의 `sacct` ElapsedRaw/AllocTRES 및 단가·전력 계측이 필요하다.

### 같은 문제 안의 승패

아래는 Ridge − 비교군의 **총토큰 / API 초**다. 모든 셀에서 정답은 동률이다. 음수는 Ridge 절감이다.

| 문제 | Vanilla | Teacher | Uniform | Position |
|---|---:|---:|---:|---:|
| L1 확률 407 | +176 / −3.67 | +113 / +3.44 | +61 / +1.83 | +62 / +1.90 |
| L2 미적분학 예비과정 352 | −44 / −10.79 | −1,310 / −38.71 | −2,037 / −61.05 | −1,554 / −46.36 |
| L3 중급대수 765 | −1,416 / −52.46 | −258 / −7.62 | −166 / −5.09 | −657 / −19.51 |
| L4 대수 1144 | −237 / −17.81 | +212 / +6.10 | +92 / +2.75 | −35 / −1.31 |
| L5 기초대수 584 | −237 / −16.74 | +157 / +5.24 | +1,418 / +42.78 | +645 / +20.00 |
| 절감 문제 수 (토큰 / 시간) | 4/5 · 5/5 | 2/5 · 2/5 | 2/5 · 2/5 | 3/5 · 3/5 |
| 차이 중앙값 | −237 / −16.74 | +113 / +3.44 | +61 / +1.83 | −35 / −1.31 |

L2 하나를 제외하면 Ridge의 총토큰 차이는 Teacher 대비 **+224**, Uniform 대비 **+1,405**, Position 대비 **+15**로 바뀐다. Teacher 대비 입력 차이는 합계 −1토큰, 문제별 −1~+1토큰이다. 출력 차이가 실측 시간 차이와 강하게 동행한다(Teacher 대비 5쌍의 Pearson r≈0.9999). 이 관측으로 계산 경로·서버 부하·생성량의 독립 효과를 추정할 수는 없다. Vanilla 대비 Ridge의 출력과 시간은 5문제 모두 줄었으나, L1의 출력 절감 111토큰보다 입력 증가 287토큰이 컸다.

문제를 단위로 보는 기술적 양측 부호 검정에서 Teacher 대비 총토큰 2승 3패의 p는 1, Vanilla 대비 API 시간 5승의 p는 0.0625다. 사전등록된 비교가 아니며 다중 비교와 선택 과정을 고려한 확증 검정으로 사용할 수 없다. 독립 무작위 문제라는 가정하에서도 5/5의 양측 정확 95% 이항 구간은 약 47.8–100%다. 현재 표본은 난이도별 1문제씩 SHA-256 순위로 뽑은 특정 MATH train 출처이며 난이도와 과목을 분리할 수 없다. 이 문제로 이미 Credit·안정성·Solver를 점검했으므로 방법 선택에 쓰면 추가 선택 편향이 생긴다. 24문제 학습·6문제 검증이라는 Probe 개발 규모와 후보 선택도 일반화 제약이다. 25건은 독립 문제 25개를 의미하지 않는다. seed 42 한 번의 비교로 prompt 변화와 샘플링 간 상호작용, TP4·서버 상태에 따른 변동을 추정할 수 없다. [데이터 선정 규칙](../../pilot_v1/data.json), [별도 Teacher 안정성 분석](../../stability_analysis_v1/REPORT_220329_220339.ko.md)

## Credit, Budget과 생성 길이

각 셀은 step 순서의 **Credit / word Budget**이다. 동일 문제의 PAB 4군은 같은 steps를 사용하며 Budget 총량은 각각 200, 250, 350, 450, 600 words로 고정됐다. 변경된 것은 step별 예산 숫자다. Vanilla는 분해와 word Budget을 사용하지 않으며 프롬프트 형식도 다르다. [80행의 subtask 원문·raw 예측·이동량](budget_comparison.csv)

| 문제 | Teacher | Uniform | Position | Ridge |
|---|---|---|---|---|
| L1 | 20·40·40 / 40·80·80 | 34·33·33 / 68·66·66 | 22·35·43 / 44·70·86 | 24·27·49 / 48·54·98 |
| L2 | 15·20·25·40 / 37·50·62·101 | 25·25·25·25 / 63·63·62·62 | 15·22·27·36 / 37·55·67·91 | 21·23·18·38 / 52·57·45·96 |
| L3 | 20·30·30·20 / 70·105·105·70 | 25·25·25·25 / 88·88·87·87 | 15·22·27·36 / 52·77·94·127 | 20·22·23·35 / 70·77·80·123 |
| L4 | 20·30·20·30 / 90·135·90·135 | 25·25·25·25 / 113·113·112·112 | 15·22·27·36 / 67·99·121·163 | 20·24·25·31 / 90·108·112·140 |
| L5 | 15·20·15·20·30 / 90·120·90·120·180 | 20·20·20·20·20 / 120·120·120·120·120 | 15·20·20·20·25 / 90·120·120·120·150 | 18·13·18·16·35 / 108·78·108·96·210 |

Credit MAE는 같은 문제의 step 절대 오차 평균을 5문제에 동일 가중해 계산했다. Teacher 기준 **Ridge 5.6133%p, Position 4.3667%p, Uniform 6.1667%p**다. Budget MAE도 **18.9467, 16.3333, 20.4333 words**로 Ridge는 Position보다 Teacher 배분에서 멀다. 그런데 Ridge의 Solver 출력은 Position보다 총 **1,539토큰 적고**, 입력 합계는 동일하다.

Credit MAE는 Teacher 배분 모사 지표다. 어느 단계의 예산을 늘리거나 줄였는지, Solver가 그 문구에 어떻게 반응했는지, 최종 정답과 생성량이 어떻게 바뀌는지를 직접 측정하지 않는다. L2에서 Ridge는 Position 대비 세 번째 step 예산을 67→45 words로 낮추고 나머지 셋은 늘렸으며 전체 출력은 3,836→2,282토큰으로 줄었다. 이 변화와 함께 L2의 전체 whitespace words도 2,432→1,410으로 줄었다. 반면 L5에서는 마지막 step 예산이 150→210으로 늘고 전체 출력도 3,348→3,994토큰으로 늘었다. L3·L4에서는 Ridge가 Position보다 Teacher에 더 가깝고 출력도 적다. L2는 반대 조합이므로 MAE가 생성량의 대리 지표라고 볼 근거가 약하다. 특정 위치의 숫자가 계획·반복 설명·종료 경로를 바꿨을 가능성과 seed에 따른 우연한 길이 차이가 모두 남는다. 동일 총량에서 특정 step에 단독으로 효과를 귀속할 수 없으며 step별 actual token 기록도 없다. Ridge의 Hidden State는 이 실험에서 Solver가 직접 읽는 입력이 아니다. frozen Credit을 Budget으로 바꾼 숫자가 prompt에 들어간다.

### Word Budget

whole-completion whitespace proxy는 raw `content.split()`으로 thinking과 최종 답변까지 세어 전체 soft word Budget 합계로 나눈 값이다. Teacher와 Uniform은 L4에서 각각 362/450 및 430/450으로만 통과했다. Ridge는 L4에서도 454/450이며 L2는 1,410/250이다. 원래 형식은 다른 풀이 방식도 허용하고 각 subtask의 고유 응답 구간을 보장하지 않는다. 따라서 subtask별 준수율은 전 군 미측정이며, 전체 초과가 특정 step의 초과를 뜻하지 않는다. `finish_reason=length`인 Smoke 두 건은 완성본 준수 판정을 비워 두었다. 모델의 8,192 **토큰** 생성 상한은 soft **word** 지시와 별도다.

## 시스템 비용과 Token Efficiency

| 방법 | Planner 입력/출력/총토큰 | Planner 재구성 초 | Planner+Solver 입력/출력/총토큰 | 합산 초 |
|---|---:|---:|---:|---:|
| Vanilla | 0 / 0 / 0 | 0.00 | 1,191 / 12,193 / 13,384 | 367.59 |
| Teacher | 12,565 / 1,973 / 14,538 | 94.34 | 15,451 / 11,799 / 27,250 | 392.02 |
| Uniform | 3,591 / 843 / 4,434 | 50.52 | 6,478 / 10,214 / 16,692 | 335.42 |
| Position | 3,591 / 843 / 4,434 | 50.52 | 6,476 / 11,123 / 17,599 | 361.91 |
| Ridge | 3,591 / 843 / 4,434 | 51.41 | 6,476 / 9,584 / 16,060 | 317.53 |

- **첫 번째 Planner 호출**은 Llama-3.1-8B HF 1×3090의 기존 관측을 사용했다. Teacher/Uniform/Position은 no-hidden control 50.516초, Ridge는 capture 51.172초와 별도 feature validation 0.216초를 반영했다. 두 경로 모두 3,591 입력 + 843 출력 토큰이다. Pilot에서 Planner를 재사용했지만, 요청 단위 비용으로 이 관측치를 계상했다. 실제 원본 실험에서는 control과 capture 모두 실행했으므로 분해 실험 자체는 10회, 8,868토큰을 기록했다.
- **Teacher 두 번째 호출**은 Llama-3.1-8B vLLM 4×A2000에서 5회, 8,974 입력 + 1,130 출력 = 10,104토큰, API 43.828초다. Ridge에서는 이 생성 호출이 빠진다. Teacher에서 Ridge로 전환한 합산 절감 11,190토큰 중 10,104토큰(90.3%)은 이 호출에 해당한다.
- **Ridge feature·Probe**는 첫 호출에서 hidden state를 저장했다. 추가 forward와 feature용 입력 토큰은 기록상 0이다. retention 0.697초, alignment 0.523초, transfer 0.142초는 capture instrumentation으로 관측한 합이며 겹치는 구간이 있어 capture wall에 재가산하지 않았다. control 다음에 capture를 실행했고 구간별 동기화와 warm-up 영향이 있다. 관측된 capture wall − control wall 0.656초에는 여러 효과가 섞여 있다. CPU feature read 0.02049초, Ridge inference 0.00483초, 정수 Credit projection 0.00034초를 온라인 추정에 계상했다. Uniform/Position의 CPU 예측+투영은 각 0.00029초/0.00032초이며 배포 경로에 필요하지 않은 오프라인 feature read는 제외했다.
- **온라인 추정 범위**는 기존 Llama 첫 호출, 두 번째 호출 또는 credit 계산, 본 실행 Solver API 시간을 순차 합산한 것이다. 모델 기동, 전체 orchestration·큐·전송, 오프라인 feature 저장, 학습 비용은 이 합계 밖이다. Teacher 대비 합산 시간 −19.00% 및 Vanilla 대비 −13.62%는 혼합 하드웨어 추정이다. Vanilla 행에는 Planner가 없으므로 Solver API 실측만 반영했다. 입출력 토큰 합은 Llama/QwQ의 각 tokenizer 계수 합이며 토큰당 같은 FLOP·가격을 가정할 근거는 없다. [문제별 측정 출처와 비용 CSV](cost_breakdown.csv), [분해 시도 CSV](decomposition_attempts.csv)

문제별로 Planner까지 합산하면 Ridge는 Teacher 대비 총토큰 5/5, 추정 시간 4/5를 절감했다. L4는 Teacher의 두 번째 호출을 빼고도 합산 시간이 2.08초 더 길었다. Vanilla 대비 총토큰 절감은 L3 한 문제뿐이고 추정 시간은 L3–L5의 3/5에서 줄었다.

절감률은 `(기준 합계 − 비교 합계) / 기준 합계 × 100`이다. 양수는 절감이다. 각 셀은 **입력 / 출력 / 총토큰 %**이며, 자세한 latency 절감률은 비용 CSV에 있다. 5군 모두 5문제 정답이므로 이 표본에서는 토큰/정답당 비용의 상대 변화도 동일하다.

| 방법 | Solver vs Teacher | Solver vs Vanilla | Planner vs Teacher | 전체 vs Teacher | 전체 vs Vanilla |
|---|---:|---:|---:|---:|---:|
| Vanilla | 58.73 / −24.09 / −5.29 | 0 / 0 / 0 | 100 / 100 / 100 | 92.29 / −3.34 / 50.88 | 0 / 0 / 0 |
| Teacher | 0 / 0 / 0 | −142.32 / 19.41 / 5.02 | 0 / 0 / 0 | 0 / 0 / 0 | −1197.31 / 3.23 / −103.60 |
| Uniform | −0.03 / 4.63 / 3.57 | −142.40 / 23.14 / 8.41 | 71.42 / 57.27 / 69.50 | 58.07 / 13.43 / 38.74 | −443.91 / 16.23 / −24.72 |
| Position | 0.03 / −4.62 / −3.56 | −142.23 / 15.69 / 1.64 | 71.42 / 57.27 / 69.50 | 58.09 / 5.73 / 35.42 | −443.74 / 8.78 / −31.49 |
| Ridge | 0.03 / 11.04 / 8.54 | −142.23 / 28.31 / 13.14 | 71.42 / 57.27 / 69.50 | 58.09 / 18.77 / 41.06 | −443.74 / 21.40 / −19.99 |

Vanilla 기준 Planner 비율은 전 군 분모 0으로 미정의다. Teacher 호출을 생략하는 Planner 토큰 절감은 Uniform/Position에서도 동일하다. Ridge 자체의 추가 가치는 그 두 대조군 대비 Solver의 생성량·정답률 또는 전체 운영 비용 개선으로 검증해야 한다.

## 결론

**입증된 사실.** 이 frozen 5문제와 저장된 decode에서 Ridge는 Teacher와 정답 5/5로 같고 Solver 합계 토큰·API 시간이 더 낮았다. Teacher를 생략한 구성에서 기존 두 번째 호출의 10,104토큰이 빠지고 첫 호출은 남는다. 합산 토큰에서 Ridge는 Teacher보다 적고 Vanilla보다 많다. L2 효과가 Teacher·Uniform·Position 대비 Solver 순위를 크게 좌우한다.

**가능한 해석.** soft Budget의 위치별 변경이 Solver 풀이 분기나 반복 길이와 상호작용했을 수 있다. Teacher Credit의 모사 정확도와 Solver 효율은 서로 다른 목표다. 풀이별 짧은 생성이 유지된다면 Teacher 호출 제거로 운영상 이득을 볼 가능성이 있다.

**아직 검증되지 않은 주장.** 독립 문제에서 정답률 비열등성, 균등 또는 위치 기반 배분 대비 Ridge 고유의 우위, step별 준수, 동일 배포 환경에서 end-to-end latency 및 금액 절감은 추가 측정이 필요하다. 배포 비교의 우선순위는 독립 문제에서 Teacher 대비 정답률 비열등성을 검정하고, 동일 요청 경로의 wall time과 GPU 사용량을 계측하며, Position/Uniform이 같은 효과를 더 단순한 경로로 얻는지 확인하는 것이다. [독립 검증 계획](INDEPENDENT_VALIDATION.ko.md)

## 재현

저장된 evaluator와 패키지 지문이 맞는 CPU 환경에서 `CUDA_VISIBLE_DEVICES='' [local-path] -B recompute.py`를 이 디렉터리에서 실행한다. 이번 재실행은 GPU·SLURM·모델 호출 없이 완료했다. 원본 파일은 읽기 전용으로 사용했고 이 분석 디렉터리에만 CSV와 로그를 작성했다. [재계산 코드](recompute.py), [실행 로그](validation.log)

두 번째 CPU 실행에서도 CSV SHA-256과 재계산 로그가 동일했다. 문제별 CSV 29행(본 25 + Smoke 4), 비용 CSV 94행, Budget CSV 80행, 분해 시도 CSV 5행 및 보고서 링크를 확인했다.
