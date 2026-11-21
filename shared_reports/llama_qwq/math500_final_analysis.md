# MATH-500 최종 분석: Llama 70B Planner를 사용하는 PAB 변형 실험


분석 시각: 2026-11-06T19:46:58.201718+09:00. 모든 계산은 기존 산출물을 읽는 CPU 작업이며 모델·GPU·새 LLM 호출은 수행하지 않았다.


## 1. 실험 환경 및 모델 구성


| 항목 | 설정 |
| --- | --- |
| Planner | casperhansen/llama-3.3-70b-instruct-awq / revision 64d255621f40b42adaf6d1f32a47e1d4534c0f14 |
| Solver | Qwen/QwQ-32B / revision 976055f8c83f394f35dbd3ab09a285a984907bd0 |
| 자원 | moana-y4, RTX 3090 × 4, TP=4, FP16; Planner AWQ, Solver 비양자화 |
| 패키지 | {"openai": "3.37.0", "vllm": "0.6.6.post1", "torch": "2.5.1+cu121", "numpy": "1.26.4", "transformers": "4.46.3"} |
| 생성 | max_tokens=8192, temperature=0.1, top_p=0.9, seed=42; 문제별 호출 순서 교대 |
| 컨텍스트 | Planner 8192 / Solver 12288; max_num_seqs=1 |
| 예산 | level 1~5 → 200/250/350/450/600 words; constant decay, 위치별 감쇠 없음 |
| 구조 | 정상 경로: Planner 분해 1회 + credit 평가 1회, Vanilla/PAB Solver 각각 1회 |


사용자가 제시한 원논문 Planner는 LLaMA-3.1-8B-Instruct이고, 본 실험은 Llama-3.3-70B-Instruct AWQ를 사용한다. 따라서 **원논문의 정확한 재현이 아닌 70B Planner PAB 변형 실험**이다. 논문 원문은 이번 분석에서 별도 검증하지 않았다. 8B 실행 결과가 없어 모델 크기 차이에 따른 실제 정확도·비용 차이는 단정할 수 없다.


## 2. 실험 완료 및 데이터 무결성 검증


| 검증 | 결과 |
| --- | --- |
| Manifest 전체 fingerprint | 통과 |
| Manifest 문제별 fingerprint | 통과 |
| Manifest 500개 ID·고유성·순서 | 통과 |
| Planner 500개 ID·고유성·순서 | 통과 |
| Planner 원본 문제·정답·난이도 매핑 | 통과 |
| Planner Manifest 설정 일치 | 통과 |
| Solver 500개 ID·고유성·순서 | 통과 |
| Solver 원본 문제·정답·난이도 매핑 | 통과 |
| Solver Manifest 설정 일치 | 통과 |
| Solver가 참조한 Planner 설정 fingerprint | 통과 |
| Planner 500개 두 단계 성공 | 통과 |
| Planner 단계 수·credit 검증 | 통과 |
| 기존 Planner 성공 494개 보존 해시 | 통과 |
| Planner 최초 실패 6개 이력 보존 | 통과 |
| Vanilla API 결과 500개 보존 | 통과 |
| PAB API 결과 500개 보존 | 통과 |
| Solver 계획·요청·raw 응답·시도·usage·평가 정합성 | 통과 |
| 성공 응답 ID 1000개 고유성 | 통과 |
| 성공 응답 ID와 서버 로그 매칭 | 통과 |
| Constant 예산 배분 및 설정 | 통과 |
| 기존 평가기 코드·의존성 일치 | 통과 |
| CPU 재평가와 저장 판정 일치 | 통과 |
| Planner 전체 시도 요약 재계산 | 통과 |
| Vanilla 전체 시도 요약 재계산 | 통과 |
| PAB 전체 시도 요약 재계산 | 통과 |
| 분석 전후 원본 파일 SHA-256 불변 | 통과 |


완료는 API 응답의 checkpoint 보존을 뜻하며 정상 종료·정답을 뜻하지 않는다. Planner는 500개 모두 두 단계 성공, Solver는 방법별 500개 API 응답을 보존했다. ID·원본 순서·문제·정답·level·레코드 fingerprint 및 plans/credits/usage/raw 응답을 대조했다.


| Job | 서버 접수 요청 | Chat HTTP 200 | 클라이언트 문제 상태 행 | 서버 ERROR/CRITICAL |
| --- | --- | --- | --- | --- |
| 153079 | 994 | 994 | 500 | 0 |
| 153165 | 12 | 12 | 6 | 0 |
| 153166 | 610 | 609 | 304 | 0 |
| 153238 | 391 | 391 | 500 | 0 |


재개 로그의 문제 상태 행은 건너뛴 문제도 출력하므로 새 API 호출 수로 해석하지 않았다. raw 응답 ID와 서버 요청 ID를 연결한 성공 응답의 작업별 분포:


| 방법 | Job별 현재 checkpoint 성공 응답 수 |
| --- | --- |
| vanilla | {"153166": 305, "153238": 195} |
| weighted_pab | {"153166": 304, "153238": 196} |


Vanilla는 500회, PAB는 501회 시도했다. 성공 응답 ID는 1000개 모두 고유하고, 기존 성공을 다시 생성한 시도는 기록상 없다. 1개 PAB 중단 시도만 재개했다. Solver의 과거 checkpoint 스냅샷은 없어 모든 과거 바이트가 불변이었다는 절대 증명은 불가하지만, 보존된 응답 ID·시도 이력·로그는 정상 resume를 뒷받침한다. Planner 494개는 복구 전 저장된 해시로 불변을 확인했다.


추가 Solver 시도: `[{"id": "292", "method": "weighted_pab", "attempts": [{"number": 1, "status": "interrupted", "request_fingerprint": "cece8662dcf76070f6a39584f43816b3bae33a4955346997df1c95e8d480bc4b", "usage": {"prompt_tokens": null, "completion_tokens": null, "latency_seconds": 6.507955}, "cost_unknown": true}, {"number": 2, "status": "success", "request_fingerprint": "cece8662dcf76070f6a39584f43816b3bae33a4955346997df1c95e8d480bc4b", "usage": {"prompt_tokens": 732, "completion_tokens": 1990, "latency_seconds": 57.591143}, "finish_reason": "stop"}]}]`


## 3. Vanilla vs PAB Accuracy 비교


| 방법 | 정답 | 오답 | 판정 불가 | Accuracy(분모 500) | stop | length |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 444 | 56 | 0 | 88.80% | 454 | 46 |
| PAB | 433 | 67 | 0 | 86.60% | 466 | 34 |


PAB−Vanilla = **-2.20 percentage points**. 다음 비교는 같은 500개 문제의 쌍 비교다.


| 정오 변화 | 문제 수 |
| --- | --- |
| 둘 다 정답 | 404 |
| Vanilla만 정답 | 40 |
| PAB만 정답 | 29 |
| 둘 다 오답 | 27 |


기존 `math_answer_eval.py`의 original 모드를 수정 없이 1000응답에 CPU 재적용했다. 원본 Qwen `extract_answer`(최종 답 문구/boxed/답 문구/마지막 숫자 fallback) → `strip_string`(LaTeX·공백·단위 등 정규화) → `math_verify.parse/verify` 순서다. 단순 문자열 일치나 수동 채점으로 바꾸지 않았다. 파싱 실패·빈 추출·예외는 판정 불가이며 전체 정확도 분모에서 제외하지 않는다. finish_reason은 정오와 별개여서 잘린 응답도 이 기준으로 채점한다. 전체 content를 평가하므로 reasoning 중 숫자나 불완전 답에서 fallback이 작동할 수 있다.


재평가와 저장 평가의 불일치: **0개**. evaluator fingerprint: `48f99eebcd002f2a2bba7e64114a749b0b8bb74429be51dad7080fb4cc7b80c6`.


| 방법/종료 | 문제 수 | 정답 | 해당 subset 정확도 |
| --- | --- | --- | --- |
| Vanilla/stop | 454 | 440 | 96.92% |
| Vanilla/length | 46 | 4 | 8.70% |
| PAB/stop | 466 | 432 | 92.70% |
| PAB/length | 34 | 1 | 2.94% |


탐색적 paired 검정: exact McNemar 양측 p=0.2284; 문제 단위 paired bootstrap 5000회(seed=42)의 정확도 차이 95% 구간=-5.40~+1.00 pp. 단일 seed·고정 데이터 결과이며 광범위한 일반화 또는 원인 증명이 아니다.


정확도 점추정은 PAB가 낮지만 이 paired 검정의 5% 기준에서 유의한 차이라고 단정할 수 없다. '동등함이 입증됐다'는 뜻도 아니다. 정상 종료 subset 정확도는 서로 다른 문제 집합을 조건으로 하므로 전체 500개 paired 정확도를 대체하지 않는다.


## 4. Vanilla vs PAB 토큰 사용량 비교


통계 단위는 문제 500개이며 표준편차는 모집단 표준편차(ddof=0)다. A/B는 최종 보존 성공 응답의 정확한 값이다. Planner/C는 실패·복구·중단 등 모든 시도를 포함한다. **†는 PAB 중단 호출 1개의 입력·출력 토큰이 미상인 기록값**이다. † 총합·평균은 실제 비용의 하한이며, 중앙값·최소/최대·표준편차는 알려진 값 벡터의 기술통계일 뿐 실제 비용 분포의 확정치가 아니다.


| 범위 | 토큰 | 총합 | 평균 | 중앙값 | 최소 | 최대 | 표준편차 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A Vanilla Solver | 입력 | 106,549 | 213.10 | 193.00 | 151 | 933 | 76.32 |
| A Vanilla Solver | 생성 | 1,446,770 | 2,893.54 | 2,038.50 | 273 | 8,192 | 2,388.30 |
| A Vanilla Solver | 전체 | 1,553,319 | 3,106.64 | 2,226.00 | 427 | 8,682 | 2,417.27 |
| B PAB Solver 성공응답 | 입력 | 292,696 | 585.39 | 572.50 | 352 | 1,289 | 141.32 |
| B PAB Solver 성공응답 | 생성 | 1,011,807 | 2,023.61 | 950.00 | 228 | 8,192 | 2,279.84 |
| B PAB Solver 성공응답 | 전체 | 1,304,503 | 2,609.01 | 1,543.50 | 627 | 9,202 | 2,356.88 |
| Planner 모든 시도 | 입력 | 1,182,477 | 2,364.95 | 2,314.50 | 2,091 | 4,685 | 245.78 |
| Planner 모든 시도 | 생성 | 165,958 | 331.92 | 323.00 | 126 | 781 | 119.24 |
| Planner 모든 시도 | 전체 | 1,348,435 | 2,696.87 | 2,644.50 | 2,218 | 5,399 | 341.27 |
| C PAB Planner+Solver 모든 시도 | 입력 | 1,475,173† | 2,950.35† | 2,875.00† | 2,443† | 5,739† | 369.11† |
| C PAB Planner+Solver 모든 시도 | 생성 | 1,177,765† | 2,355.53† | 1,304.50† | 401† | 8,788† | 2,342.01† |
| C PAB Planner+Solver 모든 시도 | 전체 | 2,652,938† | 5,305.88† | 4,286.50† | 2,914† | 12,711† | 2,546.03† |


PAB Solver 모든 시도에서 알려진 토큰 합계는 성공응답 합계와 같지만, 중단 1회의 토큰은 미상이다. 이를 0토큰 호출로 단정하지 않았다.


| 비교 | 입력 변화 | 생성 변화 | 전체 변화 |
| --- | --- | --- | --- |
| PAB Solver 성공응답 | +174.71% | -30.06% | -16.02% |
| PAB 전체 알려진 시도 비용† | +1284.50% | -18.59% | +70.79% |


감소율=100×(Vanilla−PAB)/Vanilla, 변화율=100×(PAB/Vanilla−1). 총합의 비율을 사용하며 문제별 비율의 단순 평균은 아니다. †에서 Solver 절감률은 가능한 상한, 전체 비용 증가율은 하한이다. 서로 다른 Llama/QwQ tokenizer의 토큰 단순 합계는 FLOPs·GPU 시간·금전 비용과 동일하지 않다.


## 5. Vanilla vs PAB 추론 시간 비교


| 범위(초) | 합계 | 평균 | 중앙값 | 최소 | 최대 | 표준편차 |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla Solver | 41,552.111 | 83.104 | 58.194 | 7.978 | 241.338 | 68.967 |
| PAB Solver 성공응답 | 29,568.896 | 59.138 | 27.952 | 7.251 | 246.024 | 66.497 |
| PAB Solver 모든 시도 | 29,575.404 | 59.151 | 27.952 | 7.251 | 246.024 | 66.497 |
| Planner 모든 시도 | 7,842.550 | 15.685 | 15.253 | 8.993 | 31.658 | 3.794 |
| Planner+PAB 모든 시도 | 37,417.953 | 74.836 | 44.272 | 17.426 | 265.975 | 68.426 |


실패·중단을 포함한 PAB 전체 API latency 변화율은 **-9.95%**다. API 시간은 클라이언트 요청부터 응답까지이며 서버 처리·대기·통신을 포함한다. Planner와 Solver는 별도 작업이므로 위 합은 합성 pipeline 비용이며 실제 한 번의 end-to-end wall-clock 측정이 아니다.


| Job | State | ExitCode | 시작 | 종료 | Slurm 경과 | 대기(초) |
| --- | --- | --- | --- | --- | --- | --- |
| 153079 | FAILED | 2:0 | 2026-11-04T23:59:13 | 2026-11-05T02:17:54 | 02:18:41 | 1.0 |
| 153165 | COMPLETED | 0:0 | 2026-11-05T10:04:56 | 2026-11-05T10:08:20 | 00:03:24 | 1.0 |
| 153166 | FAILED | 15:0 | 2026-11-05T10:11:21 | 2026-11-05T22:09:13 | 11:57:52 | 0.0 |
| 153238 | COMPLETED | 0:0 | 2026-11-05T22:23:20 | 2026-11-06T06:58:04 | 08:34:44 | 0.0 |


Slurm 값은 읽기 전용 sacct 조회 스냅샷이다. 첫 Solver 작업 153166은 종료 신호로 실패했고 153238에서 재개했다. 이들 경과 시간을 단순 합해 추론 latency로 사용하지 않았다. 작업 사이 공백도 추론 시간에서 제외했다.


| Job | 가중치 로딩 시작 | 로딩 종료 | 병렬 로딩 경과(초) | 엔진 준비 | 첫 API 접수 |
| --- | --- | --- | --- | --- | --- |
| 153079 | 2026-11-04 23:59:34 | 2026-11-04 23:59:43 | 9.0 | 2026-11-05 00:00:07 | 2026-11-05 00:00:12 |
| 153165 | 2026-11-05 10:05:18 | 2026-11-05 10:05:26 | 8.0 | 2026-11-05 10:05:50 | 2026-11-05 10:05:58 |
| 153166 | 2026-11-05 10:11:42 | 2026-11-05 10:12:02 | 20.0 | 2026-11-05 10:12:22 | 2026-11-05 10:12:31 |
| 153238 | 2026-11-05 22:23:45 | 2026-11-05 22:24:09 | 24.0 | 2026-11-05 22:24:29 | 2026-11-05 22:24:44 |


로딩 경과는 로그의 병렬 worker 공통 구간이며 GPU별 시간을 합하지 않았다. 엔진 준비에는 profiling·KV cache·warmup 등이 포함된다. 서버 준비와 첫 요청 사이에는 polling·클라이언트 시작 등이 있어 순수 로딩으로 간주하지 않는다.


## 6. 난이도별 성능 분석


| Level | N | Vanilla Acc | PAB Acc | 차이 pp | V만 정답 | P만 정답 | 둘 다 정답 | 둘 다 오답 | Solver 생성 변화 | Solver 시간 변화 | 전체 토큰 변화† | 전체 시간 변화 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 43 | 93.02% | 83.72% | -9.30 | 6 | 2 | 34 | 1 | -35.24% | -33.33% | +171.14% | +1.68% |
| 2 | 90 | 95.56% | 92.22% | -3.33 | 5 | 2 | 81 | 2 | -40.42% | -39.02% | +123.72% | -10.54% |
| 3 | 105 | 93.33% | 92.38% | -0.95 | 6 | 5 | 92 | 2 | -46.81% | -45.66% | +70.53% | -24.90% |
| 4 | 128 | 89.84% | 86.72% | -3.12 | 12 | 8 | 103 | 5 | -35.66% | -34.58% | +59.91% | -16.39% |
| 5 | 134 | 78.36% | 79.10% | +0.75 | 11 | 12 | 94 | 17 | -16.93% | -15.71% | +56.73% | -0.51% |


수준별 비용 변화는 그 수준의 총합 기준이다. 표시 난이도는 데이터셋 level이다. Planner가 다시 평가한 difficulty와 혼동하지 않았다. 난이도가 높을수록 PAB가 항상 유리하다고 단정하지 않으며, 각 수준의 표본 수·잘림·정확도 변화를 함께 해석해야 한다.


Level 5에서는 PAB가 1문제 더 맞혀 +0.75pp였지만, 생성 토큰 절감률은 16.93%로 다른 수준보다 작았다. Level 1~4의 정확도는 모두 낮아졌다. 따라서 고난도에서 효과가 일관되게 커진다고 주장할 근거는 없다.


| 난이도 | 데이터셋 문제 수 | Planner 재평가 문제 수 | Planner 단계 평가 수 |
| --- | --- | --- | --- |
| 1 | 43 | 69 | 549 |
| 2 | 90 | 114 | 490 |
| 3 | 105 | 69 | 347 |
| 4 | 128 | 155 | 363 |
| 5 | 134 | 93 | 108 |

## 7. 문제별 결과 변화 분석


| 개선/악화 기준 | 문제 수 |
| --- | --- |
| Solver 생성 토큰 감소 | 416 |
| Solver 총토큰 감소 | 324 |
| Solver 시간 감소 | 410 |
| Planner 포함 기록 총토큰 감소† | 30 |
| Planner 포함 전체 시간 감소 | 274 |
| 정확도 유지/개선 및 전체 시간 감소 | 264 |
| PAB 전체 생성 words가 할당 합계 초과 | 366 |

| 단계 수 | N | V Acc | P Acc | 평균 V 생성 | 평균 P 생성 | Solver 생성 변화 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 63 | 90.48% | 82.54% | 1,143.1 | 710.7 | -37.82% |
| 3 | 154 | 95.45% | 93.51% | 1,980.9 | 1,096.9 | -44.63% |
| 4 | 146 | 89.73% | 89.04% | 3,010.0 | 2,038.7 | -32.27% |
| 5 | 137 | 79.56% | 78.10% | 4,600.3 | 3,653.0 | -20.59% |


Credits는 전부 양의 정수이며 합계 100이다. 단계 수·credits·정규화 entropy·Planner 평가 난이도·단계별 words를 CSV에 저장했다. 전체 생성 words는 reasoning+answer 공백 분리 수여서 수학 표기와 언어에 민감하다. 단일 Solver 호출이므로 실제 단계별 토큰/words 배분이나 엄격한 예산 준수는 확인 불가하다.


| 지표 | 평균 | 중앙값 | 최소 | 최대 |
| --- | --- | --- | --- | --- |
| 할당 words 합계 | 411.700 | 450.000 | 200 | 600 |
| 실제 PAB words | 1,117.044 | 538.000 | 128 | 5,930 |
| PAB words/할당 words | 2.529 | 1.473 | 0.309 | 27.325 |
| PAB 생성 토큰/할당 word | 4.562 | 2.681 | 0.571 | 40.960 |
| 최대 credit | 43.690 | 40.000 | 20 | 80 |
| 정규화 credit entropy | 0.923 | 0.960 | 0.665 | 1.000 |

| 전체 단계 credit 수 | 평균 | 중앙값 | 최소 | 최대 | 표준편차 |
| --- | --- | --- | --- | --- | --- |
| 1857 | 26.925 | 20 | 5 | 80 | 14.511 |

탐색적 상관관계(난이도 등 교란요인 미보정):


| X | Y | Pearson r | Spearman ρ |
| --- | --- | --- | --- |
| level | solver_delta_completion_tokens | -0.080 | -0.097 |
| level | p_completion_tokens | 0.468 | 0.570 |
| level | p_words_budget_ratio | 0.188 | 0.176 |
| subquestions | solver_delta_completion_tokens | -0.094 | -0.105 |
| subquestions | p_completion_tokens | 0.465 | 0.560 |
| subquestions | p_words_budget_ratio | 0.215 | 0.222 |
| credit_max | solver_delta_completion_tokens | 0.097 | 0.107 |
| credit_max | p_completion_tokens | -0.377 | -0.502 |
| credit_max | p_words_budget_ratio | -0.183 | -0.204 |
| credit_normalized_entropy | solver_delta_completion_tokens | -0.069 | -0.076 |
| credit_normalized_entropy | p_completion_tokens | 0.243 | 0.298 |
| credit_normalized_entropy | p_words_budget_ratio | 0.120 | 0.128 |
| words_budget | solver_delta_completion_tokens | -0.057 | -0.097 |
| words_budget | p_completion_tokens | 0.492 | 0.570 |
| words_budget | p_words_budget_ratio | 0.208 | 0.176 |
| planner_evaluated_level | solver_delta_completion_tokens | -0.182 | -0.230 |
| planner_evaluated_level | p_completion_tokens | 0.535 | 0.699 |
| planner_evaluated_level | p_words_budget_ratio | 0.376 | 0.513 |


이 상관관계는 인과관계가 아니다. 단계 수와 words budget은 난이도·문제 유형과 함께 변하므로 credit/계획 자체의 독립 효과로 해석할 수 없다.


PAB Solver 생성 증가 상위 (정량 지표로 선택; 의미적 원인 판정은 하지 않음):


| ID | Level | 단계 | V→P 정답 | V/P 생성 | 차이 | V/P 종료 | 문제 앞부분 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | 4 | 4 | True→False | 1961/8192 | 6231 | stop/length | Two fair, 6-sided dice are thrown.  What is the probability that the product of the two numbers is a multiple  |
| 108 | 3 | 3 | True→False | 3633/8192 | 4559 | stop/length | Let $F_1$ and $F_2$ be the foci of the ellipse $kx^2 + y^2 = 1,$ where $k > 1$ is a constant. Suppose that the |
| 184 | 5 | 5 | True→False | 5101/8192 | 3091 | stop/length | Let $a,$ $b,$ and $c$ be positive real numbers.  Find the set of all possible values of \[\frac{c}{a} + \frac{ |
| 11 | 5 | 5 | True→False | 5415/8192 | 2777 | stop/length | Let $p(x)$ be a polynomial of degree 5 such that \[p(n) = \frac{n}{n^2 - 1}\]for $n = 2,$ 3, 4, $\dots,$ 7.  F |
| 361 | 4 | 4 | True→False | 5591/8192 | 2601 | stop/length | An equilateral triangle is inscribed in the parabola $x^2 = 8y,$ such that one of the vertices of the triangle |

Vanilla만 정답인 사례 (정량 지표로 선택; 의미적 원인 판정은 하지 않음):


| ID | Level | 단계 | V→P 정답 | V/P 생성 | 차이 | V/P 종료 | 문제 앞부분 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | 4 | 4 | True→False | 1961/8192 | 6231 | stop/length | Two fair, 6-sided dice are thrown.  What is the probability that the product of the two numbers is a multiple  |
| 108 | 3 | 3 | True→False | 3633/8192 | 4559 | stop/length | Let $F_1$ and $F_2$ be the foci of the ellipse $kx^2 + y^2 = 1,$ where $k > 1$ is a constant. Suppose that the |
| 184 | 5 | 5 | True→False | 5101/8192 | 3091 | stop/length | Let $a,$ $b,$ and $c$ be positive real numbers.  Find the set of all possible values of \[\frac{c}{a} + \frac{ |
| 11 | 5 | 5 | True→False | 5415/8192 | 2777 | stop/length | Let $p(x)$ be a polynomial of degree 5 such that \[p(n) = \frac{n}{n^2 - 1}\]for $n = 2,$ 3, 4, $\dots,$ 7.  F |
| 361 | 4 | 4 | True→False | 5591/8192 | 2601 | stop/length | An equilateral triangle is inscribed in the parabola $x^2 = 8y,$ such that one of the vertices of the triangle |

PAB만 정답인 사례 (정량 지표로 선택; 의미적 원인 판정은 하지 않음):


| ID | Level | 단계 | V→P 정답 | V/P 생성 | 차이 | V/P 종료 | 문제 앞부분 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 299 | 4 | 5 | False→True | 8192/1550 | -6642 | length/stop | Consider the rectangle with vertices at $(5,4),$ $(5,-4),$ $(-5,4),$ $(-5,-4)$. How many integer coordinates w |
| 396 | 3 | 3 | False→True | 8192/2957 | -5235 | length/stop | Quadrilateral $ABCD$ is a square with area 16 square inches. The figure represents the pieces of a Chinese tan |
| 421 | 5 | 5 | False→True | 8192/2960 | -5232 | length/stop | If $\arccos x + \arccos 2x + \arccos 3x = \pi,$ then $x$ satisfies a cubic polynomial of the form \[ax^3 + bx^ |
| 295 | 3 | 3 | False→True | 8192/3155 | -5037 | length/stop | The solutions of the equation $z^4+4z^3i-6z^2-4zi-i=0$ are the vertices of a convex polygon in the complex pla |
| 284 | 5 | 5 | False→True | 6097/1153 | -4944 | stop/stop | Find all solutions to \[\sin \left( \tan^{-1} (x) + \cot^{-1} \left( \frac{1}{x} \right) \right) = \frac{1}{3} |

| 변화 그룹 | N | 평균 Level | 평균 단계 수 | 평균 V 생성 | 평균 P 생성 | PAB length 수 |
| --- | --- | --- | --- | --- | --- | --- |
| 둘 다 정답 | 404 | 3.35 | 3.65 | 2,341.04 | 1,462.02 | 0 |
| Vanilla만 정답 | 40 | 3.42 | 3.70 | 3,119.18 | 3,454.50 | 14 |
| PAB만 정답 | 29 | 3.90 | 3.97 | 6,320.38 | 3,856.24 | 1 |
| 둘 다 오답 | 27 | 4.30 | 4.37 | 7,145.67 | 6,338.44 | 19 |

## 8. Planner 오버헤드 분석


| 범위 | 호출 | 입력 | 생성 | 전체 토큰 | 호출 시간(초) |
| --- | --- | --- | --- | --- | --- |
| 최초 Planner 작업 | 994 | 1164619 | 163672 | 1328291 | 7,725.086036 |
| 6문제 복구 분해+평가 | 12 | 17858 | 2286 | 20144 | 117.463801 |
| 그중 최초 무효 분해(중복 합산 금지) | 6 | 4642 | 1833 | 6475 | 64.337553 |
| Planner 총 실제 시도 | 1006 | 1182477 | 165958 | 1348435 | 7,842.549837 |


Solver checkpoint의 `paired_subset.pab_including_planner`는 최종 성공 Planner 두 응답만 합산해 최초 무효 분해 6,475토큰·64.337553초를 제외한다. 본 보고서 C는 이 비용을 복원하고, PAB Solver 중단 호출 시간도 포함한 `all_attempt_cost` 기준이다. 성공 호출만 비교하는 요약과 혼동하면 안 된다.


실제 알려진 PAB 전체 토큰은 **최소 2,652,938**, Vanilla 1,553,319 대비 **최소 70.79% 증가**다. API latency 합은 37,417.953초로 Vanilla 대비 -9.95% 변했다. 더 큰 Planner가 정확도 향상을 가져왔다거나 8B보다 경제적이라는 주장은 이 결과만으로 불가하다.


## 9. PAB의 효과 및 한계


- 정확도: Vanilla 88.80%, PAB 86.60%로 PAB가 2.20pp 낮았다.
- Solver 성공응답 생성 토큰: 30.06% 감소. 미상 중단 호출을 포함한 실제 절감률은 이보다 작을 수 있다.
- Solver 성공응답 latency: 28.84% 감소.
- Planner 포함 알려진 전체 토큰: 70.79% 이상 증가. 전체 API 시간 합은 9.95% 감소.



따라서 정확도를 유지한 채 전체 비용을 줄인다는 결론은 성립하지 않는다. 이번 설정은 Solver 출력·API 시간을 줄이는 대신 정확도와 Planner 포함 토큰 비용을 지불한 trade-off다. 시간과 토큰의 서로 다른 결론은 모델별 prefill/decode 속도·입력 길이·생성량 등이 다르기 때문일 수 있으나 이 실험으로 원인을 분리하지 못한다.


주요 한계:

- 단일 모델쌍·단일 seed·고정 MATH-500이다. 8B Planner 대조 실험이 없다.
- constant이므로 위치 decay의 효과를 검증하지 않았다. 계획·credit·words 지시·추가 난이도 안내 효과를 분리하지 않았다.
- benchmark 예시 5개는 평가 집합에서 뽑혀 정답 풀이가 Planner 문맥에 포함된다. 중복 ID: `472, 36, 2, 272, 251`. 원본 저장소 방식을 유지한 제한이며 독립 benchmark 조건과 다르다.
- 잘린 출력도 원본 evaluator의 fallback으로 정답 처리될 수 있다. 정답의 의미적 타당성을 전부 수동 검증한 것은 아니다.
- 중단 PAB 호출의 토큰은 미상이다. 세션/호출 순서 효과와 wall-clock 변동도 존재한다.
- 두 tokenizer 토큰 합산은 모델별 FLOPs·에너지·요금이 아니다.


서버 로그에는 FP16 변환·custom allreduce 비활성화·NCCL 종료 정리·공유 메모리 경고가 있다. OOM이나 HTTP 오류의 원인으로 단정할 기록은 없으며, 첫 Solver 작업의 취소/종료 신호와 1개 중단 호출은 별도로 기록됐다.


## 10. 최종 결론 및 후속 실험 제안


기존 산출물 기준 Planner/Vanilla/PAB는 모두 500문제 결과를 보존했고, resume로 작업을 완결했다. 이 70B Planner 변형은 Solver 효율에 이점이 있지만 정확도와 전체 토큰 관점에서 Vanilla를 지배하지 않는다. 시간 우선/토큰 우선/정확도 우선 중 운영 목적에 따라 평가해야 한다.

후속 실험 제안(이번에는 실행하지 않음):

1. 동일한 평가·생성 설정에서 8B Planner와 70B Planner의 직접 대조.
2. 여러 seed와 paired confidence interval, 오류·잘림 사례의 수동 감사.
3. plan-only/균등 credit/credit-only/기존 decay별 ablation.
4. 독립 benchmark 예시를 사용해 평가 집합 중복 제거.
5. 요청별 GPU 실행 시간·에너지·원가 측정과 서버 측 중단 usage 기록.


### 재현 기준 및 원본 보호


재현 명령: `[local-path] -B experiments/llama_qwq/analyze_math500_final.py` (권장 환경 `PYTHONDONTWRITEBYTECODE=1`). 보고서만 다시 생성할 때 `--reuse-evaluation`은 아래 원본·평가기 해시가 일치하는 CPU 재평가 캐시만 사용한다.


| 입력 | SHA-256 |
| --- | --- |
| experiments/llama_qwq/runs/math500_original_v1/planner.json | 19a32b1a14092a0fd55f24e6cf8d09f13ea80f38fa693c9703380878136986d4 |
| experiments/llama_qwq/runs/math500_original_v1/solver.json | 464b4f8980d72030bcfe8b4c7dfb973c50f76373f8f8967933a17ba52058bd07 |
| manifest | 71fb048c7e57ecb87479c24c3b990d4c047dae9c13eaa00933eddb781477d71e |
| evaluator_code | 9c25e0d086f2676ae83aa7712b8cfead4195768e5e3317952a721906dec49ed9 |
| original_parser | 9c359a6c0d2313b94010dfdce28f8d10ed0706eeb87728e01413b326d37aae99 |
| log_llama_planner_153079 | 84685885e458dea26f2ba5e4f62a78aa9dc00d07aac923e212ab74bea2892c1b |
| log_llama_server_153079 | 80980fd3b3d97a5404a33fca38f85c4ef7a4acf5f9e02e10af28472820636878 |
| log_llama_planner_153165 | 69b3a1549dd99e51ff4d6e39820186272451067232ea890d6da5260e2ddc22e5 |
| log_llama_server_153165 | 2609d5cb021b6eacb8367bf09458e7caa15c42f9802af837e9919965718118e7 |
| log_qwq_compare_153166 | 355f4b887d004709e76d1dfc45a6ecf98295c13af5689834cff73052a54ae71c |
| log_qwq_server_153166 | b0d20f341da83711e0b9046876f55b1c31a3c47a4600982d363573403d3c3d88 |
| log_qwq_compare_153238 | 6ed74a0c7f66b0905aecf4d284e7a678abac41c9d6de098f9a5e22b0e029aac7 |
| log_qwq_server_153238 | f0c4a452f2aa701cd0a15012ec0f4fdaf4cb80f6b9218626a5d7019a7a535f39 |


문제별 CSV는 원본 ID와 source_index를 함께 보존한다. `p_all_*`/`planner_*`/`e2e_*`는 모든 시도의 알려진 합계이며 `*_missing`은 미상 호출 수다. `v_*`/`p_*`는 최종 보존 응답이고, `solver_delta_*`는 PAB−Vanilla다. 평가 결과는 별도 분석 캐시에 저장하며 checkpoint·평가 코드·로그를 고치지 않는다.
