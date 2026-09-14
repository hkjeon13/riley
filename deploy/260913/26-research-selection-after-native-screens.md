# 연구 후보 재선정 — native 실험 이후

상태: 연구 및 다음 PR 범위. 성능 개선 구현 완료나 serving 승격을 의미하지 않는다. Runtime Rust → C ABI → CUDA, 동일 모델·수치 계약, 향후 multi-GPU/Hopper/Blackwell 지원 조건을 유지한다.

## 실험에서 얻은 제약

- Softmax 분리 및 출력 폭 확대는 strict 검증을 통과했으나 대표 native shape에서 회귀했다. `benchmarks/results/20260914-decode-softmax-reuse-native`, `20260914-decode-softmax-fused-native` 참조.
- 요청 두 개의 KV 재사용은 공유 입력에서도 느렸다. 별도 profile에서 PV 평균 10.916 → 23.511 us, register 40 → 72, 정적 resident CTA 16 → 9였다. Profile에는 warmup/correctness가 포함된다. 정적 occupancy는 achieved occupancy가 아니며 인과관계를 단독으로 증명하지 않는다. `20260914-decode-request-pair-profile` 참조.
- GPU counter는 ERR_NVGPUCTRPERM으로 수집하지 못했다. HBM/L2 병목으로 확정하지 않는다.
- CPU linked owner index는 off-batch owner 512개에서 약 0.50 → 7.7–8.0 ms로 회귀했다. 동일 page의 삽입마다 기존 owner chain을 순회하는 비용이 커진다. production 수정은 철회하고 재현 patch를 `20260914-owner-index-cpu-screen`에 보존했다.

## 구체적 검색 및 문헌의 적용 범위

| 검색어 | 원문 | Riley에 주는 판단 |
|---|---|---|
| GPU small M large K wave quantization work-centric decomposition | [Stream-K](https://arxiv.org/abs/2301.03598), [NVIDIA GEMM guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) | 출력 tile이 적을 때 작업 분배를 바꾸는 후보. K 부분합 순서 변경은 현재 BF16 경계와 충돌할 수 있으므로 동일 경계의 작업 분배부터 검토한다. |
| attention communication complexity SRAM head dimension fast matrix multiplication lower bound | [I/O Complexity of Attention](https://arxiv.org/abs/2402.07443) | 2단계 메모리 모델의 I/O 하한은 연산량 감소가 이동량 감소를 보장하지 않음을 점검하는 근거다. 짧은 paged decode의 실제 HBM/L2 비용을 직접 예측하는 식으로 쓰지 않는다. |
| attention bounded entries polynomial approximation subquadratic complexity | [Fast Attention Requires Bounded Entries](https://arxiv.org/abs/2302.13214) | 근사 attention 연구에서 입력 범위와 오차 조건을 먼저 읽는다. 학습된 모델에 무조건 대입하거나 strict greedy 동일성을 주장하지 않는다. |

원문 초록 및 공식 guide를 통한 후보 선별이다. 새로운 CUDA 구현의 효과를 입증한 결과가 아니다. 관련 선행 연구와 FA3/FA4/Hydragen 상세 대조는 `24-matrix-hardware-research.md`를 함께 본다.

## 다음 PR 단위

### PR A — 행렬 shape와 작업 분배 후보 선별

대상: 실제 serving에서 실행되는 FFN/projection shape와 그 호출 비중.

묶음: (1) 대표 M/N/K 및 전체 grid·tile padding 추출, (2) SM 수에 따른 wave 수와 마지막 wave 낭비 계산, (3) 기존 K320 BF16 부분합 경계를 유지하는 분배안과 scratch/동기화 비용 산정.

완료: 각 후보의 절약 가능한 시간 상한, 추가 비용, 대표 shape를 기록한다. 이론 상한이 작은 후보는 구현하지 않는다. 정적 계산만으로 실제 GPU 활용률을 주장하지 않는다.

다음 구현 PR은 native strict oracle와 실제 모델 일치 검증을 통과한 뒤 serving 비교를 수행한다. Hopper/Blackwell은 SM 수·자원 한도를 별도로 모델링하고 실행 장비가 없는 테스트만 미검증으로 남긴다.

### PR B — attention 재사용의 자원 비용을 제한한 설계

대상: 현재 pure-decode의 `decode_gqa_attention_v50.cuh` values 경로.

묶음: (1) 요청별 softmax 상태 live range 분석, (2) 재사용 이득과 register/shared-memory 증가를 함께 계산하는 후보 표, (3) 원래 recurrence·BF16 probability 경계를 보존할 수 있는 설계 한 개 선택. 단순 요청 수 확대나 출력 폭 확대는 앞선 실패를 반복하므로 제외한다.

완료: 값 로드 감소량뿐 아니라 활성 CTA 감소·barrier·scratch 비용과 수치 계약을 설명해야 한다. 하드웨어 counter 권한이 없으면 SASS/정적 자원/native timing으로 확인 가능한 범위까지만 주장한다. 실제 HBM 대역폭 결론은 보류한다.

### PR C — 통계적 근사의 별도 feasibility 문서

대상: bounded-logit/polynomial/random-feature 방법의 조건과 적용 불가능한 경우.

묶음: (1) 논문 가정·오차 상한 정리, (2) 실제 모델의 logits 범위와 adversarial 범위 비교 설계, (3) 정확도 계약 및 fallback 비용 정의.

완료: 현재 strict 경로의 대체 후보인지, 별도 품질 예산이 필요한 연구인지 명확히 판정한다. strict 기준을 변경하지 않는다. 위상수학은 기존 attention 계산을 보존하면서 CUDA 비용을 줄이는 구체적 연결이 확인될 때만 후보로 올린다.

## 검증 및 rollback

현재 승격된 dense-wire baseline을 유지한다. 연구 실패도 원문·가정·patch·측정값을 보존한다. 실제 변경을 선택했을 때만 related optimization batch를 구현하고 correctness → native timing → 동일 조건 serving 순서로 평가한다. 충분한 단계마다 prior/vLLM throughput·TTFT·TPOT·P95/P99 및 실패율을 표로 공개한다. 현재 owner-index 후보의 GPU serving 결과는 없으며 메모리 사전 조건 실패를 성능 수치로 취급하지 않는다.
