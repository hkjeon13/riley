# 행렬·GPU·수치 알고리즘 연구 후보 — 2026-09-14

상태: 1차 논문 초록 및 기존 코드 대조로 후보를 선별했다. 본문·공개 구현·수치 조건의 상세 검토 및 구현 성능은 아직 미검증이다. 엔진 기능명에 한정하지 않고 GPU 자원 불균형, 통신 복잡도, 선형대수와 통계적 근사를 탐색한다. 아래 우선순위는 연구 순서이며 승격 결정이 아니다.

## 구체적인 검색과 적용 가설

| 검색어 | 1차 자료 | Riley에 연결할 가설 / 다음 확인 |
|---|---|---|
| attention GPU softmax special function units asymmetric scaling conditional rescaling | [FlashAttention-4](https://arxiv.org/abs/2603.05451) | GEMM보다 느리게 확장되는 exponential/shared-memory 자원을 고려한다. `mixed_attention_v49.cuh`의 softmax 및 rescale 경로를 대조하고, 연산 생략 조건의 부동소수점 동등성을 먼저 검토한다. Blackwell 파이프라인은 별도 backend 대상이다. |
| asynchronous softmax unified maximum flat GEMM double buffering | [FlashDecoding++](https://arxiv.org/abs/2311.01282) | 작은 M의 GEMM에 대형 타일·padding이 적합한지 조사한다. Softmax unified-max는 분포 가정·overflow·fallback을 본문에서 확인하기 전 적용하지 않는다. |
| attention communication complexity IO lower bound SRAM head dimension | [The I/O Complexity of Attention](https://arxiv.org/abs/2402.07443) | 데이터 이동을 줄일 수 있는 한계와 현재 구현의 간극을 구분한다. 이론적 2단계 메모리 모델을 paged GQA·cache hit·짧은 요청에 그대로 적용하지 않는다. 소스의 중복 load만으로 HBM 병목이라고 단정하지 않는다. |
| online softmax associative reduction numerical stability exact attention memory | [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682) | 중간 attention 행렬을 저장하지 않는 원리는 이미 사용 중이다. 추가 이득은 reduction 배치·상태 보관 방식에 있어야 한다. 실수 대수의 결합법칙과 BF16/FP32 실행 동등성을 구별한다. |
| GPU warp specialization asynchronous matmul softmax TMA pipeline | [FlashAttention-3](https://arxiv.org/abs/2407.08608) | 한 번의 shared staging에서 더 나아가 copy/MMA/softmax 중첩을 검토한다. 과거 GQA staging serving 회귀와 비교해 barrier·register·occupancy 비용까지 설명해야 한다. |
| attention outlier smoothing per-thread quantization accumulation error | [SageAttention2](https://arxiv.org/abs/2411.10958) | Q/K 분포와 누적 오차를 다루는 아이디어는 저정밀 연구 후보다. 논문의 작은 품질 손실은 strict greedy 일치 증거가 아니다. BF16 경로 대체가 아닌 별도 precision 계약 대상으로 둔다. |
| softmax kernel positive orthogonal random features estimator variance | [Performers](https://arxiv.org/abs/2009.14794) | 통계적 feature 근사는 계산량을 줄일 수 있지만 원래 attention을 추정한다. 현 strict 기준의 drop-in 후보로 승격하지 않는다. 향후 품질 예산을 둔 연구에 분리한다. |
| attention persistent homology topological features inference | [Topological Attention](https://arxiv.org/abs/2107.09031) | 이번에 확인한 자료는 시계열 모델에 위상 특징을 추가하는 연구다. 기존 LLM 행렬 연산 가속 근거로 연결되지 않는다. 위상수학 전체가 무용하다는 결론이 아니라 이 자료의 적용 우선순위가 낮다는 판정이다. |

## 첫 번째 상세 검토 대상

1. **Softmax와 MMA의 자원 불균형:** FA4/FA3 본문과 공개 kernel의 producer/consumer 의존성을 읽고 현 코드와 비교한다. 기존 수치 recurrence를 보존한 실행 중첩과 수치 결과를 바꾸는 근사 exponential을 분리한다.
2. **작은 M 행렬 실행:** FlashDecoding++의 flat GEMM을 현재 FFN/projection shape와 비교한다. Double buffering, shape별 tile 선택, intermediate fusion을 연관된 batch 후보로 묶되 기존 미세 threshold 실험을 반복하지 않는다.
3. **수치 변환의 검증:** 이전 context-split의 strict 출력 불일치를 실패 사례로 유지한다. 수학적으로 같은 식도 누적 순서·rounding 때문에 다른 출력이 될 수 있으므로 native oracle → 실제 model logits/greedy → stop/cancel → serving 순서로 검증한다.

각 상세 검토는 원문 식/조건, 대상 코드, 예상 절약 자원, 추가 비용, 기각 조건, 하드웨어별 실행 가능 여부를 기록한다. 논문의 최대 kernel 배속을 serving 예상 배속으로 옮기지 않는다. 현재 C8/C16/C64 측정 중에는 추가 GPU profiling/build를 실행하지 않는다.

Runtime은 Rust → C ABI → CUDA를 유지한다. FA4의 Python/CuTe-DSL 구현을 런타임에 직접 가져온다고 가정하지 않는다. Native 이식 또는 독립적으로 검증된 사전 컴파일 연결이 가능한지 별도로 확인한다. Hopper/Blackwell 실행 장비가 없으면 해당 실행 테스트만 명시적으로 미검증으로 남긴다.

## FA4 본문과 현재 softmax 대조

[본문 §3.1.2–3.1.4](https://arxiv.org/html/2603.05451v1) 확인 결과, 세 기법을 분리해야 한다.

- **Pipeline:** TMEM을 사용하는 Blackwell 전용 구조이며 현재 SM89 kernel의 단순 치환 대상이 아니다. Correction 작업 분리의 의존성 설계를 향후 native backend에 반영할 후보로 둔다.
- **Polynomial exponential:** 일부 exponential을 FMA로 옮기지만 근사 오차와 추가 register 비용이 있다. BF16 오차 통계가 비슷하다는 사실은 개별 결과의 bitwise 일치를 뜻하지 않는다. 현재 strict 경로에 바로 적용하지 않는다.
- **Conditional rescaling:** 최대값이 변하지 않는 경우와, 양의 slack으로 최대값 갱신을 늦추는 경우를 구분한다. 후자는 probability rounding과 FP32 누적 결과가 달라질 수 있다. 전자의 항등 연산 생략도 NaN·무한대·subnormal 및 컴파일러 동작을 확인해야 한다.

현재 `mixed_attention_v49.cuh`는 scalar 경로와 2-row 경로에서 alpha를 계산하고 매 tile accumulator를 곱한다. 따라서 **기존 최대값·tile 순서를 유지하는 조건부 correction**, **동일 순서 안에서 독립 연산을 앞당기는 scheduling**, **중간값 live range 축소**를 하나의 연구 batch 후보로 좁힌다. 이는 아직 구현이나 개선 증거가 아니다. 먼저 실제 shape에서 alpha=1 빈도와 correction instruction 비용을 측정해야 하며, 낮으면 이 batch를 기각한다. 별도 bounded native 계측으로 시작하고 현재 serving matrix에 계측을 섞지 않는다.

## Concurrency 검증의 환경 중단

`dense-wire-matrix-v1`은 C8 unique prior 완료 후 다음 lane 시작 전 48°C cooldown의 120초 제한으로 실패했다. 후보 서버 실행 오류로 해석하지 않는다. Lifecycle은 serving exit 1 및 Blender 세 개의 복구 성공을 기록했다. C16/C64는 시작하지 않았다.

실패 spool과 lifecycle을 보존하고 `dense-wire-matrix-v2`에서 전체 순서를 다시 실행한다. 시작 온도 기준 48°C, workload, binary, warmup 및 retained 수는 유지한다. 대기 한도만 600초로 늘리고 lane별 온도 시계열을 기록한다. 측정 결과가 없는 lane을 성공으로 표시하거나 기존 부분 결과와 새 실행을 합치지 않는다.
