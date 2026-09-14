# PR 06 — Stream-K·LeanAttention 방식 작업 분할

상태: **attention split 계획 미구현 / 후속 adaptive projection 모델 통합·측정 완료, 기본값 비승격**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

긴 context 또는 output tile 수가 적은 decode에서 GPU 작업 불균형을 줄인다.

## 의존성과 변경 위치

선행: 03.

예상 수정 위치: attention backend plan, kernels reduction/scratch, runtime graph bucket. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. context·batch별 split 작업 계획을 만든다.
2. online softmax partial 결합과 scratch ownership을 구현한다.
3. split/unsplit 선택 및 graph bucket을 연결한다.
4. 실제 serving shape 분포를 평가 harness에 반영한다.

## 범위 경계

GEMM 전체를 동시에 Stream-K로 바꾸거나 짧은 context에도 split을 강제하지 않는다. small-M GEMM은 이 PR 결과 후 별도 후보로 남긴다.

## Correctness·수명 계약

reduction 결합은 수학적 equivalence와 bitwise equivalence가 다르므로 01 수치 계약 적용. padding·빈 split이 결과에 섞이지 않아야 한다.

## 검증과 하드웨어 skip

짧은/긴 context, ragged batch, 최대 길이, split 수 변경의 수치 검증. merge 비용과 scratch 증가 포함. 4090 지원 경로 실행; 장비 특화 경로 조건부 skip.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

long-context serving 이득과 short-context fallback의 회귀 없음이 필요하다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

unsplit backend로 복귀한다.

## 연구 근거

[Stream-K](https://arxiv.org/abs/2301.03598), [LeanAttention](https://arxiv.org/abs/2405.10480). 논문 성능 배수는 Riley의 예상 개선율이 아니다.

## 후속 범위 선택 — 2026-09-14

[C8/C16 serving 및 C8 profile](../../benchmarks/results/20260914-decode-capacity-serving/README.md)에서 낮은 concurrency의 decode 비중과 고정 32행 projection의 불필요한 MMA tile 계산을 확인했다. 기존 계획의 small-M GEMM 후속 후보로 QKV·attention 출력 projection·FFN down을 하나의 적응형 16/32행 batch로 묶는다. 이는 Stream-K 또는 LeanAttention의 split reduction을 구현했다고 주장하는 범위가 아니며 기존 attention split 계획을 대체 완료하지 않는다.

[Native gate](../../benchmarks/results/20260914-decode-adaptive-native/README.md): device active count로 1/2 tile을 선택하는 격리 후보, active0..33 graph102건 bitwise partial 일치, memcheck/racecheck, SM89 실행 및 SM90a/SM100a compile 통과. 30개 weight 집합의 세 연산 시간은 active8 약−21%, active16 약−14%였다. 모델·serving에는 아직 연결되지 않았다.

다음 묶음은 ordinary/paired future decode 연결, source/profile identity, full-model 자유 생성·logits parity, C8/C16 및 C32/C64 serving 비교다. K reduction·BF16 partial rounding·scratch stride를 유지한다. Native 시간 비율을 serving 개선율로 환산하거나 기본값을 승격하지 않는다.

### Adaptive 모델·serving 결과

[통합 및 비교표](../../benchmarks/results/20260914-adaptive-decode-serving/README.md): 위 후속 묶음을 구현했다. 자유 생성3,616토큰·자연어12,582,912 BF16 logits가 baseline과 일치했고 paired terminal/cancel·자원 회수가 통과했다. 실제 ordinary472개/future430개 graph에서 각각 새 커널90개를 확인했다. C8/C16 throughput은 직전 대비 약4% 개선됐지만 C32+0.64%, C64−0.06%이고 고부하 P99도 높아 기본값으로 승격하지 않는다. 동일 projection family의 작은 variant를 이어가기보다 PR10의 prefix ownership/transfer를 다음 구조 영역으로 선택한다. 기존 cache-off 격차와 attention split 미구현 범위는 남아 있다.


## Attention 분할 착수 전 계약 확인 — 2026-09-14

`decode_gqa_attention_v50.cuh`는 이미 QK context tile 분할과 V 출력8개 block 분산을 구현한다. 이것을 새 LeanAttention 구현으로 재포장하지 않는다. `planned_softmax_values.cuh`의 normalize-once 접근도 이미 native 성능 실패로 중단됐다([기록](../../benchmarks/results/20260913-attention-task-costs/README.md)).

[LeanAttention v2 본문](https://arxiv.org/html/2405.10480v2)의 부분 online-softmax rescaling은 수학적 결합 성질을 사용한다. 현재 Riley의128-token 역순 recurrence와 각 tile probability BF16 반올림은 별도 계약이다. 독립 split에서 local maximum으로 반올림한 뒤 재스케일하면 원래 running maximum으로 먼저 정규화해 반올림한 값과 달라질 수 있다.

이를 실제 기존 CUDA kernel과 비교하는 `attention_split_numerics_probe.cu`를 추가했다. Q dim0=1, K dim0=51/64로 실현 가능한 score51/512, 두128-token tile과 V1/0을 사용한다. Host 계산에서는 기존 결과0.4765625와 독립 split 결과0.474609375로 달랐다. 이는 CPU rounding 메커니즘 예시이며 GPU 검증을 대체하지 않는다. Probe는1-tile/동일max 대조군, 두 replay, inactive sentinel을 포함한다. 예상 mismatch 확인은 기존 backend와의 compatibility **거부**를 뜻하며 backend correctness pass나 성능 향상이 아니다. GPU·sanitizer 검증 전 결과를 확정하지 않는다.

이 gate의 목적은 무효한 exact-profile 통합을 미리 막는 것이다. 실제 LeanAttention 라이브러리 전체를 구현하거나 검증한 것이 아니며, 논문 오류를 주장하지 않는다. 별도 numerical backend를 도입하려면 결과를 보기 전에 새 profile과 모델 수준 기준을 명시해야 하고 기존 profile의 실패를 tolerance 완화로 덮지 않는다.

[GPU 확인 결과](../../benchmarks/results/20260914-attention-split-compatibility/README.md):3개 대조군은 일치했고 두-tile counterexample의576개 BF16 출력이 달랐다(0.4765625 vs0.474609375). 두 graph replay/비활성 sentinel,memcheck/racecheck0,SM90a/SM100a compile 통과 및 Blender 복구 확인. Legacy exact profile과의 독립 partial merge 호환성은 **REJECT**다. 성능 검증이나 LeanAttention 전체 구현 완료를 의미하지 않으며 PR06은 별도 numerical profile 계약이 필요한 미완료 작업으로 유지한다.
