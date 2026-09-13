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
