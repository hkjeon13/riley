# PR 05 — 메모리 수명 기반 FFN tile 실행

상태: **현재 serving trace로 영역 선정 완료 / kernel batch 구현 전**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

gate/up→activation→down 사이 중간 global tensor 이동을 줄인다. launch 수 감소만을 최적화 목표로 삼지 않는다.

## 의존성과 변경 위치

선행: 01.

예상 수정 위치: riley-runtime layer workspace/plan, kernels FFN backend, full-model benchmark. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. FFN DAG와 tensor의 마지막 consumer·수명을 명시한다.
2. 분리 실행과 tile 전달 fusion의 두 plan을 구현한다.
3. 중복 producer 계산·weight read·shared/register 용량을 포함한 후보 선택을 구현한다.
4. shape별 dispatch와 workspace 재사용을 통합한다.

## 범위 경계

범용 GPU compiler 자체 개발, QKV chain까지 동시 재작성, 전 layer persistent scheduler는 제외한다.

## Correctness·수명 계약

부동소수점 연산 순서 변경은 01의 신규 수치 계약을 따른다. 중간 buffer alias는 마지막 GPU 사용 이후에만 허용한다.

## 검증과 하드웨어 skip

작은/큰 M, 전 layer working set, cold/warm 조건을 구별한다. split/merge 오차, spill·재계산, weight traffic을 검증한다. counter가 없으면 논리 byte 추정과 실측을 구분한다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

동일 shape micro 결과 이후 full-model·serving 개선을 확인한다. V56의 일부 row 회귀처럼 shape별 손실이 있으면 범위를 제한하거나 reject한다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

분리 FFN plan을 기본값으로 되돌린다.

## 연구 근거

[Welder](https://www.usenix.org/conference/osdi23/presentation/shi), [MCFuser](https://arxiv.org/abs/2506.22169), [FLUTE](https://arxiv.org/abs/2407.10960). 논문 성능 배수는 Riley의 예상 개선율이 아니다.

## 실제 코드·trace 기반 첫 batch 확정

[현재 바이너리 영역 분석](../../benchmarks/results/20260913-serving-area-census/README.md)에서 V7 decode kernel 시간의32.90%, prefill/mixed의27.81%가 FFN gate/activation/down이다. V7의 gate/up/SwiGLU 및 merge/norm fusion은 이미 구현되어 있다. 첫 batch는 전체 FFN을 한 CTA로 합치는 대신 다음 세 변경을 함께 구현하고 검증한다.

1. `decode_gate_v56.cuh`와 별도의 후보 backend에 gate/up 두-stage global→shared pipeline을 구현한다. 64-wide K tile에서32-row input4,096B와 gate/up8-column weights2,048B를 stage마다 보관한다. 두-stage 논리 예산은12,288B다. 두 warp가 weight tile을 공유하되 기존 순차 MMA, gate/up BF16 round, exp/SwiGLU 순서를 보존한다.
2. down projection에도 같은 pipeline 소유·동기화 규칙을 적용한다. stage는 input4,096B + weight1,024B, 두-stage10,240B다. 기존320-wide 분할, 마지막256-wide chunk, partial의 BF16 반올림 후 FP32 저장과 merge 순서를 유지한다. 마지막 tile에서 기존에 수행하지 않은 MMA를 임의로 추가하지 않는다.
3. 두 kernel을 함께 선택하는 명시적 native/runtime FFN backend와 graph signature를 연결한다. default V7는 그대로 두고 shared extent·register/spill·active-row 조건을 기록한다. 처음에는 decode에 한정하되 mixed/pure 전환 시 full logits 및 수치 일관성을 검사한다. Prefill backend 확장은 같은 자료로 별도 승격한다.

비동기 copy와 MMA overlap은 [NVIDIA CUDA programming guide의 asynchronous copies](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#asynchronous-data-copies) 및 [CUTLASS SM80 copy primitive](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/memory_sm80.h)를 근거로 한다. 같은 primitive를 사용한다고 library의 GEMM 전체 구현을 사용했다고 표기하지 않는다. CUDA80+ 공통 경로이며 Hopper/Blackwell 최적화는 capability별 backend 확장을 유지한다. 현재 표의 byte 수는 계획 예산이며 ptxas·실제 실행 전 occupancy나 속도 개선을 주장하지 않는다.

모든 CTA thread는 copy commit/wait와 shared reuse barrier에 참여해야 한다. row17 미만일 때 두 번째 warp가 기존처럼 조기 return하면 안 된다. inactive row는 zero-fill하고, source alignment 및 경계 밖 pointer를 검사한다. runtime Python 호출은 없다.

검증: active1/15/16/17/31/32, K-tail, 반복 graph replay의 row 갱신, pure/mixed 전환, full-model logits·greedy·free generation, native memcheck/racecheck. 기존 MMA 순서를 유지해도 bitwise 결과는 실측으로 확인한다. register/shared spill로 이득이 사라지는 경우 원인을 기록하고 채택하지 않는다. 두 후보 kernel과 serving 선택이 연결된 시점에 기존 V7 / 새 FFN / vLLM의 자연·고정 workload C16/C32 교차 비교표를 갱신한다. kernel별 수정마다 serving 표를 반복하지 않는다.
