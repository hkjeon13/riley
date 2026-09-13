# PR 05 — 메모리 수명 기반 FFN tile 실행

상태: **decode FFN 모델·serving 검증 완료 / 실험 옵션 유지·미승격**. 공통 계약은 [README](README.md)를 따른다.

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

## Native batch 구현 결과

[FFN pipeline 검증](../../benchmarks/results/20260913-ffn-pipeline-native/README.md): gate/up과 down 두 kernel을 이중 shared stage로 구현했다. 24 graph replay 경우에서 기존 kernel의 전체 gate 출력·down partial과 bitwise 일치했고, memcheck0 errors / racecheck0 hazards다. 실제 shared 예산은12,288B/10,240B, register38/30, spill0이다. SM89 실행 및 SM90a/100a AOT 빌드는 통과했지만 latter runtime은 장비 부재 skip이다. 명시적 runtime/backend·graph identity·full-model·serving 연결이 다음 필수 단계이며, 아직 성능 개선이나 PR05 완료를 주장하지 않는다.

## 모델·serving 통합 결과

[최종 첫 batch 비교표](../../benchmarks/results/20260913-ffn-serving-screen/README.md): 독립 FFN recorder·graph fingerprint·loopback CLI 옵션을 연결했다. 자유 생성1,024토큰 및 자연어12,582,912 BF16 logits가 V7과 일치하며 full-model memcheck0 errors다. C16/C32 고정/자연어24-run screen의 throughput 이득은 V7 대비2.20–3.64%다. C32 natural은 vLLM보다 throughput9.22% 낮고 median TPOT21.21% 길어 최종 목표 미달이다. 기존 경로를 기본값으로 유지하며, prefill tile 전달·일반 shape 확장·장기 안정성은 완료되지 않았다. 다음 주요 영역은 반복적인 FFN 미세 튜닝보다 prefill/mixed attention·자원 정책으로 둔다.


## 다음 batch: prefill FFN shared staging과 copy/MMA pipeline

[현재 C32/C64 matched-request profile](../../benchmarks/results/20260913-load-shape-profile/README.md)에서 paired의 prefill/mixed는 두 부하 모두 graph envelope 시간의 약53%였다. C64 선택 구간의 기존 fused gate/up와 down projection은 각각131.24/139.04ms다. 다음 pair의 scheduler 예약 확장보다 이 GPU 실행 영역을 먼저 검증한다. 이전 decode-only pipeline의 작은 이득을 prefill 성능으로 간주하지 않는다.

하나의 구현 batch:

1. 별도 `prefill_ffn_pipeline.cuh`에16-row CTA 입력 공유와 두-stage global→shared asynchronous copy를 넣은 gate/up를 구현한다. 기존 `prefill_fused_gate_v51.cuh`의4-warp 출력 분할, K16 MMA 순서, BF16 gate/up rounding과 SwiGLU 순서를 유지한다. K64 stage 기준 입력2,048B + 두 weight8,192B, 두-stage 합20,480B는 논리 계획이며 ptxas/occupancy로 확인한다. 기존 gate/up fusion은 재구현 목표가 아니다.
2. 같은16-row 전달 규약을 down projection에 적용한다.2-warp/K64 stage 기준 입력2,048B + weight2,048B, 두-stage8,192B다. 기존 K320 구간별 BF16 rounding 및 마지막K256 누적 순서를 보존한다. CTA 전체가 copy completion과 shared reuse barrier에 참여하고 inactive M row는 안전한 zero-fill을 사용한다.
3. Native entry/profile·mixed prefill graph identity·retained workspace 수명·Rust opt-in 선택을 함께 연결한다. Pure decode와 기존 successor-overlap 경로는 보존하고 mixed/prefill 경로의 두 FFN kernel만 선택한다. 기존 FFN decode backend와 혼동하지 않는다. Backend 선택/identity와 실제 모델 연결 전 native gate가 필요하다.

검증은 live M1/15/16/17/31/32/64/128/398/512 및 capacity padding1024, row 갱신을 동반한 반복 graph replay, inactive 출력 미변경, BF16 bitwise 대조, native memcheck/racecheck, ptxas register/shared/spill, 전체30-layer 모델 logits/greedy·natural reference/stop/cancel 순서다. Native가 회귀하면 원인을 기록하고 모델 승격을 하지 않는다. 통과한 후보는 현재 single/직전 paired/새 paired/vLLM의 동일 natural serving을 C32 및 client C64/active32 두 순서로 비교한다. 최종 판정에는 TTFT/TPOT/P95/P99와 오류율을 포함한다.

이는 CUDA asynchronous-copy 및 PR05에 이미 조사한 memory-lifetime 실행 방식의 prefill 적용이다. Counter 접근이 없으면 논리 byte 수를 실제 HBM 절감량으로 표기하지 않는다. SM89 실행과 SM90a/SM100a compile을 구분하고 후자의 runtime은 장비 부재를 명시한다. 기존 수치 gate를 낮추지 않으며 실제 실패를 skip으로 바꾸지 않는다. 기본값 승격은 serving 검증 후 별도 판정한다.


### Prefill FFN native gate 결과

[Native 구현 및 대조](../../benchmarks/results/20260913-prefill-ffn-pipeline-native/README.md): gate/up와 down 두-stage copy/MMA pipeline,16-row 입력 공유, shared-bank alias를 피하는72-element stride를 구현했다. 초기64 stride 회귀를 보존하고 padding 후26개 전체 출력/반복/inactive 검사에서 bitwise 일치, native memcheck/racecheck0, SM90a/SM100a compile 통과를 확인했다. 실제 shared는 gate20,992B/down8,704B이며 spill0이다. 두 target runtime은 장비 부재 미검증이다.

Warm one-layer native 합산 시간은 M32 약−19.4%, M128−15.7%, M512−6.6%지만 M398−0.38%, M1024−1.25%는 거의 동률이다. 이를 serving 개선으로 사용하지 않는다. **이 native gate 시점에는 Rust/model recorder/serving backend에 연결하지 않았다.** 다음은 독립 profile/catalog identity와 retained model 연결, full-model gate, C32/C64 single·직전·후보·vLLM 비교다. 기본값과 기존 serving binary는 유지한다.


### Prefill FFN 모델·serving 통합 결과 (2026-09-14)

[통합 결과와 비교표](../../benchmarks/results/20260913-prefill-ffn-model-serving/README.md): retained C ABI recorder, 별도 graph fingerprint, Rust session 및 loopback CLI 선택을 연결했다. Pure decode를 유지하며 paired execution과 결합할 수 있다. 자유 생성1,024토큰 및 자연어12,582,912 BF16 logits가 기존 경로와 일치하고 stop/cancel·C32/C64 두 순서 serving도 통과했다.

현재 paired 대비 throughput은 C32 +2.57%, C64/active32 +1.69%다. 같은 screen vLLM 대비 각각−6.43%/−5.90%이며 E2E P99는 직전 대비 악화돼 기본값으로 승격하지 않는다. Native 이득을 serving 이득으로 과장하지 않는다. PR05의 일반 shape 및 장기 안정성은 남아 있으며 다음 영역은 PR04 mixed batching·시간 예산 실행 계약과 평가 workload 검토다.
