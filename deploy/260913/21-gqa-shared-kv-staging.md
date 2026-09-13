# PR21 — GQA head 그룹의 K/V shared staging

상태: 격리 native 구현·4090 GPU gate 및 SM90a/SM100a compile 완료. Retained model/server opt-in 연결 및 C8/C32/C64 serving 검증 완료, 개선 미확인으로 승격하지 않음. 기본값 변경 없음.

## 선택 근거

[현재 workload profile](../../benchmarks/results/20260914-cache-residency-profile/README.md)의 고유 prompt graph span 중 prefill/mixed가83.55%다. [Rolling serving](../../benchmarks/results/20260914-rolling-serving/README.md)은 decode overlap의 효과를 확인했지만 고유 prompt와 높은 concurrency의 vLLM 격차는 남았다. 같은 query tile의 확대는 native 일부 shape의9–11% 개선에도 serving이 거의 그대로였으므로 반복하지 않는다.

현재 `mixed_attention_v49.cuh`는 하나의8-query task·query head마다32-thread CTA를 사용한다. SmolLM2의 query head9/KV head3에서 같은 query range의 세 head가 동일 K/V를 각각 읽는다. 이는 source상 중복 load이며 실제 HBM traffic 계측은 아니다. Cache가 이미 흡수할 수 있으므로 shared staging이 더 빠르다고 가정하지 않는다.

[FlashAttention-2](https://arxiv.org/abs/2307.08691)의 warp 작업 분할 및 [CUDA asynchronous copy](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)를 적용 근거로 한다. 논문의 kernel·성능 수치를 그대로 재현하거나 FlashInfer 라이브러리를 사용한 구현은 아니다. Rust → C ABI → CUDA 경계를 유지한다.

## 하나의 optimization batch

1. 동일 KV head의 세 query head를96-thread CTA에 묶고 각 warp는 독립 softmax·output accumulator를 소유한다. Query tile8과 기존 metadata task map은 유지한다. Grouping-only 대조군을 함께 둔다.
2. Paged K/V의128-token tile을16-byte `cp.async`로 shared memory에 전달한다. BF16 저장과 page mapping, 역순128-token softmax recurrence, K16 MMA 순서, 확률 BF16 rounding을 보존한다. 한 buffer를 사용하므로 다음 tile과의 copy/compute overlap을 구현했다고 주장하지 않는다.
3. 세 warp가 동일하게 barrier에 참여한다. Causal tail은 zero-fill하며 K/V parent 범위 밖 주소를 만들지 않는다. Nonfinite V는 CTA 전체가 기존 ordered 경로로 다시 계산하도록 한다. 짧은 prefill·decode는 기존 compact body로 처리한다. 출력·coverage·captured replay·failure cases를 대조한다.
4. Native 통과 및 대표 serving shape에서 의미 있는 이득이 있으면 retained owner/catalog identity와 opt-in server 선택으로 연결한다. Full-model logits/greedy·stop/cancel·shared cache 검증 후 같은 rolling 기준과 vLLM을 C8/C32/C64 shared/unique로 비교한다. Native 회귀 시 serving 연결을 강행하지 않는다.

## 자원 예산과 검증

SM89 ptxas: staging release152 registers/thread,45,056B shared,spill0; grouping-only127 registers,12,288B shared,spill0. Shared staging은 bandwidth 절감 가능성과 occupancy·barrier 비용을 교환한다. 런타임 occupancy와 시간을 실제 측정한다. 이 후보는3warp CTA이며 기존1warp와 CTA 수만 비교하지 않는다.

기존17 shape, ragged31/32/33/47/49, context4096, scrambled pages, nonfinite V 및 captured replay/coverage 검사를 사용한다. Native memcheck/racecheck는 bounded fixture로 실행한다. 여러 prefill owner 및 추가 K/Q nonfinite 입력은 별도 보강하고 single-prefill 합성 시험을 전체 serving 증거로 대체하지 않는다. SM90a/SM100a compile과 실제 runtime 검증을 구분하며 장비 부재만 skip한다.

Blender 세션은 GPU gate 동안만 종료하고 결과 성공·실패에 관계없이 복구한다. `3d.fin-ally.net`, `3dsol.fin-ally.net`, `3dfable.fin-ally.net`의 정적 웹 서버는 계속 유지한다.

## 완료·롤백

GPU gate, 모델 수치, 실제 serving 비교 전 승격하지 않는다. 롤백은 기존 mapped/compact path이며 이 후보는 명시적 opt-in에서만 선택한다. Multi-GPU/Hopper/Blackwell 전문 backend는 별도 capability로 확장하며 이 SM89 fixture를 일반 모델 지원으로 표시하지 않는다.

## Native 판정

[최종 native gate](../../benchmarks/results/20260914-gqa-staged-native/README.md):171개 audit와171개 graph 비교가 bitwise 일치했고 memcheck/racecheck 각45개 검사에서 오류0이다. 여러 prefill owner를 포함한8개 shape의 graph 시간은2.30–16.90% 감소했다. Grouping-only는 모두 회귀했다. 이 native 결과만으로 승격하지 않았으며 후속 full-model 및 serving 판정은 아래에 기록한다.

## Retained model 및 serving 통합

별도 C ABI recorder와 Rust factory, source-bound graph identity, `RILEY_MIXED_GQA_STAGING=1` 선택을 연결했다. Query reuse 등과 동시 선택은 거부하며 기존 ordinary/future pure-decode 및 rolling reservation 경로는 유지한다. [모델 gate](../../benchmarks/results/20260914-gqa-staging-model/README.md)의2,359,296 BF16 logits bytes가 일치했고 whole-model memcheck 오류0이다. 로컬 server105개, CUDA wrapper host95개 검사가 통과했다.

[C32 serving](../../benchmarks/results/20260914-gqa-staging-serving/README.md)에서 shared/unique throughput은 기존 rolling Riley 대비 각각−1.51%/−0.95%다. Native 개선이 serving으로 이어지지 않아 승격하지 않는다. 프로파일 없이 occupancy나 데이터 분포를 확정 원인으로 쓰지 않는다. 다음 분석은 동일 binary의 control/GQA 실제 모델 graph를 비교해 attention 및 나머지 stage의 시간 변화를 분리한다. Unqualified 경로이며 전체 vLLM 성능 목표는 미달이다.

[C8/C32/C64 통합 표](../../benchmarks/results/20260914-gqa-staging-serving/README.md#c8c32c64-milestone)를 확정했다. 후보의 기존 Riley 대비 throughput 변화는 shared에서 C8 −1.00%, C32 −1.51%, C64 −0.57%; unique에서 +0.49%, −0.95%, −1.45%다. C8 shared의 vLLM 우위는 기존 rolling 경로에도 있어 이 후보의 개선으로 해석하지 않는다. C8 unique token interval P95 회귀도 유지 기록한다. 전체 12,288 요청과 stop/cancel/recovery 각312건은 통과했지만 장기 부하 및 다른 모델·하드웨어 검증은 미완료다.
