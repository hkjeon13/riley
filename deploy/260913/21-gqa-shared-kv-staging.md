# PR21 — GQA head 그룹의 K/V shared staging

상태: 격리 native 구현·4090 GPU gate 및 SM90a/SM100a compile 완료. Serving 미연결·기본값 변경 없음.

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

Blender 세션은 GPU gate 동안만 종료하고 결과 성공·실패에 관계없이 복구한다. `3d.fin-ally.net`의 정적 웹 서버는 계속 유지한다.

## 완료·롤백

GPU gate, 모델 수치, 실제 serving 비교 전 승격하지 않는다. 롤백은 기존 mapped/compact path이며 이 격리 후보는 production에서 참조하지 않는다. Multi-GPU/Hopper/Blackwell 전문 backend는 별도 capability로 확장하며 이 SM89 fixture를 일반 모델 지원으로 표시하지 않는다.

## Native 판정

[최종 native gate](../../benchmarks/results/20260914-gqa-staged-native/README.md):171개 audit와171개 graph 비교가 bitwise 일치했고 memcheck/racecheck 각45개 검사에서 오류0이다. 여러 prefill owner를 포함한8개 shape의 graph 시간은2.30–16.90% 감소했다. Grouping-only는 모두 회귀했다. Full-model 및 serving은 미검증이므로 기본값은 유지하고 명시적 retained-model backend 연결로 진행한다.
