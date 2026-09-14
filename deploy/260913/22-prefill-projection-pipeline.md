# PR22 — Prefill projection operand pipeline

상태: 실제 model profile 기반 구현 대기. 기본 경로는 기존 rolling Riley이며 GQA staging은 비활성이다.

## 근거와 범위

[현재 control/GQA profile](../../benchmarks/results/20260914-gqa-staging-profile/README.md)에서 unique prefill/mixed graph 시간의 25.65%가 Q/K/V 및 attention-output projection이다. GQA attention 시간 감소는 선택 구간에서 5.08%이며 전체 graph 감소는 1.57%에 불과하다. Native 수치를 serving으로 확대하지 않고 다음 의미 있는 영역으로 이동한다.

`kernels/src/prefill_shape_projection.cuh`는 각 K16 MMA에서 A/B를 global memory로부터 직접 읽는다. Q/K/V는 K192, output은 K128마다 BF16 중간 rounding을 수행한다. 일반 GEMM의 단일 FP32 accumulation으로 교체하면 동일 계산 계약이 아니므로 cuBLAS 호출로 무조건 대체하지 않는다. 기존 FFN pipeline은 이미 별도 구현되어 있어 중복 작업하지 않는다.

[CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)의 계층형 tile 재사용과 double-buffered mainloop를 적용 근거로 한다. CUTLASS는 shared tile과 register fragment의 이중 버퍼링을 설명한다. 이 계획은 해당 원리를 기존 수치 recurrence에 적용하며 CUTLASS 라이브러리 전체를 사용했다고 표시하지 않는다.

## 하나의 optimization batch

1. Q/K/V/output projection의 weight를 모델 준비 시 한 번 tile layout으로 pack하고 retained owner 수명·byte budget·graph identity에 포함한다. Request 경로에서 pack/allocation을 하지 않는다. 행렬별 N/K 및 기존 output storage는 유지한다.
2. 공통 prefill projection mainloop에 A shared reuse와 A/B의 두 K64 buffer, 비동기 copy와 MMA overlap을 함께 구현한다. Shared stride는 bank mapping과 실제 resource 보고로 확인한다. K16 MMA 순서와 K192/K128 BF16 rounding·합산 순서는 보존한다. Invalid/inactive row는 안전한 source로 zero-fill하고 output을 쓰지 않는다.
3. 네 projection을 하나의 명시적 backend로 선택하고 dynamic live-row graph replay, 기존 fallback 및 source-bound identity를 연결한다. Pure decode와 다른 attention 선택을 변경하지 않는다. Native-only, full-model, serving 판정을 분리한다.

## 검증·승격·롤백

Native에서는 N576/K576/interval192, N192/K576/interval192, N576/K576/interval128을 함께 검사한다. M1/8/16/17/31/32/64/128/398/512, capacity padding, live-row 변화와 inactive sentinel, bitwise full output, bounded memcheck/racecheck, register/shared/spill 및 multi-layer weight 순환 측정을 포함한다. 준비 pack 비용·추가 상주 byte도 보고한다. 하나의 warm microbenchmark만으로 모델 연결을 승격하지 않는다.

Native 통과 후 기존 logits/greedy gate를 그대로 적용하고 현재 frozen rolling Riley 및 같은 binary control과 C8/C32/C64 shared/unique serving을 vLLM과 비교한다. TTFT/TPOT/E2E P95/P99 및 실제 SSE token-interval tail, stop/cancel/recovery와 오류를 함께 기록한다. 수치 실패의 tolerance를 바꾸지 않는다. 이득 미확인 시 기본값은 유지하고 증거와 원인을 정리한다.

Rust → C ABI → CUDA를 유지한다. SM89 runtime과 SM90a/SM100a compile을 구분하고 후자의 runtime만 장비 부재로 skip한다. Multi-GPU 장치별 owner 및 packed layout identity를 유지하되 테스트하지 않은 지원을 완료로 표시하지 않는다. GPU 시험 동안 세 Blender만 일시 중지 후 복구하고 3d/3dsol/3dfable 웹 서버는 계속 유지한다. 롤백은 backend 비활성 및 기존 weight owner 선택이며 checkpoint 파일을 덮어쓰지 않는다.
