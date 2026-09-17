# CUDA Rust 도입 판정용 PR-sized spike

상태: **제안 단계**. 현재 exact Qwen candidate나 Rust serving hot path를 CUDA Rust로
이전하는 작업은 아니다. [NVIDIA의 CUDA Rust 소개](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)를 바탕으로, Rust-native kernel 작성이 Riley의 한정된 보조 연산에서 실질 이득을 주는지 판정한다.

## 왜 별도 spike인가

Riley는 이미 Rust ownership/lifecycle와 C/C++ CUDA ABI를 분리해 두었고, 현재 P13의
정확성 병목은 Q/K/V epilogue, attention, RMSNorm의 framework-specific rounding이다.
여기를 새 language backend로 바꾸면 알고리즘과 compiler/codegen 변수를 동시에 바꾸게 되어
원인을 판별할 수 없다. 따라서 현재 C++ CUDA 경로를 immutable correctness reference로 남기고,
CUDA Rust는 standalone kernel 후보 하나에서만 평가한다. Rust↔Python runtime 경로는 만들지
않으며 Python/HF는 기존처럼 offline oracle에만 쓴다.

## 도구 선택과 현재 환경

NVIDIA는 두 경로를 소개한다.

| 경로 | 요구사항 | Riley에서의 위치 |
|---|---|---|
| `cuda-oxide` SIMT Rust-to-PTX | Linux, compute capability 8 이상, CUDA 12 이상, clang/libclang, pinned nightly Rust | RTX 4090/CUDA 12.8에서 isolated compile probe 가능. nightly compiler backend이므로 default build나 production selector에 넣지 않는다. |
| `cutile-rs` Tile IR JIT | Linux, compute capability 8 이상, CUDA 13.3 이상, stable Rust 1.89 이상 | Hopper/Blackwell 및 CUDA 13.3+ lane에서만 평가한다. 현재 CUDA 12.8 serving host에는 도입하지 않는다. |

workspace의 MSRV/CI는 Rust 1.85이므로 spike crate는 nested workspace로 격리한다. Rust
toolchain 또는 CUDA version을 production workspace 전체에 올리지 않는다. CUDA Rust가 아직
early/experimental이라는 NVIDIA의 위치도 그대로 반영해, codegen hash·PTX/SM target·driver/CUDA
version을 artifact에 고정한다.

## 첫 대상

첫 후보는 exact GEMM/attention/RMSNorm이 아니라, Qwen decode에서 이미 명확한 입출력과
독립 검증이 가능한 **KV layout/repeat-gather 보조 연산**이다. 입력 BF16 `[T, KVH, D]`와
GQA group mapping, 출력 layout, alias/overlap/stream/lifecycle contract를 C++ kernel과 동일하게
고정한다. 이 연산은 attention 계산 자체가 아니므로 candidate가 실패해도 numerical profile이나
serving selection을 바꾸지 않는다.

한 PR에는 다음을 함께 넣는다.

1. `cuda-oxide` 전용 nested crate와 C ABI shim을 만들고 default Cargo feature/serving link에서 제외한다.
2. C++ reference와 Rust PTX kernel을 같은 fixed shapes, BF16 bytes, stream, alignment, alias rejection으로 ABBA 실행한다.
3. lifecycle error, poisoned plan, close, hot-path allocation counter, supported-SM dispatch를 C++ contract와 같은 test에서 검증한다.
4. CUDA 13.3+ host가 생기면 동일 interface 아래 `cutile-rs` implementation을 별도 build target으로 추가한다. 4090/CUDA 12.8에서 이 target은 `unsupported-toolchain`, Hopper/Blackwell에서만 실제 JIT execution을 기록한다.

## 승격/중단 기준

| Gate | 승격 조건 | 중단 조건 |
|---|---|---|
| Build/dispatch | pinned toolchain으로 PTX 생성, declared SM에서 explicit dispatch | default Riley build에 nightly/JIT 의존성이 새거나 unsupported SM에서 C++ fallback을 Rust success로 표기 |
| Correctness | C++ reference와 predeclared output bytes exact, repeat exact, invalid span/overlap/context fail-closed | tolerance 추가, alias 허용, lifecycle/close mismatch |
| Runtime | warm allocation-free repeats에서 event-time median이 C++ reference보다 의미 있게 낮고 variance와 겹치지 않음 | JIT/compile time을 hot latency로 숨기거나 isolated microbenchmark만으로 serving 승격 |
| Serving | cache-on/full-forward/selector gate를 다시 통과하고 matched Riley/vLLM AB/BA에서 이득 확인 | exact P13 또는 selector lifecycle이 미통과인 상태에서 serving benchmark 실행 |

첫 spike가 correctness와 isolated runtime을 통과해도, P13 cache-on gate와 P14 selector
integration은 별도로 통과해야 한다. 성능 이득이 작거나 codegen 안정성이 부족하면 C++ CUDA
reference를 유지하고 CUDA Rust lane은 future-toolchain research artifact로만 남긴다.

## 미래 하드웨어

interface는 `sm89`, Hopper, Blackwell을 명시적으로 구분한다. `cuda-oxide` PTX JIT와
`cutile-rs` Tile IR JIT의 artifact는 GPU/driver/toolkit별로 별도 기록하며, 4090에서 실행할 수
없는 CUDA 13.3+ path를 fallback 결과로 대체하지 않는다. multi-GPU에서는 이 spike가 NCCL,
KV ownership, rank failure semantics을 대신 검증하지 않으며 그 계약은 별도 TP/KV 계획을 따른다.
