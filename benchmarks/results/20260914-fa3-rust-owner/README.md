# Optional FA3 build and Rust owner — 2026-09-14

The native FA3 plan now has optional Cargo/CMake integration and a safe borrowed Rust primitive owner. This is **not a serving backend promotion**: no model recorder or dynamic batching path selects it, and no Hopper attention was executed.

## Implemented batch

1. `RILEY_FA3_SOURCE` enables pinned source export verification, isolated SM90a objects, and the native owner bridge in the existing CUDA archive. Ordinary kernels retain their configured architecture (SM89 in this run). Build configuration always writes the option, including an empty value, to prevent stale enablement. Python prepares immutable sources offline; runtime uses Rust → C ABI → CUDA only.
2. `Fa3Batch` validates lengths, offsets, pages, decode/prefill mode and bounds before constructing the raw host ABI. `PreparedFa3Attention` exclusively borrows the stream and four buffers and validates their context/idle state. Native creation copies immutable metadata; the host batch need not outlive the plan. The FFI pins the raw spec's size and pointer offset.
3. The native bridge restores Riley's current-context contract and retains stream/buffer active-use counters. Native close drains work before releasing leases. Rust exposes separate `enqueue`, `synchronize` and consuming `close`; Drop attempts close. A failed close retains native leases so subsequent buffer/stream destruction cannot release potentially live resources.
4. Close-stage failure now permanently disables destructive retry in both raw plan and owner. CUDA free may report an error after an uncertain side effect; Drop must not issue another free. Context-restoration failure after successful raw destruction likewise leaves the owner and resource leases retained. This deliberate leak-on-uncertainty behavior is implemented, **not fault-injection verified on Hopper**.

## Verification

| Check | Evidence / result |
|---|---|
| Rust host metadata validation | 2 unit tests passed, CUDA off and optional backend off |
| Bound output cannot be closed while plan remains borrowed | compile-fail doctest passed |
| Enabled CUDA release build and Rust/native link | Passed; `fa3-owner-enabled-final-v2.log` |
| Actual RTX 4090 rejection through Rust owner | 3 create attempts returned `NotSupported` |
| Rejection releases all native leases | Four buffers and stream closed; context allocation stats returned to zero and context closed |
| Independent AOT architecture | FA3 flags `sm_90a`; base flags `sm_89`, recorded in `enabled-symbols-flags.txt` |
| Fresh and repeated pinned export configuration | Passed |
| Modified exported launch header | Rejected; `overlay-check.json` |
| Same Cargo build directory, option cleared | Release build + 2 host tests passed; no FA3 symbols in rebuilt archive |
| Actual Hopper creation/launch/synchronize/close, numerical outputs | Not run: no Hopper |
| Ambiguous CUDA close/context-restoration fault injection | Not run; remains required before promotion |

The first enabled run revealed only an unused Rust import warning, which was fixed. The final v2 run additionally compiles the no-retry close guard. Unrelated existing dead-code warnings remain. The disabled build is repeated after that final native change; raw logs and archive checks distinguish it from the earlier off/on screen.

The 4090 test proves the **unsupported-device rollback** path, not successful Hopper allocation accounting or normal owner execution. FA3's internally allocated workspace is not yet integrated into Riley's context allocation statistics.

The test binary requires the isolated CUDA 13 library path used by the recorded runs. Default-shell `ldd` cannot find `libcudart.so.13`; `runtime-library-resolution.txt` shows successful resolution with that configured path. It has no Python/Torch runtime dependency and is not packaged as a standalone deployment binary.

## Usage and remaining integration

```sh
RILEY_FA3_SOURCE=/absolute/pinned/flash-attention cargo test --release -p riley-cuda --features cuda --test fa3_owner_gpu -- --nocapture
RILEY_FA3_SOURCE= cargo test --release -p riley-cuda --features cuda --lib fa3::tests
```

The source checkout must contain FA3 `98eb7998a0eba4047c7a30375522569d4b8efb20` and CUTLASS `7127592069c2fe01b041e174ba4345ef9b279671` git objects. A supported CUDA compiler/driver and the ordinary Riley CUDA build environment are also required. `FA3_COMPILED` reports build availability only.

The current safe owner retains exclusive borrows and does **not** expose graph capture or buffer updates through other Riley primitives. It therefore is not yet a composable model execution session. Next integrate retained graph resources, dynamic metadata and workspace accounting, then an explicit experimental numerical profile/model recorder. Hopper numerical, fault/lifetime, graph replay, full-model and serving tests remain required. Blackwell is not accepted by the SM90a dispatch.

No vLLM comparison is added because serving execution is unchanged. The [FFN serving comparison](../20260913-prefill-ffn-model-serving/README.md) remains the existing performance reference; it is not evidence for FA3. Disable `RILEY_FA3_SOURCE` to remove the optional archive objects. No default serving profile changed.
