# FA3 native plan and stream contract — 2026-09-14

The experimental C ABI is implemented and linked against pinned upstream FA3 CUDA code. It is **not connected to Rust serving**, and Hopper attention correctness/performance remains unverified. No default backend changed.

## Implemented batch

1. **Explicit HND/page16 input contract:** packed BF16 Q/O `[total_q,9,64]`, BF16 K/V `[physical_pages,3,16,64]`; head64, 1–32 requests, total query rows 1–1024, per-request KV length 1–4096. Causal ragged prefill and single-query decode are separate modes. Prefill queries represent the suffix of each request's KV sequence. No FP8, split-KV, KV append, softcap or dtype reinterpretation.
2. **Owned immutable metadata/workspace:** validate monotone query/page offsets and physical page bounds, then convert request page lists into a rectangular table. `seqlen_k` is rounded to page16 so upstream's `seqlen_k/page_size` table shape is correct; `seqused_k` retains true lengths. Q offsets, lengths, scheduler vectors/semaphore and LSE have separate aligned regions. A cold create copies metadata and binds external buffers, context and explicit stream. Repeated enqueue allocates nothing and performs no host copy or synchronization in adapter code.
3. **Cold kernel preparation and native error boundary:** a pinned-source overlay separates upstream function-attribute setup from launch, suppressing scheduler/kernel launches during preparation. Normal enqueue recomputes scheduler metadata on the same stream. CUDA/CUTLASS exceptions cannot cross the `noexcept` C ABI. Capability checks require SM90, tensor-map access and cluster support; cold attribute setup also validates the kernel's dynamic shared-memory requirement.
4. **Lifecycle and fallback boundary:** create rejects capture; enqueue binds the same CUDA context/stream; destroy rejects capture and synchronizes that stream before freeing workspace. Raw callers must retain Q/K/V/O and stream, serialize calls, and destroy every graph referencing the plan before destroying it. Unsupported hardware returns a distinct status without a plan. No fallback is silently selected or numerical profile changed.

Plan identity fixes shapes, page table, lengths and bound addresses. Q/K/V values can change with stream ordering; changing metadata requires a new plan. This is a primitive integration boundary, **not yet an efficient dynamic serving-plan cache**. Rust graph/lease ownership and dynamic serving metadata integration remain required.

## Evidence

| Check | Result |
|---|---|
| CUDA 13.0.88 SM90a paged prefill/decode + scheduler build/link | Pass |
| C header compilation; four unmangled C ABI symbols | Pass |
| Noncontiguous page table, true length and aligned workspace layout | Pass |
| Invalid page IDs, signed offset extremes, invalid decode shape/context/page counts | Rejected as expected |
| Maximum 32 requests / 1,024 query rows / 4,096 context / 8,192 page entries | Host layout pass |
| Null validate/enqueue/destroy arguments | Rejected as expected |
| Create during CUDA stream capture; end capture afterward | Rejected without invalidating capture |
| Valid spec on actual RTX 4090 SM89 | Unsupported, null plan |
| Python/Torch dynamic runtime dependencies | None (`link-device.txt`) |
| Hopper buffer binding, allocation/lifetime and GPU numerical execution | Not run: no Hopper |
| Hopper graph replay, sanitizers, full-model and serving comparison | Not run: no Hopper; integration/suites still required |

`receipt.json`, `build.log`, `contract.stdout`, `contract.stderr` and `link-device.txt` are final v2 evidence. v1 passed the initial contract suite; v2 adds maximum-shape and capture rejection checks. Both runs remain under `/data/riley-serving-260913-recovery/fa3-adapter-v1` and `fa3-adapter-v2`.

The compiler still reports WGMMA C7510 and paged register spills. Compile success and a working rejection path do not establish GPU correctness, overlap, latency or throughput. No new vLLM comparison is warranted before serving integration; the [last FFN serving comparison](../20260913-prefill-ffn-model-serving/README.md) remains separate.

## Reproduce and remaining work

```sh
python3 benchmarks/analysis/build_fa3_native_adapter.py FA3_CHECKOUT NVCC FRESH_OUTPUT
```

The build-only Python script exports the same immutable FA3/CUTLASS commits as the [native build probe](../20260914-fa3-native-build/README.md), retains licenses, and hashes both overlays and adapter sources. It invokes nvcc and a native executable; there is no Python serving bridge.

Next: optional native build integration and a Rust owner that retains buffer/stream/graph leases, followed by dynamic serving metadata integration and an explicit experimental numerical profile. Actual Hopper allocation failure/alias/extent tests, repeated enqueue and metadata isolation, graph replay, numerical oracle, model quality and serving gates must run before promotion. Existing exact serving stays selected. Disabling the optional backend is the rollback path; compiled SM90a is not evidence of Blackwell support.
