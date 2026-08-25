# rustinfer-cuda

This crate is the narrow Rust boundary for the production CUDA C ABI. Its
`cuda` feature AOT-compiles the sources under `../../kernels` with `nvcc`,
links the native static archive plus the shared CUDA Runtime and Driver
libraries, and exposes a safe host-runtime API.

The host runtime supports device metadata, retained primary-context ownership,
non-default streams, timing events, explicit query/synchronize/close paths,
opaque device and pinned-host byte buffers, coherent allocation accounting,
and stream-ordered asynchronous H2D/D2H copies. The diagnostic fill kernel is
only a lifecycle and ordering smoke test; model loading, tensor execution, and
inference remain out of scope. The stable C contract and ownership rules are
documented in [`../../docs/cuda-abi-v1.md`](../../docs/cuda-abi-v1.md).

## Build configuration

- Toolkit discovery order: an explicit `CUDAToolkit_ROOT`, `CUDA_HOME`, or
  `CUDA_PATH`; then `NVCC`/`CUDACXX`; then `nvcc` on `PATH`.
- Multiple explicit toolkit variables must resolve to the same directory.
- `RUSTINFER_CUDA_ARCHITECTURES` accepts numeric CMake AOT targets separated by
  commas or semicolons, for example `80;89`. It defaults to `89` and is printed
  in the Cargo build log. Runtime-dependent `native` is intentionally rejected.
- `CMAKE` may select a non-default CMake executable.
- The native archive deliberately links the toolkit-selected shared CUDA
  Runtime (`cudart`). CMake writes the exact linker-file location and the Cargo
  build validates that it resolves beneath the same toolkit as `nvcc`; no
  toolkit library path is hardcoded. A release environment must provide the
  matching shared CUDA Runtime, such as an NVIDIA CUDA runtime container.
- Driver symbols are linked through CMake's selected `CUDA::cuda_driver`
  development linker file. The deployed process resolves the host driver's
  `libcuda.so.1`; the build does not embed a toolkit-stub runtime path.

The host-only ABI/link smoke is:

```text
cargo test --locked -p rustinfer-cuda --features cuda --test abi_link
```

It calls only the host ABI/version symbols. It does not require or touch a CUDA
device, though compilation requires a complete CUDA toolkit.

Feature-off CPU checks are safe on any development host:

```text
cargo test --locked -p rustinfer-cuda --no-default-features
```

The `host_runtime_gpu` integration target contains exactly seven tests marked
`#[ignore = "remote GPU"]`. Run them only in an explicitly authorized remote
NVIDIA GPU environment; ordinary Cargo test invocations do not execute device
work:

```text
cargo test --locked -p rustinfer-cuda --no-default-features --features cuda \
  --test host_runtime_gpu -- --ignored --test-threads=1 --nocapture
```

The repository's Python-free GPU evidence workflow, sanitizer mode, and
required environment variables are described in
[`../../ci/README.md`](../../ci/README.md). None of these tests loads a model or
runs inference.

## Query-length-one decode

`PreparedDecodeAttention` cold-selects either a materialized BF16 reference or
the D64 chunked-online implementation. Selection fixes the backend, workspace
dtype and byte size, partition capacity, and provenance trace before the hot
path. `execute` never allocates and never changes backend. Both implementations
consume a BF16 query `[QH,D]` and head-major BF16 caches `[KVH,max_seq,D]`, and
write BF16 `[QH,D]`; GQA requires `KVH` to divide `QH`.

The reference workspace reserves `QH*max_seq*2` bytes once. For a call with
logical length `T`, its active prefix is densely packed BF16 `[QH,T]`; callers
must treat the remaining capacity as opaque scratch rather than a strided
`[QH,max_seq]` tensor.

The online workspace is packed F32
`[partition_capacity,QH,D+2]`. Each row is version-1 `(m,l,n[D])`: `m` is the
range maximum, `l` is the unnormalized exponential sum, and `n` is the
unnormalized weighted-value sum. Empty rows use `m=-inf`, `l=0`, and zero `n`.
Reducers merge logical slots in an explicit ascending or descending order and
normalize once at the end. The public CPU `DecodePartialState` follows the same
contract so producer/reducer arithmetic can be checked without CUDA.

`kv_cache_append` performs one validated, bit-preserving paired scatter from
token-major K/V `[T,KVH,D]` into cache positions of head-major
`[KVH,max_seq,D]`. It synchronizes before returning; callers should advance
their logical cache length only after every layer append succeeds.

The decode GPU integration target is remote-only and ignored by default. It
covers exact cache placement and sentinels, MHA/GQA reference-versus-online
comparisons across partition boundaries, allocation stability, and standalone
reducer order/empty-state behavior:

```text
cargo test --locked -p rustinfer-cuda --no-default-features --features cuda \
  --test decode_attention_gpu -- --ignored --test-threads=1 --nocapture
```

The additive PR 04 memory boundary has a separate five-test remote target. All
five tests are ignored by default and cover logical zero-byte handles,
allocation accounting returning to zero, pinned-host/device round trips,
range/owner validation, and an explicit two-stream completion handoff:

```text
cargo test --locked -p rustinfer-cuda --no-default-features --features cuda \
  --test memory_gpu -- --ignored --test-threads=1 --nocapture
```

`CudaPendingH2D` and `CudaPendingD2H` borrow their stream, device buffer, and
pinned host buffer until completion. Native active-use tokens independently
reject early access, reuse, and close. Forgetting a pending token therefore
causes a permanent busy/accounted leak instead of exposing a pointer or
allowing storage to be freed during DMA. Device and pinned buffers are `Send`,
deliberately `!Sync`, non-cloneable opaque owners; no raw device or host pointer
is part of the safe API.

## Memory round-trip lifecycle

Run this only on an authorized CUDA host. The completion value retains mutable
borrows of the stream and both buffers through each asynchronous copy. Explicit
close keeps free and context-release errors observable, while the accounting
snapshot proves the logical allocations returned to zero.

```rust,no_run
use rustinfer_cuda::{CudaResult, CudaRuntime};

fn main() -> CudaResult<()> {
    let runtime = CudaRuntime::initialize()?;
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut device_buffer = context.allocate_device_buffer(4)?;
    let mut pinned = context.allocate_pinned_host_buffer(4)?;

    pinned.write(0, &[1, 2, 3, 4])?;
    device_buffer
        .copy_from_pinned_async(0, &mut pinned, 0, 4, &mut stream)?
        .synchronize()?;
    pinned.write(0, &[0; 4])?;
    device_buffer
        .copy_to_pinned_async(0, &mut pinned, 0, 4, &mut stream)?
        .synchronize()?;
    assert_eq!(pinned.to_vec()?, [1, 2, 3, 4]);

    device_buffer.close()?;
    pinned.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
```

## Diagnostic-kernel lifecycle

Build this example with the `cuda` feature and run it only on an authorized
CUDA host. `finish` synchronizes the originating stream before copying results
to host. Close child resources before explicitly closing the retained context
lease so every destruction error remains observable.

```rust,no_run
use rustinfer_cuda::{CudaResult, CudaRuntime};

fn main() -> CudaResult<()> {
    let runtime = CudaRuntime::initialize()?;
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let kernel = context.kernel();
    let mut stream = context.create_stream()?;

    let values = kernel.launch_fill(&mut stream, 1_024, 1.25)?.finish()?;
    assert_eq!(values.len(), 1_024);

    stream.close()?;
    drop(kernel);
    context.close()?;
    Ok(())
}
```
