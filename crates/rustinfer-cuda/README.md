# rustinfer-cuda

This crate is the narrow Rust boundary for the production CUDA C ABI. Its
`cuda` feature AOT-compiles the sources under `../../kernels` with `nvcc`,
links the native static archive plus the shared CUDA Runtime and Driver
libraries, and exposes a safe host-runtime API.

PR 03 supports device metadata, retained primary-context ownership,
non-default streams, timing events, explicit query/synchronize/close paths,
and a checked diagnostic fill kernel. The fill is only a lifecycle and
ordering smoke test; model loading, tensor execution, and inference remain out
of scope. The stable C contract and ownership rules are documented in
[`../../docs/cuda-abi-v1.md`](../../docs/cuda-abi-v1.md).

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

## Safe lifecycle example

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
