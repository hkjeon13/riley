# rustinfer-cuda

This crate is the narrow Rust boundary for the production CUDA C ABI. Its
`cuda` feature AOT-compiles `../../kernels/src/version.cu` with `nvcc`, links a
static library, and exposes safe host-only ABI/version functions.

The PR 02 native library deliberately contains no kernel launch, device query,
CUDA context creation, allocation, model loading, or inference. Compile-target
architectures and runtime device capability remain separate concepts.

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

The native link smoke is:

```text
cargo test --locked -p rustinfer-cuda --features cuda --test abi_link
```

It calls only the host ABI/version symbols. It does not require or touch a CUDA
device, though compilation requires a complete CUDA toolkit.
