# PR 02–05 native CI contract

The mandatory lane is CPU-only. Native CUDA compilation is a separate
nightly/manual lane, the cumulative PR 03 host-runtime and PR 04 memory GPU
tests are an explicit self-hosted/manual lane, and the Python reference suite
is an optional offline fake-backend lane. No CI lane loads a model or runs
model inference. Only the opted-in GPU lane initializes a device: PR 03 launches
the small fill smoke kernel, while PR 04 performs allocation and byte-copy
lifecycle checks.

## Production CPU gate

Use the repository-pinned Rust 1.85.0 toolchain:

```sh
cargo fmt --all -- --check
python3 ci/check_workspace_boundaries.py --locked
cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo test --locked --workspace --no-default-features
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps --no-default-features
ci/check_feature_matrix.sh
ci/check_workspace_without_research_tools.sh
```

The boundary checker requires only Python 3.11 or newer and the standard
library. Python is used to inspect Cargo metadata in CI; Cargo never invokes it
and it is not part of a production artifact. The checker fails closed unless:

- the workspace contains exactly the seven production crates;
- `tools/python`, `tools/native`, and `experiments/triton` remain excluded;
- crate edges and feature ownership match `crates/README.md`;
- every crate inherits `publish = false`;
- the only direct third-party Cargo dependencies are exact-version `serde`,
  `serde_json`, and `sha2` requirements owned by `rustinfer-model`;
- the complete resolved third-party graph exactly matches
  `ci/approved_cargo_dependencies.toml`, including crates.io source, checksum,
  license expression, MSRV, and dependency edges;
- no git dependency is present, and every approved package's MSRV is at most
  the workspace Rust 1.85 MSRV; and
- no production build script invokes Python or Triton.

The approved dependency manifest is a reviewed allowlist, not a discovery
output. Adding or upgrading a package requires updating its exact resolved
closure and re-reviewing every changed checksum, license, and MSRV entry.

The final shell check copies the current tree to a temporary directory without
the excluded tool/research roots, then runs locked metadata and an all-targets
CPU build there.

## Python-free CUDA compile and link

The reproducible container lane is Linux/amd64 and pins both input images by
immutable manifest digest:

- Docker Official Image `rust:1.85.0-bookworm`;
- NVIDIA `cuda:12.8.1-devel-ubuntu22.04`.

Compile for the RTX 4090's compute capability 8.9 without granting the
container GPU access:

```sh
docker build \
  --file ci/cuda/Dockerfile \
  --build-arg RUSTINFER_CUDA_ARCHITECTURES=89 \
  --progress plain \
  --tag rustinfer-native-cuda:local \
  .
```

No `--gpus` flag is intentional: PR 02 validates AOT compilation and linking,
not runtime device behavior. The container gate records or checks:

1. exact Rust, Cargo, nvcc, toolkit root, and AOT architecture information;
2. locked release build plus the plan's exact root command
   `cargo build --release --features cuda,server`;
3. the host-only C ABI link test and ABI version 1;
4. compile-only `host_runtime_gpu` and `memory_gpu` test binaries plus the
   CUDA-backed `rustinfer-tensor` surface, without device access;
5. CUDA feature-on `rustinfer-cuda` and `rustinfer-tensor` Clippy across all
   targets with warnings denied;
6. `rustinfer --version` reporting the linked CUDA ABI;
7. a clear failure for an explicit nonexistent CUDA toolkit root; and
8. `ldd`, `readelf`, `nm`, and `Cargo.lock` evidence with no Python, PyTorch,
   Transformers, or Triton runtime dependency.

PR 03부터 artifact는 CUDA Driver API를 link한다. GPU를 의도적으로 주지 않는 이
compile-only image에는 실제 host driver가 없으므로, `abi_link`와 `--version`처럼
device/context를 초기화하지 않는 metadata executable에 한해서 toolkit
`libcuda.so` stub을 임시 SONAME alias로 사용한다. 이 경로는 artifact의
RPATH/RUNPATH에 기록되지 않아야 한다. 별도 GPU gate는 NVIDIA Container Runtime이
주입한 실제 `libcuda.so.1`을 `ldd`와 `ldconfig`로 다시 강제한다.

On a host with CUDA 12.8.1 installed, the equivalent direct build contract is:

```sh
export CUDAToolkit_ROOT=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export RUSTINFER_CUDA_ARCHITECTURES=89
cargo build --locked --release --features cuda,server
cargo test --locked -p rustinfer-cuda --features cuda --test abi_link
cargo test --locked -p rustinfer-cuda --no-default-features --features cuda \
  --test host_runtime_gpu --no-run
cargo test --locked -p rustinfer-cuda --no-default-features --features cuda \
  --test memory_gpu --no-run
cargo test --locked -p rustinfer-tensor --no-default-features --features cuda --no-run
./target/release/rustinfer --version
```

`native-cuda.yml` runs the pinned container nightly or by manual dispatch. A
CUDA base-image, Rust-image, toolkit, or architecture change is an explicit CI
contract change and must update the digest and captured evidence together.

## Python-free CUDA host-runtime and memory GPU gate

The image build compiles both `host_runtime_gpu` and `memory_gpu` with
`--no-run`, together with `rustinfer-tensor --features cuda`. Execution is a
separate operation and requires NVIDIA Container Toolkit GPU passthrough. On
an authorized GPU host:

```sh
GPU_EVIDENCE_DIR=$(mktemp -d)
SOURCE_ARCHIVE_PATH=$(mktemp)
git archive --format=tar --output="${SOURCE_ARCHIVE_PATH}" HEAD
SOURCE_REVISION=$(git rev-parse HEAD)
SOURCE_ARCHIVE_SHA256=$(sha256sum "${SOURCE_ARCHIVE_PATH}" | cut -d ' ' -f 1)
GPU_IMAGE_ID=$(docker image inspect --format '{{.Id}}' rustinfer-native-cuda:local)
docker run --rm \
  --network none \
  --gpus all \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --env RUSTINFER_CUDA_LEAK_ITERATIONS=128 \
  --env RUSTINFER_CUDA_COMPUTE_SANITIZER=0 \
  --env RUSTINFER_GPU_EVIDENCE_DIR=/evidence \
  --env "RUSTINFER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env "RUSTINFER_SOURCE_ARCHIVE_SHA256=${SOURCE_ARCHIVE_SHA256}" \
  --env "RUSTINFER_GPU_IMAGE_ID=${GPU_IMAGE_ID}" \
  --volume "${GPU_EVIDENCE_DIR}:/evidence" \
  rustinfer-native-cuda:local \
  ci/verify_python_free_gpu_runtime.sh
```

The verifier has no network and no Python executable. It records NVIDIA and
Rust/CUDA metadata, lists both exact integration-test inventories, then
executes each ignored target separately with
`--ignored --test-threads=1 --nocapture`. Every GPU test is marked
`#[ignore = "remote GPU"]`, so an ordinary Cargo test command cannot
accidentally execute device work.

The PR 03 `host_runtime_gpu` target remains exactly seven tests covering:

- device identity, compute capability, total memory, multiprocessor count,
  driver version, and runtime version;
- an invalid device ordinal;
- explicit event ordering across two non-default streams;
- async fill correctness after synchronization;
- launch-time error staging for invalid launch parameters;
- positive event elapsed timing; and
- repeated context/stream/event create-drop leak smoke, controlled by
  `RUSTINFER_CUDA_LEAK_ITERATIONS` (32–4096, default 128).

The additive PR 04 `memory_gpu` target is exactly five tests:

- `allocation_accounting_returns_to_zero`;
- `zero_byte_allocations_and_copies_are_logical_noops`;
- `pinned_host_device_round_trip_is_exact`;
- `two_stream_copy_handoff_prevents_early_reuse`; and
- `copy_ranges_and_context_ownership_are_validated`.

Its stable accounting marker must report all four values as zero:

```text
rustinfer-cuda-memory-accounting device_live_bytes=0 device_live_allocations=0 pinned_host_live_bytes=0 pinned_host_live_allocations=0
```

Evidence consists of `environment.txt`, `nvidia-smi-list.txt`,
`nvidia-smi-device-metadata.csv`, the existing `host-runtime-*` test/list/link
logs, additive `memory-*` test/list/link logs, SHA-256 records for both exact
test executables, the injected CUDA driver/runtime library inventory, and the
top-level `SHA256SUMS` manifest. Both binaries receive independent
`ldd`/`readelf`/`nm` inspection, including resolved `libcuda.so.1` and
`libcudart.so` checks, no driver-stub RPATH/RUNPATH, and no Python, PyTorch,
Transformers, or Triton dependency. Existing evidence is never overwritten.
The workflow also binds this output to the checked-out revision, the SHA-256
of `git archive --format=tar HEAD`, and the locally built GPU image ID.

Set `RUSTINFER_CUDA_COMPUTE_SANITIZER=1` to repeat both ignored targets serially
under `compute-sanitizer --tool memcheck --leak-check full`. The PR 03 output
retains its existing `compute-sanitizer-memcheck.log` name, while PR 04 writes
`compute-sanitizer-memory-memcheck.log`, so logs cannot collide. Each log must
independently report `ERROR SUMMARY: 0 errors` and `LEAK SUMMARY: 0 bytes
leaked`; both are included in `SHA256SUMS`. The seven-test target deliberately
exercises one invalid launch; `--report-api-errors no` prevents that expected,
already-asserted CUDA API status from polluting memcheck's error summary without
suppressing memory-access or allocation-leak findings. This optional pass is
also exposed as the manual workflow input `run_compute_sanitizer`.

The GitHub GPU job is disabled unless a manual dispatch explicitly selects
`run_gpu_tests`. It targets only `[self-hosted, linux, x64, rustinfer-gpu]`, so
standard hosted runners never receive or wait on a GPU job. Scheduled runs
continue to perform compile/link and feature-on compile validation only.

## Optional Python reference gate

`python-reference.yml` runs only by manual dispatch. It installs no project
dependencies, asserts that Torch, Transformers, and Triton are unavailable,
and runs the standard-library fake-backend unit suite. Canonical reference or
benchmark inference remains a separate remote-GPU workflow.
