# PR 02–03 native CI contract

The mandatory lane is CPU-only. Native CUDA compilation is a separate
nightly/manual lane, the PR 03 GPU host-runtime smoke is an explicit
self-hosted/manual lane, and the Python reference suite is an optional offline
fake-backend lane. No CI lane loads a model or runs model inference. Only the
opted-in PR 03 GPU lane initializes a device and launches the small fill smoke
kernel.

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
- there are no third-party Cargo packages needing license review in PR 02; and
- no production build script invokes Python or Triton.

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
4. `rustinfer --version` reporting the linked CUDA ABI;
5. a clear failure for an explicit nonexistent CUDA toolkit root; and
6. `ldd`, `readelf`, `nm`, and `Cargo.lock` evidence with no Python, PyTorch,
   Transformers, or Triton runtime dependency.

On a host with CUDA 12.8.1 installed, the equivalent direct build contract is:

```sh
export CUDAToolkit_ROOT=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export RUSTINFER_CUDA_ARCHITECTURES=89
cargo build --locked --release --features cuda,server
cargo test --locked -p rustinfer-cuda --features cuda --test abi_link
./target/release/rustinfer --version
```

`native-cuda.yml` runs the pinned container nightly or by manual dispatch. A
CUDA base-image, Rust-image, toolkit, or architecture change is an explicit CI
contract change and must update the digest and captured evidence together.

## Python-free CUDA host-runtime GPU gate

The image build compiles `host_runtime_gpu` with `--no-run`. Execution is a
separate operation and requires NVIDIA Container Toolkit GPU passthrough. On an
authorized GPU host:

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
Rust/CUDA metadata, lists the exact integration tests, then executes the whole
ignored test target with `--ignored --test-threads=1 --nocapture`. Every GPU
test is marked `#[ignore = "remote GPU"]`, so an ordinary Cargo test command
cannot accidentally execute device work. The required tests cover:

- device identity, compute capability, total memory, multiprocessor count,
  driver version, and runtime version;
- an invalid device ordinal;
- explicit event ordering across two non-default streams;
- async fill correctness after synchronization;
- launch-time error staging for invalid launch parameters;
- positive event elapsed timing; and
- repeated context/stream/event create-drop leak smoke, controlled by
  `RUSTINFER_CUDA_LEAK_ITERATIONS` (32–4096, default 128).

Evidence consists of `environment.txt`, `nvidia-smi-list.txt`,
`nvidia-smi-device-metadata.csv`, `host-runtime-test-list.txt`,
`host-runtime-tests.log`, dynamic-link inspection from `ldd`/`readelf`/`nm`,
the injected CUDA driver/runtime library inventory, and `SHA256SUMS`. Existing
evidence is never overwritten. The workflow also binds this output to the
checked-out revision, the SHA-256 of `git archive --format=tar HEAD`, and the
locally built GPU image ID.

Set `RUSTINFER_CUDA_COMPUTE_SANITIZER=1` to repeat the same ignored target
serially under `compute-sanitizer --tool memcheck --leak-check full`. Its output
is captured as `compute-sanitizer-memcheck.log` and must report zero errors.
This optional pass is also exposed as the manual workflow input
`run_compute_sanitizer`.

The GitHub GPU job is disabled unless a manual dispatch explicitly selects
`run_gpu_tests`. It targets only `[self-hosted, linux, x64, rustinfer-gpu]`, so
standard hosted runners never receive or wait on a GPU job. Scheduled runs
continue to perform compile/link validation only.

## Optional Python reference gate

`python-reference.yml` runs only by manual dispatch. It installs no project
dependencies, asserts that Torch, Transformers, and Triton are unavailable,
and runs the standard-library fake-backend unit suite. Canonical reference or
benchmark inference remains a separate remote-GPU workflow.
