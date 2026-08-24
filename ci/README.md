# PR 02 CI contract

The mandatory lane is CPU-only. Native CUDA compilation is a separate
nightly/manual lane, and the Python reference suite is an optional offline
fake-backend lane. None of these PR 02 checks loads a model, initializes a CUDA
device, launches a kernel, or runs inference.

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
  --tag rustinfer-pr02-cuda:local \
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

## Optional Python reference gate

`python-reference.yml` runs only by manual dispatch. It installs no project
dependencies, asserts that Torch, Transformers, and Triton are unavailable,
and runs the standard-library fake-backend unit suite. Canonical reference or
benchmark inference remains a separate remote-GPU workflow.
