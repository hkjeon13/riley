# syntax=docker/dockerfile:1.7

# Dependency preparation is allowed to use the network only while this image is
# built on server-4096.  The evidence runner later resolves this image by its
# sha256 ID and starts it with --network none.
FROM rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03 AS rust-toolchain

ENV RUSTUP_TOOLCHAIN=1.85.0-x86_64-unknown-linux-gnu
RUN rustup component add --toolchain "${RUSTUP_TOOLCHAIN}" clippy \
    && rustup component list --installed --toolchain "${RUSTUP_TOOLCHAIN}" \
        | grep -Fx 'clippy-x86_64-unknown-linux-gnu' \
    && cargo clippy --version | grep -E '^clippy 0\.1\.85 '

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup

ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV RUSTUP_TOOLCHAIN=1.85.0-x86_64-unknown-linux-gnu
ENV PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV RILEY_CUDA_ARCHITECTURES=89
ENV CARGO_INCREMENTAL=0
ENV CARGO_NET_OFFLINE=true
ENV CARGO_TERM_COLOR=never

# Build this stage with --no-cache.  The bind supplies the selected checkout's
# complete workspace manifests and lockfile without baking source into the
# reusable execution image.
RUN --mount=type=bind,source=.,target=/seed,readonly \
    CARGO_NET_OFFLINE=false cargo fetch --locked --manifest-path /seed/Cargo.toml

RUN rustc --version \
    && cargo --version \
    && nvcc --version \
    && test "$(rustc --version | cut -d ' ' -f 2)" = 1.85.0 \
    && test "$(cargo --version | cut -d ' ' -f 2)" = 1.85.0 \
    && nvcc --version | grep -F 'Cuda compilation tools, release 12.8, V12.8.93' \
    && for command_name in python python3 pip pip3; do \
        if command -v "${command_name}" >/dev/null 2>&1; then \
            echo "forbidden optimizer evidence executable: ${command_name}" >&2; \
            exit 1; \
        fi; \
    done

WORKDIR /workspace
