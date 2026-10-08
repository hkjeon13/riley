# syntax=docker/dockerfile:1.7

# Docker Official Image linux/amd64 manifest for rust:1.85.0-bookworm.
FROM rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03 AS rust-toolchain

# NVIDIA-published linux/amd64 manifest for CUDA 12.8.1 devel on Ubuntu 22.04.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7 AS build-environment

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        pkg-config \
        python3 \
        python3-tomli \
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

# Network access is permitted only while preparing this image.  The reviewed
# A/B builds run this content-addressed image by sha256 ID with --network none.
RUN --mount=type=bind,source=.,target=/seed,readonly \
    CARGO_NET_OFFLINE=false cargo fetch --locked --manifest-path /seed/Cargo.toml

RUN rustc --version \
    && cargo --version \
    && nvcc --version \
    && python3 -c 'import tomli; assert tomli.loads("value = 1")["value"] == 1' \
    && test "$(rustc --version | cut -d ' ' -f 2)" = 1.85.0 \
    && test "$(cargo --version | cut -d ' ' -f 2)" = 1.85.0 \
    && nvcc --version | grep -F 'Cuda compilation tools, release 12.8, V12.8.93'

WORKDIR /workspace
