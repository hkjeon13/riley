# Docker Official Image linux/amd64 manifest for rust:1.85.0-bookworm.
FROM rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03 AS rust-toolchain

# NVIDIA-published linux/amd64 manifest for
# nvidia/cuda:12.8.1-devel-ubuntu22.04, resolved 2026-08-25.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7

ARG RUSTINFER_CUDA_ARCHITECTURES=89

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
ENV PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV RUSTINFER_CUDA_ARCHITECTURES=${RUSTINFER_CUDA_ARCHITECTURES}
ENV CARGO_INCREMENTAL=0
ENV CARGO_TERM_COLOR=always

WORKDIR /workspace
COPY . .

RUN ci/verify_python_free_cuda.sh
