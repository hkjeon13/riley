# Distribution image: `docker run` a bundled model preset or any prepared
# checkpoint. Unlike ci/release/Dockerfile (the pinned single-architecture
# release contract), this image targets the common deployment case:
#   - CUDA 12.4 base, so hosts with driver 550 or newer can run it;
#   - kernels for sm_80 (A100), sm_86 (RTX 30xx/A10) and sm_89 (RTX 40xx/L4/L40);
#   - riley-checkpoint with Hub download plus verified model presets;
#   - listens on 0.0.0.0:8080 by default.
#
# Build from the repository root:
#   docker build -f deploy/docker/Dockerfile -t riley:cu124 .

# Docker Official Image linux/amd64 manifest for rust:1.85.0-bookworm.
FROM rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03 AS rust-toolchain

# NVIDIA-published index digest for nvidia/cuda:12.4.1-devel-ubuntu22.04,
# resolved 2026-10-29.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5 AS builder

ARG RILEY_CUDA_ARCHITECTURES="80;86;89"
# Optional caps for shared build hosts, e.g. --build-arg CARGO_BUILD_JOBS=16.
ARG CARGO_BUILD_JOBS
ARG CMAKE_BUILD_PARALLEL_LEVEL

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
ENV RILEY_CUDA_ARCHITECTURES=${RILEY_CUDA_ARCHITECTURES}
ENV CARGO_INCREMENTAL=0

WORKDIR /workspace
COPY . .

# Same feature set as the binaries behind the published A100 numbers.
RUN cargo build --locked --release -p riley-server --features server,hf-exact-probe \
    && cargo build --locked --release --features hub \
        --manifest-path tools/riley-checkpoint/Cargo.toml \
    && mkdir -p /runtime-root/bin /runtime-root/presets /runtime-root/licenses/flashinfer \
    && install -m 0755 target/release/riley /runtime-root/bin/riley \
    && install -m 0755 tools/riley-checkpoint/target/release/riley-checkpoint \
        /runtime-root/bin/riley-checkpoint \
    && install -m 0755 deploy/docker/entrypoint.sh /runtime-root/bin/riley-docker \
    && install -m 0644 deploy/docker/presets/*.json /runtime-root/presets/ \
    && install -m 0644 LICENSE /runtime-root/licenses/LICENSE \
    && install -m 0644 third_party/flashinfer/LICENSE /runtime-root/licenses/flashinfer/LICENSE

# NVIDIA-published index digest for nvidia/cuda:12.4.1-runtime-ubuntu22.04,
# resolved 2026-10-29.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9 AS runtime

COPY --from=builder /runtime-root/ /opt/riley/

ENV PATH=/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV RILEY_MODEL_DIR=/models
ENV RILEY_BIND=0.0.0.0:8080

# A named volume mounted at /models inherits this owner, so the non-root
# user can download presets into it.
RUN mkdir -p /models \
    && chown 65532:65532 /models \
    && ldd /opt/riley/bin/riley \
    && /opt/riley/bin/riley-checkpoint --help >/dev/null

USER 65532:65532
VOLUME ["/models"]
EXPOSE 8080
ENTRYPOINT ["/opt/riley/bin/riley-docker"]
CMD ["help"]
