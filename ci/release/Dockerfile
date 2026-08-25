# Docker Official Image linux/amd64 manifest for rust:1.85.0-bookworm.
FROM rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03 AS rust-toolchain

# NVIDIA-published linux/amd64 manifest for CUDA 12.8.1 devel on Ubuntu 22.04.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7 AS builder

ARG RUSTINFER_CUDA_ARCHITECTURES=89
ARG RUSTINFER_SOURCE_REVISION
ARG SOURCE_DATE_EPOCH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup

ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV RUSTUP_TOOLCHAIN=1.85.0-x86_64-unknown-linux-gnu
ENV PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV RUSTINFER_CUDA_ARCHITECTURES=${RUSTINFER_CUDA_ARCHITECTURES}
ENV CARGO_INCREMENTAL=0

WORKDIR /workspace
COPY . .

# Licensing and immutable provenance fail before the expensive CUDA build.
RUN python3 ci/release/check_release_preflight.py \
        --source-revision "${RUSTINFER_SOURCE_REVISION}" \
        --source-date-epoch "${SOURCE_DATE_EPOCH}"
RUN cargo build --locked --release --features cuda,server
RUN mkdir -p /release \
    && python3 ci/release/build_release_bundle.py \
        --binary target/release/rustinfer \
        --output /release/rustinfer.tar.gz \
        --source-revision "${RUSTINFER_SOURCE_REVISION}" \
        --source-date-epoch "${SOURCE_DATE_EPOCH}" \
    && python3 ci/release/verify_release_bundle.py /release/rustinfer.tar.gz \
    && mkdir -p /runtime-root \
    && tar --extract --gzip --file /release/rustinfer.tar.gz \
        --strip-components=1 --directory /runtime-root

# NVIDIA-published linux/amd64 index digest for
# nvidia/cuda:12.8.1-runtime-ubuntu22.04, resolved 2026-08-26.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd AS runtime

COPY --from=builder /runtime-root/ /opt/rustinfer/

ENV PATH=/opt/rustinfer/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# The final image receives only verified bundle contents. It has neither the
# source/build toolchain nor Python-family executables.
RUN test -x /opt/rustinfer/bin/rustinfer \
    && test -s /opt/rustinfer/SHA256SUMS \
    && test -s /opt/rustinfer/manifest/native-dependencies.txt \
    && test -s /opt/rustinfer/manifest/release.json \
    && ldd /opt/rustinfer/bin/rustinfer \
    && for command in python python3 pip pip3 cargo rustc nvcc cmake make cc c++; do \
        if command -v "${command}" >/dev/null 2>&1; then \
            echo "forbidden runtime executable: ${command}" >&2; exit 1; \
        fi; \
    done \
    && test ! -e /workspace \
    && if find / -xdev -type f \( \
        -name '*.py' -o -name '*.pyc' -o -name '*.whl' \
        -o -name '*.pkl' -o -name '*.pickle' \) | grep -q .; then \
        echo 'forbidden Python artifact in runtime image' >&2; exit 1; \
    fi

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/opt/rustinfer/bin/rustinfer"]
CMD ["--help"]
