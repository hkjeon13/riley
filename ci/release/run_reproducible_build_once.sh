#!/usr/bin/env bash
# Run inside one clean, network-disabled reproducibility container.

set -euo pipefail
umask 022

: "${RILEY_REPRO_BUILD_ID:?RILEY_REPRO_BUILD_ID is required}"
: "${RILEY_SOURCE_REVISION:?RILEY_SOURCE_REVISION is required}"
: "${RILEY_SOURCE_ARCHIVE_SHA256:?RILEY_SOURCE_ARCHIVE_SHA256 is required}"
: "${RILEY_BUILD_IMAGE_ID:?RILEY_BUILD_IMAGE_ID is required}"
: "${SOURCE_DATE_EPOCH:?SOURCE_DATE_EPOCH is required}"

case "${RILEY_REPRO_BUILD_ID}" in
    A|B) ;;
    *) echo "RILEY_REPRO_BUILD_ID must be A or B" >&2; exit 2 ;;
esac
[[ "${RILEY_SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${RILEY_SOURCE_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${RILEY_BUILD_IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ "${SOURCE_DATE_EPOCH}" =~ ^[0-9]+$ ]]

test "$(pwd -P)" = /workspace
test -f Cargo.toml
test -f Cargo.lock
test -f ci/release/Dockerfile
test -f /input/source.tar
test -s /input/container-inspect.json
test -s /input/builder-image-inspect.json
test -d /evidence
test -z "$(find /evidence -mindepth 1 -print -quit)"

actual_source_sha256=$(sha256sum /input/source.tar | cut -d ' ' -f 1)
test "${actual_source_sha256}" = "${RILEY_SOURCE_ARCHIVE_SHA256}"

export CARGO_INCREMENTAL=0
export CARGO_NET_OFFLINE=true
export CARGO_TERM_COLOR=never
export LANG=C
export LC_ALL=C
export RILEY_CUDA_ARCHITECTURES=89
export TZ=UTC

mkdir -p /workspace/logs /workspace/release /workspace/release-root /workspace/tmp
export TMPDIR=/workspace/tmp

{
    printf 'rustc_version='
    rustc --version
    printf 'cargo_version='
    cargo --version
    printf 'nvcc_version='
    nvcc --version | sed -n '/Cuda compilation tools/p'
} > /workspace/logs/toolchain.txt

python3 ci/release/run_release_python.py ci/release/check_release_preflight.py \
    --source-revision "${RILEY_SOURCE_REVISION}" \
    --source-date-epoch "${SOURCE_DATE_EPOCH}" \
    > /workspace/logs/preflight.log 2>&1

cargo build --locked --offline --release --features cuda,server \
    > /workspace/logs/cargo-build.log 2>&1

cargo build --locked --offline --release --features bench,cuda \
    --bin riley-profile \
    > /workspace/logs/profile-build.log 2>&1

python3 ci/release/run_release_python.py ci/release/build_release_bundle.py \
    --binary target/release/riley \
    --output /workspace/release/riley.tar.gz \
    --source-revision "${RILEY_SOURCE_REVISION}" \
    --source-date-epoch "${SOURCE_DATE_EPOCH}" \
    > /workspace/logs/bundle-build.log 2>&1

python3 ci/release/run_release_python.py ci/release/verify_release_bundle.py \
    /workspace/release/riley.tar.gz \
    > /workspace/logs/bundle-verify.log 2>&1

tar --extract --gzip --file /workspace/release/riley.tar.gz \
    --strip-components=1 --directory /workspace/release-root

mkdir /evidence/artifacts /evidence/logs
install -m 0755 target/release/riley /evidence/artifacts/riley
install -m 0755 target/release/riley-profile /evidence/artifacts/riley-profile
install -m 0644 /workspace/release/riley.tar.gz /evidence/artifacts/riley.tar.gz
install -m 0644 /workspace/release-root/manifest/native-dependencies.txt \
    /evidence/artifacts/native-dependencies.txt
install -m 0644 /workspace/logs/toolchain.txt /evidence/logs/toolchain.txt
install -m 0644 /workspace/logs/preflight.log /evidence/logs/preflight.log
install -m 0644 /workspace/logs/cargo-build.log /evidence/logs/cargo-build.log
install -m 0644 /workspace/logs/profile-build.log /evidence/logs/profile-build.log
install -m 0644 /workspace/logs/bundle-build.log /evidence/logs/bundle-build.log
install -m 0644 /workspace/logs/bundle-verify.log /evidence/logs/bundle-verify.log

# This is deliberately the last in-container action. The host captures the
# exited/zero-status Docker receipt before packaging these raw bytes.
python3 ci/release/run_release_python.py \
    ci/release/write_reproducible_build_completion.py \
    --build-id "${RILEY_REPRO_BUILD_ID}" \
    --source-revision "${RILEY_SOURCE_REVISION}" \
    --source-archive-sha256 "${RILEY_SOURCE_ARCHIVE_SHA256}" \
    --source-date-epoch "${SOURCE_DATE_EPOCH}" \
    --build-image-id "${RILEY_BUILD_IMAGE_ID}" \
    --container-inspect /input/container-inspect.json \
    --binary /evidence/artifacts/riley \
    --profile-binary /evidence/artifacts/riley-profile \
    --bundle /evidence/artifacts/riley.tar.gz \
    --native-manifest /evidence/artifacts/native-dependencies.txt \
    --toolchain-log /evidence/logs/toolchain.txt \
    --preflight-log /evidence/logs/preflight.log \
    --cargo-build-log /evidence/logs/cargo-build.log \
    --profile-build-log /evidence/logs/profile-build.log \
    --bundle-build-log /evidence/logs/bundle-build.log \
    --bundle-verify-log /evidence/logs/bundle-verify.log \
    --output /evidence/logs/build-completion.json
