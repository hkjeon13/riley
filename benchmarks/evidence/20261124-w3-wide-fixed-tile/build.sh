#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
source_stage="$evidence/w4-pointer-syntax-20261124.HgwYVqCf"
soak="$evidence/w5-pointer-soak-20261124.wa1UJ2ZB"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c dispatch.sha256
test "$(cat "$soak/campaign.pid")" = 787131
test "$(cat "$soak/campaign.started")" = 2026-11-24T05:05:53Z
test "$(cat "$soak/campaign.exit")" = 0
test "$(cat "$soak/recovery/campaign.exit")" = 0
test "$(cat "$source_stage/campaign.exit")" = 0
printf '%s  %s\n' \
    4398f3e068a79f043270bc0ec2df638719f43e8834e5c2d290c52b6d2101384b "$source_stage/source.tar" \
    8436c243060357615f9ac5499edd98ba404a079e44d55a8df0f76bcf9ce0b6d8 "$source_stage/cpu-evidence/report.json" |
    sha256sum --strict -c
mkdir src
tar -xf "$source_stage/source.tar" -C src --no-same-owner
(cd src && sha256sum --strict -c ../before.sha256 &&
    git apply --check --whitespace=error ../wide-fixed-tile.patch &&
    git apply --whitespace=error ../wide-fixed-tile.patch &&
    sha256sum --strict -c ../after.sha256)
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$stage/target" CUDA_VISIBLE_DEVICES= RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
printf 'BUILD_START %s\n' "$(date -u +%FT%TZ)"
(cd src && taskset -c 24-27 cargo build --offline --locked --release -p riley-server --features server,cuda) >compiler.log 2>&1
(cd src && taskset -c 24-27 cargo test --offline --locked --release -p riley-cuda --features cuda \
    --test flashinfer_attention_gpu --no-run --message-format=json) >gpu-test-build.jsonl 2>gpu-test-build.log
install -m 0755 target/release/riley riley-diagnostic
gpu_test=$(/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
items = [json.loads(line) for line in Path("gpu-test-build.jsonl").read_text().splitlines()]
artifact, = [item for item in items if item["reason"] == "compiler-artifact"
             and item.get("executable") and item["target"]["name"] == "flashinfer_attention_gpu"]
print(artifact["executable"])
PY
)
[[ "$gpu_test" == "$stage/target/release/deps/"* ]]
install -m 0755 "$gpu_test" flashinfer-attention-test
ldd riley-diagnostic >dependencies.txt
ldd flashinfer-attention-test >gpu-test-dependencies.txt
! grep -Fq 'not found' dependencies.txt gpu-test-dependencies.txt
sha256sum riley-diagnostic flashinfer-attention-test dependencies.txt gpu-test-dependencies.txt \
    toolchain.txt after.sha256 >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
