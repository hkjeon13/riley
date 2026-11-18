#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
root=$(cd -- "$run/../.." && pwd -P)
hold="$root/evidence/w2-sm17-20261118.6UMfeHh2"
soak="$root/evidence/w5-mixed-soak-20261119.YJg0Xzhk"
holdout="$root/evidence/w4-holdout-20261119.cgmtjz4x"
profile="$root/evidence/w3-rag-profile-20261119.FE1ZUFeE"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
for dependency in "$soak" "$holdout" "$profile"; do
    printf 'WAITING FOR %s %s\n' "$dependency/campaign.exit" "$(date -u +%FT%TZ)"
    for attempt in $(seq 1 400); do
        if test -s "$dependency/campaign.exit"; then break; fi
        if ! ps -o args= -p "$(cat "$dependency/campaign.pid")" | grep -Fq -- "$dependency/campaign.sh"; then
            test -s "$dependency/campaign.exit" && break
            printf 'Missing campaign %s\n' "$dependency" >&2
            exit 75
        fi
        sleep 15
    done
    test -s "$dependency/campaign.exit"
done
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo"
export CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export RILEY_CUDA_ARCHITECTURES=80
export CARGO_TARGET_DIR="$run/target"
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CARGO_BUILD_JOBS=4
export PYTHONDONTWRITEBYTECODE=1
export W4_GPU_INDEX=6
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
mkdir "$run/src"
tar -xzf "$run/source.tar.gz" -C "$run/src"
existing="$root/evidence/w4-schema-build-20261119.QWnmhslP/src"
(cd "$existing" && sha256sum -c "$run/kernels.sha256") >"$run/kernel-verification.log"
while read -r hash path; do
    install -D -m 0644 "$existing/$path" "$run/src/$path"
done <"$run/kernels.sha256"
(cd "$run/src" && sha256sum -c "$run/kernels.sha256") >>"$run/kernel-verification.log"
(
    cd "$run/src"
    git apply --check "$run/final-prefill.patch"
    git apply "$run/final-prefill.patch"
    sha256sum crates/riley-runtime/src/llama/executor/dispatch.rs
    "$toolchain/cargo" --version
    "$toolchain/rustc" --version
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) >"$run/build.log" 2>&1
sha256sum "$run/target/release/riley" >"$run/binary.sha256"
failures=0
for phase in native3 hf3 native9 hf9; do
    printf 'START %s %s\n' "$phase" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 run -- /usr/bin/python3 -B "$run/tools/run_trace.py" "$phase" || rc=$?
    if test -d "$run/$phase"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status >"$run/$phase/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$phase" "$(date -u +%FT%TZ)" "$rc"
    if test "$rc" -ne 0; then
        failures=1
        if test "$phase" = native3; then break; fi
    fi
done
for n in 3 9; do
    if test -s "$run/native$n/index.tsv" && test -s "$run/hf$n/hf-rows/metadata.json"; then
        "$root/tools/vllm-venv/bin/python" -B "$run/tools/layer_trace.py" compare \
            "$run/native$n" "$run/hf$n/hf-rows" "$run/fixtures/legacy.json" "$n" "$run/case$n-comparison.json"
    fi
done
exit "$failures"
