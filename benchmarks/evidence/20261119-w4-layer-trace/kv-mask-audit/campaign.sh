#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
root=$(dirname "$evidence")
source="$evidence/w4-layer-trace-r3-20261119.qpoSLydn"
soak="$evidence/w5-cache-off-repeat-20261119.gcvXdzAj"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$source/campaign.exit")" = 0
printf 'WAIT %s %s\n' "$soak" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 36000))
until test -s "$soak/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$soak/campaign.pid")" | grep -Fq -- "$soak/campaign.sh"; then
        test -s "$soak/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
mkdir src tools fixtures
cp -a -- "$source/src/." src/
for file in run_trace.py layer_trace.py long_accuracy.py compare_runner.py; do
    cp -- "$source/tools/$file" "tools/$file"
done
cp -- "$source/holder.py" "$source/final-prefill.patch" .
cp -- "$source/fixtures/legacy.json" "$source/fixtures/reference.json" fixtures/
sha256sum -c runner-base.sha256
gzip -dc runner.patch.gz | git apply --check -
gzip -dc runner.patch.gz | git apply -
sha256sum -c runner.sha256
mv -- replay_attention_mask.py tools/
sha256sum tools/*.py fixtures/*.json > staged-inputs.sha256
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$run/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1 W4_GPU_INDEX=6
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    sha256sum -c "$source/kernels.sha256"
    sha256sum -c "$run/dispatch-base.sha256"
    gzip -dc "$run/kv-trace.patch.gz" | git apply --check -
    gzip -dc "$run/kv-trace.patch.gz" | git apply -
    sha256sum -c "$run/dispatch.sha256"
    "$toolchain/cargo" --version
    "$toolchain/rustc" --version
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) > build.log 2>&1
sha256sum target/release/riley > binary.sha256
result=0
for case in 3 9; do
    phase="native$case"
    rc=0
    printf 'START %s %s\n' "$phase" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 run -- \
        /usr/bin/python3 -B "$run/tools/run_trace.py" "$phase" || rc=$?
    if test -d "$phase"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status > "$phase/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$phase" "$(date -u +%FT%TZ)" "$rc"
    if (( rc == 0 )); then
        CUDA_VISIBLE_DEVICES= "$root/tools/vllm-venv/bin/python" -B tools/replay_attention_mask.py \
            "$phase" fixtures/legacy.json "$case" "case$case-replay.json"
    else
        result=1
    fi
done
exit "$result"
