#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
root=$(realpath ../..)
old=$(realpath ../w3-prefill-cohort-20261124.fC0iBjq3)
package=$(realpath ../w4-pointer-syntax-20261124.HgwYVqCf)
h=$(realpath ../w2-sm17-20261118.6UMfeHh2)
checker=$(realpath ../w5-prefix-package-gpu-20261121.jBCeUWfR)
trap 'echo "$?" > exit-status' EXIT
cp "$old"/{arm.py,client.py,replay.py,holdout.py,schedules.json} .
printf '%s  %s\n' 4398f3e068a79f043270bc0ec2df638719f43e8834e5c2d290c52b6d2101384b "$package/source.tar" | sha256sum --check --strict
export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
if [[ -n ${PRESSURE_BUILD_FROM:-} ]]; then
    for input in arrival-pressure.patch arrival-pressure-registration.json; do
        cmp "$input" "$PRESSURE_BUILD_FROM/$input"
    done
    (cd "$PRESSURE_BUILD_FROM" && sha256sum --strict -c binaries.sha256)
    cp "$PRESSURE_BUILD_FROM"/{riley-diagnostic,binaries.sha256,cpu-tests.log,build.log} .
else
    mkdir src
    tar xf "$package/source.tar" -C src --no-same-owner
    (cd src && git apply --check ../arrival-pressure.patch && git apply ../arrival-pressure.patch)
    (cd src && CUDA_VISIBLE_DEVICES= taskset -c 24-27 cargo test --offline --locked -p riley-scheduler --no-default-features) > cpu-tests.log 2>&1
    (cd src && CUDA_VISIBLE_DEVICES= taskset -c 24-27 cargo build --offline --locked --release -p riley-server --features server,cuda) > build.log 2>&1
    install -m 0755 target/release/riley riley-diagnostic
    sha256sum riley-diagnostic src/crates/riley-scheduler/src/scheduler.rs > binaries.sha256
fi
unset LD_LIBRARY_PATH
export PYTHONPATH="$stage:$old/tools:$checker/harness/ci:${PRESSURE_BUILD_FROM:-$stage}/src/ci" PYTHONDONTWRITEBYTECODE=1
"$root/tools/vllm-venv/bin/python" -B -c 'import check_distribution_gpu, holdout, replay, run_profile'
if [[ -f tool-return-registration.json ]]; then
    cp tool_return_client.py client.py
    if [[ -f tool-return-projection-registration.json ]]; then
        cp tool_return_projection.py client.py
        sha256sum tool-return-projection-registration.json tool_return_projection.py agent_tool_projection.py validate_projection.py > selection-inputs.sha256
    fi
    "$root/tools/vllm-venv/bin/python" -B client.py schedules.json
    sha256sum agent_workflow.py long_tool_records.py tool_return_client.py tool-return-registration.json schedules.json > tool-inputs.sha256
fi
sha256sum arrival-pressure.patch arrival-pressure-registration.json arm.py client.py replay.py holdout.py schedules.json > inputs.sha256
run_arm() {
    local gpu=$1 mode=$2 cpus
    cpus=$(cat "../w3-aged-decode-native-20261124.tNZ3fCMo/gpu$gpu/server-cpus.txt") || return 1
    test -n "$cpus" || return 1
    local ctl=(python3 -B "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
    mkdir -p "gpu$gpu"
    W3_GPU_INDEX="$gpu" W3_PREFILL_ORDER="$mode" W3_CLIENT_CPU=$((gpu+38)) W3_SERVER_CPUS="$cpus" \
        "${ctl[@]}" run -- taskset -c "$cpus" "$root/tools/vllm-venv/bin/python" -B arm.py riley > "gpu$gpu/$mode.log" 2>&1 || return 1
    "${ctl[@]}" status > "gpu$gpu/$mode/lease-after.json"
}
round() {
    run_arm 6 "$1" & first=$!
    rc=0
    run_arm 7 "$2" || rc=$?
    wait "$first" || rc=$?
    return "$rc"
}
if [[ -f tool-return-projection-registration.json ]]; then
    round control control
else
    round control candidate
    round candidate control
fi
touch released
