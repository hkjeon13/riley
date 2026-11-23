#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w3-c32-short-burst-admission-build-20261123.jPY8bnjT"
reference="$evidence/w3-c32-short-burst-admission-gpu-20261123.TA5Cmib0"
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
recovery="$evidence/w5-pattern-class-recovery-20261123.HcickTXi"
warm="$evidence/w3-coder-startup-warmup-20261121.db8Pexwl"
prior="$evidence/w3-coder-rag-cache-20261121.Ngz53FnV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$recovery"
    test "$(cat campaign.exit)" = 0
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$previous"
    test "$(cat build.exit)" = 0
    printf '83791301b0a1dc27e4479a855fbf31ae1818a7978a3ec8f19b20b7b752743097  build-result.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c build-result.sha256 >/dev/null
    cd src
    sha256sum --strict -c ../patched-source.sha256 >/dev/null)
(cd "$reference"
    test "$(cat campaign.exit)" = 0
    printf 'a368c89328bba1e8d629e15cc2b55079b16cce55065fcddca4aa370c58482df0  schedules.json\n' | sha256sum --strict -c
    sha256sum --strict -c runner.sha256 >/dev/null)
(cd "$run" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
grep -qx 'checkpoint verify: verified' "$prior/checkpoint-verify.log"

printf 'BUILD_START %s\n' "$(date -u +%FT%TZ)"
cp -r -- "$previous/src" src
cp -- "$reference/"{load.py,replay.py,run_case.py,schedules.json} .
"$python" -B prepare.py "$stage/src" "$stage" "$stage/load-transport.patch"
sha256sum load.py replay.py run_case.py schedules.json src/crates/riley-server/src/{service,w3_trace}.rs >patched-inputs.sha256
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=3 CARGO_BUILD_JOBS=3
export CUDA_VISIBLE_DEVICES=
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
(cd src && cargo build --locked --offline --release -p riley-server --features server,hf-exact-probe) >compiler.log 2>&1
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ldd target/release/riley >dependencies.txt
if grep -F 'not found' dependencies.txt; then exit 1; fi
sha256sum target/release/riley toolchain.txt dependencies.txt patched-inputs.sha256 >build-result.sha256
printf '0\n' >build.exit
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"

export PYTHONPATH="$stage:$prior:$warm:$run/harness/ci:$stock/src/ci"
for index in 6 7; do "${control[@]}" --index "$index" status >"gpu$index-before.json"; done

run_sequence() {
    local index=$1 layout limit trace label rc
    shift
    for layout in "$@"; do
        limit="${layout%-*}"
        trace="${layout##*-}"
        label="private-short-d$limit-$trace-gpu$index"
        mkdir "$label"
        printf 'START %s %d %s\n' "$trace" "$index" "$(date -u +%FT%TZ)"
        rc=0
        "${control[@]}" --index "$index" run -- \
            "$python" -B "$stage/run_case.py" "$stage" "$stage" "$stage/replay.py" \
            "$index" private 256 64 "$trace" "$limit" >"$label/driver.log" 2>&1 || rc=$?
        printf 'END %s %d %s exit=%d\n' "$trace" "$index" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || return "$rc"
    done
}
run_sequence 6 0-transport 8-transport 8-control 0-control &
left=$!
run_sequence 7 0-control 8-control 8-transport 0-transport &
right=$!
rc=0
wait "$left" || rc=1
wait "$right" || rc=1
for index in 6 7; do "${control[@]}" --index "$index" status >"gpu$index-after.json"; done
test "$rc" = 0
"$python" -B report.py "$stage" >verified.json
{
    printf '%s\n' schedules.json patched-inputs.sha256 build-result.sha256 verified.json gpu{6,7}-{before,after}.json
    find private-* -maxdepth 1 -type f
} | LC_ALL=C sort | xargs sha256sum >result.sha256
