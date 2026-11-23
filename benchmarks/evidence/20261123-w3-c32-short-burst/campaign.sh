#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w3-c32-steady256-burst-gpu-20261123.yOsEoY51"
build="${1:?diagnostic build required}"
[[ $build =~ ^"$evidence"/w3-c32-short-burst-build-20261123\.[[:alnum:]]+$ ]]
old_http="$evidence/w3-c32-http-workers-gpu-20261122.iU3I4Dcq"
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
warm="$evidence/w3-coder-startup-warmup-20261121.db8Pexwl"
prior="$evidence/w3-coder-rag-cache-20261121.Ngz53FnV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$previous"
    test "$(cat campaign.exit)" = 0
    printf '17c9061163e071ae14a27720d80989eac37397b806cf1d3730363a95399bf1d9  result.sha256\n5e199f69bf7e550287cafc092a7b4260bce38ee40b5f3b8ec9838d58a659740b  runner.sha256\na368c89328bba1e8d629e15cc2b55079b16cce55065fcddca4aa370c58482df0  schedules.json\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 runner.sha256 >/dev/null)
while [ ! -e "$build/build.exit" ]; do
    pid=$(cat "$build/build.pid")
    test "$(readlink "/proc/$pid/cwd")" = "$build"
    ps -p "$pid" -o stat= | awk 'NR==1 && $1 !~ /^Z/ { live=1 } END { exit !live }'
    sleep 5
done
(cd "$build"
    test "$(cat build.exit)" = 0
    sha256sum --strict -c build-inputs.sha256 build-result.sha256
    grep -q '1 passed; 0 failed' scheduler-test.log
    cd src && sha256sum --strict -c ../patched-source.sha256)
(cd "$old_http"
    printf 'fcdf930431e27582c77ca229ec19ead6d2df6c0c6ade512bbcd00f7467cf8be1  fixed-inputs.sha256\n4e81c401abab145a91a46dd683a60ddda067875edb02ce634c88b6d6391b0ae4  runtime-inputs.sha256\n' | sha256sum --strict -c)
(cd "$evidence" && sha256sum --strict -c "$old_http/fixed-inputs.sha256" "$old_http/runtime-inputs.sha256")
printf '%s\n' "$build" >build-path.txt
cp "$previous/"{schedules.json,replay.py,load.py,run_case.py} .
python3 -B prepare-runner.py run_case.py
sha256sum load.py replay.py run_case.py >runner.sha256
export PYTHONPATH="$stage:$prior:$warm:$run/harness/ci:$stock/src/ci"
grep -qx 'checkpoint verify: verified' "$prior/checkpoint-verify.log"
reserve() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
run_sequence() {
    local index="$1" layout label rc
    shift
    for layout in "$@"; do
        label="private-$layout-gpu$index"
        mkdir "$label"
        reserve "$index" >"$label/reservation-before.json"
        printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
        rc=0
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
            "$python" -B "$stage/run_case.py" "$stage" "$build" "$stage/replay.py" "$index" private 256 64 control "${layout#short-d}" >"$label/driver.log" 2>&1 || rc=$?
        reserve "$index" >"$label/reservation-after.json"
        printf '%d\n' "$rc" >"$label/exit"
        printf 'END %s %s exit=%s\n' "$label" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || return "$rc"
    done
}
for index in 6 7; do reserve "$index" >"gpu$index-before.json"; done
run_sequence 7 short-d0 short-d8 short-d16 &
left=$!
run_sequence 6 short-d16 short-d8 short-d0 &
right=$!
rc=0
wait "$left" || rc=1
wait "$right" || rc=1
for index in 6 7; do reserve "$index" >"gpu$index-after.json"; done
test "$rc" = 0
{
    printf '%s\n' schedules.json build-path.txt runner.sha256 gpu{6,7}-{before,after}.json
    find private-* -maxdepth 1 -type f
} | LC_ALL=C sort | xargs sha256sum >result.sha256
