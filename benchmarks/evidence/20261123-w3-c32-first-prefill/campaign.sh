#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w3-c32-http-workers-gpu-20261122.iU3I4Dcq"
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
warm="$evidence/w3-coder-startup-warmup-20261121.db8Pexwl"
prior="$evidence/w3-coder-rag-cache-20261121.Ngz53FnV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
build="${1:?diagnostic build path required}"
[[ $build =~ ^"$evidence"/w3-c32-prefill-build-20261123\.[[:alnum:]]+$ ]]
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$previous"
    test "$(cat campaign.exit)" = 0
    printf '769fa22dbb1a4b09820b560dcfb87faa71c36b6b626140768bc6547fe5634171  result.sha256\ne5066545b7fa2215245ab9f157a6d2e25469d1576308624108f83bb71545a995  runner.sha256\nfcdf930431e27582c77ca229ec19ead6d2df6c0c6ade512bbcd00f7467cf8be1  fixed-inputs.sha256\n4e81c401abab145a91a46dd683a60ddda067875edb02ce634c88b6d6391b0ae4  runtime-inputs.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 runner.sha256 >/dev/null)
(cd "$evidence" && sha256sum --strict -c "$previous/fixed-inputs.sha256" "$previous/runtime-inputs.sha256")
cp "$previous/"{schedules.json,replay.py,load.py,run_case.py} .
python3 -B prepare-runner.py run_case.py
sha256sum load.py replay.py run_case.py >runner.sha256
export PYTHONPATH="$stage:$prior:$warm:$run/harness/ci:$stock/src/ci"
grep -qx 'checkpoint verify: verified' "$prior/checkpoint-verify.log"
while [ ! -e "$build/build.exit" ]; do
    pid=$(cat "$build/build.pid")
    test "$(readlink "/proc/$pid/cwd")" = "$build"
    ps -p "$pid" -o stat= | awk 'NR==1 && $1 !~ /^Z/ { live=1 } END { exit !live }'
    sleep 5
done
(cd "$build" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-inputs.sha256 build-result.sha256 >/dev/null)
printf '%s\n' "$build" >build-path.txt
reserve() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
run_sequence() {
    local index="$1" layout budget workers trace label rc
    shift
    for layout in "$@"; do
        IFS=- read -r budget workers trace <<<"$layout"
        label="private-$layout-gpu$index"
        mkdir "$label"
        reserve "$index" >"$label/reservation-before.json"
        printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
        rc=0
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
            "$python" -B "$stage/run_case.py" "$stage" "$build" "$stage/replay.py" "$index" private "$budget" "${workers#w}" "$trace" >"$label/driver.log" 2>&1 || rc=$?
        reserve "$index" >"$label/reservation-after.json"
        printf '%d\n' "$rc" >"$label/exit"
        printf 'END %s %s exit=%s\n' "$label" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || return "$rc"
    done
}
for index in 6 7; do reserve "$index" >"gpu$index-before.json"; done
run_sequence 7 128-w32-control 128-w64-control 128-w32-trace 128-w64-trace 256-w32-trace 256-w64-trace &
left=$!
run_sequence 6 256-w64-trace 256-w32-trace 128-w64-trace 128-w32-trace 128-w64-control 128-w32-control &
right=$!
rc=0
wait "$left" || rc=1
wait "$right" || rc=1
for index in 6 7; do reserve "$index" >"gpu$index-after.json"; done
test "$rc" = 0
{
    printf '%s\n' schedules.json build-path.txt runner.sha256 gpu{6,7}-{before,after}.json
    find private-* -maxdepth 1 -type f ! -name driver.log
} | LC_ALL=C sort | xargs sha256sum >result.sha256
