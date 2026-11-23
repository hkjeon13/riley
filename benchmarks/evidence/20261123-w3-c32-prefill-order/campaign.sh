#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
build="$evidence/w3-c32-prefill-only-20261123.sDaHWRWD"
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
warm="$evidence/w3-coder-startup-warmup-20261121.db8Pexwl"
prior="$evidence/w3-coder-rag-cache-20261121.Ngz53FnV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$build"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        60ad187fe2230d29eedb9d91d5c8f0193814bccf306dcf5ee560f502d62d9582 result.sha256 \
        85f3301b703ea13409483f35ab5a858395c15672c34a7e9967b4da75a1859d21 build-result.sha256 \
        c5de9aa6baac113dbedce15a5e0ae07e42302dc576b7be23d3176c6bdc8cb35f patched-inputs.sha256 | sha256sum --strict -c
    sha256sum --strict -c result.sha256 build-result.sha256 patched-inputs.sha256 >/dev/null)
(cd "$run" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
cp -- "$build/"{load.py,replay.py,run_case.py,schedules.json} .
cp -- "$build/build-result.sha256" previous-build.sha256
export PYTHONPATH="$stage:$prior:$warm:$run/harness/ci:$stock/src/ci"
export CUDA_VISIBLE_DEVICES=

for index in 6 7; do "${control[@]}" --index "$index" status >"gpu$index-before.json"; done
run_sequence() {
    local index=$1 limit label rc
    shift
    for limit in "$@"; do
        label="private-short-d$limit-control-gpu$index"
        mkdir "$label"
        printf 'START %s %d %s\n' "$limit" "$index" "$(date -u +%FT%TZ)"
        rc=0
        "${control[@]}" --index "$index" run -- \
            "$python" -B "$stage/run_case.py" "$stage" "$build" "$stage/replay.py" \
            "$index" private 256 64 control "$limit" >"$label/driver.log" 2>&1 || rc=$?
        printf 'END %s %d %s exit=%d\n' "$limit" "$index" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || return "$rc"
    done
}
run_sequence 6 0 prefill-only &
left=$!
run_sequence 7 prefill-only 0 &
right=$!
rc=0
wait "$left" || rc=1
wait "$right" || rc=1
for index in 6 7; do "${control[@]}" --index "$index" status >"gpu$index-after.json"; done
test "$rc" = 0
"$python" -B check.py "$stage" >verified.json
"$python" -B compare-rounds.py "$build" "$stage" >order-comparison.json
{
    printf '%s\n' schedules.json previous-build.sha256 verified.json order-comparison.json gpu{6,7}-{before,after}.json
    find private-* -maxdepth 1 -type f
} | LC_ALL=C sort | xargs sha256sum >result.sha256
