#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w3-c32-page-slots-gpu-20261123.wUKuuOzr"
original_build="$evidence/w3-c32-page-slots-build-20261123.52mgCzYU"
build="${1:?diagnostic build required}"
[[ $build =~ ^"$evidence"/w3-c32-fixed-slots-build-20261123\.[[:alnum:]]+$ ]]
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
    [[ $(cat campaign.exit) =~ ^[01]$ ]]
    printf 'b523c6afda14a29a04dc278d342eaedbdb2ba9669a16e9dd71a004a73a57eaf8  result.sha256\n379f2fad79b5a43a0d2e943aaf08b4b65381f8d4a075b740b37433a38acfe2e2  runner.sha256\na368c89328bba1e8d629e15cc2b55079b16cce55065fcddca4aa370c58482df0  schedules.json\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 runner.sha256 >/dev/null)
while [ ! -e "$build/build.exit" ]; do
    pid=$(cat "$build/build.pid")
    test "$(readlink "/proc/$pid/cwd")" = "$build"
    ps -p "$pid" -o stat= | awk 'NR==1 && $1 !~ /^Z/ { live=1 } END { exit !live }'
    sleep 5
done
(cd "$build" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-inputs.sha256 build-result.sha256
    cd src && sha256sum --strict -c ../patched-source.sha256)
(cd "$original_build"
    printf '21a68110a07e848807f2ac408110be8cfa395c0a00d69aa56f4c219ef389f2e6  build-result.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c build-result.sha256)
(cd "$old_http"
    printf 'fcdf930431e27582c77ca229ec19ead6d2df6c0c6ade512bbcd00f7467cf8be1  fixed-inputs.sha256\n4e81c401abab145a91a46dd683a60ddda067875edb02ce634c88b6d6391b0ae4  runtime-inputs.sha256\n' | sha256sum --strict -c)
(cd "$evidence" && sha256sum --strict -c "$old_http/fixed-inputs.sha256" "$old_http/runtime-inputs.sha256")
printf '%s\n' "$build" >build-path.txt
cp "$previous/"{schedules.json,replay.py,load.py,run_case.py} .
python3 -B prepare-runner.py run_case.py
sha256sum load.py replay.py run_case.py >runner.sha256
export PYTHONPATH="$stage:$prior:$warm:$run/harness/ci:$stock/src/ci"
grep -qx 'checkpoint verify: verified' "$prior/checkpoint-verify.log"
"$python" -B check-cli.py "$stage" "$build" "$previous" "$original_build" >cli.json.tmp
mv -- cli.json.tmp cli.json
reserve() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
run_sequence() {
    local index="$1" layout budget mode slots label rc failures=0
    shift
    for layout in "$@"; do
        IFS=- read -r budget mode slots <<<"$layout"
        label="private-$layout-gpu$index"
        mkdir "$label"
        reserve "$index" >"$label/reservation-before.json"
        printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
        rc=0
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
            "$python" -B "$stage/run_case.py" "$stage" "$build" "$stage/replay.py" "$index" private "${budget#s}" 64 control "$mode" "${slots#a}" >"$label/driver.log" 2>&1 || rc=$?
        reserve "$index" >"$label/reservation-after.json"
        printf '%d\n' "$rc" >"$label/exit"
        printf 'END %s %s exit=%s\n' "$label" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || failures=1
    done
    return "$failures"
}
for index in 6 7; do reserve "$index" >"gpu$index-before.json"; done
run_sequence 7 s256-fixed-a32 s256-fixed-a48 s128-fixed-a48 s128-fixed-a32 &
left=$!
run_sequence 6 s128-fixed-a32 s128-fixed-a48 s256-fixed-a48 s256-fixed-a32 &
right=$!
rc=0
wait "$left" || rc=1
wait "$right" || rc=1
for index in 6 7; do reserve "$index" >"gpu$index-after.json"; done
{
    printf '%s\n' cli.json schedules.json build-path.txt runner.sha256 gpu{6,7}-{before,after}.json
    find private-* -maxdepth 1 -type f
} | LC_ALL=C sort | xargs sha256sum >result.sha256
test "$rc" = 0
