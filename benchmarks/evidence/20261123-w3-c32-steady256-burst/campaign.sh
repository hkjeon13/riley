#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w3-c32-prefill-gpu-20261123.K9sfsf0T"
build="$evidence/w3-c32-prefill-build-20261123.nnAtltun"
parent_burst="$evidence/w3-c32-burst-gpu-20261123.3HaIq7VW"
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
    printf '422565c545fcf56213729cc33b4cde907fd5f27a920b3dfb5bc5488bacdc41bf  result.sha256\nfe794936fc571c0f4dfa1f10b39386678aa4b837b813002383dc196e88e5d738  runner.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 runner.sha256 >/dev/null)
(cd "$parent_burst"
    test "$(cat campaign.exit)" = 0
    printf '76f2ca64ebe217c325ff9862d98be3d399627fbfa8eaef24155e9488930161c5  result.sha256\ne67f8b2906470d028573b4d314dbedb47481f15ba739fa023df4dc2a441d425d  runner.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 runner.sha256 >/dev/null)
(cd "$build"
    test "$(cat build.exit)" = 0
    printf '036bda0aed79f54e3d9ac8f4178b7ea51287b185b04d6a9b3cb9b0c7bce9d179  build-result.sha256\n' | sha256sum --strict -c
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
reserve() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
run_sequence() {
    local index="$1" layout budget cap floor label rc
    shift
    for layout in "$@"; do
        IFS=- read -r budget cap floor <<<"$layout"
        label="private-$layout-gpu$index"
        mkdir "$label"
        reserve "$index" >"$label/reservation-before.json"
        printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
        rc=0
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
            "$python" -B "$stage/run_case.py" "$stage" "$build" "$stage/replay.py" "$index" private "${budget#s}" 64 control "${cap#d}" "${floor#p}" >"$label/driver.log" 2>&1 || rc=$?
        reserve "$index" >"$label/reservation-after.json"
        printf '%d\n' "$rc" >"$label/exit"
        printf 'END %s %s exit=%s\n' "$label" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || return "$rc"
    done
}
for index in 6 7; do reserve "$index" >"gpu$index-before.json"; done
run_sequence 7 s256-d32-p512 s256-d8-p64 s256-d16-p64 &
left=$!
run_sequence 6 s256-d16-p64 s256-d8-p64 s256-d32-p512 &
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
