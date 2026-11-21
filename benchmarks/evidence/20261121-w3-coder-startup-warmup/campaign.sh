#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
stock="$evidence/w5-current-package-20261121.f6SEbjvA"
structured="$evidence/w4-structured-holdout-20261121.w0hBehiZ"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
previous="$evidence/w3-prefix-min-gpu-20261121.8gBDZBBM"
prior="$evidence/w3-coder-rag-cache-20261121.Ngz53FnV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
build="$evidence/w3-prefix-min-build-20261121.vdX2vOQA"
python="$(dirname "$evidence")/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$evidence" && sha256sum --strict -c "$stage/runtime-inputs.sha256")
(cd "$previous" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c gpu-inputs.sha256 result.sha256 >/dev/null)
(cd "$prior" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null)
(cd "$build" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-inputs.sha256 build-result.sha256 >/dev/null)
export PYTHONPATH="$previous:$prior:$structured:$rag:$stock/harness/ci:$stock/src/ci"
grep -qx 'checkpoint verify: verified' "$prior/checkpoint-verify.log"

reserve_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}

run_pair() {
    local index=$1 condition label mode rc
    shift
    for condition in "$@"; do
        mode=${condition%-*}
        label="$condition-gpu$index"
        mkdir "$label"
        reserve_status "$index" >"$label/reservation-before.json"
        printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
        rc=0
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
            "$python" -B run_startup.py "$stage" "$build" "$index" "$mode" "${condition##*-}" >"$label/driver.log" 2>&1 || rc=$?
        reserve_status "$index" >"$label/reservation-after.json"
        printf '%s\n' "$rc" >"$label/exit"
        printf 'END %s %s exit=%s\n' "$label" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || return "$rc"
    done
}

for index in 6 7; do reserve_status "$index" >"gpu$index-before.json"; done
run_pair 7 off-cold filtered-cold filtered-warm off-warm &
first=$!
run_pair 6 off-warm filtered-warm filtered-cold off-cold &
second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
for index in 6 7; do reserve_status "$index" >"gpu$index-after.json"; done
test "$rc" = 0
sha256sum {off,filtered}-{cold,warm}-gpu{6,7}/{exit,receipt.json,readiness.json,reservation-before.json,reservation-after.json} \
    gpu{6,7}-{before,after}.json >result.sha256
