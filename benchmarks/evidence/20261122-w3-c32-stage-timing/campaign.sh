#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
previous="$evidence/w3-coder-startup-warmup-20261121.db8Pexwl"
proof="$evidence/w3-private-warm-gpu-20261121.Zdv0ggSI"
prior="$evidence/w3-coder-rag-cache-20261121.Ngz53FnV"
fixed="$evidence/w3-c32-arrival-replay-20261121.knwCy7jZ"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
build="${1:?diagnostic build path required}"
[[ $build =~ ^"$evidence"/w3-c32-stage-build-20261122\.[[:alnum:]]+$ ]]
python="$(dirname "$evidence")/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$evidence" && sha256sum --strict -c "$stage/fixed-inputs.sha256" "$stage/runtime-inputs.sha256")
(cd "$previous" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null)
(cd "$prior" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null)
(cd "$run" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
(cd "$stock" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-result.sha256 >/dev/null)
(cd "$fixed" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null)
for ((attempt=0; attempt<90; attempt++)); do
    test ! -e "$build/build.exit" || break
    sleep 5
done
(cd "$build" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-inputs.sha256 build-result.sha256 >/dev/null)
printf '%s\n' "$build" >build-path.txt
cp "$fixed/"{schedules.json,replay.py,load.py} "$stage/"
cp "$proof/run_case.py" "$stage/"
git apply --check replay-timing.patch
git apply replay-timing.patch
sha256sum load.py replay.py run_case.py >runner.sha256
export PYTHONPATH="$stage:$prior:$previous:$run/harness/ci:$stock/src/ci"
grep -qx 'checkpoint verify: verified' "$prior/checkpoint-verify.log"

reserve_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}

run_sequence() {
    local index=$1 label rc condition variant trace
    shift
    for label in "$@"; do
        condition=${label%-*}
        variant=${label##*-}
        trace=0
        if [[ $variant == trace ]]; then trace=1; fi
        label="$label-gpu$index"
        mkdir "$label"
        reserve_status "$index" >"$label/reservation-before.json"
        printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
        rc=0
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
            "$python" -B "$stage/run_case.py" "$stage" "$build" "$stage/replay.py" "$index" "$condition" "$trace" >"$label/driver.log" 2>&1 || rc=$?
        reserve_status "$index" >"$label/reservation-after.json"
        printf '%s\n' "$rc" >"$label/exit"
        printf 'END %s %s exit=%s\n' "$label" "$(date -u +%FT%TZ)" "$rc"
        test "$rc" = 0 || return "$rc"
    done
}

for index in 6 7; do reserve_status "$index" >"gpu$index-before.json"; done
run_sequence 7 off-warm-control private-control off-warm-trace private-trace &
first=$!
run_sequence 6 private-trace off-warm-trace private-control off-warm-control &
second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
for index in 6 7; do reserve_status "$index" >"gpu$index-after.json"; done
test "$rc" = 0
sha256sum schedules.json build-path.txt runner.sha256 {off-warm,private}-{control,trace}-gpu{6,7}/{exit,receipt.json,reservation-before.json,reservation-after.json} \
    gpu{6,7}-{before,after}.json >result.sha256
