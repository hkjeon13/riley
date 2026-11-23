#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
reference="$evidence/w3-c32-short-burst-admission-gpu-20261123.TA5Cmib0"
build="$evidence/w3-c32-short-burst-admission-build-20261123.jPY8bnjT"
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
warm="$evidence/w3-coder-startup-warmup-20261121.db8Pexwl"
prior="$evidence/w3-coder-rag-cache-20261121.Ngz53FnV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$reference"
    test "$(cat campaign.exit)" = 0
    test "$(cat build-path.txt)" = "$build"
    printf '7a641e1843016b5198ac7e627b263555f010273899084baa52fdcdd4b785f021  runner.sha256\na368c89328bba1e8d629e15cc2b55079b16cce55065fcddca4aa370c58482df0  schedules.json\n' |
        sha256sum --strict -c
    sha256sum --strict -c runner.sha256)
(cd "$build"
    test "$(cat build.exit)" = 0
    printf '83791301b0a1dc27e4479a855fbf31ae1818a7978a3ec8f19b20b7b752743097  build-result.sha256\n29960186880b03767d1cd6cfee29fa5c25eb8376a41741239d86601279983de2  build-inputs.sha256\n' |
        sha256sum --strict -c
    sha256sum --strict -c build-result.sha256 build-inputs.sha256)
grep -qx 'checkpoint verify: verified' "$prior/checkpoint-verify.log"

reserve() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
for variant in original late12; do
    case_dir="$stage-$variant"
    mkdir -- "$case_dir"
    python3 -B "$stage/schedules.py" "$reference/schedules.json" "$variant" "$case_dir/schedules.json"
    cp -- "$reference/"{load.py,replay.py,run_case.py,runner.sha256} "$case_dir/"
    printf '%s\n' "$build" >"$case_dir/build-path.txt"
    for index in 6 7; do reserve "$index" >"$case_dir/gpu$index-before.json"; done
done

record_event() {
    local case_dir="$1"
    shift
    printf '%s\n' "$*" >>"$case_dir/campaign.log"
    printf '%s %s\n' "${case_dir##*-}" "$*"
}

run_variant() {
    local index="$1" variant="$2" case_dir layout label rc short_limit trace
    shift 2
    case_dir="$stage-$variant"
    export PYTHONPATH="$case_dir:$prior:$warm:$run/harness/ci:$stock/src/ci"
    for layout in "$@"; do
        short_limit="${layout%-*}"
        trace="${layout##*-}"
        label="private-$layout-gpu$index"
        mkdir -- "$case_dir/$label"
        reserve "$index" >"$case_dir/$label/reservation-before.json"
        record_event "$case_dir" START "$label" "$(date -u +%FT%TZ)"
        rc=0
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
            "$python" -B "$case_dir/run_case.py" "$case_dir" "$build" "$case_dir/replay.py" \
            "$index" private 256 64 "$trace" "${short_limit#short-d}" >"$case_dir/$label/driver.log" 2>&1 || rc=$?
        reserve "$index" >"$case_dir/$label/reservation-after.json"
        printf '%d\n' "$rc" >"$case_dir/$label/exit"
        record_event "$case_dir" END "$label" "$(date -u +%FT%TZ)" "exit=$rc"
        test "$rc" = 0 || return "$rc"
    done
    reserve "$index" >"$case_dir/gpu$index-after.json"
}

gpu6=(short-d0-admission short-d8-admission short-d8-control short-d0-control)
gpu7=(short-d0-control short-d8-control short-d8-admission short-d0-admission)
(run_variant 6 original "${gpu6[@]}" && run_variant 6 late12 "${gpu6[@]}") &
left=$!
(run_variant 7 late12 "${gpu7[@]}" && run_variant 7 original "${gpu7[@]}") &
right=$!
rc=0
wait "$left" || rc=1
wait "$right" || rc=1
for index in 6 7; do reserve "$index" >"gpu$index-after.json"; done
test "$rc" = 0
for variant in original late12; do
    (cd "$stage-$variant"
        {
            printf '%s\n' schedules.json build-path.txt runner.sha256 campaign.log gpu{6,7}-{before,after}.json
            find private-* -maxdepth 1 -type f
        } | LC_ALL=C sort | xargs sha256sum >result.sha256
        sha256sum result.sha256)
done >variant-results.sha256
