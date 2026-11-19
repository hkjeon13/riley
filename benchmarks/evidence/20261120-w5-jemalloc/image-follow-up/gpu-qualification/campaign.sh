#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w5-allocator-image-recheck-20261120.qPXAvyfv"
soak="$evidence/w5-jemalloc-20261120.hLPizqfG"
accuracy="$evidence/w5-jemalloc-accuracy-20261120.CJrnRg9W"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w4-dist-d3d8fe6e5.FE5PU6Y4"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
(cd "$harness" && sha256sum -c "$stage/gpu-soak-inputs.sha256")
(cd "$prior" && sha256sum -c "$stage/gpu-accuracy-inputs.sha256")
export PYTHONPATH="$package:$accuracy:$soak:$harness/ci:$base/src/ci"

test "$(cat "$package/campaign.exit")" = 0
(cd "$package" && sha256sum -c dispatch.sha256 result.sha256)
require_image=$(cat "$package/image.id")
test "$(docker image inspect --format '{{.Id}}' "$require_image")" = "$require_image"
(cd "$accuracy" && sha256sum -c dispatch.sha256 result.sha256)
(cd "$soak" && sha256sum -c gpu-dispatch.sha256 preflight.sha256 result.sha256)
"$python" -B qualify_packaged_gpu.py bind "$package" "$evidence" "$stage" >gpu-checker-input.json

reservations() {
    local directory=$1 point=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$directory/idle-$point.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$directory/gpu7-$point.json"
}

run_scope() {
    local scope=$1 rc=0
    mkdir "$scope"
    reservations "$scope" before
    "$python" -B qualify_packaged_gpu.py admission "$package" "$evidence" "$stage" "$scope"
    printf 'START %s %s\n' "$scope" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/qualify_packaged_gpu.py" run "$package" "$evidence" "$stage" "$scope" \
        >"$scope/gpu-check.log" 2>&1 || rc=$?
    reservations "$scope" after
    printf '%d\n' "$rc" >"$scope/exit"
    printf 'END %s %s exit=%d\n' "$scope" "$(date -u +%FT%TZ)" "$rc"
    "$python" -B qualify_packaged_gpu.py verify "$package" "$evidence" "$stage" "$scope" >"$scope/verified.json"
}

run_scope cache-on
run_scope cache-off
"$python" -B qualify_packaged_gpu.py verify "$package" "$evidence" "$stage" cache-on cache-off >verified-soaks.json
sha256sum gpu-checker-input.json verified-soaks.json cache-{on,off}/verified.json >result.sha256
if "$python" -B -c 'import json; d=json.load(open("verified-soaks.json")); raise SystemExit(not all(item["status"] == "passed" for item in d["results"].values()))'; then
    run_scope accuracy
    "$python" -B qualify_packaged_gpu.py verify "$package" "$evidence" "$stage" cache-on cache-off accuracy >verified-all.json
    sha256sum verified-all.json >accuracy-result.sha256
else
    printf 'SKIP_STRICT_SOAK %s\n' "$(date -u +%FT%TZ)"
fi
