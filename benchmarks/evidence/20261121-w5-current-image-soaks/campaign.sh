#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
image="$evidence/w5-current-package-20261121.f6SEbjvA"
model="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l/model"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
model_sha=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
cpu="$image/cpu-evidence/report.json"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum -c dispatch.sha256
test "$(cat "$image/campaign.exit")" = 0
(cd "$image" && sha256sum -c runtime-tools.sha256 >/dev/null)
printf '%s  %s\n' 9aba66838fb889e749e9a9235e0d633408d79917bc71a5924abb4cdc5396ddf3 "$cpu" | sha256sum -c
printf '%s  %s\n' "$model_sha" "$model/riley-checkpoint.json" | sha256sum -c

reserve_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}

for index in 6 7; do
    reserve_status "$index" >"gpu$index-before.json"
done
"$python" -B verify_soaks.py --reservations "$stage" before

run_arm() {
    local mode=$1 index=$2 uuid=$3 rc=0
    printf 'START %s %s %s\n' "$mode" "$index" "$(date -u +%FT%TZ)"
    PYTHONPATH="$image/src/ci" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
        "$python" -B "$image/harness/ci/check_distribution_gpu.py" "$cpu" "$uuid" \
        "$model" "$model_sha" "$stage/$mode" --tools --mixed-soak "$mode" >"$mode.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$mode.exit"
    printf 'END %s %s %s exit=%d\n' "$mode" "$index" "$(date -u +%FT%TZ)" "$rc"
}

run_arm cache-off 6 GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7 &
off=$!
run_arm cache-on 7 GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0 &
on=$!
wait "$off"
wait "$on"
for index in 6 7; do
    reserve_status "$index" >"gpu$index-after.json"
done
"$python" -B verify_soaks.py "$stage" >verified.json
sha256sum verified.json cache-{off,on}.exit cache-{off,on}/{api,soak,report,shutdown}.json gpu{6,7}-{before,after}.json >result.sha256
