#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
stock="$evidence/w4-dot-package-20261123.nKyEd9x9"
variant="$evidence/w5-dot-allocator-20261123.uoCr9cHQ"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
cpu="$stock/cpu-evidence/report.json"
model_sha=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$stock"
    test "$(cat campaign.exit)" = 0
    printf 'd412c16ac878610b3470352b1e6f97a7da408f68aa2d0e13072e8da134e202dc  result.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$variant"
    test "$(cat campaign.exit)" = 0
    printf 'c0af16e2c614fffd56401d758411ba64d0c4c724354270066b29b306214d933a  result.sha256\n6cbece97b900d4d60c512d87355f01672bd549c455621e75986f7697b52f8369  allocator.py\n4e2b6d74bd401b6e2a33cff39c174a1dbc5e1b33fb7b1094fd20f68e60a972cc  verify.py\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$run" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
export PYTHONPATH="$variant:$run/harness/ci:$stock/src/ci:$rag:$prior/src/ci"

reservations() {
    local phase=$1
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"gpu7-$phase.json"
    "$runner" -B - "$phase" <<'PY'
from pathlib import Path
import sys
from current_binding import check_reservations
check_reservations(Path.cwd(), sys.argv[1])
PY
}

run_arm() {
    local label=$1 index=$2 gpu=$3 rc=0
    printf 'START %s %s %s\n' "$label" "$index" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
        "$runner" -B "$run/harness/ci/check_distribution_gpu.py" "$cpu" "$gpu" "$base/model" "$model_sha" \
        "$stage/$label" --tools --mixed-soak "$label" --require-prefix-cache-metrics >"$label.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$label.exit"
    printf 'END %s %s %s exit=%d\n' "$label" "$index" "$(date -u +%FT%TZ)" "$rc"
    test -s "$label/report.json"
}

reservations before
run_arm cache-off 6 GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7 &
off=$!
run_arm cache-on 7 GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0 &
on=$!
rc=0
wait "$off" || rc=1
wait "$on" || rc=1
reservations after-soaks
test "$rc" = 0
CUDA_VISIBLE_DEVICES= "$runner" -B compare-soaks.py "$stage" "$variant" "$cpu" >verified.json
sha256sum verified.json {cache-off,cache-on}.exit {cache-off,cache-on}/{report,api,shutdown,soak}.json \
    idle-{before,after-soaks}.json gpu7-{before,after-soaks}.json >result.sha256
