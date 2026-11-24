#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
stock="$evidence/w4-pointer-syntax-20261124.HgwYVqCf"
previous="$evidence/w5-dot-allocator-20261123.uoCr9cHQ"
builder="$evidence/w5-allocator-rebuild-20261123.oN3VEEWP"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
package="$evidence/w5-jemalloc-20261120.hLPizqfG/libjemalloc2_5.2.1-4ubuntu1_amd64.deb"
cpu="$stock/cpu-evidence/report.json"
candidate="$stage/allocator"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
model_sha=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu6=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
gpu7=GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0

trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$stock"
    test "$(cat campaign.exit)" = 0
    printf 'ecb3f7b5293406f6e94eab0a107aecd24f156958dd2d364e66164463b78185a8  result.sha256\n8436c243060357615f9ac5499edd98ba404a079e44d55a8df0f76bcf9ce0b6d8  cpu-evidence/report.json\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$previous"
    printf '6cbece97b900d4d60c512d87355f01672bd549c455621e75986f7697b52f8369  allocator.py\n4e2b6d74bd401b6e2a33cff39c174a1dbc5e1b33fb7b1094fd20f68e60a972cc  verify.py\n5e506712f4f1be0b955874179a40f4af490b07847630fd756472ad7e29376a33  verify_allocator_image.py\n' | sha256sum --strict -c)
(cd "$builder"
    printf 'c7c4d0f99bf7b51f9712ee39f8f123f19fe31e3262fbfa1c24653986ec33e984  benchmarks/live/build_jemalloc_trial_image.py\n' | sha256sum --strict -c)
(cd "$run" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
export PYTHONPATH="$previous:$run/harness/ci:$stock/src/ci:$rag:$prior/src/ci"

reservations() {
    local folder=$1 phase=$2
    "${control[@]}" --index 6 status >"$folder/idle-$phase.json"
    "${control[@]}" --index 7 status >"$folder/gpu7-$phase.json"
    "$runner" -B - "$folder" "$phase" <<'PY'
from pathlib import Path
import sys
from verify import check_reservations
check_reservations(Path(sys.argv[1]), sys.argv[2])
PY
}

run_arm() {
    local folder=$1 label=$2 index=$3 rc=0
    shift 3
    printf 'START %s %s %d %s\n' "${folder##*/}" "$label" "$index" "$(date -u +%FT%TZ)"
    "${control[@]}" --index "$index" run -- "$runner" -B "$@" >"$folder/$label.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$folder/$label.exit"
    printf 'END %s %s %d %s exit=%d\n' "${folder##*/}" "$label" "$index" "$(date -u +%FT%TZ)" "$rc"
    test -s "$folder/$label/report.json"
}

candidate_arm() {
    local label=$1 index=$2
    run_arm "$candidate" "$label" "$index" "$previous/allocator.py" run "$candidate" "$cpu" "$base" "$label"
}

stock_arm() {
    local label=$1 index=$2 gpu=$3
    run_arm "$stage/stock" "$label" "$index" "$run/harness/ci/check_distribution_gpu.py" \
        "$cpu" "$gpu" "$base/model" "$model_sha" "$stage/stock/$label" --tools --mixed-soak "$label" --require-prefix-cache-metrics
}

"$runner" -B "$builder/benchmarks/live/build_jemalloc_trial_image.py" "$cpu" "$package" "$base/driver/libcuda.so.1" "$candidate"
"$runner" -B "$previous/allocator.py" bind "$candidate" "$cpu" >"$candidate/gpu-checker-input.json"
printf 'IMAGE_VERIFIED %s %s\n' "$(cat "$candidate/image.id")" "$(date -u +%FT%TZ)"
mkdir stock
reservations "$candidate" before
reservations "$stage/stock" before

round() {
    local first=$1 second=$2 left right rc=0
    "$first" cache-off 6 "$gpu6" &
    left=$!
    "$second" cache-on 7 "$gpu7" &
    right=$!
    wait "$left" || rc=1
    wait "$right" || rc=1
    return "$rc"
}
round stock_arm candidate_arm
reservations "$stage/stock" between
round candidate_arm stock_arm
reservations "$candidate" after-soaks
reservations "$stage/stock" after-soaks
candidate_arm accuracy 6
reservations "$candidate" after
test "$(cat "$candidate/accuracy.exit")" = 0
"$runner" -B "$previous/allocator.py" verify "$candidate" "$cpu" >"$candidate/verified.json"

"$runner" -B compare-soaks.py "$stage/stock" "$candidate" "$cpu" >verified.json
sha256sum verified.json allocator/{verified-image,verified,gpu-checker-input}.json allocator/context.sha256 \
    allocator/{cache-off,cache-on,accuracy}.exit allocator/{cache-off,cache-on,accuracy}/{report,api,shutdown}.json \
    allocator/cache-{off,on}/soak.json allocator/{cache-off,cache-on,accuracy}-allocator-{before,after}.json \
    allocator/idle-{before,after-soaks,after}.json allocator/gpu7-{before,after-soaks,after}.json \
    stock/{cache-off,cache-on}.exit stock/{cache-off,cache-on}/{report,api,shutdown,soak}.json \
    stock/idle-{before,between,after-soaks}.json stock/gpu7-{before,between,after-soaks}.json >result.sha256
printf 'SOAKS_VERIFIED %s\n' "$(date -u +%FT%TZ)"
mkdir recovery
(cp recovery.sh recovery/ && cd recovery && bash recovery.sh >campaign.log 2>&1)
printf 'RECOVERY_VERIFIED %s\n' "$(date -u +%FT%TZ)"
