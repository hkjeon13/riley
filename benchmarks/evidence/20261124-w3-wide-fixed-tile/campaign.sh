#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w3-prefill-continuation-20261124.A2s5mCyZ"
w5="$evidence/w5-pointer-soak-20261124.wa1UJ2ZB"
stock="$evidence/w4-pointer-syntax-20261124.HgwYVqCf"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c campaign-dispatch.sha256
test "$(cat build-wait.pid)" = 899144
test "$(cat build-wait.started)" = 2026-11-24T05:25:44Z
test "$(cat build-wait.exit)" = 0
test "$(cat build.exit)" = 0
sha256sum --strict -c build-wait-script.sha256
sha256sum --strict -c dispatch.sha256 build-result.sha256
test "$(cat "$w5/campaign.pid")" = 787131
test "$(cat "$w5/campaign.exit")" = 0
test "$(cat "$w5/recovery/campaign.exit")" = 0
test "$(cat "$prior/campaign.exit")" = 0
printf '%s  %s\n' \
    adf20db16066d71c3f38d9aa4a284368a6aea94cb051db9fbfb4596407f711e2 "$prior/dispatch.sha256" \
    5311e09b061adc3819ade1437945dd600992cc4e23cb8bb0a3d9c77a3833cbb0 "$prior/runtime.sha256" \
    a798aee83321633f6347b02ece564d8d42997d4d9d619df8cf83f1ac904be46c "$prior/result.sha256" \
    8436c243060357615f9ac5499edd98ba404a079e44d55a8df0f76bcf9ce0b6d8 "$stock/cpu-evidence/report.json" |
    sha256sum --strict -c
(cd "$prior" && sha256sum --strict -c runtime.sha256 result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
mkdir -p tools gpu{6,7}/rag/{1,2,3}
cp "$prior"/{arm.py,client.py,holdout.py,replay.py,holder.py,schedules.json,cpu-placement.json} .
cp "$prior"/tools/*.py tools/
"$runner" -B prepare.py
sha256sum arm.py client.py holdout.py replay.py holder.py schedules.json cpu-placement.json tools/*.py \
    riley-diagnostic flashinfer-attention-test >runtime.sha256
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci" PYTHONDONTWRITEBYTECODE=1

reservations() {
    local phase=$1
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$phase.json"
    done
    "$runner" -B - "$phase" "$w5" <<'PY'
import json
from pathlib import Path
import sys

for gpu, file in [(6, "idle"), (7, "gpu7")]:
    row = json.loads(Path(f"gpu{gpu}/{sys.argv[1]}.json").read_bytes())
    prior = json.loads((Path(sys.argv[2]) / f"recovery/{file}-after-stock.json").read_bytes())
    if (row["phase"] != "idle" or row["other_processes"] or
            row["allocation_bytes"] != 20 * 2**30 or row["torch_reserved_bytes"] != 20 * 2**30 or
            any(row[key] != prior[key] for key in ("gpu", "physical_index", "pid", "start_ticks"))):
        raise RuntimeError(f"original reservation is unavailable on GPU {gpu}")
PY
}

arm() {
    local index=$1 pass=$2 mode=$3 cpus rc=0
    case "$index" in
        6) cpus=24-33,72-81 ;;
        7) cpus=34-43,82-91 ;;
        *) return 2 ;;
    esac
    printf 'START gpu%s %s %s %s\n' "$index" "$pass" "$mode" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" W3_RAG_HISTORY=rag W3_REPEAT_PASS="$pass" W3_PREFILL_ORDER="$mode" \
        W3_CLIENT_CPU=$((index + 38)) W3_SERVER_CPUS="$cpus" \
        "${control[@]}" --index "$index" run -- taskset -c "$cpus" "$runner" -B arm.py riley \
        >"gpu$index/rag/$pass/$mode.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s %s exit=%s\n' "$index" "$pass" "$mode" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}

round() {
    local pass=$1 first second rc=0
    arm 6 "$pass" "$2" & first=$!
    arm 7 "$pass" "$3" & second=$!
    wait "$first" || rc=1
    wait "$second" || rc=1
    return "$rc"
}

reservations before
printf 'START_KERNEL gpu6 %s\n' "$(date -u +%FT%TZ)"
gpu_rc=0
"${control[@]}" --index 6 run -- timeout 180 taskset -c 24-27 \
    env LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 \
    "$stage/flashinfer-attention-test" --ignored --test-threads=1 \
    --exact flashinfer_rows_are_batch_invariant_and_close_to_the_page_path --nocapture \
    >gpu6/flashinfer-attention-test.log 2>&1 || gpu_rc=$?
printf 'END_KERNEL gpu6 %s exit=%s\n' "$(date -u +%FT%TZ)" "$gpu_rc"
reservations after-kernel
test "$gpu_rc" = 0
grep -Fq '1 passed; 0 failed' gpu6/flashinfer-attention-test.log

rc=0
if round 1 control candidate; then
    reservations between-1
    if round 2 candidate control; then
        reservations between-2
        round 3 control candidate || rc=1
    else
        rc=1
    fi
else
    rc=1
fi
reservations after
find gpu6 gpu7 -type f -print0 | sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
