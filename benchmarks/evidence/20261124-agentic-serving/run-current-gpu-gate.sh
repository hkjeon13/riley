#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in
    6) modes=(vllm riley model application) ;;
    7) modes=(application model riley vllm) ;;
    *) exit 2 ;;
esac
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
controller=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
sha256sum -c build.sha256
export RILEY_TRIAL_BINARY="$PWD/current-riley"
RILEY_TRIAL_BINARY_SHA256=$(sha256sum "$RILEY_TRIAL_BINARY" | cut -d' ' -f1)
export RILEY_TRIAL_BINARY_SHA256 AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct
export RILEY_TRIAL_VLLM_RESERVATION=0.75
mkdir -p "gpu$gpu"
for mode in "${modes[@]}"; do
    dir="gpu$gpu/$mode"
    marker="$dir.verified-run"
    [[ ! -f "$marker" ]] || continue
    attempt=$(mktemp -d "$dir.XXXXXXXX")
    out="$attempt/result"
    unset CODER_CATALOG_COMPARE CODER_NAMED_MODE
    case "$mode" in
        vllm) export CODER_CATALOG_COMPARE=1; command=(python3 -B run-reference.py "$out" --seed 5091) ;;
        riley) export CODER_CATALOG_COMPARE=1; command=(bash run-profile.sh "$out" stock 5091 wide) ;;
        *) export CODER_NAMED_MODE="$mode"; command=(bash run-profile.sh "$out" stock 5091 wide) ;;
    esac
    "${controller[@]}" run -- "${command[@]}" > "$attempt/controller.log" 2>&1
    "${controller[@]}" status > "$attempt/lease-after.json"
    python3 - "$out" "$mode" <<'PY'
import json, sys
from pathlib import Path
out, mode = Path(sys.argv[1]), sys.argv[2]
waves = json.loads((out/"waves.json").read_text())
assert [len(w["jobs"]) for w in waves] == [4,8,16,16]
if mode in ("riley","vllm"):
    fixed=json.loads((out/"fixed.json").read_text())
    assert not (out/"fixed-error.json").exists()
    assert sum(row["phase"]=="measured" for row in fixed)==3
PY
    printf '%s\n' "$attempt" > "$marker"
done
