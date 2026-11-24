#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'echo "$?" > exit-status' EXIT
base=$(realpath ../long-tile)
root=$(realpath ../../..)
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
model="$root/models/Qwen2.5-7B-Instruct"
(cd "$base" && sha256sum -c binary.sha256)
cp "$base/binary.sha256" .
sha256sum *.py *.sh *registration.json > inputs.sha256
cp "$model/riley-checkpoint.json" checkpoint.json
python3 - <<'PY' > model-files.sha256
import json
for row in json.load(open("checkpoint.json"))["files"]:
    print(row["sha256"],row["path"],sep="  ")
PY
(cd "$model" && sha256sum -c -) < model-files.sha256 > validated-files.txt
export TRIAL_REQUEST_TIMEOUT_MS=900000 TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048 TRIAL_PREFIX_BLOCKS=0
arm() {
    local gpu=$1 mode=$2 attention=fixed-tile-unsplit-v1
    if [[ $mode == page ]]; then attention=page-v1; fi
    local ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
    REGISTER_TOKEN_WITNESS=1 TRIAL_ATTENTION_MODE="$attention" "${ctl[@]}" run -- bash run-profile.sh "$mode" "$base/target/release/riley" 0 control > "$mode.log" 2>&1 || return 1
    "${ctl[@]}" status > "$mode/lease-after.json"
}
arm 6 page & first=$!
rc=0
arm 7 tile || rc=1
wait "$first" || rc=1
test "$rc" = 0
ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index 7)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 "${ctl[@]}" run -- taskset -c 34-37 "$root/tools/vllm-venv/bin/python" -B register_reference.py hf.json >hf.log 2>&1
"${ctl[@]}" status > hf-lease-after.json
touch released
