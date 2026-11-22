#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
down="$evidence/w4-fp32-down-serving-20261120.Saj3XXSY"
attention="$evidence/w4-o31-serving-20261120.7W3nr5JH"
model="$root/models/Qwen2.5-7B-Instruct"
venv="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$evidence" && sha256sum --strict -c "$stage/captures.sha256")
for archived in "$attention" "$down"; do
    (cd "$archived" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c result.sha256 >/dev/null)
done
printf '%s  %s\n' 655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f "$model/riley-checkpoint.json" | sha256sum -c
export PYTHONPATH="$down/tools"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1
"$venv" -B - "$evidence" <<'PY'
import sys
from pathlib import Path
from replay import cases
rows, _ = cases(Path(sys.argv[1]))
print(f"PRECHECK {len(rows)} inputs; {sum(len(row[2]['points']) for row in rows)} pinned comparison positions", flush=True)
PY
image=sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
docker run --rm --network none --env NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)"     --mount "type=bind,src=$model,dst=/models,readonly" "$image" riley-checkpoint verify --checkpoint /models >checkpoint-verify.log
grep -qx 'checkpoint verify: verified' checkpoint-verify.log
reservation() {
    local index=$1 point=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point.json"
}
for index in 6 7; do reservation "$index" before; done
gpu=$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)
test "$gpu" = GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0
printf 'START gpu7 %s\n' "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run --     env CUDA_VISIBLE_DEVICES="$gpu" "$venv" -B "$stage/replay.py" "$evidence" "$model" "$stage/replay.json" >replay.log 2>&1 || rc=$?
for index in 6 7; do reservation "$index" after; done
printf 'END gpu7 %s rc=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
sha256sum replay.json replay.log checkpoint-verify.log gpu{6,7}-{before,after}.json >result.sha256
