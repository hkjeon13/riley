#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in
    6) model=Qwen2.5-3B-Instruct; order=(riley vllm vllm riley);;
    7) model=Qwen2.5-Coder-7B-Instruct; order=(vllm riley riley vllm);;
    *) exit 2;;
esac
export AGENT_DECODE_MODEL="$model" FROZEN_INPUT_DIR="$PWD/frozen"
root=$(realpath ../../..)
h="$root/evidence/w2-sm17-20261118.6UMfeHh2"
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
mkdir -p "gpu$gpu"
cp "$root/models/$model/riley-checkpoint.json" "gpu$gpu/checkpoint.json"
(cd "$root/models/$model" && sha256sum *.json *.safetensors) > "gpu$gpu/files.sha256"
for i in "${!order[@]}"; do
    engine=${order[$i]}
    out="gpu$gpu/$i-$engine"
    test ! -e "$out"
    if [[ $engine == riley ]]; then
        "${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 wide
    else
        "${ctl[@]}" run -- python3 -B run-reference.py "$out" --seed 761
    fi
    "${ctl[@]}" status > "$out/lease-after.json"
done
