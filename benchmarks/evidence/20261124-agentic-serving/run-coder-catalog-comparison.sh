#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) modes=(vllm riley);; 7) modes=(riley vllm);; *) exit 2;; esac
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct CODER_CATALOG_COMPARE=1
mkdir "gpu$gpu"
for engine in "${modes[@]}"; do
    out="gpu$gpu/$engine"
    if [[ $engine == riley ]]; then cmd=(bash run-profile.sh "$out" stock 5091 wide)
    else cmd=(python3 -B run-reference.py "$out" --seed 5091); fi
    "${ctl[@]}" run -- "${cmd[@]}" > "$out.controller.log" 2>&1
    "${ctl[@]}" status > "$out/lease-after.json"
done
