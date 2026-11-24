#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
trap 'echo "$?" > "exit-$gpu"' EXIT
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
case "$gpu" in
  6) arms=(riley-hide riley-keep vllm-keep vllm-hide);;
  7) arms=(vllm-hide vllm-keep riley-keep riley-hide);;
  *) exit 2;;
esac
for arm in "${arms[@]}"; do
    out=gpu$gpu-$arm
    export FINAL_LAYOUT=${arm#*-}
    case "$arm" in
      riley-*) cmd=(bash run-profile.sh "$out" stock 761 wide);;
      vllm-*) cmd=(python3 -B run-reference.py "$out" --seed 761);;
    esac
    rc=0
    "${ctl[@]}" run -- "${cmd[@]}" > "$out.log" 2>&1 || rc=$?
    "${ctl[@]}" status > "$out.lease.json"
    ((rc == 0)) || exit "$rc"
done
