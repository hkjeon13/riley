#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
trap 'echo "$?" > "exit-$gpu"' EXIT
until [[ -s build.exit ]]; do sleep 3; done
test "$(cat build.exit)" = 0
sha256sum -c binary.sha256
if [[ $gpu == cpu ]]; then
    CUDA_VISIBLE_DEVICES= taskset -c 62 python3 -B chat_render_trial.py cpu . ../../../models/Qwen2.5-7B-Instruct recorded.json linux.json
    exit 0
fi
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
case "$gpu" in
    6) order=(control candidate);;
    7) order=(candidate control);;
    *) exit 2;;
esac
for arm in "${order[@]}"; do
    out=gpu$gpu-$arm
    rc=0
    CHAT_RENDER_TRIAL=1 "${ctl[@]}" run -- bash run-profile.sh "$out" "$PWD/server-$arm" 761 control > "$out.log" 2>&1 || rc=$?
    "${ctl[@]}" status > "$out.lease.json"
    ((rc==0)) || exit "$rc"
done
