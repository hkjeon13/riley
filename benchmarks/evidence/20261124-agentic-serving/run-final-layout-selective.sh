#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
previous=$(realpath "$1")
gpu=$2
trap 'echo "$?" > "exit-$gpu"' EXIT
until [[ -s "$previous/exit-$gpu" ]]; do sleep 3; done
test "$(cat "$previous/exit-$gpu")" = 0
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
case "$gpu" in
  6) order=(selective defined);;
  7) order=(defined selective);;
  *) exit 2;;
esac
for mode in "${order[@]}"; do
  out=gpu$gpu-$mode
  rc=0
  FINAL_LAYOUT=keep FINAL_REPORT_MODE="$mode" "${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 wide > "$out.log" 2>&1 || rc=$?
  "${ctl[@]}" status > "$out.lease.json"
  ((rc==0)) || exit "$rc"
done
