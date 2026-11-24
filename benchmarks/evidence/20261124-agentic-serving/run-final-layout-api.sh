#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
parent=$(realpath "$1")
gpu=$2
trap 'echo "$?" > "exit-$gpu"' EXIT
until [[ -s build.exit && -s "$parent/exit-$gpu" ]]; do sleep 4; done
test "$(cat build.exit)" = 0
test "$(cat "$parent/exit-$gpu")" = 0
sha256sum -c binary.sha256
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
case "$gpu" in
  6) out=named; export FINAL_LAYOUT_NAMED=1;;
  7) out=duplicate;;
  *) exit 2;;
esac
rc=0
FINAL_LAYOUT=keep "${ctl[@]}" run -- bash run-profile.sh "$out" "$PWD/target/release/riley" 761 wide > "$out.log" 2>&1 || rc=$?
"${ctl[@]}" status > "$out.lease.json"
exit "$rc"
