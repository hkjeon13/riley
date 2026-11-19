#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
candidate="$evidence/w5-jemalloc-20261120.hLPizqfG"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w4-dist-d3d8fe6e5.FE5PU6Y4"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
(cd "$candidate" && sha256sum -c dispatch.sha256 gpu-dispatch.sha256 preflight.sha256)
(cd "$prior" && sha256sum -c "$stage/accuracy-harness.sha256")
export PYTHONPATH="$candidate:$prior/harness/ci:$base/src/ci"

printf 'WAIT %s %s\n' "$candidate" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$candidate/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$candidate/campaign.pid")" | grep -Fq -- "$candidate/campaign.sh"; then
        test -s "$candidate/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
test "$(cat "$candidate/campaign.exit")" = 0
(cd "$candidate" && sha256sum -c result.sha256)
"$python" -B qualify_allocator.py decision "$candidate" "$evidence" "$stage" >decision.json
if ! "$python" -B -c 'import json; raise SystemExit(not json.load(open("decision.json"))["run_accuracy"])'; then
    printf 'SKIP_STRICT_SOAK %s\n' "$(date -u +%FT%TZ)"
    sha256sum decision.json >result.sha256
    exit 0
fi
"$python" -B qualify_allocator.py inputs "$candidate" "$evidence" "$stage" >inputs-verified.json
for idx in 6 7; do
    if test "$idx" = 6; then file=idle-before.json; else file=gpu7-before.json; fi
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$idx" status >"$file"
done
"$python" -B qualify_allocator.py admission "$candidate" "$evidence" "$stage"
printf 'START accuracy %s\n' "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/qualify_allocator.py" run "$candidate" "$evidence" "$stage" >gpu-check.log 2>&1 || rc=$?
for idx in 6 7; do
    if test "$idx" = 6; then file=idle-after.json; else file=gpu7-after.json; fi
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$idx" status >"$file"
done
printf '%d\n' "$rc" >gpu-exit
printf 'END accuracy %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B qualify_allocator.py verify "$candidate" "$evidence" "$stage" >verified-accuracy.json
sha256sum decision.json inputs-verified.json idle-{before,after}.json gpu7-{before,after}.json \
    allocator-{before,after}.json verified-accuracy.json >result.sha256
