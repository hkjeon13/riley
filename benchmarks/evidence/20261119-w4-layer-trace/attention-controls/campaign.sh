#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
root=$(cd -- "$run/../.." && pwd -P)
evidence="$root/evidence"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
baseline="$evidence/w4-layer-trace-r3-20261119.qpoSLydn"
preceding="$evidence/w3-qwen4-profile-r2-20261119.Ydyk8B8n"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$baseline/campaign.exit")" = 0
mkdir tools
for input in run_trace.py layer_trace.py long_accuracy.py compare_runner.py; do
    cp -- "$baseline/tools/$input" "tools/$input"
done
cp -- "$baseline/holder.py" "$baseline/fixtures/legacy.json" .
sha256sum -c runner-base.sha256
git apply --check runner.patch
git apply runner.patch
sha256sum -c runner.sha256
mv -- riley_attention_controls.py tools/
sha256sum tools/*.py legacy.json > staged-inputs.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 36000))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
export W4_GPU_INDEX=6 PYTHONDONTWRITEBYTECODE=1
rc=0
for arm in baseline shared-kv separate-write; do
    printf 'START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
    code=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 run -- \
        /usr/bin/python3 -B "$run/tools/run_trace.py" "$arm" || code=$?
    if test -d "$arm"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status > "$arm/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$arm" "$(date -u +%FT%TZ)" "$code"
    if (( code != 0 )); then
        rc=1
        if [[ "$arm" == baseline ]]; then break; fi
    fi
done
exit "$rc"
