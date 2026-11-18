#!/usr/bin/env bash
set -u
run=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-mixed-soak-20261119.YJg0Xzhk
hold=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w2-sm17-20261118.6UMfeHh2
export W5_GPU_INDEX=7
export PYTHONDONTWRITEBYTECODE=1
exit_code=0
for arm in cache-off cache-on; do
    printf 'START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- /usr/bin/python3 -B "$run/tools/run_soak.py" "$arm" || rc=$?
    if test -d "$run/$arm"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status >"$run/$arm/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$arm" "$(date -u +%FT%TZ)" "$rc"
    if test -f "$run/$arm/client.log"; then tail -n 1 "$run/$arm/client.log"; fi
    if test "$rc" -ne 0; then
        exit_code=1
        if ! test -s "$run/$arm/soak.json" || ! test -s "$run/$arm/shutdown-metrics.json"; then
            break
        fi
    fi
done
for index in 6 7; do
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index "$index" status >"$run/gpu${index}-final.json"
done
printf '%d\n' "$exit_code" >"$run/campaign.exit"
exit "$exit_code"
