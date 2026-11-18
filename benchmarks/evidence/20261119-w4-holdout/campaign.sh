#!/usr/bin/env bash
set -u
root=/mnt/efs/nlp/psyche/riley-gap-20261102
run="$root/evidence/w4-holdout-20261119.cgmtjz4x"
soak="$root/evidence/w5-mixed-soak-20261119.YJg0Xzhk"
hold="$root/evidence/w2-sm17-20261118.6UMfeHh2"
export PATH="$root/tools/vllm-venv/bin:$PATH"
export VLLM_CACHE_ROOT="$run/vllm-cache"
export TORCHINDUCTOR_CACHE_DIR="$run/torch-cache"
export W4_GPU_INDEX=7
export PYTHONDONTWRITEBYTECODE=1
printf 'WAITING FOR %s %s\n' "$soak/campaign.exit" "$(date -u +%FT%TZ)"
for attempt in $(seq 1 300); do
    if test -s "$soak/campaign.exit"; then
        break
    fi
    if ! ps -o args= -p "$(cat "$soak/campaign.pid")" | grep -Fq -- "$soak/campaign.sh"; then
        printf 'W5 process no longer matches its campaign\n' >&2
        break
    fi
    sleep 15
done
if ! test -s "$soak/campaign.exit"; then
    printf '75\n' >"$run/campaign.exit"
    exit 75
fi
exit_code=0
for phase in riley hf vllm; do
    printf 'START %s %s\n' "$phase" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- /usr/bin/python3 -B "$run/tools/run_accuracy.py" "$phase" || rc=$?
    if test -d "$run/$phase"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status >"$run/$phase/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$phase" "$(date -u +%FT%TZ)" "$rc"
    if test "$rc" -ne 0; then
        exit_code=1
        if test "$phase" = riley; then break; fi
    fi
done
printf '%d\n' "$exit_code" >"$run/campaign.exit"
exit "$exit_code"
