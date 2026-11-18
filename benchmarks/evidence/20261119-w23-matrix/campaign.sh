#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
base="$(dirname "$run")"
hold="$base/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
dependencies=(
  w3-rag-profile-20261119.FE1ZUFeE
  w4-layer-trace-r2-20261119.P5fMeEP9
  w4-gsm8k-20261119.7FpfoueC
)
for item in "${dependencies[@]}"; do
    dependency="$base/$item"
    printf 'WAIT %s %s\n' "$item" "$(date -u +%FT%TZ)"
    deadline=$((SECONDS + 28800))
    until test -s "$dependency/campaign.exit"; do
        if ! ps -o args= -p "$(cat "$dependency/campaign.pid")" | grep -Fq -- "$dependency/campaign.sh"; then
            test -s "$dependency/campaign.exit" && break
            exit 75
        fi
        if (( SECONDS >= deadline )); then exit 75; fi
        sleep 20
    done
done
export W3_GPU_INDEX=6 PYTHONDONTWRITEBYTECODE=1
arms=(
    'immediate 32 1'
    'after-submit 2 1'
    'immediate 2 1'
    'after-submit 32 1'
    'after-submit 32 2'
    'immediate 2 2'
    'after-submit 2 2'
    'immediate 32 2'
)
result=0
for row in "${arms[@]}"; do
    read -r issue cap seed <<<"$row"
    label="$issue-$cap-s$seed"
    printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 run -- \
        /usr/bin/python3 -B "$run/tools/measure_matrix.py" "$issue" "$cap" "$seed" || rc=$?
    if [[ -d "$run/$label" ]]; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status >"$run/$label/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$label" "$(date -u +%FT%TZ)" "$rc"
    if (( rc != 0 )); then result=1; fi
done
exit "$result"
