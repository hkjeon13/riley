#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
profile="$evidence/w3-qwen4-profile-r2-20261119.Ydyk8B8n"
attention="$evidence/w4-riley-attention-controls-20261119.iJqg07GO"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
mkdir tools
cp -- "$profile"/tools/*.py tools/
cp -- "$profile/holder.py" .
/usr/bin/python3 -B - <<'PY'
from pathlib import Path

runner = Path('tools/run_profile_retry.py')
source = runner.read_text()
old = 'model == "qwen4" and mode != "profile"'
if source.count(old) != 1:
    raise ValueError('unexpected Qwen3-4B dispatch guard')
runner.write_text(source.replace(old, 'model == "qwen4" and mode == "vllm"'))
PY
sha256sum tools/*.py holder.py > staged-inputs.sha256
printf 'WAIT %s %s\n' "$attention" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 36000))
until test -s "$attention/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$attention/campaign.pid")" | grep -Fq -- "$attention/campaign.sh"; then
        test -s "$attention/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
test "$(cat "$profile/campaign.exit")" = 0
export PYTHONDONTWRITEBYTECODE=1
printf 'START Qwen3-4B without profiling %s\n' "$(date -u +%FT%TZ)"
status=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
    /usr/bin/python3 -B "$run/tools/run_profile_retry.py" qwen4 riley || status=$?
if test -d qwen4-riley; then
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > qwen4-riley/idle-after.json
fi
printf 'END Qwen3-4B without profiling %s exit=%d\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"
