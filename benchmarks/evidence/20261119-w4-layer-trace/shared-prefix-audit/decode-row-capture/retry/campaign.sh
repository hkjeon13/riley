#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
first="$evidence/w4-decode-logits-20261119.8VHxJvdK"
framing="$evidence/w5-http-framing-20261119.IyUaTevw"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
framing_paused=
finalize() {
    status=$?
    if [[ -n "$framing_paused" ]] && ps -o args= -p "$framing_paused" | grep -Fxq -- "bash $framing/campaign.sh" \
        && kill -CONT "$framing_paused"; then
        printf 'FRAMING_RESUMED %s %s\n' "$framing_paused" "$(date -u +%FT%TZ)"
    fi
    printf '%d\n' "$status" >"$stage/campaign.exit.tmp"
    mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"
}
trap finalize EXIT
cd "$stage"
sha256sum -c dispatch.sha256
test "$(cat "$first/campaign.exit")" = 1
grep -Fq "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 1" "$first/riley/client.log"
test -z "$(find "$first/riley/rows" -name '*.bf16' -print -quit)"
mkdir -p tools target/release
cp -- "$first/holder.py" .
cp -- "$first/tools/"{run_accuracy.py,long_accuracy.py,compare_runner.py,capture_decode_rows.py} tools/
cp -- "$first/binary.sha256" .
cp -R -- "$first/fixtures" .
ln -s "$first/target/release/riley" target/release/riley
sha256sum -c binary.sha256
sha256sum -c tools-base.sha256
gzip -dc restore-gzip-input.patch.gz | git apply --check -
gzip -dc restore-gzip-input.patch.gz | git apply -
printf '%s  %s\n' 15dbf1acd1f97907249b53ea84de291438a78d6ad570c80328e002fe6dfdd10d     tools/long_accuracy.py | sha256sum -c
PYTHONPATH="$stage/tools" "$python" -B - >cpu-probe.json <<'PY'
import gzip
import json
from pathlib import Path
from hashlib import sha256

from capture_decode_rows import files

root = Path("fixtures")
binding = {}
for name, expected in (("long", "699f1efd9cb26ccc5b848e641c5c9dc1ebc294904d005837ec3584ecf40e1de9"),
                       ("holdout", "2b28ec348efd0048cacec7e61ab2a3ffdbfc09787c7d40bcb0241982d87f8b36")):
    file = root / f"20261119-w4-{name}/riley/capture.json.gz"
    if sha256(gzip.decompress(file.read_bytes())).hexdigest() != expected:
        raise ValueError(f"{name} capture differs")
    binding[name] = expected
cases = [(suite, case.label, len(base["prompt_tokens"]), offset) for suite, case, base, _, offset, _ in files(root)]
if len(cases) != 8:
    raise ValueError("expected eight original requests before occupying a GPU")
print(json.dumps({"source_sha256": binding, "cases": cases}, indent=2))
PY

holder_status() {
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
if ! test -s "$framing/campaign.exit"; then
    framing_pid=$(cat "$framing/campaign.pid")
    test "$framing_pid" = 3949016
    ps -o args= -p "$framing_pid" | grep -Fxq -- "bash $framing/campaign.sh"
    if grep -Eq '^START cache-(on|off) ' "$framing/campaign.log"; then
        printf 'WAIT_FOR_FRAMING_GPU %s\n' "$(date -u +%FT%TZ)"
        until test -s "$framing/campaign.exit"; do
            ps -o args= -p "$framing_pid" | grep -Fxq -- "bash $framing/campaign.sh" || test -s "$framing/campaign.exit"
            sleep 25
        done
    else
        framing_paused=$framing_pid
        kill -STOP "$framing_paused"
        ps -o stat= -p "$framing_paused" | grep -Eq '^T'
        ! grep -Eq '^START cache-(on|off) ' "$framing/campaign.log"
        printf 'FRAMING_WAIT_FOR_W4 %s %s\n' "$framing_paused" "$(date -u +%FT%TZ)"
    fi
fi
holder_status 6 >idle-before-6.json
holder_status 7 >idle-before.json
"$python" -B - <<'PY'
import json
from pathlib import Path

for name, index, pid in (("idle-before-6.json", 6, 2431228), ("idle-before.json", 7, 2431237)):
    state = json.loads(Path(name).read_bytes())
    if (state["physical_index"] != index or state["pid"] != pid or state["phase"] != "idle"
            or state["other_processes"] or state["allocation_bytes"] != state["torch_reserved_bytes"]
            or state["allocation_bytes"] != 20 * 1024**3):
        raise ValueError(f"GPU {index} is not in the original 20 GiB reservation")
PY
export PYTHONDONTWRITEBYTECODE=1 W3_GPU_INDEX=7
printf 'START riley %s\n' "$(date -u +%FT%TZ)"
result=0
"$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run --     "$python" -B "$stage/tools/run_accuracy.py" riley || result=$?
holder_status 7 >idle-after.json
printf 'END riley %s exit=%d\n' "$(date -u +%FT%TZ)" "$result"
test "$result" = 0
"$python" -B - <<'PY'
from pathlib import Path
import json

read = lambda name: json.loads(Path(name).read_bytes())
before, after = read("idle-before.json"), read("idle-after.json")
if not all(before[key] == after[key] for key in before if key != "updated_ns"):
    raise ValueError("the original GPU reservation was not restored")
if len(read("riley/client.log")["cases"]) != 8 or len(list(Path("riley/rows").glob("*.bf16"))) != 8:
    raise ValueError("the expected eight BF16 rows were not recovered")
PY
sha256sum riley/{receipt.json,client.log,server.log} idle-{before,after}.json riley/rows/*.bf16 >result.sha256
