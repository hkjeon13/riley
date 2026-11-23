#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-kv-history-gpu-20261123\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact GPU evidence path.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
checksum=$(shasum -a 256 "$here/result.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$checksum'" <<'REMOTE'
set -euo pipefail
stage=$1
cd "$stage"
printf '%s  verification/result.sha256\n' "$2" | sha256sum --strict -c >/dev/null
sha256sum --strict -c verification/result.sha256 >/dev/null
evidence=$(dirname "$stage")
prior="$evidence/w4-long-code-gpu-20261122.QAbtnxKV"
build=$(cat "$prior/build-path.txt")
export PYTHONPATH="$stage/verification:$stage:$prior:$build:$build/tools"
"$(dirname "$evidence")/tools/vllm-venv/bin/python" -B - "$2" <<'PY'
import json
from pathlib import Path
import sys
from trial import OLD_KEYS, capture, differences, require

results = {}
start = OLD_KEYS.index(("attention_residual", "1"))
for role, donor in (("baseline", "candidate"), ("candidate", "baseline")):
    swapped = capture(Path(role) / "history", swapped=True, boundary=True)
    original = capture(Path(donor) / "calibration")
    equal = []
    for name, layer in OLD_KEYS[start:]:
        key = name, layer
        delta = differences(swapped[key], original[key])
        if delta["changed_words"]:
            first = {"stage": name, "layer": int(layer) if layer != "-" else None, **delta}
            break
        equal.append({"stage": name, "layer": int(layer) if layer != "-" else None})
    else:
        raise ValueError(f"no difference after layer-1 attention: {role}")
    require(equal and equal[0]["stage"] == "attention_residual", f"layer-1 attention differs: {role}")
    results[role] = {"donor": donor, "equal_rows": equal, "first_difference": first}
print(json.dumps({"capture_checksums_sha256": sys.argv[1], "directions": results}, indent=2))
PY
REMOTE
