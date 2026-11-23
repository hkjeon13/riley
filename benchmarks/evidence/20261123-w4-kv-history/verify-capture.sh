#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
verification=$(pwd -P)
stage=$(dirname "$verification")
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
[[ $stage =~ ^"$evidence"/w4-kv-history-gpu-20261123\.[[:alnum:]]+$ ]]
[[ $(basename "$verification") == verification ]]
(cd "$verification" && sha256sum --strict -c dispatch.sha256)
cd "$stage"
cmp dispatch.sha256 verification/capture-dispatch.sha256
sha256sum --strict -c dispatch.sha256
test "$(cat campaign.exit)" = 1
grep -Fxq "ValueError: GPU lease changed: $stage/gpu6-after.json" campaign.log
test "$(cat baseline/worker.exit)" = 0
test "$(cat candidate/worker.exit)" = 0
python3 -B - <<'PY'
from pathlib import Path
old = Path("trial.py").read_text()
new = Path("verification/trial.py").read_text()
wrong = 'reservation(stage / f"gpu{i}-after.json", i)'
fixed = 'reservation(stage / f"gpu{i}-after.json", int(i))'
if old.count(wrong) != 1 or old.replace(wrong, fixed, 1) != new:
    raise ValueError("expected only the GPU index conversion in the post-capture checker")
PY
build=$(cat build-path.txt)
[[ $build =~ ^"$evidence"/w4-kv-history-build-20261123\.[[:alnum:]]+$ ]]
(cd "$build" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-inputs.sha256 build-result.sha256 patched-source.sha256)
prior="$evidence/w4-long-code-gpu-20261122.QAbtnxKV"
prior_build=$(cat "$prior/build-path.txt")
export PYTHONPATH="$stage:$prior:$prior_build:$prior_build/tools"
"$root/tools/vllm-venv/bin/python" -B verification/trial.py verify "$stage" "$build" >verification/verified.json.tmp
mv -- verification/verified.json.tmp verification/verified.json
{
    printf '%s\n' build-path.txt boundary_trial.py campaign.exit campaign.log checkpoint-verify.log dispatch.sha256 trial.py gpu{6,7}-{before,after}.json \
        verification/capture-dispatch.sha256 verification/dispatch.sha256 verification/trial.py verification/verify-capture.sh verification/verified.json
    find baseline candidate -type f ! -name calibrated -print
} | LC_ALL=C sort | xargs sha256sum >verification/result.sha256
