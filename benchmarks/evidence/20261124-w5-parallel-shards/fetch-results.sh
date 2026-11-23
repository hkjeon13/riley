#!/usr/bin/env bash
set -euo pipefail

stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-parallel-shards-20261124.iMsJcmnF
bundle=${1:?provide the destination archive path}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
sha256sum --strict -c dispatch.sha256 result.sha256 binary.sha256 >&2
root=$(dirname "$(dirname "$(pwd -P)")")
/usr/bin/python3 -B - "$root/models/Qwen2.5-Coder-14B-Instruct" <<'PYTHON' >model-layout.json
from hashlib import sha256
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
raw = (model / "riley-checkpoint.json").read_bytes()
digest = sha256(raw).hexdigest()
assert digest == "82a56cdab6e16e31478b7449be768dce15992b448d543fea4b2f2ed163237d00"
plan = json.loads(raw)
weights = [entry for entry in plan["files"] if entry["path"].endswith(".safetensors")]
assert len(weights) == 6 and all((model / entry["path"]).stat().st_size == entry["bytes"] for entry in weights)
print(json.dumps({
    "checkpoint_sha256": digest,
    "source_model": plan["source_model"],
    "source_revision": plan["source_revision"],
    "weight_shards": len(weights),
    "weight_bytes": sum(entry["bytes"] for entry in weights),
}, indent=2))
PYTHON
tar -czf - campaign.log campaign.exit dispatch.sha256 result.sha256 binary.sha256 \
    *.time *.stdout *.stderr cpu.json affinity.txt rustc.txt cargo.txt linux-tests.log model-layout.json
REMOTE
