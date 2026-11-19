#!/usr/bin/env bash
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
export PYTHONPATH="$1/tools"
source=$(dirname "$1")/w3-qkv-serving-20261119.MhydzNsX
cmp verified.json <(/usr/bin/python3 -B verify_fixed.py "$1" "$source")
/usr/bin/python3 -B - "$1" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
from fixed_w3 import ORDER, require

stage=Path(sys.argv[1])
name=stage.name
raw=(stage/"verified.json").read_bytes()
source=json.loads((stage/"preflight.json").read_bytes())
chronology=[]
for line in (stage/"campaign.log").read_text().splitlines():
    words=line.split()
    if words and words[0] in ("START", "END"):
        chronology.append(words)
required=[(event,arm) for arm in ORDER for event in ("START","END")]
require([tuple(row[:2]) for row in chronology]==required and
        all(row[3:]==["exit=0"] for row in chronology[1::2]) and
        [row[2] for row in chronology]==sorted(row[2] for row in chronology),
        "incomplete or inconsistent experiment chronology")
selection={}
for arm in ORDER:
    log=(stage/f"qwen14-{arm}"/"server.log").read_bytes()
    markers=[line for line in log.splitlines() if line.startswith(b"RILEY_W3_QKV1024")]
    require(bool(markers)==arm.startswith("qkv-") and
            all(line==b"RILEY_W3_QKV1024 recorded=1 cuda=120405" for line in markers),
            "recorded QKV selection did not match the declared arm")
    selection[arm]={"successful_runtime_selection_records":len(markers),"server_log_sha256":sha256(log).hexdigest()}
scopes={key:{"seed":spec["seed"],"requests":spec["requests"],"warmup":spec["warmup"],
             "origin":Path(spec["schedule_origin"]).name,"source_receipt_sha256":spec["source_receipt_sha256"]}
        for key,spec in source.items()}
print(json.dumps({"result":json.loads(raw), "source_schedules":scopes, "runtime_selection":selection, "execution":{
    "stage":name, "campaign_exit":0, "chronology":chronology, "physical_gpu":7,
    "remote_verified_json_sha256":sha256(raw).hexdigest(),
    "remote_dispatch_and_result_digests_verified":True, "remote_independent_comparison_byte_exact":True,
}}, indent=2,allow_nan=False))
PY
