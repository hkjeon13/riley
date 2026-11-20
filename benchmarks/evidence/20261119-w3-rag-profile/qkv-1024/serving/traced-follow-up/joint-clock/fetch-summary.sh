#!/usr/bin/env bash
set -euo pipefail
initial=$1
retry=$2
analysis=$3
evidence=$(dirname "$initial")
root=$(dirname "$evidence")
prior="$evidence/w3-qkv-tracing-20261119.s9zZEnqW"
test "$(cat "$initial/campaign.exit")" = 1
test "$(cat "$retry/campaign.exit")" = 0
(cd "$initial" && sha256sum -c dispatch.sha256 binary.sha256 patched-driver.sha256 >/dev/null)
(cd "$retry" && sha256sum -c dispatch.sha256 result.sha256 binary.sha256 patched-driver.sha256 >/dev/null)
(cd "$analysis" && sha256sum -c analysis.sha256 >/dev/null)
python="$root/tools/vllm-venv/bin/python"
PYTHONPATH="$initial/tools" "$python" -B - "$initial" "$prior" "$retry" "$analysis" <<'PY'
from hashlib import file_digest
import json
from pathlib import Path
import sys
initial, prior, retry, analysis = map(Path, sys.argv[1:])
sys.path.insert(0, str(analysis))
from verify_joint import summarize

baseline = initial/"qwen14-riley-first"
with (baseline/"trace.nsys-rep").open("rb") as report:
    assert file_digest(report,"sha256").hexdigest() == json.loads((baseline/"receipt.json").read_bytes())["raw_report_sha256"]
result = summarize(initial, prior, retry)
holds = {}
for mode,stage in (("riley-first",initial),("qkv-first",retry)):
    holds[mode]={}
    for index,pid,ticks in ((6,2431228,"6339570869"),(7,2431237,"6339570875")):
        for moment in ("before","after"):
            item=json.loads((stage/f"gpu{index}-{moment}-{mode}.json").read_bytes())
            assert (item["pid"],item["start_ticks"],item["physical_index"],item["phase"]) == (pid,ticks,index,"idle")
            assert not item["other_processes"]
            assert item["allocation_bytes"] == item["torch_reserved_bytes"] == 20*1024**3
        holds[mode][str(index)]={"pid":pid,"start_ticks":ticks,"allocation_bytes":20*1024**3}
candidate=initial/"qwen14-qkv-first"
failed=json.loads((candidate/"receipt.json").read_bytes())
assert failed["status"]=="failed"
assert not (candidate/"server-exit.json").exists()
assert failed["error"] == "RuntimeError: unexpected process on selected GPU pid="+(candidate/"server.pid").read_text().strip()
print(json.dumps({"result":result,"execution":{
    "initial_stage":initial.name, "retry_stage":retry.name, "physical_gpu":7,
    "baseline_finished":True, "candidate_retry_exit":0, "original_candidate_has_server_exit_receipt":False,
    "original_candidate_monitor_rejected_the_riley_pid":True,
    "reservations_after_each_accepted_arm":holds,
    "boundaries":{stage.name:[line for line in (stage/"campaign.log").read_text().splitlines()
                               if line.startswith(("START ","END "))] for stage in (initial,retry)},
}},indent=2,allow_nan=False))
PY
