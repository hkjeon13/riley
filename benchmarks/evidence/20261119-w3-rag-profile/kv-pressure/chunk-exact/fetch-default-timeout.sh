#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-chunk-exact-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute remote stage of the default-timeout attempt.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$1' && sha256sum -c dispatch.sha256 >/dev/null && /usr/bin/python3 -B -" <<'PY'
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from long_accuracy import same_completion

root = Path.cwd()
before = root / "qwen14-chunk1024-kv2576"
after = root / "qwen14-chunk512-kv768"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


assert (root / "campaign.exit").read_text() == "1\n"
base = read(before / "receipt.json")
pressure = read(after / "receipt.json")
assert base["status"] == "passed" and pressure["status"] == "failed"
assert read(before / "server-exit.json")["exit"] == 0
assert base["common_kv_blocks"] == 2576 and pressure["common_kv_blocks"] == 768
argv = read(after / "plan.json")["server"]
assert "--admission-timeout-ms" not in argv
assert argv[argv.index("--prefill-chunk-tokens") + 1] == "512"
assert pressure["holder_pid"] == base["holder_pid"] == 2431237
probe = before / "token-id-probe.json"
assert base["results"]["token-id-probe"] == digest(probe)
raw = read(probe)["rows"]
matched = {}
for level in ("c8", "c32"):
    matched[level] = sum(
        same_completion(a, b) and a["prompt_tokens"] == b["prompt_tokens"] and a["usage"] == b["usage"]
        for a, b in zip(raw["c1"] * 3, raw[level], strict=True)
    )
assert matched == {"c8": 33, "c32": 33}
http = re.findall(r"urllib\.error\.HTTPError: HTTP Error ([0-9]{3}):", (after / "client.log").read_text())
assert http == ["408"]

log = (after / "server.log").read_text()
startup, warmup_count, serving = re.split(r"(?m)^riley startup warm-up: ([0-9]+) internal requests in .* s$", log)
assert int(warmup_count) == 48
frontend = [line.split()[1] for line in serving.splitlines()
            if line.startswith("W3-PHASE ") and " front_submit " in line]
mapping = {parts[1]: parts[4] for line in serving.splitlines()
           if line.startswith("W3-PHASE ") and " kv_scheduler_id " in line
           for parts in [line.split()]}
assert len(frontend) == 76 and set(frontend) == set(mapping)
events = defaultdict(list)
for line in serving.splitlines():
    if line.startswith("W3-KV "):
        _, native, name, stamp, need, held, cap, *_ = line.split()
        events[native].append((name, int(stamp), int(need), int(held), int(cap)))
assert set(events) == set(mapping.values())
unadmitted = [(i, events[mapping[key]]) for i, key in enumerate(frontend)
              if not any(event[0] in {"submit_active", "admitted"} for event in events[mapping[key]])]
(i, trace), = unadmitted
assert 44 <= i < 76 and trace[0][0] == "submit_queued"
last = trace[-1]
assert last[0] == "blocked" and last[2] + last[3] > last[4] == 768
milliseconds = (last[1] - trace[0][1]) / 1e6
assert 29900 <= milliseconds < 30000
assert sum(any(item[0] == "released" for item in records) for records in events.values()) == 75

for index, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    state = read(root / f"gpu{index}-after-chunk512-kv768.json")
    assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (
        index, pid, ticks, "idle",
    )
    assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
    assert not state["other_processes"]

print(json.dumps({
    "remote_stage": str(root),
    "campaign_exit": 1,
    "baseline": {
        "chunk_limit": 1024, "kv_blocks": 2576, "c1_completed": len(raw["c1"]),
        "within_arm_exact": matched, "probe_sha256": digest(probe),
    },
    "failed_arm": {
        "chunk_limit": 512, "kv_blocks": 768, "default_admission_timeout_ms": 30000,
        "http_status_observed_in_client_error": int(http[0]),
        "client_reported_error_count": len(http),
        "user_requests_submitted": len(frontend),
        "released_after_active": 75,
        "unadmitted_requests": len(unadmitted),
        "client_group": "c32",
        "time_from_queue_to_last_insufficient_capacity_ms": milliseconds,
        "last_blocked_required_blocks": last[2],
        "last_blocked_already_reserved_blocks": last[3],
        "server_log_sha256": digest(after / "server.log"),
        "client_log_sha256": digest(after / "client.log"),
        "receipt_sha256": digest(after / "receipt.json"),
    },
    "gpu_reservation_restored_gib": {"6": 20, "7": 20},
}, indent=2, allow_nan=False))
PY
