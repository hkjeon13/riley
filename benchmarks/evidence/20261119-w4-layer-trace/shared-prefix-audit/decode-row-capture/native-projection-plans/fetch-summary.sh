#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ||
      ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-native-projection-plans-20261120\.[A-Za-z0-9]+$ ||
      ! $2 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-native-projection-replay-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute native and replay stages.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "
    set -eu
    cd '$1'
    test \"\$(cat campaign.exit)\" = 0
    sha256sum -c dispatch.sha256 result.sha256 >/dev/null
    cd '$2'
    test \"\$(cat campaign.exit)\" = 0
    sha256sum -c dispatch.sha256 result.sha256 >/dev/null
    /usr/bin/python3 -B - '$1'
" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


native = Path(sys.argv[1])
root = Path.cwd()
for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    for point in ("before", "after"):
        for directory, prefix in ((native, "riley-"), (root, "")):
            state = read(directory / f"gpu{gpu}-{prefix}{point}.json")
            assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
            assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
            assert not state["other_processes"]

report = read(root / "verified.json")
assert report["gpu_uuid"] == read(root / "gpu7-after.json")["gpu"]
assert report["riley_capture"] == read(native / "plans.json")
assert report["plan_capture_manifest_sha256"] == digest(native / "result.sha256")
assert report["cpu_oracle_manifest_sha256"] == "0fc62e2b3cffacfa2cc638bd73e6e54a77935a15a400b02728e41882b037f8f5"
prior = read(root.parent / "w4-projection-oracle-20261120.RB5VX8Md/verified.json")["comparisons"]
for operator, row in report["operators"].items():
    assert set(row["cases"]) == {"669-9", "1023-67"}
    checks = {case: prior[f"{case}/{operator}"]["observations"]["riley"] for case in row["cases"]}
    for case, candidates in row["cases"].items():
        assert set(candidates) == set(row["plans"])
        assert candidates["recorded"]["sha256"] == checks[case]["output_sha256"]
        assert candidates["recorded"]["different_numeric_values"] == checks[case]["different_numeric_values"]
    for label, cfg in row["plans"].items():
        if label == "recorded":
            continue
        assert (cfg["split_k"] <= 1 and cfg["reduction"] == 0) or (cfg["split_k"] > 1 and cfg["reduction"] == 2)

down = report["operators"]["down_proj"]["plans"]
assert down["recorded"]["reduction"] == 4 and down["fixed-reduction"]["reduction"] == 2
assert all(down["recorded"][field] == down["fixed-reduction"][field]
           for field in down["recorded"] if field not in {"reduction", "workspace"})

report.update(native_stage=str(native), replay_stage=str(root),
              result_manifest_sha256=digest(root / "result.sha256"),
              gpu_idle_reservation_gib={"6": 20, "7": 20})
print(json.dumps(report, indent=2, allow_nan=False))
PY
