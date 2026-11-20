#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-projection-forward-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute projection replay stage.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "
    set -eu
    cd '$1'
    test \"\$(cat campaign.exit)\" = 0
    sha256sum -c dispatch.sha256 result.sha256 capture-source.sha256 >/dev/null
    /usr/bin/python3 -B -
" <<'PY'
from hashlib import sha256
import json
from pathlib import Path


root = Path.cwd()


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    for point in ("before", "after"):
        state = read(root / f"gpu{gpu}-riley-{point}.json")
        assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
        assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
        assert not state["other_processes"]
receipt = read(root / "riley/receipt.json")
assert receipt["status"] == "passed" and receipt["server_exit"] == 0
report = read(root / "verified.json")
assert set(report["cases"]) == {"669-9", "1023-67"}
expected_modes = ("baseline", "identity-down", "fp32-reduction", "down-13-unsplit",
                  "down-13-split64", "attention-31", "baseline-repeat")
assert tuple(report["mode_order"]) == expected_modes
source = root.parent / "w4-native-projection-plans-20261120.yvjrWvOj"
prior = read(source / "plans.json")
assert digest(source / "result.sha256") == report["original_native_manifest_sha256"]
for case, item in report["cases"].items():
    modes = item["modes"]
    assert tuple(modes) == expected_modes
    baseline = modes["baseline"]
    assert baseline["trace_sha256"] == prior["native_rows_reproduced"][case]["trace_sha256"]
    for mode in ("baseline", "identity-down", "baseline-repeat"):
        control = modes[mode]
        assert control["actual_token"] == item["baseline_riley_id"] and control["same_completion_as_riley"]
        assert control["trace_sha256"] == baseline["trace_sha256"]
        assert control["full_tokens_sha256"] == baseline["full_tokens_sha256"]
    for mode, result in modes.items():
        assert result["common_prefix_with_riley"] >= item["decode_offset"]
        assert result["common_prefix_with_hf"] >= item["decode_offset"]
        directory = root / "riley" / "rows" / f"{case}.{mode}"
        assert result["trace_sha256"] == digest(directory / "rows.bf16")
        if result["applied_projection_sha256"] is not None:
            row, = directory.glob("*override.bf16")
            assert digest(row) == result["applied_projection_sha256"]

report.update(
    remote_stage=str(root),
    build_binary_sha256=(root / "binary.sha256").read_text().split()[0],
    server_receipt_sha256=digest(root / "riley/receipt.json"),
    result_manifest_sha256=digest(root / "result.sha256"),
    gpu_idle_reservation_gib={"6": 20, "7": 20},
)
print(json.dumps(report, indent=2, allow_nan=False))
PY
