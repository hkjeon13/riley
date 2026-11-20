#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-projection-buckets-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute projection bucket stage.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "
    set -eu
    cd '$1'
    test \"\$(cat campaign.exit)\" = 0
    sha256sum -c dispatch.sha256 result.sha256 >/dev/null
    /usr/bin/python3 -B -
" <<'PY'
from hashlib import sha256
import json
from pathlib import Path


root = Path.cwd()
buckets = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
candidates = ("attention-31", "down-13-unsplit", "down-13-split64", "fp32-reduction")
for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    for point in ("before", "after"):
        state = json.loads((root / f"gpu{gpu}-{point}.json").read_bytes())
        assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
        assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
        assert not state["other_processes"]

results = {}
for candidate in candidates:
    p = root / f"{candidate}.log"
    rows = []
    for m, line in zip(buckets, p.read_text().splitlines(), strict=True):
        row, outcome, *detail = line.split()
        assert int(row) == m
        a, b, c = map(int, detail)
        if outcome == "ran":
            assert 0 <= a <= 16*1024**2 and 0 <= b <= 2*m and b <= c <= 2*m*3584
            rows.append({"m": m, "executed": True, "workspace_bytes": a,
                         "different_rows": b, "different_bf16_words": c})
        else:
            assert outcome == "unsupported" and (a or b or c > 16*1024**2)
            rows.append({"m": m, "executed": False, "lt_status": a, "plan_state": b, "workspace_bytes": c})
    assert rows[0]["executed"] and rows[0]["different_bf16_words"] == 0
    results[candidate] = {"rows": rows, "source_log_sha256": sha256(p.read_bytes()).hexdigest(),
                          "executed_buckets": [r["m"] for r in rows if r["executed"]],
                          "bit_exact_buckets": [r["m"] for r in rows if r.get("different_bf16_words") == 0]}
print(json.dumps({
    "remote_stage": str(root), "m_buckets": buckets, "m1_outputs_per_candidate": 2,
    "row_orderings_per_bucket": 2, "workspace_limit_bytes": 16 * 1024**2,
    "source_replay_manifest_sha256": "bec6273c272a0af78e9b8e04cbe53b54b6d945c0abb38085ef035bf5f3599457",
    "result_manifest_sha256": sha256((root / "result.sha256").read_bytes()).hexdigest(),
    "gpu_idle_reservation_gib": {"6": 20, "7": 20}, "candidates": results,
}, indent=2, allow_nan=False))
PY
