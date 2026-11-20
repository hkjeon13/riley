#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-decode-residual-boundaries-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute AWS replay stage.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "
    set -eu
    cd '$1'
    test \"\$(cat campaign.exit)\" = 0
    sha256sum -c dispatch.sha256 tools.sha256 result.sha256 >/dev/null
    /usr/bin/python3 -B -
" <<'PY'
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_bytes())


for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    for point in ("before", "after"):
        state = read(f"gpu{gpu}-{point}.json")
        assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
        assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
        assert not state["other_processes"]

result = read("result.json")
assert set(result["cases"]) == {"669-9", "1023-67"}
assert result["addressable_vocabulary"] == 151_665
assert result["gpu_uuid"] == read("gpu7-before.json")["gpu"]
boundaries = [(layer, name) for layer in range(28) for name in ("attention_residual", "layer_residual")]
for case in result["cases"].values():
    baseline = case["hf_baseline"]
    riley, hf = case["riley_token"], case["hf_token"]
    assert baseline["lowest_max_token_id"] == hf
    assert case["exact_self_replacement_controls"] == len(boundaries)
    rows = case.pop("replacement_trials")
    assert [(row["layer"], row["stage"]) for row in rows] == boundaries
    compressed = []
    outcomes = Counter()
    for row in rows:
        by_id = row["scores"]
        riley_score, hf_score = by_id[str(riley)], by_id[str(hf)]
        selected = row["lowest_max_token_id"]
        assert 0 <= selected < result["addressable_vocabulary"]
        assert all(entry["rank"] >= 1 and entry["tied_count"] >= 1 for entry in by_id.values())
        outcome = "riley" if selected == riley else "hf" if selected == hf else "other"
        outcomes[outcome] += 1
        compressed.append([
            row["layer"], row["stage"], selected,
            riley_score["bf16_score"], hf_score["bf16_score"],
            riley_score["rank"], hf_score["rank"], row["logits_sha256"], row["riley_residual_sha256"],
        ])
    case["selection_counts"] = dict(outcomes)
    case["boundaries_choosing_riley"] = [row[:2] for row in compressed if row[2] == riley]
    case["trial_columns"] = ["layer", "boundary", "lowest_max_token_id", "riley_token_bf16_score",
                             "hf_token_bf16_score", "riley_token_rank", "hf_token_rank",
                             "logits_sha256", "riley_residual_sha256"]
    case["trials"] = compressed

result.update(
    remote_stage=str(Path.cwd()),
    result_sha256=sha256(Path("result.json").read_bytes()).hexdigest(),
    evidence_list_sha256=sha256(Path("result.sha256").read_bytes()).hexdigest(),
    idle_reservation_gib={"6": 20, "7": 20},
)
print(json.dumps(result, indent=2, allow_nan=False))
PY
