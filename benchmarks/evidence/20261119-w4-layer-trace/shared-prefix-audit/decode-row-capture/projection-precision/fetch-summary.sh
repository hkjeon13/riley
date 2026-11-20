#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-projection-oracle-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute AWS projection-oracle stage.' >&2
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


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    for point in ("before", "after"):
        state = read(f"gpu{gpu}-{point}.json")
        assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
        assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
        assert not state["other_processes"]

report = read("verified.json")
assert report["gpu_comparison"] == read("gpu-receipt.json")
assert report["gpu_comparison"]["gpu_uuid"] == read("gpu7-after.json")["gpu"]
assert report["shard_sha256"] == read("shard.json")["sha256"]
source = Path("../w4-matched-mlp-20261120.wOWcpjAr/hf/verified.json")
original = read(source)["cases"]
assert digest(source) == report["matched_operator_result_sha256"]
for level in ("669-9", "1023-67"):
    for op, width, trial, input_point, output_point in (
        ("o_proj", 3584, "attention_context", "attention_context", "attention_projection"),
        ("down_proj", 18944, "same_gated_mlp_input", "mlp_gated", "mlp_down"),
    ):
        comparison = report["comparisons"][f"{level}/{op}"]
        ref = original[level]["trials"][trial]["operators_vs_riley"]
        assert comparison["shape_m_n_k"] == [1, 3584, width]
        assert comparison["input_sha256"] == ref[input_point]["riley_sha256"]
        obs = comparison["observations"]
        assert obs["riley"]["output_sha256"] == ref[output_point]["riley_sha256"]
        assert obs["hf_default"]["output_sha256"] == ref[output_point]["emitted_sha256"]
        for row in obs.values():
            assert 0 <= row["more_than_one_bf16_step"] <= row["different_numeric_values"] <= 3584

report.update(
    remote_stage=str(Path.cwd()),
    result_manifest_sha256=digest("result.sha256"),
    gpu_idle_reservation_gib={"6": 20, "7": 20},
)
print(json.dumps(report, indent=2, allow_nan=False))
PY
