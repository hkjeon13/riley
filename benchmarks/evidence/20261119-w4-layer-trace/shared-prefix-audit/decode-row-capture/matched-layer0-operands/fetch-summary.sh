#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-matched-mlp-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute AWS replay stage.' >&2
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


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    for phase in ("riley", "hf"):
        for point in ("before", "after"):
            state = read(f"gpu{gpu}-{phase}-{point}.json")
            assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
            assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
            assert not state["other_processes"]

prior = Path("../w4-decode-residual-boundaries-20261120.iOM87n62/result.json")
assert digest(prior) == "eaa70ff7e85ab93160c8d4b3e91037ed4f7f958d59af6cfd781ce0853c29e9b2"
baseline = read(prior)["cases"]
result = read("hf/verified.json")
assert set(result["cases"]) == set(baseline) == {"669-9", "1023-67"}
assert result["gpu_uuid"] == read("gpu7-hf-after.json")["gpu"]
checks = {
    "attention_context": "attention_context",
    "attention_projection": "attention_projection",
    "same_attention_residual": "attention_residual",
    "same_post_attention_norm": "post_attention_norm",
    "same_gated_mlp_input": "mlp_gated",
    "same_mlp_down": "mlp_down",
}
for name, case in result["cases"].items():
    ref = baseline[name]
    assert all(case[key] == ref[key] for key in ("position", "input_token", "riley_token", "hf_token",
                                                 "hf_prefix_cache_sha256", "hf_baseline"))
    assert set(case["trials"]) == set(checks)
    for setting, replacement in checks.items():
        point = case["trials"][setting]["operators_vs_riley"][replacement]
        assert point["effective_sha256"] == point["riley_sha256"]
    for setting, boundary in (("same_attention_residual", "attention_residual"),
                              ("same_mlp_down", "layer_residual")):
        expected = ref["replacement_trials"][int(boundary == "layer_residual")]
        assert (expected["stage"], expected["layer"]) == (boundary, 0)
        assert all(case["trials"][setting][field] == expected[field]
                   for field in ("logits_sha256", "lowest_max_token_id", "scores"))
    case["prior_residual_replay_exact"] = True
result.update(
    remote_stage=str(Path.cwd()),
    scope="Two pinned C1 decode locations, matched layer-zero operands.",
    source_binary_sha256=Path("binary.sha256").read_text().split()[0],
    result_manifest_sha256=digest("result.sha256"),
    prior_residual_replay_sha256=digest(prior),
    operator_result_sha256=digest("hf/verified.json"),
    gpu_idle_reservation_gib={"6": 20, "7": 20},
)
print(json.dumps(result, indent=2, allow_nan=False))
PY
