"""Check both former controls and the pinned requests without loading a model."""

from hashlib import sha256
import json
from pathlib import Path
import sys

from layer_trace import pinned_case


def verify(fixture, model):
    bias = json.loads(Path("bias-control.json").read_bytes())
    partition = json.loads(Path("partition-control.json").read_bytes())
    assert bias["torch"] == partition["torch"] and bias["backend"] == partition["backend"] == "sdpa"
    cases = {}
    for case in (3, 9):
        pinned, _ = pinned_case(fixture, case)
        left, right = (record["cases"][str(case)] for record in (bias, partition))
        assert pinned["prompt_sha256"] == left["prompt_sha256"] == right["prompt_sha256"]
        assert left["runs"]["baseline"]["logits_sha256"] == right["runs"]["baseline"]["logits_sha256"]
        chunks = right["runs"]["budget_1024_with_native_final_tail"]["partition"]
        assert sum(chunks) == len(pinned["prompt_tokens"])
        cases[str(case)] = {"partition": chunks, "prompt_sha256": pinned["prompt_sha256"],
                            "full_hf_logits_sha256": right["runs"]["baseline"]["logits_sha256"]}
    manifest = sha256((model / "riley-checkpoint.json").read_bytes()).hexdigest()
    assert manifest == "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
    return {"fixture_sha256": sha256(fixture.read_bytes()).hexdigest(), "checkpoint_sha256": manifest, "cases": cases}


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
