"""Cross the two validated HF SDPA controls on the pinned first-token cases."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from hf_bias_order import split_bias
from hf_prefill_partition import measure
from layer_trace import pinned_case


def run(model, tokens, partition, split):
    hooks = []
    try:
        if split:
            for layer in model.model.layers:
                attention = layer.self_attn
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                    if projection.bias is None:
                        raise ValueError("expected QKV bias")
                    hooks.append(projection.register_forward_hook(split_bias))
        return measure(model, tokens, partition)
    finally:
        for hook in hooks:
            hook.remove()


def margin(record):
    tracked = record["tracked_logits"]
    return tracked["5257"] - tracked["264"]


def compare(model, fixture, bias_control, partition_control):
    checkpoint = sha256((model / "riley-checkpoint.json").read_bytes()).hexdigest()
    if checkpoint != "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f":
        raise ValueError("the checkpoint differs from the recorded first-token cases")
    bias_bytes, partition_bytes = bias_control.read_bytes(), partition_control.read_bytes()
    bias_cases, partition_cases = (json.loads(raw)["cases"] for raw in (bias_bytes, partition_bytes))
    reference = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    cases = {}
    for case in (3, 9):
        ordinal = str(case)
        pinned, _ = pinned_case(fixture, case)
        if not pinned["prompt_sha256"] == bias_cases[ordinal]["prompt_sha256"] == partition_cases[ordinal]["prompt_sha256"]:
            raise ValueError("comparison inputs differ from the established HF controls")
        old_bias = bias_cases[ordinal]["runs"]
        old_partition = partition_cases[ordinal]["runs"]
        tokens = torch.tensor([pinned["prompt_tokens"]], device="cuda")
        whole = [tokens.numel()]
        chunks = old_partition["budget_1024_with_native_final_tail"]["partition"]
        variants = {
            "fused_whole": (whole, False),
            "split_whole": (whole, True),
            "fused_partitioned": (chunks, False),
            "split_partitioned": (chunks, True),
            "fused_whole_repeat": (whole, False),
        }
        measured = {name: run(reference, tokens, pieces, split) for name, (pieces, split) in variants.items()}
        if (measured["fused_whole"] != old_partition["baseline"]
            or measured["fused_whole"] != measured["fused_whole_repeat"]
            or measured["fused_partitioned"] != old_partition["budget_1024_with_native_final_tail"]
            or measured["split_whole"]["logits_sha256"] != old_bias["all_qkv"]["logits_sha256"]):
            raise ValueError(f"known HF control logits changed for case {case}")
        scores = {name: margin(record) for name, record in measured.items()}
        interaction = scores["split_partitioned"] - scores["fused_partitioned"] - scores["split_whole"] + scores["fused_whole"]
        cases[ordinal] = {
            "prompt_sha256": pinned["prompt_sha256"], "riley_first_id": pinned["tokens"][0],
            "historical_control_logits_reproduced": True,
            "runs": measured, "logit_margin_5257_minus_264": scores,
            "logit_margin_interaction": interaction,
        }
    return {
        "checkpoint_sha256": checkpoint, "torch": str(torch.__version__),
        "transformers": transformers.__version__, "hf_backend": "sdpa",
        "bias_reference_sha256": sha256(bias_bytes).hexdigest(),
        "partition_reference_sha256": sha256(partition_bytes).hexdigest(),
        "fixture_sha256": sha256(fixture.read_bytes()).hexdigest(), "cases": cases,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("bias_control", type=Path)
    parser.add_argument("partition_control", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = compare(args.model, args.fixture, args.bias_control, args.partition_control)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
