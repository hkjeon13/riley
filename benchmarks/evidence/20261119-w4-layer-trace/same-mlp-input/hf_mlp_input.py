"""Replay selected HF MLPs with the same last-row input captured from Riley."""

from argparse import ArgumentParser
import json
from pathlib import Path

from replay_final_head import digest, read_arms, read_payload, require

CASES = (3, 9)
LAYERS = (22, 27)
WIDTH = 3584
VOCAB = 152064
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
CROSS_RESULT = "96250d798bdbcb03ea6b002bbe2d24d7720b8f8de39fcf320c1290b486e5f773"
STAGES = ("post_attention_norm", "mlp_down")


def capture_rows(trace, case):
    paths = (trace / f"native{case}", trace / f"hf{case}")
    receipts = [json.loads((path / "receipt.json").read_bytes()) for path in paths]
    digests = [receipt["result_sha256"] for receipt in receipts]
    payloads = [read_payload(path, "rows.bf16", hashes["rows.bf16"])
                for path, hashes in zip((paths[0], paths[1] / "hf-rows"), digests, strict=True)]
    lines = read_payload(paths[0], "index.tsv", digests[0]["index.tsv"]).decode().splitlines()
    captured = {}
    for line in lines[1:]:
        stage, layer, offset, length = line.split("\t")
        if stage in STAGES and int(layer) in LAYERS:
            begin, size = int(offset), int(length)
            require(size == 2 * WIDTH, "unexpected MLP vector dimensions")
            rows = tuple(payload[begin:begin + size] for payload in payloads)
            require(all(len(row) == size for row in rows), "truncated MLP row")
            key = (int(layer), stage)
            require(key not in captured, "duplicate captured MLP row")
            captured[key] = dict(zip(("riley", "hf"), rows, strict=True))
    require(captured.keys() == {(layer, stage) for layer in LAYERS for stage in STAGES}, "missing MLP vectors")
    return captured


def load_inputs(model, trace, crossed):
    checkpoint = digest((model / "riley-checkpoint.json").read_bytes())
    require(checkpoint == CHECKPOINT, "unexpected model checkpoint")
    case_file = trace / "fixtures/legacy.json"
    fixture = case_file.read_bytes()
    legacy = json.loads(fixture)["baseline"]
    raw_control = crossed.read_bytes()
    require(digest(raw_control) == CROSS_RESULT, "unexpected prior crossed run")
    control = json.loads(raw_control)
    require(control["hf_backend"] == "sdpa" and control["checkpoint_sha256"] == checkpoint
            and control["fixture_sha256"] == digest(fixture), "crossed run uses different inputs")
    results = {}
    for case in CASES:
        _, identity = read_arms(trace, fixture, case, WIDTH, VOCAB, checkpoint)
        pinned = legacy[case]
        previous = control["cases"][str(case)]
        require(pinned["prompt_sha256"] == previous["prompt_sha256"]
                and previous["historical_control_logits_reproduced"]
                and identity["hf"]["selected_id"] == previous["runs"]["fused_whole"]["first_id"],
                "pinned prompt or reference first token differs")
        results[case] = (pinned, previous["runs"], capture_rows(trace, case), identity)
    return results


def preflight(inputs):
    summary = {}
    for case, (pinned, reference, rows, identity) in inputs.items():
        summary[str(case)] = {
            "prompt_sha256": pinned["prompt_sha256"],
            "prompt_tokens": len(pinned["prompt_tokens"]),
            "capture_sha256": {engine: data["captured_rows_sha256"] for engine, data in identity.items()},
            "baseline_logits": {arm: reference[arm]["logits_sha256"] for arm in ("fused_whole", "fused_partitioned")},
            "last_row_sha256": {f"{layer}/{stage}": {engine: digest(raw) for engine, raw in engines.items()}
                                for (layer, stage), engines in rows.items()},
        }
    return {"checkpoint_sha256": CHECKPOINT, "cross_result_sha256": CROSS_RESULT, "layers": list(LAYERS), "cases": summary}


def execute(model, tokens, partition, rows, replace_layer=None):
    import torch

    from hf_prefill_partition import measure

    last = len(partition)
    calls = dict.fromkeys(LAYERS, 0)
    captured = {}
    hooks = []

    def save(key, tensor):
        row = tensor[0, -1].detach().contiguous()
        require(row.shape == (WIDTH,) and row.dtype == torch.bfloat16 and key not in captured,
                f"unexpected or duplicate last BF16 row for {key}")
        captured[key] = row.view(torch.uint16).cpu().numpy().tobytes()

    def norm_hook(layer):
        def apply(_module, _args, output):
            calls[layer] += 1
            if calls[layer] != last:
                return None
            if replace_layer == layer:
                native = rows[layer, "post_attention_norm"]["riley"]
                replacement = torch.frombuffer(bytearray(native), dtype=torch.bfloat16).to(output.device)
                modified = output.clone()
                modified[0, -1].copy_(replacement)
                save((layer, "post_attention_norm"), modified)
                return modified
            save((layer, "post_attention_norm"), output)
            return None
        return apply

    def down_hook(layer):
        def capture(_module, _args, output):
            if calls[layer] == last:
                save((layer, "mlp_down"), output)
        return capture

    try:
        for layer in LAYERS:
            block = model.model.layers[layer]
            hooks.append(block.post_attention_layernorm.register_forward_hook(norm_hook(layer)))
            hooks.append(block.mlp.down_proj.register_forward_hook(down_hook(layer)))
        result = measure(model, tokens, partition)
    finally:
        for hook in hooks:
            hook.remove()
    require(all(count == last for count in calls.values()) and captured.keys() == rows.keys(),
            "the full forward did not capture every specified final row")
    if replace_layer is not None:
        require(captured[replace_layer, "post_attention_norm"] == rows[replace_layer, "post_attention_norm"]["riley"],
                "the injected input differs from Riley's captured BF16 row")
    return result, captured


def compare_rows(raw, reference):
    from math import fsum, sqrt
    from struct import iter_unpack

    from replay_final_head import bf16

    values, expected = bf16(raw), bf16(reference)
    squared = (float(a - b) ** 2 for a, b in zip(values, expected, strict=True))
    matches = sum(a == b for a, b in zip(iter_unpack("<H", raw), iter_unpack("<H", reference), strict=True))
    return {"bit_identical_elements": matches, "elements": len(values), "rms_error": sqrt(fsum(squared) / len(values))}


def compare(model, inputs):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    report = preflight(inputs)
    report.update({"torch": str(torch.__version__), "transformers": transformers.__version__, "backend": "sdpa"})
    for case, (pinned, reference, rows, _identity) in inputs.items():
        tokens = torch.tensor([pinned["prompt_tokens"]], device="cuda")
        modes = {}
        for arm in ("fused_whole", "fused_partitioned"):
            partition = reference[arm]["partition"]
            baseline, captures = execute(hf, tokens, partition, rows)
            require(baseline == reference[arm], f"case {case}/{arm}: pinned HF output changed")
            if arm == "fused_whole":
                require(all(raw == rows[key]["hf"] for key, raw in captures.items()),
                        f"case {case}: the original HF MLP rows did not reproduce")
            variants = {}
            for layer in LAYERS:
                altered, outputs = execute(hf, tokens, partition, rows, replace_layer=layer)
                key = (layer, "mlp_down")
                variants[str(layer)] = {
                    "after_injection": altered,
                    "baseline_down_vs_riley": compare_rows(captures[key], rows[key]["riley"]),
                    "same_riley_input_down_vs_riley": compare_rows(outputs[key], rows[key]["riley"]),
                    "injected_down_sha256": digest(outputs[key]),
                }
            modes[arm] = {"control": baseline, "interventions": variants}
        report["cases"][str(case)]["riley_first_token"] = pinned["tokens"][0]
        report["cases"][str(case)]["modes"] = modes
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("crossed", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    inputs = load_inputs(args.model, args.trace, args.crossed)
    result = preflight(inputs) if args.preflight else compare(args.model, inputs)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
