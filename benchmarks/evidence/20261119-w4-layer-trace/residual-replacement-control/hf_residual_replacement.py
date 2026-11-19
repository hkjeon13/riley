"""Recompute HF downstream layers after replacing the last-token residual updates."""

from argparse import ArgumentParser
import json
from pathlib import Path

from hf_mlp_input import CASES, WIDTH, load_inputs
from replay_final_head import digest, read_payload, require


SITES = ("attention_projection", "mlp_down")
LAYERS = 28
VARIANTS = {
    "layer_22_mlp": [(22, "mlp_down")],
    "all_attention": [(layer, "attention_projection") for layer in range(LAYERS)],
    "all_mlp": [(layer, "mlp_down") for layer in range(LAYERS)],
    "all_attention_and_mlp": [(layer, site) for layer in range(LAYERS) for site in SITES],
}


def captured_updates(trace, case):
    native, hf = (trace / f"{engine}{case}" for engine in ("native", "hf"))
    receipts = [json.loads((directory / "receipt.json").read_bytes())["result_sha256"]
                for directory in (native, hf)]
    payloads = [read_payload(directory, "rows.bf16", receipt["rows.bf16"])
                for directory, receipt in zip((native, hf / "hf-rows"), receipts, strict=True)]
    index = read_payload(native, "index.tsv", receipts[0]["index.tsv"]).decode().splitlines()[1:]
    rows = {}
    for line in index:
        site, layer, offset, length = line.split("\t")
        if site in SITES:
            start, length = int(offset), int(length)
            key = (int(layer), site)
            require(length == 2 * WIDTH and key not in rows
                    and 0 <= start <= len(payloads[0]) - length, "invalid traced residual update")
            native_bytes, hf_bytes = (payload[start:start + length] for payload in payloads)
            require(len(hf_bytes) == length, "incomplete traced HF residual update")
            rows[key] = {"riley": native_bytes, "hf": hf_bytes}
    require(rows.keys() == set(VARIANTS["all_attention_and_mlp"]), "missing layer projection data")
    return rows


def prepare(trace, model, crossed, frozen_result):
    raw = frozen_result.read_bytes()
    frozen = json.loads(raw)
    rows = {}
    inputs = load_inputs(model, trace, crossed)
    for case in CASES:
        _, _, _, recorded = inputs[case]
        require(all(frozen["cases"][str(case)][engine]["capture_sha256"] == receipt["captured_rows_sha256"]
                    for engine, receipt in recorded.items()), "frozen replay used a different trace")
        captured = captured_updates(trace, case)
        rows[case] = captured
    summary = {
        "frozen_replay_sha256": digest(raw),
        "checkpoint_sha256": frozen["checkpoint_manifest_sha256"],
        "cases": {str(case): {
            "prompt_sha256": inputs[case][0]["prompt_sha256"],
            "baseline_logits": {arm: inputs[case][1][arm]["logits_sha256"]
                                for arm in ("fused_whole", "fused_partitioned")},
            "capture_sha256": {engine: receipt["captured_rows_sha256"]
                               for engine, receipt in inputs[case][3].items()},
            "update_rows": len(rows[case]),
        } for case in CASES},
    }
    return inputs, rows, summary


def update_hook(last_chunk, saved, key, replacement=None):
    import torch

    calls = 0

    def apply(_module, _args, projected):
        nonlocal calls
        calls += 1
        if calls != last_chunk:
            return None
        row = projected[0, -1]
        require(row.shape == (WIDTH,) and row.dtype == torch.bfloat16 and key not in saved,
                f"bad last-row projection at {key}")
        if replacement is not None:
            value = torch.frombuffer(bytearray(replacement), dtype=torch.bfloat16).to(projected.device)
            projected = projected.clone()
            projected[0, -1].copy_(value)
            row = projected[0, -1]
        saved[key] = row.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
        return projected if replacement is not None else None

    return apply


def execute(hf, tokens, partition, controls, replacements=None):
    from hf_prefill_partition import measure

    saved, hooks = {}, []
    try:
        for key in controls:
            layer, site = key
            block = hf.model.layers[layer]
            module = block.self_attn.o_proj if site == "attention_projection" else block.mlp.down_proj
            replacement = None if replacements is None else replacements[key]["riley"]
            hook = update_hook(len(partition), saved, key, replacement)
            hooks.append(module.register_forward_hook(hook))
        result = measure(hf, tokens, partition)
    finally:
        for hook in hooks:
            hook.remove()
    require(saved.keys() == set(controls), "the HF run did not see every planned final-row projection")
    if replacements is not None:
        require(all(raw == replacements[key]["riley"] for key, raw in saved.items()),
                "a last-row replacement changed the BF16 bytes")
    return result, saved


def compare(model, inputs, rows, report):
    import torch
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    report["torch"] = str(torch.__version__)
    report["transformers"] = transformers.__version__
    for case, (pinned, reference, _, _) in inputs.items():
        tokens = torch.tensor([pinned["prompt_tokens"]], device="cuda")
        receipts = {}
        for arm in ("fused_whole", "fused_partitioned"):
            partition = reference[arm]["partition"]
            before, captured = execute(hf, tokens, partition, rows[case])
            require(before == reference[arm], f"case {case}/{arm}: historical HF logits changed")
            if arm == "fused_whole":
                require(all(raw == rows[case][key]["hf"] for key, raw in captured.items()),
                        f"case {case}: whole-prefill updates differ from capture")
            variants = {}
            for name, locations in VARIANTS.items():
                result, installed = execute(hf, tokens, partition, locations, rows[case])
                variants[name] = {
                    "result": result,
                    "replacement_sha256": {f"{layer}/{site}": digest(raw)
                                           for (layer, site), raw in installed.items()},
                }
            after, _ = execute(hf, tokens, partition, ())
            require(after == before, f"case {case}/{arm}: HF baseline changed after removing hooks")
            receipts[arm] = {"baseline": before, "variants": variants, "unhooked_baseline_reproduced": True}
        report["cases"][str(case)]["runs"] = receipts
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("crossed", type=Path)
    parser.add_argument("frozen_result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    inputs, rows, report = prepare(args.trace, args.model, args.crossed, args.frozen_result)
    if not args.preflight:
        report = compare(args.model, inputs, rows, report)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
