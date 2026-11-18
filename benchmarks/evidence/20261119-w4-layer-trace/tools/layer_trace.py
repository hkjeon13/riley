"""Capture and compare BF16 rows for the pinned Qwen2.5-7B first-token cases."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from long_accuracy import complete, prompts, same_completion, save

CASES = (3, 9)
LAYERS = 28
STAGES = {
    "input_norm": 3584,
    "qkv_prebias": 4608,
    "query_rope": 3584,
    "attention_context": 3584,
    "attention_projection": 3584,
    "attention_residual": 3584,
    "post_attention_norm": 3584,
    "mlp_down": 3584,
    "layer_residual": 3584,
}
EXPECTED = [("embedding", "-", 3584)]
EXPECTED += [(stage, str(layer), width) for layer in range(LAYERS) for stage, width in STAGES.items()]
EXPECTED += [("final_norm", "-", 3584), ("logits", "-", 152064)]


def pinned_case(source, ordinal):
    row = json.loads(source.read_bytes())["baseline"][ordinal]
    text = prompts()[ordinal]
    if sha256(text.encode()).hexdigest() != row["prompt_sha256"]:
        raise ValueError("prompt does not match historical capture")
    return row, text


def native(address, model, source, ordinal, output):
    row, text = pinned_case(source, ordinal)
    result = complete(address, model, text, expected=row["prompt_tokens"])
    save(output, {"source_first_id": row["tokens"][0], "response": result,
                  "all_original_ids_text_and_finish": same_completion(result, row)})
    if not same_completion(result, row):
        raise ValueError("instrumented route did not reproduce the original output")


def row_index(payload, index, tokens):
    width_sum = 0
    rows = {}
    for (stage, layer, width), (s, layer_key, offset, size) in zip(EXPECTED, index, strict=True):
        expected = (stage, layer, str(width_sum), str(2 * width))
        if expected != (s, layer_key, offset, size):
            raise ValueError(f"unexpected stage row: expected {expected}, got {(s, layer_key, offset, size)}")
        begin = width_sum
        width_sum += 2 * width
        rows[(stage, layer)] = payload[begin:width_sum]
    if width_sum != len(payload) or len(tokens) < 1:
        raise ValueError("incomplete BF16 trace")
    return rows


def read_native(directory, tokens):
    payload = (directory / "rows.bf16").read_bytes()
    header, *raw = (directory / "index.tsv").read_text().splitlines()
    prefix, position, token, row, active, dense = header.split()
    if (prefix != "#" or (int(position), int(token)) != (len(tokens) - 1, tokens[-1])
            or not 0 <= int(row) < int(active) <= int(dense)):
        raise ValueError("trace output row does not identify the expected final prompt token")
    rows = row_index(payload, [line.split("\t") for line in raw], tokens)
    return rows, {"position": int(position), "last_token_id": int(token), "row": int(row),
                  "active_rows": int(active), "dense_rows": int(dense)}


def hf_trace(model_dir, source, reference, ordinal, directory):
    import torch
    from transformers import AutoModelForCausalLM
    from torch.nn.functional import linear

    row, _ = pinned_case(source, ordinal)
    tokens = row["prompt_tokens"]
    historical = json.loads(reference.read_bytes())["cases"][ordinal]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    if len(model.model.layers) != LAYERS:
        raise ValueError("expected the pinned Qwen2.5-7B geometry")

    captured = {}

    def keep(key, from_tensor):
        if isinstance(from_tensor, tuple):
            from_tensor = from_tensor[0]
        last = from_tensor[0, -1].detach().contiguous().clone()
        if last.ndim != 1 or last.dtype != torch.bfloat16 or key in captured:
            raise ValueError(f"unexpected shape or duplicate capture for {key}")
        captured[key] = last

    def hook_after(module, stage, layer="-"):
        return module.register_forward_hook(lambda _module, _args, out: keep((stage, str(layer)), out))

    def hook_before(module, stage, layer):
        return module.register_forward_pre_hook(lambda _module, args: keep((stage, str(layer)), args[0]))

    def hook_qkv(layer, index):
        embeddings = {}

        def after_input_norm(_module, _args, out):
            keep(("input_norm", str(index)), out)
            # Separate HF-shaped matmuls replayed without bias, before rounding
            # differences from bias fusion can affect the comparison.
            parts = [linear(out, projection.weight, None)[:, -1:, :]
                     for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)]
            keep(("qkv_prebias", str(index)), torch.cat(parts, dim=-1))

        def attention_positions(_module, _args, kwargs):
            embeddings["rotary"] = kwargs["position_embeddings"]

        def after_query(_module, _args, out):
            cos, sin = (values[0, -1] for values in embeddings.pop("rotary"))
            query = out[0, -1].reshape(-1, cos.numel())
            left, right = query.chunk(2, dim=-1)
            rotated = query * cos + torch.cat((-right, left), dim=-1) * sin
            keep(("query_rope", str(index)), rotated.reshape(1, 1, -1))

        return [
            layer.input_layernorm.register_forward_hook(after_input_norm),
            layer.self_attn.register_forward_pre_hook(attention_positions, with_kwargs=True),
            layer.self_attn.q_proj.register_forward_hook(after_query),
        ]

    hooks = [hook_after(model.model.embed_tokens, "embedding"), hook_after(model.model.norm, "final_norm")]
    for i, layer in enumerate(model.model.layers):
        hooks.extend(hook_qkv(layer, i))
        hooks.extend([
            hook_before(layer.self_attn.o_proj, "attention_context", i),
            hook_after(layer.self_attn.o_proj, "attention_projection", i),
            hook_before(layer.post_attention_layernorm, "attention_residual", i),
            hook_after(layer.post_attention_layernorm, "post_attention_norm", i),
            hook_after(layer.mlp.down_proj, "mlp_down", i),
            hook_after(layer, "layer_residual", i),
        ])
    try:
        with torch.inference_mode():
            output = model(torch.tensor([tokens], device="cuda"), use_cache=True, logits_to_keep=1)
            keep(("logits", "-"), output.logits)
            chosen = output.logits[0, -1].argmax().item()
            if historical["hf_tokens"][0] != chosen:
                raise ValueError("instrumented HF first token differs from the historical SDPA reference")
    finally:
        for hook in hooks:
            hook.remove()
    directory.mkdir()
    with (directory / "rows.bf16").open("xb") as buffer:
        for stage, layer, width in EXPECTED:
            data = captured[(stage, layer)].view(torch.uint16).cpu().numpy().tobytes()
            if len(data) != 2 * width:
                raise ValueError(f"bad width for {stage} layer {layer}: {len(data)}")
            buffer.write(data)
    save(directory / "metadata.json", {"case": ordinal, "prompt_sha256": row["prompt_sha256"],
                                       "prompt_tokens": len(tokens), "last_token_id": tokens[-1],
                                       "selected_id": chosen, "backend": "sdpa", "torch": torch.__version__})


def compare(source, ordinal, native_dir, hf_dir, output):
    import numpy as np

    base, _ = pinned_case(source, ordinal)
    ids = base["prompt_tokens"]
    native_rows, header = read_native(native_dir, ids)
    meta = json.loads((hf_dir / "metadata.json").read_bytes())
    if ((meta["case"], meta["prompt_sha256"], meta["prompt_tokens"], meta["last_token_id"])
            != (ordinal, base["prompt_sha256"], len(ids), ids[-1])):
        raise ValueError("mismatched HF source binding")
    raw = (hf_dir / "rows.bf16").read_bytes()
    layout = []
    offset = 0
    for stage, layer, width in EXPECTED:
        layout.append((stage, layer, str(offset), str(2 * width)))
        offset += 2 * width
    hf_rows = row_index(raw, layout, ids)

    def values(data):
        bits = np.frombuffer(data, dtype="<u2")
        return bits, (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    comparisons = []
    for stage, layer, width in EXPECTED:
        key = (stage, layer)
        left_bytes, left = values(native_rows[key])
        right_bytes, right = values(hf_rows[key])
        delta = np.abs(left - right)
        if not np.all(np.isfinite(delta)):
            raise ValueError(f"nonfinite stage delta at {key}")
        comparisons.append({
            "stage": stage, "layer": layer, "width": width,
            "different_bf16_elements": int(np.count_nonzero(left_bytes != right_bytes)),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
        })
    _, left = values(native_rows[("logits", "-")])
    _, right = values(hf_rows[("logits", "-")])
    selected = {str(i): {"riley": float(left[i]), "hf": float(right[i])}
                for i in sorted({base["tokens"][0], meta["selected_id"]})}
    save(output, {"case": ordinal, "native_row": header, "stages": comparisons,
                  "first_different_stage": next((row for row in comparisons if row["different_bf16_elements"]), None),
                  "raw_logits_argmax": {"riley": int(left.argmax()), "hf": int(right.argmax())},
                  "selected_token_logits": selected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    native_parser = phases.add_parser("native")
    native_parser.add_argument("address")
    native_parser.add_argument("model")
    hf = phases.add_parser("hf")
    hf.add_argument("model_dir", type=Path)
    hf.add_argument("reference", type=Path)
    comparison = phases.add_parser("compare")
    comparison.add_argument("native_dir", type=Path)
    comparison.add_argument("hf_dir", type=Path)
    for phase in (native_parser, hf, comparison):
        phase.add_argument("source", type=Path)
        phase.add_argument("case", choices=CASES, type=int)
        phase.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "native":
        native(args.address, args.model, args.source, args.case, args.output)
    elif args.mode == "hf":
        hf_trace(args.model_dir, args.source, args.reference, args.case, args.output)
    else:
        compare(args.source, args.case, args.native_dir, args.hf_dir, args.output)


if __name__ == "__main__":
    main()
