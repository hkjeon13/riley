"""Compare the first differing residual stage at two pinned decode offsets."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct

from capture_final_rows import GPU, HIDDEN, VOCAB, checkpoint, row_bytes, stem, targets
from capture_reference import complete, require, same_completion
from replay_final_head import bf16


LAYERS = 28
ROWS = [("embedding", "-", HIDDEN)]
ROWS += [(key, str(layer), HIDDEN) for layer in range(LAYERS)
         for key in ("attention_residual", "layer_residual")]
ROWS += [("final_norm", "-", HIDDEN), ("logits", "-", VOCAB)]
PRIOR_SHA256 = "39bfb851792074b968ead3f597d69d58cb2b9e1515b81e926f641921728d03bb"


def digest(data):
    return sha256(data).hexdigest()


def prior_report(source):
    data = source.read_bytes()
    require(digest(data) == PRIOR_SHA256, "recorded uninstrumented decode endpoints differ")
    return json.loads(data)


def position(original, offset):
    return len(original["prompt_tokens"]) + offset - 1, original["tokens"][offset - 1]


def read_payload(raw):
    values = {}
    start = 0
    for stage, layer, width in ROWS:
        end = start + width * 2
        require(end <= len(raw), f"truncated trace row {stage}/{layer}")
        values[(stage, layer)] = raw[start:end]
        start = end
    require(start == len(raw), "unexpected trace bytes after the logits")
    return values


def read_native(directory, previous, offset):
    header, *index = (directory / "index.tsv").read_text().splitlines()
    marker, raw_position, token, row, active, dense = header.split()
    require(marker == "#" and (int(raw_position), int(token)) == position(previous, offset)
            and int(row) == 0 and int(active) == 1 and int(dense) >= 1,
            "native trace does not identify the intended C1 decode row")
    start = 0
    for line, (stage, layer, count) in zip(index, ROWS, strict=True):
        require(line.split("\t") == [stage, layer, str(start), str(count * 2)], "unexpected native trace extent")
        start += count * 2
    raw = (directory / "rows.bf16").read_bytes()
    return raw, read_payload(raw)


def check_endpoint(rows, evidence, label):
    for key, recorded in ((("layer_residual", str(LAYERS - 1)), "residual_sha256"),
                          (("final_norm", "-"), "final_norm_sha256"), (("logits", "-"), "logits_sha256")):
        require(digest(rows[key]) == evidence[recorded],
                f"{label}: traced {key[0]} differs from the uninstrumented decode row")


def native(root, address, model, rows, previous):
    evidence = prior_report(previous)
    result = {}
    for label, (case, original, _, offset, _) in targets(root).items():
        name = stem(original, offset)
        target = rows / name
        target.mkdir()
        response = complete(address, model, case.prompt, diagnostic=True,
                            expected=original["prompt_tokens"], limit=case.limit)
        require(same_completion(original, response) and original["usage"] == response["usage"],
                f"{label}: diagnostic route changed the original output")
        raw, captured = read_native(target, original, offset)
        check_endpoint(captured, evidence["cases"][name]["engines"]["riley"], f"{label} Riley")
        result[name] = {"trace_sha256": digest(raw), "position": position(original, offset)[0],
                        "token_id": position(original, offset)[1], "output_match": True}
    return result


def hf_trace(root, model_dir, destination, previous):
    import torch
    from transformers import AutoModelForCausalLM

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and torch.cuda.device_count() == 1,
            "HF must use only the reserved A100 GPU 7")
    checkpoint(model_dir)
    evidence = prior_report(previous)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    require(len(model.model.layers) == LAYERS, "pinned HF layer count differs")
    destination.mkdir()
    result = {}
    for label, (_, original, _, offset, _) in targets(root).items():
        prefix = original["tokens"][:offset]
        captured = {}

        def keep(key, value):
            require(key not in captured, f"duplicate HF trace at {key}")
            if isinstance(value, tuple):
                value = value[0]
            captured[key] = row_bytes(value, VOCAB if key[0] == "logits" else HIDDEN)

        def after(module, key):
            return module.register_forward_hook(lambda _module, _args, output: keep(key, output))

        def before(module, key):
            return module.register_forward_pre_hook(lambda _module, args: keep(key, args[0]))

        with torch.inference_mode():
            prompt = torch.tensor([original["prompt_tokens"]], device="cuda")
            output = model(prompt, use_cache=True, logits_to_keep=1)
            for token in prefix[:-1]:
                current = torch.tensor([[token]], device="cuda")
                output = model(current, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
            hooks = [after(model.model.embed_tokens, ("embedding", "-")),
                     after(model.model.norm, ("final_norm", "-"))]
            for index, layer in enumerate(model.model.layers):
                hooks.append(before(layer.post_attention_layernorm, ("attention_residual", str(index))))
                hooks.append(after(layer, ("layer_residual", str(index))))
            try:
                current = torch.tensor([[prefix[-1]]], device="cuda")
                output = model(current, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
                keep(("logits", "-"), output.logits)
            finally:
                for hook in hooks:
                    hook.remove()
        name = stem(original, offset)
        check_endpoint(captured, evidence["cases"][name]["engines"]["hf"], f"{label} HF")
        raw = b"".join(captured[(stage, layer)] for stage, layer, _ in ROWS)
        with (destination / f"{name}.bf16").open("xb") as handle:
            handle.write(raw)
        result[name] = {"trace_sha256": digest(raw), "position": position(original, offset)[0],
                        "token_id": position(original, offset)[1], "endpoints_match": True}
    return result


def difference(left, right):
    require(len(left) == len(right), "trace widths differ")
    if left == right:
        return {"width": len(left) // 2, "different_bf16_words": 0, "max_abs": 0, "rms": 0}
    different = sum(a != b for a, b in zip(struct.iter_unpack("<H", left), struct.iter_unpack("<H", right), strict=True))
    deltas = [a - b for a, b in zip(bf16(left), bf16(right), strict=True)]
    return {"width": len(deltas), "different_bf16_words": different,
            "max_abs": max(map(abs, deltas)), "rms": math.sqrt(math.fsum(x*x for x in deltas) / len(deltas))}


def verify(root, stage, previous):
    evidence = prior_report(previous)
    receipts = {engine: json.loads((stage / engine / file).read_bytes())
                for engine, file in (("riley", "client.log"), ("hf", "trace-receipt.json"))}
    result = {}
    for key, (_, orig, _, offset, _) in targets(root).items():
        name = stem(orig, offset)
        r_raw, riley = read_native(stage / "riley/rows" / name, orig, offset)
        h_raw = (stage / "hf/rows" / f"{name}.bf16").read_bytes()
        hf = read_payload(h_raw)
        for engine, (data, rows) in (("riley", (r_raw, riley)), ("hf", (h_raw, hf))):
            receipt = receipts[engine][name]
            require(receipt["trace_sha256"] == digest(data) and
                    (receipt["position"], receipt["token_id"]) == position(orig, offset),
                    f"{name}: incorrect {engine} trace receipt")
            check_endpoint(rows, evidence["cases"][name]["engines"][engine], f"{name} {engine}")
        comparisons = [{"stage": stage_name, "layer": layer, **difference(riley[stage_name, layer], hf[stage_name, layer])}
                       for stage_name, layer, _ in ROWS if stage_name != "logits"]
        earliest = next((entry for entry in comparisons if entry["different_bf16_words"]), None)
        result[name] = {"suite": key[0], "case": key[1], "position": position(orig, offset)[0],
                        "input_token": position(orig, offset)[1],
                        "preexisting_endpoints_exact": True, "earliest_observed_difference": earliest,
                        "source_trace_sha256": {"riley": digest(r_raw), "hf": digest(h_raw)},
                        "comparisons": comparisons}
    return {"scope": "Earliest differing measured residual boundary in two actual C1 decode steps.",
            "baseline_endpoint_summary_sha256": PRIOR_SHA256, "cases": result}


def main():
    parser = ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    native_parser = actions.add_parser("riley")
    native_parser.add_argument("root", type=Path)
    native_parser.add_argument("address")
    native_parser.add_argument("model")
    native_parser.add_argument("rows", type=Path)
    native_parser.add_argument("log", type=Path)
    native_parser.add_argument("previous", type=Path)
    hf_parser = actions.add_parser("hf")
    for name in ("root", "model_dir", "destination", "previous"):
        hf_parser.add_argument(name, type=Path)
    verifier = actions.add_parser("verify")
    for name in ("root", "stage", "previous"):
        verifier.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    args.pop("log", None)
    print(json.dumps({"riley": native, "hf": hf_trace, "verify": verify}[action](**args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
