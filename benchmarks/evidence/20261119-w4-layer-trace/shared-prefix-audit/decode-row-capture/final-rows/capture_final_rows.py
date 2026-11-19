"""Trace two fixed decode decisions while retaining all BF16 arrays on the GPU host."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
import os
from pathlib import Path

from capture_reference import MARKER, complete, files, require, same_completion, score_row
from long_accuracy import top_scores
from replay_final_head import bf16, round_bf16

TARGETS = {("long", "12"): (669, 9), ("holdout", "numbers-1023"): (1023, 67)}
HIDDEN, VOCAB = 3584, 152064
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"


def fingerprint(data):
    return sha256(data).hexdigest()


def digest(path):
    return fingerprint(path.read_bytes())


def targets(root):
    selected = {}
    for suite, case, original, reference, offset, other in files(root):
        key = (suite, str(case.label))
        if key not in TARGETS:
            continue
        count = len(original["prompt_tokens"])
        require((count, offset) == TARGETS[key] and key not in selected
                and original["tokens"][:offset] == reference["hf_tokens"][:offset]
                and original["tokens"][offset] != other, f"{key}: the decode fixture changed")
        selected[key] = (case, original, reference, offset, other)
    require(selected.keys() == TARGETS.keys(), "one or more decode fixtures are missing")
    return selected


def stem(original, offset):
    return f"{len(original['prompt_tokens'])}-{offset}"


def capture_riley(root, address, model, rows, log):
    result = {}
    for key, (case, original, _, offset, other) in targets(root).items():
        prompt = original["prompt_tokens"]
        response = complete(address, model, case.prompt, diagnostic=True, expected=prompt, limit=case.limit)
        require(same_completion(original, response) and original["usage"] == response["usage"],
                f"{key}: the probe changed the original response")
        name = stem(original, offset)
        markers = [(gpu, int(vocab), int(extent)) for plen, step, gpu, vocab, extent in MARKER.findall(log.read_text())
                   if (int(plen), int(step)) == TARGETS[key]]
        chosen = response["tokens"][offset]
        require(len(markers) == 1 and markers[0][0] == str(chosen)
                and markers[0][1] == VOCAB, f"{key}: missing original greedy path")
        row = (rows / f"{name}.bf16").read_bytes()
        hidden = (rows / f"{name}.hidden.bf16").read_bytes()
        require(len(hidden) == 4 * HIDDEN, f"{key}: unexpected hidden extent")
        result[name] = {
            "response_token_sha256": fingerprint(json.dumps(response["tokens"]).encode()),
            "hidden_sha256": fingerprint(hidden),
            "row": score_row(row, VOCAB, markers[0][2], chosen, other),
        }
    return result


def row_bytes(tensor, width):
    import torch

    require(tensor.shape == (1, 1, width) and tensor.dtype == torch.bfloat16,
            f"unexpected captured tensor shape or dtype for width {width}")
    return tensor[0, 0].detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()


def checkpoint(model):
    require(digest(model / "riley-checkpoint.json") == CHECKPOINT, "checkpoint manifest differs")
    config = json.loads((model / "config.json").read_bytes())
    require((config["hidden_size"], config["vocab_size"], config["tie_word_embeddings"]) ==
            (HIDDEN, VOCAB, False), "incompatible checkpoint geometry")
    return config


def capture_hf(root, model_dir, destination):
    import torch
    import transformers

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and torch.cuda.device_count() == 1,
            "HF must use only the reserved A100 GPU 7")
    checkpoint(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    destination.mkdir()
    result = {}
    for key, (_, original, reference, offset, other) in targets(root).items():
        prefix = original["tokens"][:offset]
        records = []

        def save_norm(_module, args, output):
            records.extend((row_bytes(args[0], HIDDEN), row_bytes(output, HIDDEN)))

        with torch.inference_mode():
            prompt = torch.tensor([original["prompt_tokens"]], device="cuda")
            output = model(prompt, use_cache=True, logits_to_keep=1)
            for token in prefix[:-1]:
                inputs = torch.tensor([[token]], device="cuda")
                output = model(inputs, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
            hook = model.model.norm.register_forward_hook(save_norm)
            try:
                inputs = torch.tensor([[prefix[-1]]], device="cuda")
                output = model(inputs, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
            finally:
                hook.remove()
        chosen = original["tokens"][offset]
        scores = top_scores(output.logits[0, 0], (chosen, other))
        require(len(records) == 2 and scores == reference["first_divergence"]["hf_sdpa_replay"],
                f"{key}: SDPA decode did not reproduce its recorded scores")
        raw_logits = row_bytes(output.logits, VOCAB)
        raw_hidden = b"".join(records)
        name = stem(original, offset)
        for suffix, raw in (("bf16", raw_logits), ("hidden.bf16", raw_hidden)):
            with (destination / f"{name}.{suffix}").open("xb") as stream:
                stream.write(raw)
        result[name] = {"hidden_sha256": fingerprint(raw_hidden), "logits_sha256": fingerprint(raw_logits),
                        "scores": scores}
    return {"gpu_uuid": GPU, "torch": torch.__version__, "transformers": transformers.__version__, "cases": result}


def weight_rows(model, ids):
    from safetensors import safe_open
    import torch

    weight_map = json.loads((model / "model.safetensors.index.json").read_bytes())["weight_map"]
    head_file = weight_map["lm_head.weight"]
    norm_file = weight_map["model.norm.weight"]
    require(all(Path(name).name == name and name.endswith(".safetensors") for name in (head_file, norm_file)),
            "unexpected checkpoint shard path")
    with safe_open(model / head_file, framework="pt", device="cpu") as shard:
        matrix = shard.get_slice("lm_head.weight")
        require(matrix.get_shape() == [VOCAB, HIDDEN], "output head geometry changed")
        heads = {token: row_bytes(matrix[token:token + 1].unsqueeze(0), HIDDEN) for token in ids}
    with safe_open(model / norm_file, framework="pt", device="cpu") as shard:
        weight = shard.get_tensor("model.norm.weight")
        require(weight.shape == (HIDDEN,) and weight.dtype == torch.bfloat16, "final norm dtype or width changed")
        gamma = row_bytes(weight.reshape(1, 1, HIDDEN), HIDDEN)
    return gamma, heads


def analyze(raw_hidden, raw_logits, weights, raw_gamma, epsilon):
    require(len(raw_hidden) == HIDDEN * 4 and len(raw_logits) == VOCAB * 2, "unexpected captured extent")
    residual, norm = bf16(raw_hidden[:2 * HIDDEN]), bf16(raw_hidden[2 * HIDDEN:])
    gamma = bf16(raw_gamma)
    inverse = 1 / math.sqrt(math.fsum(value * value for value in residual) / HIDDEN + epsilon)
    replay_norm = [round_bf16(round_bf16(value * inverse) * g) for value, g in zip(residual, gamma, strict=True)]
    dot = {token: math.fsum(value * coefficient for value, coefficient in zip(norm, bf16(raw), strict=True))
           for token, raw in weights.items()}
    observed = {token: bf16(raw_logits[2 * token:2 * token + 2])[0] for token in weights}
    rounded = {token: round_bf16(value) for token, value in dot.items()}
    return {
        "residual_sha256": fingerprint(raw_hidden[:2 * HIDDEN]), "final_norm_sha256": fingerprint(raw_hidden[2 * HIDDEN:]),
        "logits_sha256": fingerprint(raw_logits),
        "two_round_fp64_norm_matching_elements": sum(a == b for a, b in zip(norm, replay_norm, strict=True)),
        "projection_fp64": dot, "projection_rounded_bf16": rounded,
        "observed_scores": observed, "projection_reproduced_both_scores": rounded == observed,
    }


def verify(root, model, stage, original_run):
    config = checkpoint(model)
    epsilon = config["rms_norm_eps"]
    require(math.isfinite(epsilon) and 0 < epsilon < 1, "invalid RMSNorm epsilon")
    riley_manifest = json.loads((stage / "riley/client.log").read_bytes())
    hf_manifest = json.loads((stage / "hf/scores.json").read_bytes())
    require(hf_manifest["gpu_uuid"] == GPU, "the reference did not use the reserved A100 GPU 7")
    old_cases = json.loads((original_run / "riley/client.log").read_bytes())["cases"]
    old = {f"{row['prompt_len']}-{row['offset']}": row for row in old_cases}
    result = {}
    for key, (_, original, reference, offset, other) in targets(root).items():
        chosen = original["tokens"][offset]
        name = stem(original, offset)
        gamma, weights = weight_rows(model, (chosen, other))
        arrays = {}
        for engine, directory in (("riley", stage / "riley/rows"), ("hf", stage / "hf/rows")):
            arrays[engine] = tuple((directory / f"{name}.{suffix}").read_bytes()
                                   for suffix in ("hidden.bf16", "bf16"))
        riley_hidden, riley_logits = arrays["riley"]
        hf_hidden, hf_logits = arrays["hf"]
        require(riley_manifest[name]["hidden_sha256"] == fingerprint(riley_hidden)
                and riley_manifest[name]["row"] == old[name]["row"]
                and fingerprint(riley_logits) == old[name]["row"]["sha256"]
                and riley_manifest[name]["response_token_sha256"] == old[name]["observed_output_hash"],
                f"{key}: Riley output did not reproduce the prior capture")
        hf = hf_manifest["cases"][name]
        require(hf["hidden_sha256"] == fingerprint(hf_hidden) and hf["logits_sha256"] == fingerprint(hf_logits)
                and hf["scores"] == reference["first_divergence"]["hf_sdpa_replay"],
                f"{key}: HF output or source receipt changed")
        reports = {engine: analyze(*raw, weights, gamma, epsilon) for engine, raw in arrays.items()}
        require(reports["hf"]["observed_scores"] == {
            token: hf["scores"]["selected_scores"][str(token)] for token in weights
        }, f"{key}: score receipt does not match the HF logits")
        different = {}
        for label, start, end in (("residual", 0, 2 * HIDDEN), ("final_norm", 2 * HIDDEN, 4 * HIDDEN)):
            differences = [a - b for a, b in zip(bf16(riley_hidden[start:end]), bf16(hf_hidden[start:end]), strict=True)]
            different[label] = {"changed_elements": sum(delta != 0 for delta in differences),
                                "rms": math.sqrt(math.fsum(delta * delta for delta in differences) / len(differences))}
        result[name] = {"suite": key[0], "case": key[1], "riley_token": chosen, "hf_token": other,
                        "width": HIDDEN, "input_sha256": original["prompt_sha256"],
                        "prefix_sha256": fingerprint(json.dumps(original["tokens"][:offset]).encode()),
                        "weight_sha256": {token: fingerprint(raw) for token, raw in weights.items()},
                        "norm_weight_sha256": fingerprint(gamma), "engine_comparison": different, "engines": reports}
    return {"checkpoint_manifest_sha256": CHECKPOINT, "gpu_uuid": GPU, "hf_torch": hf_manifest["torch"],
            "hf_transformers": hf_manifest["transformers"], "cases": result}


def main():
    parser = ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    riley = actions.add_parser("riley")
    riley.add_argument("root", type=Path)
    riley.add_argument("address")
    riley.add_argument("model")
    riley.add_argument("rows", type=Path)
    riley.add_argument("log", type=Path)
    hf = actions.add_parser("hf")
    hf.add_argument("root", type=Path)
    hf.add_argument("model_dir", type=Path)
    hf.add_argument("destination", type=Path)
    summary = actions.add_parser("verify")
    for name in ("root", "model", "stage", "original_run"):
        summary.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    print(json.dumps({"riley": capture_riley, "hf": capture_hf, "verify": verify}[action](**args),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
