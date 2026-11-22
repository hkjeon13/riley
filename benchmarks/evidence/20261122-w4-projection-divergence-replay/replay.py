"""Compare archived token choices and rescore only their first split on a shared history."""

from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

from long_accuracy import common_prefix, read_capture, read_workload, terminal_normalized

CANDIDATES = {"attention31": "w4-o31-serving-20261120.7W3nr5JH",
              "fp32_down21": "w4-fp32-down-serving-20261120.Saj3XXSY"}
VLLM = {"long": "w4-long-vllm-20261119.pwcEpuzL", "holdout": "w4-holdout-20261119.cgmtjz4x"}
NATIVE = "w4-native-projection-plans-20261120.yvjrWvOj"
ARMS = ("baseline", *CANDIDATES)
FIELDS = ("tokens", "text", "finish", "usage")
LADDER = ("sdpa", "eager")
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
CONTROL = (("long", "3"), ("long", "14"), ("holdout", "numbers-1023"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def hash_ids(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def reported_margin(row, step):
    values = sorted(row["logprobs"]["top_logprobs"][step].values(), reverse=True)
    require(len(values) >= 2 and all(math.isfinite(value) for value in values), "invalid diagnostic log probabilities")
    return values[0] - values[1]


def cases(evidence):
    suite_runtime = None
    output = []
    for suite in ("long", "holdout"):
        root = evidence / NATIVE / "fixtures" / f"20261119-w4-{suite}"
        workload = read_workload(root / "holdout.json.gz" if suite == "holdout" else None)
        hf_doc = read(root / "hf/reference.json")
        eos = hf_doc["eos_token_ids"]
        require(suite_runtime in (None, hf_doc["runtime"]), "HF reference environments differ")
        suite_runtime = hf_doc["runtime"]
        serving = {tag: read_capture(evidence / path / "candidate/rows" / f"{suite}.json", workload)
                   for tag, path in CANDIDATES.items()}
        baselines = [read_capture(evidence / path / "baseline/rows" / f"{suite}.json", workload)
                     for path in CANDIDATES.values()]
        require(baselines[0] == baselines[1], "candidate runs did not share an identical Riley baseline")
        serving["baseline"] = baselines[0]
        vllm_doc = read(evidence / VLLM[suite] / "vllm/results.json")
        require(vllm_doc.get("workload_sha256") == workload.sha256, "vLLM workload differs")
        references = zip(hf_doc["cases"], vllm_doc["reference"], workload.cases, strict=True)
        for i, (hf, vllm, case) in enumerate(references):
            label = str(case.label)
            arms = {name: serving[name]["baseline"][i] for name in ARMS}
            diagnose = {name: serving[name]["diagnostic"][i] for name in ARMS}
            prompt = arms["baseline"]["prompt_tokens"]
            require(hf["case"] == i and hf["prompt_tokens"] == len(prompt) and hf["hf_tokenizer_ids_match"]
                    and vllm["prompt_tokens"] == prompt
                    and all(row["prompt_tokens"] == prompt for row in arms.values()), f"prompt mismatch at {suite}:{label}")
            require(all(all(row[key] == diagnose[name][key] for key in FIELDS) for name, row in arms.items()),
                    f"diagnostic path differs at {suite}:{label}")
            tokens = {name: terminal_normalized(row["tokens"], row["finish"], eos) for name, row in arms.items()}
            hf_tokens = terminal_normalized(hf["hf_tokens"], hf["hf_finish"], eos)
            vllm_tokens = terminal_normalized(vllm["tokens"], vllm["finish"], eos)
            base = tokens["baseline"]
            require((base == hf_tokens) == hf["exact_token_match"], "baseline pairing changed")
            matches = {name: {"hf_exact": row == hf_tokens, "vllm_exact": row == vllm_tokens,
                              "hf_prefix": common_prefix(row, hf_tokens), "vllm_prefix": common_prefix(row, vllm_tokens),
                              "hf_finish": arms[name]["finish"] == hf["hf_finish"],
                              "vllm_finish": arms[name]["finish"] == vllm["finish"]}
                       for name, row in tokens.items()}
            splits = {}
            for name in CANDIDATES:
                row = tokens[name]
                offset = common_prefix(base, row)
                if base == row:
                    continue
                require(offset < min(len(row), len(base)), "candidate split is a termination boundary")
                splits.setdefault(offset, {})[name] = row[offset]
            if (suite, label) in CONTROL:
                control = hf["first_divergence"]
                require(control and control["hf_id"] is not None, "missing archived positive control")
                splits.setdefault(control["offset"], {})
            points = []
            for offset, changed in sorted(splits.items()):
                prefix = base[:offset]
                canonical = offset < len(hf_tokens) and hf_tokens[:offset] == prefix
                point = {"step": offset, "history_sha256": hash_ids(prefix), "baseline_id": base[offset],
                         "candidate_ids": changed, "archived_hf_id_on_same_history": hf_tokens[offset] if canonical else None,
                         "riley_reported_top_two_gap": {
                             name: reported_margin(diagnose[name], offset) for name in ("baseline", *changed)},
                         "archived_control": hf["first_divergence"] if (suite, label) in CONTROL
                         and hf["first_divergence"]["offset"] == offset else None}
                points.append(point)
            output.append((prompt, base, {"suite": suite, "case": label, "input_tokens": len(prompt),
                                          "input_sha256": hash_ids(prompt), "matches": matches, "points": points}))
    require(len(output) == 29, "expected fixed 29-input corpus")
    return output, suite_runtime


def scores(logits, choices, torch):
    values = logits.float()
    require(torch.isfinite(values).all().item(), "nonfinite logit")
    best = values.max()
    margin = torch.topk(values, 2).values.tolist()
    maxima = torch.nonzero(values == best).flatten().tolist()
    return {"max_logit": best.item(), "argmax_ids": maxima, "margin": margin[0] - margin[1],
            "selected_scores": {str(token): values[token].item() for token in sorted(set(choices))}}


def run_model(inputs, model_dir, expected):
    import torch
    from transformers import AutoModelForCausalLM
    require(torch.cuda.device_count() == 1, "expected one GPU visible to HF")
    require(sha256((model_dir / "riley-checkpoint.json").read_bytes()).hexdigest() == CHECKPOINT, "checkpoint manifest differs")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    runtime = {"torch": torch.__version__, "transformers": version("transformers"), "safetensors": version("safetensors"),
               "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "dtype": str(model.dtype)}
    require(runtime == expected, "reference GPU or package versions changed")
    with torch.inference_mode():
        for backend in LADDER:
            model.set_attn_implementation(backend)
            for prompt, history, item in inputs:
                points = {pt["step"]: pt for pt in item["points"]}
                if not points:
                    continue
                step_output = model(torch.tensor([prompt], device="cuda"), use_cache=True, logits_to_keep=1)
                for offset in range(max(points) + 1):
                    if offset in points:
                        point = points[offset]
                        choices = [point["baseline_id"], *point["candidate_ids"].values()]
                        if point["archived_hf_id_on_same_history"] is not None:
                            choices.append(point["archived_hf_id_on_same_history"])
                        control = point["archived_control"]
                        if control:
                            choices.extend(map(int, control[f"hf_{backend}_replay"]["selected_scores"]))
                        point[backend] = scores(step_output.logits[0, -1], choices, torch)
                        if control:
                            known = control[f"hf_{backend}_replay"]["selected_scores"]
                            require(all(point[backend]["selected_scores"][key] == value for key, value in known.items()),
                                    f"archived {backend} replay changed at {item['suite']}:{item['case']}")
                    if offset < max(points):
                        previous = step_output.past_key_values
                        del step_output
                        step_output = model(torch.tensor([[history[offset]]], device="cuda"), past_key_values=previous,
                                            use_cache=True, logits_to_keep=1)
                        del previous
                del step_output
                print(backend, item["suite"], item["case"], len(points), flush=True)
    return runtime


def main():
    evidence, model, output = map(Path, sys.argv[1:])
    inputs, expected = cases(evidence)
    runtime = run_model(inputs, model, expected)
    with output.open("x") as stream:
        json.dump({"checkpoint_sha256": CHECKPOINT, "runtime": runtime,
                   "cases": [record for _, _, record in inputs]}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
