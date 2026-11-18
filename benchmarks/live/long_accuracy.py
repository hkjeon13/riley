#!/usr/bin/env python3
"""Reproduce the historical 18-case long-context comparison in isolated phases.

The serving phase records the exact prompt IDs observed by Riley. The
reference phase passes those IDs directly to offline Hugging Face inference.
The phases can run in turn on one GPU.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import urllib.request

MAX_NEW_TOKENS = 32
CONTEXT = 8192
WORDS = ("the river market station garden winter engine letter mountain copper signal harbor forest "
         "planet window silver thread summer lantern village theory circuit pattern shadow bridge ocean").split()


def prompts():
    rng = random.Random(5)
    return [
        " ".join(rng.choice(WORDS) + ("." if rng.random() < .1 else "") for _ in range(length))
        + "\n\nIn short, the document is about"
        for length in [600, 1200, 2000, 3000, 4000, 5500] * 3
    ]


def terminal_normalized(tokens, finish, eos):
    """Remove at most the EOS that actually terminated the response."""
    return tokens[:-1] if finish == "stop" and tokens and tokens[-1] in eos else tokens


def common_prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def token_ids(value):
    if not isinstance(value, list) or any(type(n) is not int or not 0 <= n < 2**32 for n in value):
        raise ValueError("token_ids must be an array of unsigned integer IDs")
    return value


def same_completion(left, right):
    return all(left[key] == right[key] for key in ("tokens", "text", "finish"))


def complete(address, model, prompt, diagnostic=False, expected=None):
    body = {"model": model, "prompt": prompt, "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0, "return_token_ids": True}
    if diagnostic:
        body["logprobs"] = 5
    req = urllib.request.Request(f"http://{address}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as response:
        reply = json.load(response)
    if len(reply["choices"]) != 1:
        raise ValueError("expected one choice")
    row = reply["choices"][0]
    prompt_tokens = token_ids(row.get("prompt_token_ids", reply.get("prompt_token_ids")))
    tokens = token_ids(row["token_ids"])
    finish = row["finish_reason"]
    usage = reply["usage"]
    if expected is not None and prompt_tokens != expected:
        raise ValueError("served prompt token IDs differ from the pinned input")
    if (not prompt_tokens or len(prompt_tokens) + MAX_NEW_TOKENS > CONTEXT
            or len(tokens) > MAX_NEW_TOKENS or finish not in ("stop", "length")
            or (finish == "length" and len(tokens) != MAX_NEW_TOKENS)
            or usage["prompt_tokens"] != len(prompt_tokens)
            or usage["completion_tokens"] not in (len(tokens), len(tokens) + int(finish == "stop"))
            or not isinstance(row["text"], str)):
        raise ValueError(f"incoherent completion: {len(prompt_tokens)=}, {len(tokens)=}, {finish=}, {usage=}")
    result = {"prompt_tokens": prompt_tokens, "tokens": tokens, "finish": finish,
              "text": row["text"], "usage": usage}
    if diagnostic:
        logprobs = row["logprobs"]
        if logprobs is None or len(logprobs["top_logprobs"]) != len(tokens):
            raise ValueError("logprob rows do not match the emitted token IDs")
        result["logprobs"] = logprobs
    return result


def read_capture(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    baseline = document["baseline"]
    expected = [sha256(text.encode()).hexdigest() for text in prompts()]
    if len(baseline) != len(expected) or [row["prompt_sha256"] for row in baseline] != expected:
        raise ValueError("capture does not match the legacy 18-prompt workload")
    return document


def save(path, document):
    with path.open("x", encoding="utf-8") as output:
        json.dump(document, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def capture(address, model, output):
    workload = prompts()
    baseline = []
    for text in workload:
        row = complete(address, model, text)
        row["prompt_sha256"] = sha256(text.encode()).hexdigest()
        baseline.append(row)
    diagnostic = []
    for i, text in enumerate(workload):
        row = complete(address, model, text, diagnostic=True, expected=baseline[i]["prompt_tokens"])
        row.pop("prompt_tokens")
        diagnostic.append(row)
    duration_results = {}
    for clients in (8, 32):
        jobs = [(i, text) for _ in range(3) for i, text in enumerate(workload)]

        def replay(job):
            i, text = job
            row = complete(address, model, text, expected=baseline[i]["prompt_tokens"])
            row.pop("prompt_tokens")
            return {"case": i, **row}

        with ThreadPoolExecutor(clients) as pool:
            results = list(pool.map(replay, jobs))
        duration_results[str(clients)] = results
        matches = sum(same_completion(row, baseline[row["case"]]) for row in results)
        print(f"C{clients} exact tokens, text and finish: {matches}/{len(results)}", flush=True)
    save(output, {"baseline": baseline, "diagnostic": diagnostic, "repeats": duration_results})


def vllm_capture(address, model, source, output):
    baseline = read_capture(source)["baseline"]
    outcomes = [
        complete(address, model, row["prompt_tokens"], expected=row["prompt_tokens"])
        for row in baseline
    ]
    save(output, {"reference": outcomes})


def top_scores(scores, extra_ids=()):
    import torch

    scores = scores.reshape(-1).float()
    if not torch.isfinite(scores).all().item():
        raise ValueError("model returned nonfinite logits")
    vals, ids = torch.topk(scores, 5)
    pairs = [{"id": token_id, "logit": value} for token_id, value in zip(ids.tolist(), vals.tolist())]
    return {"top": pairs, "margin": pairs[0]["logit"] - pairs[1]["logit"],
            "selected_scores": {str(i): scores[i].item() for i in sorted(set(extra_ids))}}


def riley_margin(baseline, diagnostic, step):
    if (step >= len(baseline["tokens"]) or step >= len(diagnostic["tokens"])
            or baseline["tokens"][:step + 1] != diagnostic["tokens"][:step + 1]):
        return None
    values = sorted(diagnostic["logprobs"]["top_logprobs"][step].values(), reverse=True)
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        return None
    return values[0] - values[1]


def replay_prefix(model, prompt, prefix, token_pair):
    import torch

    # Reproduce a prefill followed by one-token cached decoding, rather than
    # scoring a new concatenated prefill of a different shape.
    inputs = torch.tensor([prompt], device="cuda")
    with torch.inference_mode():
        output = model(inputs, use_cache=True, logits_to_keep=1)
        for token in prefix:
            inputs = torch.tensor([[token]], device="cuda")
            output = model(inputs, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        return top_scores(output.logits[0, -1], token_pair)


def hf_reference(model_dir, source, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    capture_doc = read_capture(source)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    eos_value = model.generation_config.eos_token_id
    eos = eos_value if isinstance(eos_value, list) else [eos_value]
    if not eos or any(type(n) is not int for n in eos):
        raise ValueError("pinned model must have EOS token IDs")
    result = []
    for i, (prompt, base, diag) in enumerate(zip(prompts(), capture_doc["baseline"], capture_doc["diagnostic"])):
        original = base["prompt_tokens"]
        input_ids = torch.tensor([original], device="cuda")
        with torch.inference_mode():
            generated = model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None,
                top_p=None, top_k=None, repetition_penalty=None, pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, output_logits=True,
            )
        ids = generated.sequences[0, len(original):].tolist()
        reason = "stop" if ids and ids[-1] in eos else "length"
        candidate = terminal_normalized(base["tokens"], base["finish"], eos)
        expected = terminal_normalized(ids, reason, eos)
        divergence = common_prefix(candidate, expected)
        row = {"case": i, "prompt_tokens": len(original),
               "hf_tokenizer_ids_match": tokenizer.encode(prompt) == original,
               "hf_tokens": ids, "hf_finish": reason,
               "hf_text": tokenizer.decode(ids, skip_special_tokens=True),
               "exact_token_match": candidate == expected, "common_prefix": divergence,
               "source_finish": base["finish"], "first_divergence": None}
        if candidate != expected:
            row["first_divergence"] = {
                "offset": divergence,
                "riley_id": candidate[divergence] if divergence < len(candidate) else None,
                "hf_id": expected[divergence] if divergence < len(expected) else None,
            }
        if divergence < min(len(candidate), len(expected)):
            pair = (candidate[divergence], expected[divergence])
            row["first_divergence"]["riley_reported_top_logprob_gap"] = riley_margin(base, diag, divergence)
            row["first_divergence"]["hf_generation_scores"] = top_scores(generated.logits[divergence], pair)
            del generated
            shared = candidate[:divergence]
            row["first_divergence"]["hf_sdpa_replay"] = replay_prefix(model, original, shared, pair)
            model.set_attn_implementation("eager")
            try:
                row["first_divergence"]["hf_eager_replay"] = replay_prefix(model, original, shared, pair)
            finally:
                model.set_attn_implementation("sdpa")
        else:
            del generated
        result.append(row)
        print(f"case={i} prompt={len(original)} common={divergence} match={row['exact_token_match']}", flush=True)
    save(output, {
        "scope": "Qwen legacy prompts, same Riley prompt token IDs, SDPA generation and SDPA/eager shared-prefix replay",
        "runtime": {"torch": torch.__version__, "transformers": version("transformers"),
                    "safetensors": version("safetensors"), "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(), "dtype": str(model.dtype)},
        "eos_token_ids": eos, "cases": result,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    kinds = parser.add_subparsers(dest="phase", required=True)
    for mode in ("capture", "vllm"):
        sub = kinds.add_parser(mode)
        sub.add_argument("address")
        sub.add_argument("model")
        if mode == "vllm":
            sub.add_argument("source", type=Path)
        sub.add_argument("output", type=Path)
    hf = kinds.add_parser("hf")
    hf.add_argument("model_dir", type=Path)
    hf.add_argument("source", type=Path)
    hf.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "capture":
        capture(args.address, args.model, args.output)
    elif args.phase == "vllm":
        vllm_capture(args.address, args.model, args.source, args.output)
    else:
        hf_reference(args.model_dir, args.source, args.output)


if __name__ == "__main__":
    main()
