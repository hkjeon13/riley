#!/usr/bin/env python3
"""Compare long-context greedy generation in isolated phases.

The serving phase records the exact prompt IDs observed by Riley. The
reference phase passes those IDs directly to offline Hugging Face inference.
Use the historical 18 cases by default or pass a pinned holdout workload.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
from typing import NamedTuple
import urllib.request

MAX_NEW_TOKENS = 32
CONTEXT = 8192
WORDS = ("the river market station garden winter engine letter mountain copper signal harbor forest "
         "planet window silver thread summer lantern village theory circuit pattern shadow bridge ocean").split()

HOLDOUT_TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
# Use a separate, pinned set of tasks at each side of the prefill/KV boundaries.
HOLDOUT = (
    ("korean-255", 255, 64, "항구 조사팀은 화요일 열 시에 3번 부두에서 만난다.",
     "언제 어디서 만나는지 한국어 한 문장으로 답하세요."),
    ("code-256", 256, 64, "def delta(x, y):\n    return (x-y)**2\n",
     "What is delta(7, 2)? Show the calculation in one line."),
    ("numbers-257", 257, 64, "An order contains 7 boxes of 18 screws. The team returns 5 screws.",
     "How many screws remain? Return the number and units."),
    ("repeat-511", 511, 64, "The locked tag is ABA-ABA-ABB. All repeated words below are filler.",
     "Copy the locked tag with the exact punctuation."),
    ("japanese-512", 512, 64, "受付は午前九時に開きます。予約番号は青の二十六です。",
     "受付の開始時刻と予約番号を日本語で答えてください。"),
    ("code-513", 513, 64, "values = [3, 8, 2, 4]\nanswer = [v*2 for v in values if v % 2 == 0]",
     "Write the value of answer as a Python literal."),
    ("numbers-1023", 1023, 128, "Warehouse A has 312 L of water. It sends 4 lots of 26 L to B.",
     "Give the water remaining in Warehouse A and explain the arithmetic."),
    ("korean-1024", 1024, 128, "연구팀은 표본 가를 영하 18도, 표본 나를 영상 4도에 보관했다.",
     "두 표본의 보관 온도와 온도 차이를 한국어로 정리하세요."),
    ("repeat-1025", 1025, 128, "Use these labels in this exact order: pine-7, pine-7, pine-8.",
     "Write one sentence for each of the three labels, preserving their order."),
    ("long-code-7836", 7836, 256, "A batch has columns time, value, quality. Bad rows must be excluded.",
     "Write Python code that filters bad rows, computes the median by date, and explains each step."),
    ("long-korean-7917", 7917, 256, "기록의 지역은 강릉, 대전, 부여이며 각 지역에서 오전과 오후에 관측했다.",
     "세 지역에 대한 관측 일지 예시를 각각 작성하세요. 시간, 관측값, 다음 확인 내용을 자세히 쓰세요."),
)


class Case(NamedTuple):
    label: str
    prompt: str
    limit: int
    expected_ids: list[int] | None


class Workload(NamedTuple):
    cases: list[Case]
    sha256: str | None


def prompts():
    rng = random.Random(5)
    return [
        " ".join(rng.choice(WORDS) + ("." if rng.random() < .1 else "") for _ in range(length))
        + "\n\nIn short, the document is about"
        for length in [600, 1200, 2000, 3000, 4000, 5500] * 3
    ]


def fit_prompt(encode, source, question, length):
    start = source + "\nIgnore the padding words below.\n"
    end = "\n\nQuestion: " + question + "\nAnswer:"
    guess = length - len(encode(start + end))
    for offset in range(-12, 13):
        count = guess + offset
        if count < 0:
            continue
        prompt = start + " archive" * count + end
        ids = encode(prompt)
        if len(ids) == length:
            return prompt, ids
    raise ValueError(f"could not fit {length} tokenizer IDs for {question}")


def prepare_holdout(tokenizer_path, output):
    from tokenizers import Tokenizer

    tokenizer_sha = sha256(tokenizer_path.read_bytes()).hexdigest()
    if tokenizer_sha != HOLDOUT_TOKENIZER_SHA256:
        raise ValueError("holdout requires the pinned Qwen2.5-7B tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    cases = []
    for label, length, limit, source, question in HOLDOUT:
        prompt, ids = fit_prompt(encode, source, question, length)
        cases.append({"label": label, "prompt": prompt, "expected_ids": ids, "limit": limit})
    save(output, {"tokenizer_sha256": tokenizer_sha, "cases": cases})


def read_workload(path):
    if path is None:
        cases = [Case(str(i), prompt, MAX_NEW_TOKENS, None) for i, prompt in enumerate(prompts())]
        return Workload(cases, None)
    raw = path.read_bytes()
    document = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    if document["tokenizer_sha256"] != HOLDOUT_TOKENIZER_SHA256:
        raise ValueError("unexpected holdout tokenizer binding")
    source = document["cases"]
    expected = [(label, length, limit) for label, length, limit, _, _ in HOLDOUT]
    cases = []
    for row, (label, length, limit) in zip(source, expected, strict=True):
        prompt, ids = row["prompt"], token_ids(row["expected_ids"])
        if (row["label"] != label or type(row["limit"]) is not int or row["limit"] != limit
                or not isinstance(prompt, str) or not prompt or len(ids) != length
                or len(ids) + limit > CONTEXT):
            raise ValueError(f"invalid holdout case {label}")
        cases.append(Case(label, prompt, limit, ids))
    return Workload(cases, sha256(raw).hexdigest())


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


def complete(address, model, prompt, diagnostic=False, expected=None, *, limit=MAX_NEW_TOKENS):
    body = {"model": model, "prompt": prompt, "max_tokens": limit,
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
    if (not prompt_tokens or len(prompt_tokens) + limit > CONTEXT
            or len(tokens) > limit or finish not in ("stop", "length")
            or (finish == "length" and len(tokens) != limit)
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


def read_capture(path, workload):
    document = json.loads(path.read_text(encoding="utf-8"))
    baseline = document["baseline"]
    expected = [sha256(case.prompt.encode()).hexdigest() for case in workload.cases]
    if len(baseline) != len(expected) or [row["prompt_sha256"] for row in baseline] != expected:
        raise ValueError("capture does not match the pinned prompt content")
    if document.get("workload_sha256") != workload.sha256:
        raise ValueError("capture does not match the holdout file digest")
    for case, row in zip(workload.cases, baseline, strict=True):
        if case.expected_ids is not None and row["prompt_tokens"] != case.expected_ids:
            raise ValueError(f"captured prompt IDs differ for {case.label}")
    if len(document["diagnostic"]) != len(expected):
        raise ValueError("incomplete diagnostic rows")
    return document


def save(path, document):
    with path.open("x", encoding="utf-8") as output:
        json.dump(document, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def capture(address, model, output, workload):
    baseline = []
    for case in workload.cases:
        row = complete(address, model, case.prompt, expected=case.expected_ids, limit=case.limit)
        row["prompt_sha256"] = sha256(case.prompt.encode()).hexdigest()
        baseline.append(row)
    diagnostic = []
    for i, case in enumerate(workload.cases):
        row = complete(address, model, case.prompt, diagnostic=True,
                       expected=baseline[i]["prompt_tokens"], limit=case.limit)
        row.pop("prompt_tokens")
        diagnostic.append(row)
    duration_results = {}
    for clients in (8, 32):
        jobs = [(i, case) for _ in range(3) for i, case in enumerate(workload.cases)]

        def replay(job):
            i, case = job
            row = complete(address, model, case.prompt, expected=baseline[i]["prompt_tokens"], limit=case.limit)
            row.pop("prompt_tokens")
            return {"case": i, **row}

        with ThreadPoolExecutor(clients) as pool:
            results = list(pool.map(replay, jobs))
        duration_results[str(clients)] = results
        matches = sum(same_completion(row, baseline[row["case"]]) for row in results)
        print(f"C{clients} exact tokens, text and finish: {matches}/{len(results)}", flush=True)
    save(output, {"workload_sha256": workload.sha256, "baseline": baseline,
                  "diagnostic": diagnostic, "repeats": duration_results})


def vllm_capture(address, model, source, output, workload):
    baseline = read_capture(source, workload)["baseline"]
    outcomes = [
        complete(address, model, row["prompt_tokens"], expected=row["prompt_tokens"], limit=case.limit)
        for row, case in zip(baseline, workload.cases, strict=True)
    ]
    save(output, {"workload_sha256": workload.sha256, "reference": outcomes})


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


def hf_reference(model_dir, source, output, workload):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    capture_doc = read_capture(source, workload)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    eos_value = model.generation_config.eos_token_id
    eos = eos_value if isinstance(eos_value, list) else [eos_value]
    if not eos or any(type(n) is not int for n in eos):
        raise ValueError("pinned model must have EOS token IDs")
    result = []
    paired = zip(workload.cases, capture_doc["baseline"], capture_doc["diagnostic"], strict=True)
    for i, (case, base, diag) in enumerate(paired):
        original = base["prompt_tokens"]
        tokenizer_ids_match = tokenizer.encode(case.prompt) == original
        if case.expected_ids is not None and not tokenizer_ids_match:
            raise ValueError(f"HF tokenizer differs from pinned IDs for {case.label}")
        input_ids = torch.tensor([original], device="cuda")
        with torch.inference_mode():
            generated = model.generate(
                input_ids, max_new_tokens=case.limit, do_sample=False, temperature=None,
                top_p=None, top_k=None, repetition_penalty=None, pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, output_logits=True,
            )
        ids = generated.sequences[0, len(original):].tolist()
        reason = "stop" if ids and ids[-1] in eos else "length"
        candidate = terminal_normalized(base["tokens"], base["finish"], eos)
        expected = terminal_normalized(ids, reason, eos)
        divergence = common_prefix(candidate, expected)
        row = {"case": i, "label": case.label, "prompt_tokens": len(original),
               "hf_tokenizer_ids_match": tokenizer_ids_match,
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
        "scope": "Same Riley prompt token IDs, SDPA generation and SDPA/eager shared-prefix replay",
        "workload_sha256": workload.sha256,
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
        sub.add_argument("--workload", type=Path)
    hf = kinds.add_parser("hf")
    hf.add_argument("model_dir", type=Path)
    hf.add_argument("source", type=Path)
    hf.add_argument("output", type=Path)
    hf.add_argument("--workload", type=Path)
    prepared = kinds.add_parser("prepare-holdout")
    prepared.add_argument("tokenizer", type=Path)
    prepared.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare-holdout":
        prepare_holdout(args.tokenizer, args.output)
        return
    workload = read_workload(args.workload)
    if args.phase == "capture":
        capture(args.address, args.model, args.output, workload)
    elif args.phase == "vllm":
        vllm_capture(args.address, args.model, args.source, args.output, workload)
    else:
        hf_reference(args.model_dir, args.source, args.output, workload)


if __name__ == "__main__":
    main()
