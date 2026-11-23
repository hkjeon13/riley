"""Render and score the fixed long-answer cases through the chat API."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.append(str(Path(__file__).resolve().parents[2] / "live"))
from long_accuracy import CONTEXT, HOLDOUT_TOKENIZER_SHA256, save, token_ids
from long_answer_check import LIMIT, load_fixture, same_completion, score


FIXTURE_SHA = "e68ed38a73caad88ae8de0a0000a439f732331a9b118849de28d92d4366efda3"
CONFIG_SHA = "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
VARIANTS = ("wrapped", "without_answer_cue")
CONCURRENCIES = (1, 8)
CUE = "\nAnswer:"


def digest(raw):
    return sha256(raw).hexdigest()


def messages(case, variant):
    prompt = case["prompt"]
    if not prompt.endswith(CUE):
        raise ValueError(f"missing answer cue in {case['label']}")
    if variant == "without_answer_cue":
        prompt = prompt[:-len(CUE)]
    return [{"role": "user", "content": prompt}]


def prepare(model, fixture, output):
    from transformers import AutoTokenizer

    if (digest((model / "tokenizer.json").read_bytes()) != HOLDOUT_TOKENIZER_SHA256
            or digest((model / "tokenizer_config.json").read_bytes()) != CONFIG_SHA
            or (model / "chat_template.jinja").exists()):
        raise ValueError("expected the pinned Qwen chat template in tokenizer_config.json")
    cases = load_fixture(fixture, FIXTURE_SHA)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    template = json.loads((model / "tokenizer_config.json").read_bytes())["chat_template"]
    if tokenizer.chat_template != template:
        raise ValueError("Transformers selected a different chat template")

    def prepare_case(case, variant):
        conversation = messages(case, variant)
        rendered = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        if rendered.count(conversation[0]["content"]) != 1:
            raise ValueError(f"chat template changed the user content in {case['label']}")
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if ids != tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                               return_dict=False):
            raise ValueError(f"chat template and explicit tokenization differ in {case['label']}")
        return {"label": case["label"], "user_sha256": digest(conversation[0]["content"].encode()),
                "rendered_sha256": digest(rendered.encode()), "prompt_tokens": ids}

    payload = {"fixture_sha256": FIXTURE_SHA, "tokenizer_sha256": HOLDOUT_TOKENIZER_SHA256,
               "tokenizer_config_sha256": CONFIG_SHA, "chat_template_sha256": digest(template.encode()),
               "transformers": version("transformers"), "tokenizers": version("tokenizers"),
               "variants": {variant: [prepare_case(case, variant) for case in cases] for variant in VARIANTS}}
    raw = gzip.compress((json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0)
    with output.open("xb") as stream:
        stream.write(raw)
    _, bound = load_inputs(cases, output)
    lengths = {variant: sorted(len(row["expected_ids"]) for row in rows) for variant, rows in bound.items()}
    print(json.dumps({"inputs_sha256": digest(raw), "fixture_sha256": FIXTURE_SHA,
                      "tokenizer_config_sha256": CONFIG_SHA, "chat_template_sha256": payload["chat_template_sha256"],
                      "transformers": payload["transformers"], "tokenizers": payload["tokenizers"],
                      "tokens_by_variant": {variant: [sizes[0], sizes[-1]] for variant, sizes in lengths.items()}},
                     indent=2, allow_nan=False))


def load_inputs(cases, path):
    raw = path.read_bytes()
    inputs = json.loads(gzip.decompress(raw))
    if (inputs["fixture_sha256"] != FIXTURE_SHA or inputs["tokenizer_sha256"] != HOLDOUT_TOKENIZER_SHA256
            or inputs["tokenizer_config_sha256"] != CONFIG_SHA or inputs["variants"].keys() != set(VARIANTS)):
        raise ValueError("chat inputs differ from the pinned fixture and tokenizer")
    bound = {}
    for variant, rows in inputs["variants"].items():
        workload = []
        for case, row in zip(cases, rows, strict=True):
            ids = token_ids(row["prompt_tokens"])
            if (row["label"] != case["label"] or not len(case["expected_ids"]) < len(ids) <= CONTEXT - LIMIT
                    or ids == case["expected_ids"] or len(row["rendered_sha256"]) != 64
                    or row["user_sha256"] != digest(messages(case, variant)[0]["content"].encode())):
                raise ValueError(f"invalid chat input in {variant}/{case['label']}")
            workload.append({**case, "expected_ids": ids})
        bound[variant] = workload
    if any(a["expected_ids"] == b["expected_ids"] for a, b in zip(*(bound[key] for key in VARIANTS), strict=True)):
        raise ValueError("removing the answer cue did not change the served input")
    return digest(raw), bound


def run_chat(address, model, case, variant):
    body = {"model": model, "messages": messages(case, variant), "add_generation_prompt": True,
            "max_tokens": LIMIT, "temperature": 0, "return_token_ids": True}
    request = Request(f"http://{address}/v1/chat/completions", json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        reply = json.load(response)
    (choice,) = reply["choices"]
    message = choice["message"]
    tokens = token_ids(choice["token_ids"])
    text = message["content"]
    finish = choice["finish_reason"]
    if text is None and finish == "stop" and (not tokens or tokens in ([151643], [151645])):
        text = ""
    if message["role"] != "assistant" or message.get("tool_calls"):
        raise ValueError(f"expected a text-only assistant response for {case['label']}")
    row = {"label": case["label"], "prompt_tokens": token_ids(choice.get("prompt_token_ids", reply.get("prompt_token_ids"))),
           "tokens": tokens, "finish": finish, "text": text, "usage": reply["usage"]}
    score(case, row)
    return row


def capture(cases, address, model, runtime, path, inputs_path):
    inputs_sha, bound = load_inputs(cases, inputs_path)
    results = {}
    for variant, workload in bound.items():
        modes = {}
        for concurrency in CONCURRENCIES:
            with ThreadPoolExecutor(concurrency) as pool:
                modes[f"c{concurrency}"] = list(pool.map(lambda case: run_chat(address, model, case, variant), workload))
            print(f"{runtime} {variant} c{concurrency}: {len(workload)}", flush=True)
        results[variant] = modes
    save(path, {"fixture_sha256": FIXTURE_SHA, "inputs_sha256": inputs_sha, "runtime": runtime, "variants": results})


def report(root):
    cases = load_fixture(root / "facts.json.gz", FIXTURE_SHA)
    inputs_sha, bound = load_inputs(cases, root / "inputs.json.gz")
    results = {"fixture_sha256": FIXTURE_SHA, "inputs_sha256": inputs_sha, "engines": {}, "cases": {}}
    for engine in ("riley", "vllm"):
        raw = (root / engine / "capture.json").read_bytes()
        capture_data = json.loads(raw)
        if (capture_data["fixture_sha256"] != FIXTURE_SHA or capture_data["inputs_sha256"] != inputs_sha
                or capture_data["runtime"] != engine or capture_data["variants"].keys() != set(VARIANTS)):
            raise ValueError(f"{engine} capture differs from the registered inputs")
        engine_report = {"capture_sha256": digest(raw), "variants": {}}
        for variant, workload in bound.items():
            modes = capture_data["variants"][variant]
            if modes.keys() != {"c1", "c8"}:
                raise ValueError(f"missing capture mode in {engine}/{variant}")
            counts = {mode: {name: dict.fromkeys(("requests", "exact", "exact_stop", "first_line", "length"), 0)
                             for name in ("lookup", "arithmetic-ko", "join")} for mode in modes}
            preserved = 0
            for case, c1, c8 in zip(workload, modes["c1"], modes["c8"], strict=True):
                outcomes = {}
                for mode, row in (("c1", c1), ("c8", c8)):
                    summary = score(case, row)
                    first_line = next((line.strip() for line in row["text"].splitlines() if line.strip()), "")
                    outcome = {"exact": summary["exact_answer"], "exact_stop": summary["complete_exact_answer"],
                               "first_line": first_line == case["answer"], "length": summary["finish"] == "length"}
                    outcomes[mode] = outcome
                    totals = counts[mode][case["task"]]
                    totals["requests"] += 1
                    for key, value in outcome.items():
                        totals[key] += int(value)
                invariant = same_completion(c1, c8) and c1["usage"] == c8["usage"]
                preserved += int(invariant)
                results["cases"].setdefault(case["label"], {}).setdefault(variant, {})[engine] = {
                    **outcomes, "batch_invariant": invariant}
            engine_report["variants"][variant] = {"batch_invariant": preserved, **counts}
        results["engines"][engine] = engine_report
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for name in ("model", "fixture", "output"):
        preparation.add_argument(name, type=Path)
    run = commands.add_parser("capture")
    run.add_argument("runtime", choices=("riley", "vllm"))
    run.add_argument("address")
    run.add_argument("model")
    run.add_argument("output", type=Path)
    run.add_argument("fixture", type=Path)
    run.add_argument("fixture_sha256")
    commands.add_parser("report").add_argument("stage", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.model, args.fixture, args.output)
    elif args.command == "capture":
        if args.fixture_sha256 != FIXTURE_SHA:
            raise ValueError("unexpected fixture hash")
        capture(load_fixture(args.fixture, FIXTURE_SHA), args.address, args.model, args.runtime,
                args.output, args.fixture.with_name("inputs.json.gz"))
    else:
        print(json.dumps(report(args.stage), indent=2, allow_nan=False))
