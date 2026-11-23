#!/usr/bin/env python3
"""Measure exact short answers on pinned long-context requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from typing import NamedTuple

from long_accuracy import CONTEXT, HOLDOUT_TOKENIZER_SHA256, complete, same_completion, save, token_ids

LENGTHS = (255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 7836, 7917)
LIMIT = 32
POSITIONS = ("first", "middle", "last")
NOTES = ("north west inlet bridge moss ridge quarry marsh wind estuary delta pier wharf bluff ice stream "
         "slope sand rain track rail road branch valley grove forest hill lake creek plain coast chalk "
         "pine fern reed frost snow rust rock light shade mud").split()


class Task(NamedTuple):
    label: str
    facts: str
    question: str
    answer: str


TASKS = (
    Task("lookup", "Station Kestrel has archive code 4721.",
         "What is the archive code for Station Kestrel? Return only the integer.", "4721"),
    Task("arithmetic-ko", "설악 선반에는 유리병 318개가 있었다. 팀이 유리병 27개씩 든 쟁반 4개를 옮겨 갔다.",
         "설악 선반에 남은 유리병은 몇 개인가요? 다른 문자 없이 정수만 쓰세요.", "210"),
    Task("join", "Case Alder was assigned to crew Juniper. Crew Juniper uses tag JQ58.",
         "Which tag is used by the crew assigned to Case Alder? Return only the tag.", "JQ58"),
)


def record(i):
    stations = ("Harbor", "Quarry", "Meadow", "Depot", "Valley")
    states = ("inspection pending", "seal intact", "recount scheduled", "transit complete")
    station, state = stations[i % len(stations)], states[(i // 3) % len(states)]
    return f"{station} bay {i + 1}. Batch {6000 + 11 * i}. Quantity {(i * 19) % 197 + 1}. {state}."


def pack(encode, task, position, length, logs):
    header = "Read the record and answer the question.\nRecords:\n"
    suffix = f"\n\nQuestion: {task.question}\nAnswer:"

    def render(count, notes=""):
        location = {"first": 0, "middle": count // 2, "last": count}[position]
        prefix = header + "".join(line + "\n" for line in logs[:location])
        suffix_logs = "".join("\n" + line for line in logs[location:count])
        return prefix + task.facts + suffix_logs + notes + suffix, len(prefix)

    low, high = 0, len(logs)
    while low < high:
        mid = (low + high + 1) // 2
        if len(encode(render(mid)[0]).ids) <= length:
            low = mid
        else:
            high = mid - 1
    for count in range(low, max(low - 4, -1), -1):
        for width in range(len(NOTES) + 1):
            notes = "\nIndex labels: " + " ".join(NOTES[:width]) if width else ""
            prompt, offset = render(count, notes)
            encoded = encode(prompt)
            if len(encoded.ids) == length:
                token_offset = next(i for i, (_, end) in enumerate(encoded.offsets) if end > offset)
                return {"label": f"{task.label}-{length}", "task": task.label, "position": position,
                        "fact_start_token": token_offset, "prompt": prompt, "expected_ids": encoded.ids,
                        "max_tokens": LIMIT, "answer": task.answer}
    raise ValueError(f"could not fit {task.label} at {length} tokens")


def prepare(tokenizer_path, output):
    from tokenizers import Tokenizer

    if sha256(tokenizer_path.read_bytes()).hexdigest() != HOLDOUT_TOKENIZER_SHA256:
        raise ValueError("long-answer check requires the pinned Qwen tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    logs = [record(i) for i in range(512)]
    cases = [pack(encode, task, position, length, logs)
             for task, position in zip(TASKS, POSITIONS, strict=True) for length in LENGTHS]
    data = {"tokenizer_sha256": HOLDOUT_TOKENIZER_SHA256, "tokenizers_version": version("tokenizers"), "cases": cases}
    with output.open("xb") as stream:
        stream.write(gzip.compress((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode(), mtime=0))


def load_fixture(path, expected_digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_digest:
        raise ValueError("long-answer fixture hash differs")
    data = json.loads(gzip.decompress(raw))
    if data["tokenizer_sha256"] != HOLDOUT_TOKENIZER_SHA256:
        raise ValueError("long-answer tokenizer binding differs")
    expected = [(task, position, length)
                for task, position in zip(TASKS, POSITIONS, strict=True) for length in LENGTHS]
    for case, (task, position, length) in zip(data["cases"], expected, strict=True):
        ids = token_ids(case["expected_ids"])
        if (case["label"] != f"{task.label}-{length}" or case["task"] != task.label or case["position"] != position
                or case["answer"] != task.answer or case["max_tokens"] != LIMIT or len(ids) != length
                or length + LIMIT > CONTEXT or task.facts not in case["prompt"] or task.question not in case["prompt"]
                or not 0 <= case["fact_start_token"] < length):
            raise ValueError(f"unexpected long-answer fixture case {task.label}/{length}")
    return data["cases"]


def capture(cases, digest, address, model, mode, output):
    def request(case):
        prompt = case["prompt"] if mode == "riley" else case["expected_ids"]
        result = complete(address, model, prompt, expected=case["expected_ids"], limit=LIMIT)
        return {"label": case["label"], **result}

    modes = {}
    for concurrency in (1, 8):
        with ThreadPoolExecutor(concurrency) as pool:
            modes[f"c{concurrency}"] = list(pool.map(request, cases))
    save(output, {"fixture_sha256": digest, "runtime": mode, "modes": modes})


def score(case, response):
    ids = token_ids(response["tokens"])
    reason, text = response["finish"], response["text"]
    if (response["label"] != case["label"] or response["prompt_tokens"] != case["expected_ids"]
            or reason not in ("stop", "length") or not isinstance(text, str)
            or len(ids) > LIMIT or (reason == "length" and len(ids) != LIMIT)
            or response["usage"]["prompt_tokens"] != len(case["expected_ids"])
            or response["usage"]["completion_tokens"] not in (len(ids), len(ids) + int(reason == "stop"))):
        raise ValueError(f"response differs from the request protocol for {case['label']}")
    matches = text.strip() == case["answer"]
    return {"exact_answer": matches, "complete_exact_answer": matches and reason == "stop", "finish": reason}


def verify(cases, digest, riley_path, vllm_path):
    result = {"fixture_sha256": digest, "runtimes": {}, "cases": {}}
    for runtime, path in (("riley", riley_path), ("vllm", vllm_path)):
        raw = path.read_bytes()
        source = json.loads(raw)
        if (source["runtime"] != runtime or source["fixture_sha256"] != digest
                or source["modes"].keys() != {"c1", "c8"}):
            raise ValueError(f"{runtime} capture is not bound to the same fixture and modes")
        c1, c8 = (source["modes"][mode] for mode in ("c1", "c8"))
        counts = {mode: dict.fromkeys(("exact_answer", "complete_exact_answer", "length"), 0)
                  for mode in ("c1", "c8")}
        preserved = 0
        for case, isolated, concurrent in zip(cases, c1, c8, strict=True):
            mode_results = {}
            for mode, response in (("c1", isolated), ("c8", concurrent)):
                outcome = score(case, response)
                for metric in ("exact_answer", "complete_exact_answer"):
                    counts[mode][metric] += int(outcome[metric])
                counts[mode]["length"] += int(outcome["finish"] == "length")
                mode_results[mode] = outcome
            invariant = same_completion(isolated, concurrent) and isolated["usage"] == concurrent["usage"]
            preserved += int(invariant)
            result["cases"].setdefault(case["label"], {})[runtime] = {**mode_results, "batch_invariant": invariant}
        result["runtimes"][runtime] = {"capture_sha256": sha256(raw).hexdigest(),
                                      "requests_per_mode": len(cases), "batch_invariant": preserved, **counts}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("tokenizer", type=Path)
    prep.add_argument("output", type=Path)
    run = commands.add_parser("capture")
    run.add_argument("runtime", choices=("riley", "vllm"))
    run.add_argument("address")
    run.add_argument("model")
    run.add_argument("output", type=Path)
    review = commands.add_parser("verify")
    review.add_argument("riley", type=Path)
    review.add_argument("vllm", type=Path)
    for command in (run, review):
        command.add_argument("fixture", type=Path)
        command.add_argument("fixture_sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.tokenizer, args.output)
    else:
        cases = load_fixture(args.fixture, args.fixture_sha256)
        if args.command == "capture":
            capture(cases, args.fixture_sha256, args.address, args.model, args.runtime, args.output)
        else:
            print(json.dumps(verify(cases, args.fixture_sha256, args.riley, args.vllm), indent=2))


if __name__ == "__main__":
    main()
