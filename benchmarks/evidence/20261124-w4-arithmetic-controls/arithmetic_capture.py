"""Run the fixed Korean arithmetic controls through each engine's chat API."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from chat_answers import FIXTURE_SHA, load_fixture, run_chat, same_completion, save, token_ids
from prepare_arithmetic import inputs


INPUTS_SHA = "a7b2dbca08cdbc50477665cd930d4e21c2b8f0bd0f32df4e904599c18a2557de"
TEMPLATE_SHA = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"


def workload(stage):
    fixture = stage / "facts.json.gz"
    load_fixture(fixture, FIXTURE_SHA)
    data = (stage / "inputs.json").read_bytes()
    if sha256(data).hexdigest() != INPUTS_SHA:
        raise ValueError("pinned arithmetic control inputs differ")
    document = json.loads(data)
    if document["fixture_sha256"] != FIXTURE_SHA or document["template_sha256"] != TEMPLATE_SHA:
        raise ValueError("arithmetic controls differ in source cases or tokenizer template")
    cases = []
    for row, (name, original, text, answer) in zip(document["controls"], inputs(fixture), strict=True):
        ids = token_ids(row["expected_ids"])
        if (row["control"] != name or row["baseline_label"] != original or row["answer"] != answer
                or row["messages"] != [{"role": "user", "content": text}] or not ids or len(ids) > 8160):
            raise ValueError("unexpected arithmetic control")
        label = name if original is None else f"{name}/{original}"
        cases.append({"label": label, "control": name, "prompt": text + "\nAnswer:", "answer": answer, "expected_ids": ids})
    return cases


def capture(cases, engine, endpoint, model, output):
    def request(case):
        return run_chat(endpoint, model, case, "without_answer_cue")

    modes = {}
    for parallel in (1, 8):
        with ThreadPoolExecutor(parallel) as executor:
            modes[f"c{parallel}"] = list(executor.map(request, cases))
        print(f"{engine} c{parallel}: {len(cases)}", flush=True)
    save(output, {"inputs_sha256": INPUTS_SHA, "runtime": engine, "modes": modes})


def report(stage):
    cases = workload(stage)
    results = {"inputs_sha256": INPUTS_SHA, "engines": {}}
    for engine in ("riley", "vllm"):
        raw = (stage / engine / "capture.json").read_bytes()
        data = json.loads(raw)
        if data["runtime"] != engine or data["inputs_sha256"] != INPUTS_SHA or data["modes"].keys() != {"c1", "c8"}:
            raise ValueError("invalid engine capture")
        output = {"capture_sha256": sha256(raw).hexdigest(), "cases": {}, "counts": {}}
        for case, c1, c8 in zip(cases, data["modes"]["c1"], data["modes"]["c8"], strict=True):
            row_report = {"batch_invariant": same_completion(c1, c8) and c1["usage"] == c8["usage"]}
            for mode, row in (("c1", c1), ("c8", c8)):
                if row["label"] != case["label"] or row["prompt_tokens"] != case["expected_ids"]:
                    raise ValueError("captured prompt ID sequence differs")
                answer = ",".join(part.strip() for part in row["text"].strip().split(","))
                correct = answer == case["answer"] and row["finish"] == "stop"
                row_report[mode] = {"output": row["text"], "finish": row["finish"], "exact_and_stop": correct}
                totals = output["counts"].setdefault(case["control"], {}).setdefault(mode, {"requests": 0, "exact_and_stop": 0})
                totals["requests"] += 1
                totals["exact_and_stop"] += int(correct)
            output["cases"][case["label"]] = row_report
        results["engines"][engine] = output
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("capture")
    run.add_argument("engine", choices=("riley", "vllm"))
    run.add_argument("endpoint")
    run.add_argument("model")
    run.add_argument("output", type=Path)
    run.add_argument("fixture", type=Path)
    run.add_argument("fixture_sha256")
    commands.add_parser("report").add_argument("stage", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        if args.fixture_sha256 != FIXTURE_SHA:
            raise ValueError("unexpected source fixture")
        capture(workload(args.fixture.parent), args.engine, args.endpoint, args.model, args.output)
    else:
        print(json.dumps(report(args.stage), ensure_ascii=False, indent=2))
