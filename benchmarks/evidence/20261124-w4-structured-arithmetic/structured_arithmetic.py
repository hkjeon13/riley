"""Compare direct arithmetic and schema-constrained operand extraction on held-out inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from chat_answers import CONFIG_SHA, FIXTURE_SHA, HOLDOUT_TOKENIZER_SHA256, LIMIT, run_chat, same_completion, save
from long_answer_check import load_fixture, score
from prepare_arithmetic import SEPARATOR, TASK


LENGTHS = (255, 257, 512, 513, 1024, 1025, 7836, 7917)
OPERANDS = ((287, 19, 6), (451, 38, 7), (627, 22, 5), (394, 56, 4),
            (548, 37, 6), (639, 24, 3), (728, 53, 6), (915, 46, 7))
FIELDS = ("initial", "per_tray", "trays")
EXTRACT = ("설악 선반의 처음 유리병 수는 initial, 한 쟁반의 유리병 수는 per_tray, "
           "옮긴 쟁반 수는 trays에 넣어서 JSON 객체만 출력하세요. 계산은 하지 마세요.")
TEMPLATE_SHA = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "operands", "schema": {
    "type": "object", "additionalProperties": False, "required": list(FIELDS),
    "properties": {field: {"type": "integer", "minimum": 1, "maximum": 999} for field in FIELDS},
}}}


def digest(raw):
    return sha256(raw).hexdigest()


def holdout(fixture):
    cases = {case["label"]: case for case in load_fixture(fixture, FIXTURE_SHA)}
    for length, (initial, per_tray, trays) in zip(LENGTHS, OPERANDS, strict=True):
        label = f"arithmetic-ko-{length}"
        raw = cases[label]["prompt"].removesuffix("\nAnswer:")
        stem, question = raw.split(SEPARATOR)
        if stem.count(TASK.facts) != 1 or question != TASK.question:
            raise ValueError(f"fact or question differs for {label}")
        changed = (f"설악 선반에는 유리병 {initial}개가 있었다. 팀이 유리병 {per_tray}개씩 든 쟁반 "
                   f"{trays}개를 옮겨 갔다.")
        stem = stem.replace(TASK.facts, changed)
        yield label, dict(zip(FIELDS, (initial, per_tray, trays), strict=True)), stem + SEPARATOR + question, stem + SEPARATOR + EXTRACT


def prepare(model, fixture, output):
    from transformers import AutoTokenizer

    if (digest((model / "tokenizer.json").read_bytes()) != HOLDOUT_TOKENIZER_SHA256
            or digest((model / "tokenizer_config.json").read_bytes()) != CONFIG_SHA):
        raise ValueError("tokenizer metadata differs")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if digest(tokenizer.chat_template.encode()) != TEMPLATE_SHA:
        raise ValueError("chat template differs")
    cases = []
    for label, operands, direct, structured in holdout(fixture):
        cases.append({"label": label, "operands": operands, "prompts": {name: {"text": text,
                      "ids": tokenizer.apply_chat_template([{"role": "user", "content": text}],
                           add_generation_prompt=True, tokenize=True, return_dict=False)}
                      for name, text in (("direct", direct), ("structured", structured))}})
    save(output, {"fixture_sha256": FIXTURE_SHA, "template_sha256": TEMPLATE_SHA, "cases": cases})
    workload(output.parent)
    print(json.dumps({"input_sha256": digest(output.read_bytes()),
                      "token_ranges": {mode: [min(sizes), max(sizes)] for mode in ("direct", "structured")
                                       if (sizes := [len(case["prompts"][mode]["ids"]) for case in cases])}}))


def workload(stage):
    raw = (stage / "inputs.json").read_bytes()
    data = json.loads(raw)
    if data["fixture_sha256"] != FIXTURE_SHA or data["template_sha256"] != TEMPLATE_SHA:
        raise ValueError("input fixture identity differs")
    for case, (label, operands, direct, structured) in zip(data["cases"], holdout(stage / "facts.json.gz"), strict=True):
        if case["label"] != label or case["operands"] != operands:
            raise ValueError("held-out operands differ")
        for mode, text in (("direct", direct), ("structured", structured)):
            row = case["prompts"][mode]
            if row["text"] != text or not 0 < len(row["ids"]) <= 8192 - LIMIT:
                raise ValueError("held-out prompt differs or exceeds supported context")
    return digest(raw), data["cases"]


def remaining(operands):
    return operands["initial"] - operands["per_tray"] * operands["trays"]


def chat_case(case, mode):
    prompt = case["prompts"][mode]
    return {"label": case["label"], "prompt": prompt["text"] + "\nAnswer:",
            "expected_ids": prompt["ids"], "answer": str(remaining(case["operands"]))}


def capture(stage, address, model, output):
    inputs_sha, cases = workload(stage)
    completed = {}
    for mode in ("direct", "structured"):
        def request(case):
            response_format = RESPONSE_FORMAT if mode == "structured" else None
            return run_chat(address, model, chat_case(case, mode), "without_answer_cue", response_format=response_format)

        completed[mode] = {}
        for parallel in (1, 8):
            with ThreadPoolExecutor(parallel) as executor:
                completed[mode][f"c{parallel}"] = list(executor.map(request, cases))
            print(f"{mode} c{parallel}: {len(cases)}", flush=True)
    save(output, {"inputs_sha256": inputs_sha, "modes": completed})


def parse_operands(text):
    def unique(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("duplicate operand field")
        return result

    try:
        data = json.loads(text, object_pairs_hook=unique)
    except ValueError:
        return None
    if (not isinstance(data, dict) or data.keys() != set(FIELDS)
            or any(type(value) is not int or not 1 <= value <= 999 for value in data.values())):
        return None
    return data


def report(stage):
    sha, cases = workload(stage)
    raw = (stage / "riley/capture.json").read_bytes()
    captures = json.loads(raw)
    if captures["inputs_sha256"] != sha or captures["modes"].keys() != {"direct", "structured"}:
        raise ValueError("capture does not match the held-out inputs")
    result = {"inputs_sha256": sha, "capture_sha256": digest(raw), "modes": {}}
    for mode, data in captures["modes"].items():
        c1, c8 = (data[key] for key in ("c1", "c8"))
        counts = {key: {"exact_and_stop": 0, "length": 0} if mode == "direct"
                  else {"valid_schema_and_stop": 0, "exact_operands": 0, "correct_after_integer_math": 0, "length": 0}
                  for key in data}
        rows = []
        for case, isolated, concurrent in zip(cases, c1, c8, strict=True):
            row = {"label": case["label"], "expected_operands": case["operands"], "expected_answer": remaining(case["operands"]),
                   "batch_invariant": same_completion(isolated, concurrent) and isolated["usage"] == concurrent["usage"]}
            for parallel, response in (("c1", isolated), ("c8", concurrent)):
                direct_score = score(chat_case(case, mode), response)
                outcome = {"text": response["text"], "finish": response["finish"]}
                total = counts[parallel]
                total["length"] += response["finish"] == "length"
                if mode == "direct":
                    total["exact_and_stop"] += direct_score["complete_exact_answer"]
                else:
                    operands = parse_operands(response["text"]) if response["finish"] == "stop" else None
                    answer = remaining(operands) if operands else None
                    total["valid_schema_and_stop"] += operands is not None
                    total["exact_operands"] += operands == case["operands"]
                    total["correct_after_integer_math"] += answer == row["expected_answer"]
                    outcome.update(operands=operands, calculated_answer=answer)
                row[parallel] = outcome
            rows.append(row)
        result["modes"][mode] = {"counts": counts, "rows": rows}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("model", "fixture", "output"):
        prep.add_argument(name, type=Path)
    run = commands.add_parser("capture")
    run.add_argument("engine", choices=("riley",))
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
            raise ValueError("source fixture differs")
        capture(args.fixture.parent, args.address, args.model, args.output)
    else:
        print(json.dumps(report(args.stage), indent=2, ensure_ascii=False, allow_nan=False))
