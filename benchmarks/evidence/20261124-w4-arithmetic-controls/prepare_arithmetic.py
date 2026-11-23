"""Render matched controls for the Korean arithmetic failures."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import sys

from chat_answers import CONFIG_SHA, CONTEXT, FIXTURE_SHA, HOLDOUT_TOKENIZER_SHA256, LIMIT, load_fixture
from long_answer_check import TASKS

TASK = next(task for task in TASKS if task.label == "arithmetic-ko")
SEPARATOR = "\n\nQuestion: "
START = "Read the record and answer the question.\nRecords:\n"
EXTRACT = "설악 선반의 처음 유리병 수, 한 쟁반에 담긴 유리병 수, 옮긴 쟁반 수를 쉼표로 구분해 이 순서대로 쓰세요."
EXPRESSION = "318 - (27 × 4) 결과를 구하세요. 다른 문자 없이 정수 하나만 쓰세요."


def digest(raw):
    return sha256(raw).hexdigest()


def inputs(fixture):
    arithmetic = [case for case in load_fixture(fixture, FIXTURE_SHA) if case["task"] == TASK.label]
    yield "fact-only", None, START + TASK.facts + SEPARATOR + TASK.question, TASK.answer
    yield "expression-only", None, EXPRESSION, TASK.answer
    for case in arithmetic:
        stem, question = case["prompt"].removesuffix("\nAnswer:").split(SEPARATOR)
        if question != TASK.question or not stem.startswith(START):
            raise ValueError("unexpected original arithmetic fixture")
        lines = stem.removeprefix(START).split("\n")
        if lines.count(TASK.facts) != 1 or lines.index(TASK.facts) in (0, len(lines) - 1):
            raise ValueError("expected the fact in the middle of the original records")
        lines.remove(TASK.facts)
        reordered = "\n".join([*lines, TASK.facts])
        yield "facts-last", case["label"], START + reordered + SEPARATOR + question, TASK.answer
        yield "extract-operands", case["label"], stem + SEPARATOR + EXTRACT, "318,27,4"


def prepare(model, fixture, output):
    from transformers import AutoTokenizer

    assert digest((model / "tokenizer.json").read_bytes()) == HOLDOUT_TOKENIZER_SHA256
    assert digest((model / "tokenizer_config.json").read_bytes()) == CONFIG_SHA
    config = json.loads((model / "tokenizer_config.json").read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    assert tokenizer.chat_template == config["chat_template"]
    controls = []
    for mode, original, text, expected in inputs(fixture):
        chat = [{"role": "user", "content": text}]
        ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=False)
        assert 0 < len(ids) <= CONTEXT - LIMIT
        controls.append({"control": mode, "baseline_label": original, "messages": chat, "answer": expected, "expected_ids": ids})
    assert len(controls) == 24
    data = {"fixture_sha256": FIXTURE_SHA, "template_sha256": digest(tokenizer.chat_template.encode()),
            "transformers": version("transformers"), "controls": controls}
    with output.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"total": len(controls), "sha256": digest(output.read_bytes()),
                      "tokens": {mode: [len(row["expected_ids"]) for row in controls if row["control"] == mode]
                                 for mode in {row["control"] for row in controls}}}, ensure_ascii=False))


if __name__ == "__main__":
    prepare(*map(Path, sys.argv[1:]))
