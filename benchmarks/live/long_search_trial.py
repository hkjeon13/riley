"""Matched long-context free-text lookup with source/amount grading."""

import argparse
import json
import random
import re
import time
from pathlib import Path

from examples.long_search import Passage, retrieve, search_request

from .agent_workflow import fingerprint, stream, tool, turn_request

SIZES = (192, 480, 720)
SEEDS = (31653, 42701, 47593, 50023, 52567, 55609, 58391, 60943)
TITLES = ("Alder Quartz", "Beacon Saffron", "Delta Marigold", "Echo Cypress")
STAGES = ("initial approval", "final revision")
MONEY = r"\d[\d,]*"
SOURCE = r"r[0-9a-f]{4}-[0-9a-f]{4}"


def test_case(size, index, seed):
    rng = random.Random(seed + size)
    title = TITLES[index % len(TITLES)]
    ids = [f"r{seed:04x}-{i:04x}" for i in range(size)]
    amounts = rng.sample(range(10000, 99999), size)
    early = rng.randrange(size // 10, size * 3 // 10)
    late = rng.randrange(size * 7 // 10, size * 9 // 10)
    pivot = rng.randrange(size * 4 // 10, size * 6 // 10)
    positions = ((early, late), (pivot, early), (late, pivot), (late, early))
    chosen = positions[index % len(positions)]
    findings = dict(zip(chosen, STAGES, strict=True))
    regions = ("harbor", "valley", "ridge", "plateau", "coastal", "inland")
    topics = ("culvert", "pipeline", "signal", "drainage", "electrical", "warehouse")
    decoys = set(rng.sample([i for i in range(size) if i not in findings], 6))
    passages = []
    for i, (source, amount) in enumerate(zip(ids, amounts, strict=True)):
        region = rng.choice(regions)
        topic = rng.choice(topics)
        stage = rng.choice(STAGES)
        if i in findings:
            text = f"The {title} project {findings[i]} confirmed a funding amount of {amount} dollars."
        elif i in decoys:
            text = f"The {title} project recorded a separate {topic} maintenance estimate of {amount} dollars."
        else:
            text = f"The {region} {topic} project {stage} confirmed a funding amount of {amount} dollars."
        passages.append(Passage(source, text))

    if index % 4 == 0:
        ask = f"How much funding did the initial approval for {title} establish?"
        targets = (chosen[0],)
    elif index % 4 == 1:
        ask = f"What amount was approved for {title} after the final revision?"
        targets = (chosen[1],)
    elif index % 4 == 2:
        ask = f"For {title}, what were the amounts at initial approval and at final revision?"
        targets = chosen
    else:
        ask = f"How much was approved for {title} at the outset, and how much did the final revision establish?"
        targets = chosen
    return ask, passages, {ids[i]: amounts[i] for i in targets}


def corpus_cases():
    for size in SIZES:
        for index, seed in enumerate(SEEDS):
            yield f"search-{size}-{seed}", *test_case(size, index, seed)


def grade(answer, expected):
    citation = re.compile(rf"\[({SOURCE})\]", re.IGNORECASE)
    cited = [value.casefold() for value in citation.findall(answer)]
    values = {
        int(v.replace(",", "")) for v in re.findall(MONEY, citation.sub("", answer))
    }
    pairs = {
        (source.casefold(), int(value.replace(",", "")))
        for value, source in re.findall(
            rf"(?<![\w,])({MONEY})\s*(?:dollars?\s*)?\[({SOURCE})\]",
            answer,
            re.IGNORECASE,
        )
    }
    return {
        "missing_values": sorted(set(expected.values()) - values),
        "missing_sources": sorted(set(expected) - set(cited)),
        "unexpected_sources": sorted(set(cited) - set(expected)),
        "duplicate_sources": len(cited) - len(set(cited)),
        "missing_pairs": sorted(set(expected.items()) - pairs),
    }


def sample(address, request, start):
    measurement = {}
    answer, calls = stream(address, request, measurement, timeout_s=120)
    if calls or measurement["status"] != 200 or not measurement["content_ns"]:
        raise ValueError("expected a visible answer stream")
    if int(
        measurement["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    ):
        raise ValueError("expected an uncached request")
    return {
        "body_sha256": fingerprint(request),
        "answer": answer,
        "first_ms": (measurement["content_ns"][0] - start) / 1e6,
        "complete_ms": (measurement["done_ns"] - start) / 1e6,
        "last_content_ms": (measurement["content_ns"][-1] - start) / 1e6,
        "measurement": measurement,
    }


def measure(address, question, corpus, expected, technique):
    start = time.monotonic_ns()
    evidence = retrieve(question, corpus) if technique == "ranked" else corpus
    prepared = time.monotonic_ns()
    request = search_request("m", question, evidence, token_witness=True)
    row = sample(address, request, start)
    problems = grade(row["answer"], expected)
    return {
        **row,
        "expected": expected,
        "problems": problems,
        "correct": not any(problems.values())
        and row["measurement"]["finish"] == "stop",
        "question": question,
        "corpus_sha256": fingerprint([(p.source, p.text) for p in corpus]),
        "evidence_count": len(evidence),
        "target_coverage": sorted(set(expected) & {p.source for p in evidence}),
        "prepare_ms": (prepared - start) / 1e6,
        **(
            {"evidence_sources": [p.source for p in evidence]}
            if technique == "ranked"
            else {}
        ),
    }


def diagnostics(address):
    results = {}
    for size in (24000, 0):
        content = (
            "The catalog value of ALPHA is 736251. Background records follow. "
            + " river" * size
            + " End of background. The catalog value of ZULU is 914603. "
            "Return only the two catalog values, ALPHA first and ZULU second."
        )
        request = search_request("m", "", (), token_witness=True)
        request["messages"][0]["content"] = content
        request["max_tokens"] = 40
        row = sample(address, request, time.monotonic_ns())
        row["correct"] = re.findall(MONEY, row["answer"]) == ["736251", "914603"]
        results[f"river-{size}"] = row

    seed, size = 862, 890
    rng = random.Random(seed)
    records = [
        {
            "source": f"F{seed}-{i:04d}",
            "region": rng.choice(("north", "south", "west", "east", "central")),
            "service": rng.choice(("rail", "water", "power", "road", "port")),
            "approved": rng.randrange(10000, 90000),
        }
        for i in range(size)
    ]
    targets = [records[size // 5], records[size * 4 // 5]]
    expected = {r["source"]: r["approved"] for r in targets}
    first, second = expected
    ask = (
        f"Report the approved amounts for the two source IDs {first} and {second}. "
        "Return only a JSON object mapping each requested source ID to its integer amount."
    )
    for name, evidence in (("whole", records), ("gold", targets)):
        request = turn_request(
            "m",
            [
                {"role": "user", "content": ask},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_register",
                            "type": "function",
                            "function": {"name": "read_register", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_register",
                    "content": json.dumps(evidence, separators=(",", ":")),
                },
            ],
            [tool("read_register", "Return the complete register.", {})],
            None,
        )
        request.update(max_tokens=96, return_token_ids=True)
        row = sample(address, request, time.monotonic_ns())
        try:
            entries = json.loads(
                row["answer"]
                .strip()
                .removeprefix("```json")
                .removesuffix("```")
                .strip(),
                object_pairs_hook=list,
            )
            row["correct"] = (
                isinstance(entries, list)
                and len(entries) == len(expected)
                and dict(entries) == expected
            )
        except (TypeError, ValueError):
            row["correct"] = False
        row["expected"] = expected
        results[f"record-{name}"] = row
    return results


def run(address, out, *, reverse):
    sample(
        address,
        search_request(
            "m",
            "Which amount is in the report?",
            [Passage("r0000-0000", "The amount is 50 dollars.")],
        ),
        time.monotonic_ns(),
    )
    results = {"search": {}, "diagnostics": diagnostics(address)}
    out.write_text(json.dumps(results) + "\n")
    cases = list(corpus_cases())
    for index, (name, question, corpus, expected) in enumerate(
        reversed(cases) if reverse else cases
    ):
        techniques = (
            ("whole", "ranked") if (index + reverse) % 2 == 0 else ("ranked", "whole")
        )
        cell = results["search"][name] = {}
        for technique in techniques:
            cell[technique] = measure(address, question, corpus, expected, technique)
        out.write_text(json.dumps(results) + "\n")
        print(
            name,
            *(
                f"{mode}={row['correct']}/{row['complete_ms']:.0f}ms"
                for mode, row in cell.items()
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("output", type=Path)
    p.add_argument("--reverse", action="store_true")
    args = p.parse_args()
    run(args.address, args.output, reverse=args.reverse)
