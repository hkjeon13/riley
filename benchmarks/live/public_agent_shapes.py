"""Replay public token-length budgets with synthetic content and capture observed output lengths."""

import argparse
import json
import subprocess
from pathlib import Path

from agent_workflow import fingerprint, stream


def body(idx, words, limit):
    text = (
        f"Study record {idx}. Write a detailed account of the work of a team of scientists. "
        "Use the following observation list as background." + " river" * words
    )
    return {
        "model": "m",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": limit,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def build(source, binary, tokenizer, *, llama3=False):
    sets = source["timing_replay_sample"] + source["extended_context_sample"]
    labels = ["within_8k"] * len(source["timing_replay_sample"]) + ["extended"] * len(
        source["extended_context_sample"]
    )
    if llama3:
        sets = source["long_context_sample"]
        labels = ["long_context"] * len(sets)
    offsets = [0 for _ in sets]

    def lengths():
        raw = "".join(
            json.dumps(body(i, offsets[i], max(1, row["next_completion_tokens"])))
            + "\n"
            for i, row in enumerate(sets)
        )
        r = subprocess.run(
            [binary, tokenizer, *(["--llama3"] if llama3 else [])],
            input=raw,
            text=True,
            capture_output=True,
            check=True,
        )
        return [len(json.loads(line)["ids"]) for line in r.stdout.splitlines()]

    for i, (row, n) in enumerate(zip(sets, lengths(), strict=True)):
        offsets[i] = max(0, row["next_prompt_tokens"] - n)
    actual = lengths()
    prepared = []
    for i, (row, n) in enumerate(zip(sets, actual, strict=True)):
        limit = row["next_completion_tokens"]
        matched = n == row["next_prompt_tokens"] and limit > 0
        prepared.append(
            {
                **row,
                "index": i,
                "repetitions": offsets[i],
                "encoded_prompt_tokens": n,
                "exact_input_and_positive_budget": matched,
                "request_sha256": fingerprint(body(i, offsets[i], max(1, limit))),
                "set": labels[i],
            }
        )
    return {"source_sha256": source["archive_sha256"], "rows": prepared}


def run(address, config, out):
    result = []
    for r in config["rows"]:
        data = {**r, "attempted": False}
        if r["exact_input_and_positive_budget"]:
            req = body(r["index"], r["repetitions"], r["next_completion_tokens"])
            if fingerprint(req) != r["request_sha256"]:
                raise ValueError("prepared request changed")
            data["attempted"] = True
            measurement = {}
            try:
                answer, calls = stream(
                    address,
                    req,
                    measurement,
                    timeout_s=900 if r["set"] == "long_context" else 90,
                )
                data["answer_sha256"] = fingerprint(answer)
                data["usage"] = measurement["usage"]
                data["finish"] = measurement["finish"]
                data["prompt_exact"] = (
                    measurement["usage"]["prompt_tokens"] == r["next_prompt_tokens"]
                )
                data["output_exact"] = (
                    measurement["usage"]["completion_tokens"]
                    == r["next_completion_tokens"]
                )
                data["complete"] = not calls and bool(measurement.get("done_ns"))
                content = measurement["content_ns"]
                if content:
                    data["first_content_ms"] = (
                        content[0] - measurement["sent_ns"]
                    ) / 1e6
                    data["last_content_ms"] = (
                        content[-1] - measurement["sent_ns"]
                    ) / 1e6
            except (ValueError, OSError) as error:
                data["error"] = str(error)
            data["http_status"] = measurement.get("status")
        result.append(data)
        out.write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    build_args = sub.add_parser("prepare")
    build_args.add_argument("source", type=Path)
    build_args.add_argument("binary", type=Path)
    build_args.add_argument("tokenizer", type=Path)
    build_args.add_argument("--llama3", action="store_true")
    run_args = sub.add_parser("run")
    run_args.add_argument("address")
    run_args.add_argument("manifest", type=Path)
    run_args.add_argument("out", type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                build(
                    json.loads(args.source.read_text()),
                    args.binary,
                    args.tokenizer,
                    llama3=args.llama3,
                ),
                indent=2,
            )
        )
    else:
        run(args.address, json.loads(args.manifest.read_text()), args.out)


if __name__ == "__main__":
    main()
