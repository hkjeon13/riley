"""Check all published BF16 ties on token prefixes where Riley and HF agree."""

from argparse import ArgumentParser
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected(root):
    from long_accuracy import common_prefix, read_capture, read_workload, riley_margin, same_completion, terminal_normalized

    hashes, requests, lengths = {}, [], Counter()
    for suite in ("long", "holdout"):
        directory = root/f"20261119-w4-{suite}"
        capture_path = directory/"riley/capture.json.gz"
        reference_path = directory/"hf/reference.json"
        fixture = directory/"holdout.json.gz" if suite == "holdout" else None
        for path in filter(None, (capture_path, reference_path, fixture)):
            encoded = path.read_bytes()
            decoded = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
            hashes[path.relative_to(root).as_posix()] = sha256(decoded).hexdigest()
        workload = read_workload(fixture)
        riley = read_capture(capture_path, workload)
        reference = load(reference_path)
        require(reference.get("workload_sha256") == workload.sha256, f"{suite}: HF workload differs")
        eos = set(reference["eos_token_ids"])
        for case, original, diagnostic, hf in zip(
                workload.cases, riley["baseline"], riley["diagnostic"], reference["cases"], strict=True):
            require(same_completion(original, diagnostic) and hf["hf_tokenizer_ids_match"]
                    and hf["prompt_tokens"] == len(original["prompt_tokens"]), "Riley or HF input differs")
            left = terminal_normalized(original["tokens"], original["finish"], eos)
            right = terminal_normalized(hf["hf_tokens"], hf["hf_finish"], eos)
            shared = common_prefix(left, right)
            prompt_len = len(original["prompt_tokens"])
            lengths[prompt_len] += 1
            positions = [{"offset": i, "agreed_id": left[i]} for i in range(shared)
                         if riley_margin(original, diagnostic, i) == 0]
            if positions:
                requests.append((case, original, {"suite": suite, "case": str(case.label),
                                 "prompt_len": prompt_len, "full_generation_matches": left == right,
                                 "positions": positions}))
    require(all(lengths[row["prompt_len"]] == 1 for _, _, row in requests),
            "prompt-length capture selectors are ambiguous")
    return {"input_content_sha256": hashes, "requests": [row for _, _, row in requests]}, requests


def replace(source, previous, current):
    require(source.count(previous) == 1, f"source anchor differs: {previous}")
    return source.replace(previous, current, 1)


def prepare(source, plan):
    engine = source/"crates/riley-server/src/engine.rs"
    runtime = source/"crates/riley-runtime/src/llama/batch_executor.rs"
    require(digest(engine) == "4f91f1acdb0d0b56448926dec58dcf40dceeaaf50819edfe68dd3306c34de378"
            and digest(runtime) == "0d56d9d596f3c675e22200ce44b4594d01b570fdb0779c4d8c46c661203db546",
            "parent capture source changed")
    positions = [f"({row['prompt_len']}, {p['offset']})" for row in plan["requests"] for p in row["positions"]]
    src = engine.read_text()
    src = replace(src, "        matches!((prompt_tokens, offset), (669, 9) | (1023, 67))",
                  "        matches!((prompt_tokens, offset), " + " | ".join(positions) + ")")
    src = replace(src, "// The two decode divergences where the Riley logit was a unique maximum.",
                  "// Published ties on the Riley and HF shared prefixes.")
    src = replace(src, "|| !plan.prefill_items().is_empty()\n                            || plan.decode_items().len() != 1",
                  "|| plan.prefill_items().len() + plan.decode_items().len() != 1")
    src = replace(src, ".capture_w4_decode_final_rows(stream)",
                  ".capture_w4_decode_final_rows(item.input_tokens().len() - 1, stream)")
    engine.write_text(src)
    src = runtime.read_text()
    start = src.index("    /// Captures the hidden row before")
    end = src.index("    /// Downloads one vocabulary row", start)
    function = src[start:end]
    function = replace(function, "of a C1 decode.", "for a one-item prefill or decode step.")
    function = replace(function,
                       "    /// The diagnostic caller checks that the step contains one decode input and\n    /// no prefill.",
                       "    /// The diagnostic caller checks that the step contains one work item.")
    function = replace(function, "        stream: &mut CudaStream,", "        dense_row: usize,\n        stream: &mut CudaStream,")
    function = replace(function, "        let mut bytes = vec![0; row_bytes * 2];",
                       """        let offset = u64::try_from(dense_row.checked_mul(row_bytes).ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "dense_row",
                reason: "hidden row offset overflowed",
            },
        )?).map_err(|_| LlamaBatchExecutorError::InvalidConfiguration {
            field: "dense_row",
            reason: "hidden row offset does not fit u64",
        })?;
        let mut bytes = vec![0; row_bytes * 2];""")
    function = replace(function, "buffer.download_to_slice(0, destination", "buffer.download_to_slice(offset, destination")
    runtime.write_text(src[:start]+function+src[end:])


def capture():
    from capture_reference import MARKER, complete, same_completion, score_row

    parser = ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("rows", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    plan, requests = selected(args.fixtures)
    require(plan == load(args.rows.parent.parent/"targets.json"), "selected positions changed")
    result = []
    for case, original, row in requests:
        response = complete(args.address, args.model, case.prompt, diagnostic=True,
                            expected=original["prompt_tokens"], limit=case.limit)
        require(same_completion(original, response) and original["usage"] == response["usage"],
                f"{row['case']}: full response changed while recording logits")
        markers = MARKER.findall(args.log.read_text())
        for position in row["positions"]:
            offset = position["offset"]
            prompt_len = row["prompt_len"]
            name = f"{prompt_len}-{offset}"
            hits = [(int(token), int(vocab), int(addressable)) for prompt, index, token, vocab, addressable in markers
                    if (int(prompt), int(index)) == (prompt_len, offset)]
            require(len(hits) == 1 and hits[0][0] == response["tokens"][offset] == position["agreed_id"],
                    f"{name}: the device and served ID were not the expected token")
            _, vocabulary, addressable = hits[0]
            logits = (args.rows/f"{name}.bf16").read_bytes()
            result.append({**position, "prompt_len": prompt_len, "case": row["case"], "suite": row["suite"],
                           "full_generation_matches": row["full_generation_matches"],
                           "response_ids_sha256": sha256(json.dumps(response["tokens"]).encode()).hexdigest(),
                           "hidden_sha256": digest(args.rows/f"{name}.hidden.bf16"),
                           "bf16_row": score_row(logits, vocabulary, addressable, position["agreed_id"], position["agreed_id"])})
    print(json.dumps(result, indent=2, allow_nan=False))


def verify(stage, model):
    from capture_final_rows import GPU, HIDDEN, VOCAB, CHECKPOINT, checkpoint, weight_rows
    from replay_final_head import bf16, round_bf16

    checkpoint(model)
    plan, _ = selected(stage/"fixtures")
    require(plan == load(stage/"targets.json"), "CPU selection differs from pinned fixtures")
    before, after = (load(stage/f"gpu7-{point}.json") for point in ("before", "after"))
    require(all(row["phase"] == "idle" and row["status"] == "running" and row["physical_index"] == 7
                and row["gpu"] == GPU and row["allocation_bytes"] == 20 * 2**30 and not row["other_processes"]
                for row in (before, after)) and all(before[k] == after[k] for k in ("pid","start_ticks")),
            "A100 7 reservation did not recover")
    raw_rows = load(stage/"riley/client.log")
    expected = {(row["suite"], row["case"], place["offset"], place["agreed_id"])
                for row in plan["requests"] for place in row["positions"]}
    require(len(raw_rows) == len(expected)
            and {(r["suite"], r["case"], r["offset"], r["agreed_id"]) for r in raw_rows} == expected,
            "served captures are missing or duplicated")
    require(load(stage/"riley/receipt.json")["status"] == "passed", "server did not exit cleanly")
    token_ids = sorted({token for row in raw_rows for token in row["bf16_row"]["tied_maximum_ids"]})
    _, weights = weight_rows(model, token_ids)
    weights_fp64 = {token: bf16(raw) for token, raw in weights.items()}
    result = []
    for row in raw_rows:
        name = f"{row['prompt_len']}-{row['offset']}"
        directory = stage/"riley/rows"
        hidden_file, logits_file = directory/f"{name}.hidden.bf16", directory/f"{name}.bf16"
        require(digest(hidden_file) == row["hidden_sha256"] and digest(logits_file) == row["bf16_row"]["sha256"],
                f"{name}: captured arrays changed")
        hidden, logits = hidden_file.read_bytes(), logits_file.read_bytes()
        require(len(hidden) == 4 * HIDDEN and len(logits) == 2 * VOCAB, f"{name}: capture geometry differs")
        norm = hidden[2 * HIDDEN:]
        norm_fp64 = bf16(norm)
        maxima = row["bf16_row"]["tied_maximum_ids"]
        scores = {token: math.fsum(a*b for a,b in zip(norm_fp64, weights_fp64[token], strict=True)) for token in maxima}
        observed = {token: bf16(logits[2*token:2*token+2])[0] for token in maxima}
        chosen = min(maxima, key=lambda token: (-scores[token], token))
        result.append({**row, "final_norm_sha256": sha256(norm).hexdigest(),
                       "fp64_choice": chosen, "original_choice_changes": chosen != row["agreed_id"],
                       "rounded_scores_match": all(round_bf16(scores[token]) == observed[token] for token in maxima),
                       "fp64_scores": scores})
    cases_changed = {(row["suite"], row["case"]) for row in result if row["original_choice_changes"]}
    full_cases_changed = {(row["suite"], row["case"]) for row in result
                          if row["original_choice_changes"] and row["full_generation_matches"]}
    return {"status": "verified", "checkpoint_sha256": CHECKPOINT, "target_manifest_sha256": digest(stage/"targets.json"),
            "selected_positions": len(result), "capture_requests": len(plan["requests"]),
            "prefill_positions": sum(row["offset"] == 0 for row in result),
            "bf16_reproduced_positions": sum(row["rounded_scores_match"] for row in result),
            "changed_positions": sum(row["original_choice_changes"] for row in result),
            "affected_requests": len(cases_changed), "affected_exact_generation_requests": len(full_cases_changed),
            "weights_sha256": {token: sha256(raw).hexdigest() for token, raw in weights.items()},
            "reservation": {key: after[key] for key in ("pid","start_ticks","allocation_bytes")}, "rows": result}


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "select":
        print(json.dumps(selected(Path(sys.argv[1]))[0], indent=2, allow_nan=False))
    elif action == "prepare":
        prepare(Path(sys.argv[1]), load(Path(sys.argv[2])))
    elif action == "verify":
        print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
    else:
        raise ValueError(action)
