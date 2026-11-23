"""Replay the existing greedy path and rescore its tied BF16 candidates."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

TARGETS = {
    ("long", "0"): (681, 2), ("long", "6"): (667, 14),
    ("long", "12"): (669, 9), ("long", "14"): (2196, 9),
    ("holdout", "numbers-1023"): (1023, 67),
    ("holdout", "korean-1024"): (1024, 77),
}


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(root):
    path = root / "crates/riley-server/src/engine.rs"
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == "4f91f1acdb0d0b56448926dec58dcf40dceeaaf50819edfe68dd3306c34de378",
            "parent capture source changed")
    old = "        matches!((prompt_tokens, offset), (669, 9) | (1023, 67))"
    source = raw.decode()
    require(source.count(old) == 1, "unexpected trace selector")
    matchers = " | ".join(f"({length}, {offset})" for length, offset in TARGETS.values())
    path.write_text(source.replace(old, f"        matches!((prompt_tokens, offset), {matchers})", 1))


def capture():
    import capture_final_rows as helper

    helper.TARGETS = TARGETS
    parser = ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("rows", type=Path)
    parser.add_argument("server_log", type=Path)
    args = parser.parse_args()
    print(json.dumps(helper.capture_riley(args.evidence, args.address, args.model, args.rows, args.server_log),
                     indent=2, allow_nan=False))


def verify(stage, previous, model):
    import capture_final_rows as helper
    from replay_final_head import bf16, round_bf16

    checkpoint = helper.checkpoint(model)
    require(checkpoint["rms_norm_eps"] > 0, "invalid checkpoint configuration")
    before, after = (load(stage / f"gpu7-{phase}.json") for phase in ("before", "after"))
    require(all(r["phase"] == "idle" and r["status"] == "running" and r["physical_index"] == 7
                and r["gpu"] == helper.GPU and r["allocation_bytes"] == 20 * 2**30
                and not r["other_processes"] for r in (before, after))
            and all(before[key] == after[key] for key in ("pid", "start_ticks")),
            "the original A100 reservation was not restored")
    original = load(previous / "riley/client.log")["cases"]
    decoded = {(row["suite"], str(row["case"])): row for row in original}
    require(len(original) == len(decoded) == 8 and TARGETS.keys() <= decoded.keys(), "wrong source corpus")
    new = load(stage / "riley/client.log")
    require(set(new) == {f"{p}-{i}" for p, i in TARGETS.values()}, "incomplete GPU capture")
    receipt = load(stage / "riley/receipt.json")
    require(receipt["status"] == "passed" and receipt.get("returncode", 0) == 0, "Riley did not exit cleanly")
    entries = []
    for key, (prompt_len, offset) in TARGETS.items():
        name = f"{prompt_len}-{offset}"
        row = decoded[key]
        require((row["prompt_len"], row["offset"]) == (prompt_len, offset), f"{key}: capture position changed")
        current = new[name]
        require(row["row"] == current["row"] and row["observed_output_hash"] == current["response_token_sha256"],
                f"{key}: greedy outputs or original BF16 logits changed")
        logits = (stage/"riley/rows"/f"{name}.bf16").read_bytes()
        raw = (stage/"riley/rows"/f"{name}.hidden.bf16").read_bytes()
        require(len(raw) == 4 * helper.HIDDEN and sha256(raw).hexdigest() == current["hidden_sha256"]
                and sha256(logits).hexdigest() == row["row"]["sha256"], f"{key}: capture checksum differs")
        norm = raw[helper.HIDDEN * 2:]
        norm_values = bf16(norm)
        tied = row["row"]["tied_maximum_ids"]
        riley, ref = row["riley_id"], row["hf_id"]
        _, weights = helper.weight_rows(model, sorted({*tied, riley, ref}))
        scores = {token: math.fsum(a*b for a, b in zip(norm_values, bf16(weight), strict=True))
                  for token, weight in weights.items()}
        observed = {token: bf16(logits[2*token:2*token+2])[0] for token in weights}
        reproduced = all(round_bf16(value) == observed[token] for token, value in scores.items())
        chosen = min(tied, key=lambda token: (-scores[token], token))
        entries.append({"case": key[1], "suite": key[0], "position": offset, "prompt_tokens": prompt_len,
                        "original_riley_id": riley, "hf_sdpa_id": ref, "bf16_maximizers": tied,
                        "fp64_choice_among_bf16_maximizers": chosen, "rounded_scores_match": reproduced,
                        "observed_bf16": observed, "fp64_scores": scores,
                        "raw_margin_hf_minus_riley": scores[ref] - scores[riley],
                        "final_norm_sha256": sha256(norm).hexdigest(), "logits_sha256": row["row"]["sha256"],
                        "weight_sha256": {token: sha256(raw).hexdigest() for token, raw in weights.items()}})
    prefill = load(stage/"prefill-reference.json")
    require(prefill["checkpoint_manifest_sha256"] == helper.CHECKPOINT, "prefill checkpoint changed")
    for case in ("3", "9"):
        prior = decoded["long", case]
        evidence = prefill["cases"][case]["engines"]["riley"]
        ties = prior["row"]["tied_maximum_ids"]
        require(ties == evidence["observed_argmax_ties"]
                and set(map(str, ties)) == set(evidence["fp64_dot_products"])
                and evidence["rounded_bf16_matches_observed_pair"]
                and evidence["observed_pair"][str(prior["riley_id"])] == prior["row"]["riley_logit"]
                and evidence["selected_id"] == prior["riley_id"], "prefill comparison differs from archived evidence")
        chosen = min(ties, key=lambda token: (-evidence["fp64_dot_products"][str(token)], token))
        entries.append({"case": case, "suite": "long", "position": 0, "prompt_tokens": prior["prompt_len"],
                        "original_riley_id": prior["riley_id"], "hf_sdpa_id": prior["hf_id"],
                        "bf16_maximizers": ties, "fp64_choice_among_bf16_maximizers": chosen,
                        "rounded_scores_match": True, "observed_bf16": evidence["observed_pair"],
                        "fp64_scores": evidence["fp64_dot_products"], "final_norm_sha256": evidence["final_norm_sha256"],
                        "historical_prefill": True})
    return {"status": "verified", "checkpoint_sha256": helper.CHECKPOINT, "decode_capture_count": len(new),
            "prefill_reference_count": 2, "prior_bf16_rows_sha256": sha256((previous/"riley/client.log").read_bytes()).hexdigest(),
            "prefill_reference_sha256": sha256((stage/"prefill-reference.json").read_bytes()).hexdigest(),
            "rows": entries, "reservation": {key: after[key] for key in ("pid","start_ticks","allocation_bytes")}}


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "prepare":
        prepare(Path(sys.argv[1]))
    elif command == "capture":
        capture()
    elif command == "verify":
        print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
    else:
        raise ValueError(command)
