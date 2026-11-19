"""Measure the exact BF16 encoding distance between paired Riley and HF scores."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
import struct


def bf16_bits(score):
    encoded = struct.pack(">f", score)
    bits, = struct.unpack(">I", encoded)
    round_trip, = struct.unpack(">f", encoded)
    if not 0 < score < math.inf or bits & 0xffff or round_trip != score:
        raise ValueError(f"expected an exactly representable positive BF16 score: {score}")
    return bits >> 16


def compare(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    rows = []
    for sample in report["cases"]:
        observed = sample["prior_first_divergence"]
        if observed is None:
            continue
        native = sample["row"]
        reference = observed["hf_sdpa"]["scores"]
        scores = [(native["riley_logit"], reference["riley"]),
                  (native["hf_choice_logit"], reference["hf"])]
        distances = [bf16_bits(riley) - bf16_bits(hf) for riley, hf in scores]
        rows.append({
            "suite": sample["suite"], "case": sample["case"], "offset": sample["offset"],
            "tokens": [sample["riley_id"], sample["hf_id"]],
            "riley_bf16_scores": [riley for riley, _ in scores],
            "hf_sdpa_bf16_scores": [hf for _, hf in scores],
            "riley_minus_hf_bf16_steps": distances,
            "exactly_one_score_changed_by_one_bf16_step": sorted(map(abs, distances)) == [0, 1],
        })
    if len(rows) != report["decisions"]["divergences"] or len(rows) != 7:
        raise ValueError("missing first divergence evidence")
    return {
        "scope": "Two original selected token IDs per first divergence at the same token prefix.",
        "source_sha256": sha256(raw).hexdigest(),
        "all_first_divergences_one_bf16_step": all(row["exactly_one_score_changed_by_one_bf16_step"] for row in rows),
        "positive_shift_on_riley_id": sum(row["riley_minus_hf_bf16_steps"] == [1, 0] for row in rows),
        "negative_shift_on_hf_id": sum(row["riley_minus_hf_bf16_steps"] == [0, -1] for row in rows),
        "cases": rows,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("verified_rows", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.verified_rows), indent=2, allow_nan=False))
