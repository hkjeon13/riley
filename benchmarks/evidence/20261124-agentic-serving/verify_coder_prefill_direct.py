"""Compare the direct prompt reduction candidate with the matched page baseline."""

import argparse
import json
from pathlib import Path

from verify_coder_prefill_trace import SEEDS, arm
from verify_decode import compare
from verify_model_decode import read


def score(root):
    assert (root / "direct-exits.log").read_text().split() == ["0", "0"]
    result = {}
    for gpu in (6, 7):
        d = root / f"gpu{gpu}"
        assert set((d / "direct.env").read_text().splitlines()) == {
            "RILEY_NATIVE_D128_PROMPT_DIRECT=1",
            "RILEY_NATIVE_D128_PROMPT_DIRECT_MAX_PAGES=0",
        }
        assert (d / "direct/server-arguments.txt").read_text() == (
            d / "page/server-arguments.txt"
        ).read_text()
        baseline, candidate = [arm(root, gpu, mode) for mode in ("page", "direct")]
        raws = {
            m: {r["seed"]: r for r in read(d / m / "natural.json")}
            for m in ("page", "direct")
        }
        rows = []
        for seed in SEEDS:
            a, b = baseline[seed], candidate[seed]
            assert (
                a["prompt_ids_sha256"] == b["prompt_ids_sha256"]
                and a["cached"] == b["cached"]
            )
            assert [
                (c["rows"], c["end_position"], c["members"]) for c in a["chunks"]
            ] == [(c["rows"], c["end_position"], c["members"]) for c in b["chunks"]]
            rows.append(
                {
                    "seed": seed,
                    "ttft_ms": [a["ttft_ms"], b["ttft_ms"]],
                    "gpu_prefill_ms": [
                        a["prefill_participating_gpu_ms"],
                        b["prefill_participating_gpu_ms"],
                    ],
                    "valid": [a["valid"], b["valid"]],
                    "natural_completion_ms": [a["last_final_ms"], b["last_final_ms"]],
                    "tokens": [a["final_tokens"], b["final_tokens"]],
                    "same_natural_text": raws["page"][seed]["turns"][-1]["answer"]
                    == raws["direct"][seed]["turns"][-1]["answer"],
                }
            )
        cells = compare(d / "direct", d / "page")["cells"]
        fixed = [
            {
                "history": c["history"],
                "stream": c["stream"],
                "tpot_ratio": c["tpot_ratio"],
                "gap_ratio": c["gap_ratio"],
                "same_output_ids": c["identity_output_ids_equal"],
            }
            for c in cells
        ]
        gates = {
            "lower_long_ttft_and_gpu": all(
                r["ttft_ms"][1] < r["ttft_ms"][0]
                and r["gpu_prefill_ms"][1] < r["gpu_prefill_ms"][0]
                for r in rows
                if r["seed"] != 761
            ),
            "no_lost_correct": all(not r["valid"][0] or r["valid"][1] for r in rows),
            "natural_completion_guard": all(
                r["natural_completion_ms"][1] <= 1.05 * r["natural_completion_ms"][0]
                for r in rows
            ),
            "fixed_decode_guard": all(
                max(c["tpot_ratio"], c["gap_ratio"]) <= 1.05 for c in fixed
            ),
        }
        result[gpu] = {
            "natural": rows,
            "fixed": fixed,
            "gates": gates,
            "passed": all(gates.values()),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(score(p.parse_args().directory), indent=2))
