"""Compare each CUDA 12.4 plan to the frozen BF16 projection rows."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re

from projection_crossover import CASES, OUT, PARTS, difference, loaded, require


FIELDS = ("id", "tile", "stages", "split_k", "reduction", "cta_swizzling", "custom_option", "numeric_flags")
PINNED = {"id": 21, "tile": 15, "stages": 18, "split_k": 1, "reduction": 0,
          "cta_swizzling": 0, "custom_option": 0, "numeric_flags": 131586}
PRIOR = "9bb140a4290330432613cd1a861bd629620b085be021a3fcfabd0e2e48fd9491"


def compare(stage, crossover):
    proof = (crossover / "verified.json").read_bytes()
    require(sha256(proof).hexdigest() == PRIOR, "unexpected crossover receipt")
    previous = json.loads(proof)
    require(set(previous["cases"]) == set(CASES), "unexpected captured cases")
    rows = stage / "operands"
    references = {}
    for name in CASES:
        binding = previous["binding"]["cases"][name]
        references[name] = {arm: loaded(rows / f"{name}.{arm}.bf16", binding[arm], OUT) for arm in ("riley", "hf")}

    first, *lines = (stage / "plans.txt").read_text().splitlines()
    header, returned = first.split()
    require(header == "heuristic_results" and 0 <= int(returned) <= 32, "invalid heuristic result count")
    plans = {}
    for line in lines:
        label, *raw = line.split()
        require(len(raw) == len(FIELDS) and label not in plans, "unknown plan record")
        if label != "recorded":
            require(re.fullmatch(r"rank-[0-9]+", label) and int(label.removeprefix("rank-")) < int(returned),
                    "unknown heuristic rank")
        fields = dict(zip(FIELDS, map(int, raw), strict=True))
        require(fields["split_k"] <= 1 and fields["reduction"] == 0, "unexpected reduction configuration")
        cases = {}
        for name in CASES:
            actual = (rows / f"{name}.{label}.bf16").read_bytes()
            comparisons = {}
            for arm, baseline in references[name].items():
                offset = 0
                by_projection = {}
                for part, width in PARTS:
                    end = offset + width * 2
                    by_projection[part] = difference(actual[offset:end], baseline[offset:end])["different_bf16_words"]
                    offset = end
                comparisons[arm] = {**difference(actual, baseline), "different_by_projection": by_projection}
            cases[name] = {"sha256": sha256(actual).hexdigest(), **comparisons}
        plans[label] = {"config": fields, "cases": cases}

    require("recorded" in plans and plans["recorded"]["config"] == PINNED, "missing recorded Riley plan")
    for name in CASES:
        require(plans["recorded"]["cases"][name]["sha256"] == previous["cases"][name]["replays"]["native-lt"]["sha256"],
                "Riley control did not replay exactly")
    return {"prior_verified_sha256": PRIOR, "native_cuda": 12040, "native_cublaslt": 120405,
            "m": 1, "n": OUT, "k": 3584, "workspace": 0, "heuristic_results": int(returned), "plans": plans}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("crossover", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.stage, args.crossover), indent=2, allow_nan=False))
