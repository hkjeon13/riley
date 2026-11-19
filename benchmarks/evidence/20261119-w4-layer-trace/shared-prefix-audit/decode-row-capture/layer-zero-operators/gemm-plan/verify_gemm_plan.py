"""Bind each recorded CUDA GEMM plan to a reproduced layer-zero activation trace."""

import argparse
import json
from pathlib import Path

from trace_decode_layers import digest, require

COMPLETED_OPERATOR_SHA = "b7c459a5558d1d371228c751b83ad6bf29aeed4ff6100f3d472300bf59ece10a"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
FIELDS = ("position", "token", "m", "n", "k", "backend", "id", "tile", "stages", "split_k", "reduction",
          "cta_swizzling", "custom_option", "workspace", "numeric_flags", "deterministic", "sm", "cuda", "cublaslt")


def read(path):
    return json.loads(path.read_bytes())


def plans(stage, reference, model):
    raw = (reference / "verified.json").read_bytes()
    require(digest(raw) == COMPLETED_OPERATOR_SHA, "completed operator reference differs")
    require(digest((model / "riley-checkpoint.json").read_bytes()) == CHECKPOINT, "checkpoint manifest changed")
    baseline = json.loads(raw)
    run = read(stage / "riley/receipt.json")
    require(run["status"] == "passed" and run["server_exit"] == 0, "Riley capture failed")
    client = read(stage / "riley/client.log")
    labels = [line.split()[1:] for line in (stage / "riley/server.log").read_text().splitlines()
              if line.startswith("W4_QKV_PLAN ")]
    markers = [dict(piece.split("=", 1) for piece in line) for line in labels]
    require(len(markers) == 2 and all(tuple(p) == FIELDS for p in markers), "expected two selected plan records")
    rows = {}
    for case, known in baseline["cases"].items():
        trace = stage / "riley/rows" / case
        active = client[case]
        _, pos, token, row, count, dense = (trace / "index.tsv").read_text().splitlines()[0].split()
        earlier = known["source_trace_sha256"]["riley"]
        require(active["output_match"] and active["trace_sha256"] == digest((trace / "rows.bf16").read_bytes()) == earlier
                and (int(pos), int(token), int(row), int(count)) == (known["position"], known["input_token"], 0, 1),
                "current operator capture differs from the matched reference")
        hits = [r for r in markers if (r["position"], r["token"]) == (pos, token)]
        require(len(hits) == 1, "the trace has no unique GEMM record")
        info = hits[0]
        require(info["deterministic"] == "true" and info["sm"] == "8.0" and info["backend"] == "1"
                and (info["m"], info["n"], info["k"]) == (dense, "4608", "3584"),
                "GEMM evidence does not identify the actual decode extent")
        converted = {key: int(value) for key, value in info.items() if key not in ("deterministic", "sm")}
        rows[case] = {"identical_operator_trace_sha256": earlier, "gemm": {**converted, "deterministic": True, "sm": "8.0"}}
    return {"scope": "Selected canonical cuBLASLt merged QKV plans for the two reproduced C1 rows.",
            "operator_reference_sha256": COMPLETED_OPERATOR_SHA, "checkpoint_manifest_sha256": CHECKPOINT,
            "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("stage", "reference", "model"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(plans(args.stage, args.reference, args.model), indent=2, allow_nan=False))
