"""Check the profiled HF projections against the full, previously verified trace."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


REFERENCE = "b7c459a5558d1d371228c751b83ad6bf29aeed4ff6100f3d472300bf59ece10a"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
WIDTHS = {"q": 3584, "k": 512, "v": 512}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def verify(stage, reference, model):
    require(digest(reference / "verified.json") == REFERENCE, "completed operator proof changed")
    require(digest(model / "riley-checkpoint.json") == CHECKPOINT, "checkpoint manifest changed")
    previous = json.loads((reference / "verified.json").read_bytes())
    traces = json.loads((stage / "hf/trace-receipt.json").read_bytes())
    require(traces.keys() == previous["cases"].keys(), "different HF cases were profiled")
    cases = {}
    for name, case in previous["cases"].items():
        source = traces[name]
        original_hash = case["source_trace_sha256"]["hf"]
        require(source["endpoints_match"] and (source["position"],source["token_id"]) ==
                (case["position"],case["input_token"])
                and digest(stage / "hf/rows" / f"{name}.bf16") == original_hash == source["trace_sha256"],
                f"{name}: operator trace changed under the profiler")
        plans = source["registered_prebias_projections"]
        require(plans.keys() == WIDTHS.keys(), "the QKV profile is incomplete")
        for projection, width in WIDTHS.items():
            ops = plans[projection]
            require(any(op["operator"] == "aten::linear" and op["input_shapes"][:2] ==
                        [[1,1,3584],[width,3584]] for op in ops), "linear shape differs")
            require(any(op["device_kernels"] for op in ops), "missing executed CUDA kernel labels")
        cases[name] = {"identical_hf_trace_sha256": original_hash, "profiled_projections": plans}
    return {"scope": "Executed PyTorch/CUDA kernel evidence for the three reconstructed HF bias-free QKV products.",
            "operator_reference_sha256":REFERENCE,"checkpoint_manifest_sha256":CHECKPOINT,"cases":cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage","reference","model"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.stage,args.reference,args.model),indent=2,allow_nan=False))
