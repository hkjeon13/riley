"""Crossover workload for the independent generated-row attention score intervention."""

import argparse
import json
import os
from pathlib import Path

from agent_workflow import fingerprint, stream
from full_register import body, case
from register_reference import MODEL

HERE = Path(__file__).resolve().parent


def inputs():
    reg = json.loads((HERE / "register-score-holdout.json").read_bytes())
    for kind, seed, count in reg["cases"]:
        req, records, answer, schema = case(kind, seed, count)
        req = body(req, records, schema, False, False)
        req["return_token_ids"] = True
        yield f"{kind}-{seed}", req, answer


def workload(address, out):
    rows = {}
    for name, req, expected in inputs():
        rows[name] = {}
        for label, cap in (("fixed", 24), ("natural", req["max_tokens"])):
            q = {**req, "max_tokens": cap}
            m = {}
            answer, calls = stream(address, q, m, timeout_s=600)
            rows[name][label] = {
                "answer": answer,
                "expected": expected,
                "calls": calls,
                "measurement": m,
            }
            out.write_text(json.dumps(rows) + "\n")


def hf(out):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("RILEY_GPU_HOLDER_LEASE_DIR"):
        raise ValueError("GPU lease missing")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        local_files_only=True,
    ).eval()
    report = {
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cases": {},
    }
    with torch.inference_mode():
        for name, req, exp in inputs():
            messages = req["messages"]
            for m in messages:
                for c in m.get("tool_calls", []):
                    c["function"]["arguments"] = json.loads(c["function"]["arguments"])
            prompt = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False
            )
            ids = torch.tensor([prompt], device="cuda")
            seq = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=req["max_tokens"],
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
            chosen = seq[0, len(prompt) :].tolist()
            report["cases"][name] = {
                "generated_ids": chosen,
                "answer": tok.decode(chosen, skip_special_tokens=True),
                "expected": exp,
                "prompt_ids_sha256": fingerprint(prompt),
                "prompt_tokens": len(prompt),
            }
            out.write_text(json.dumps(report) + "\n")
            print(name, len(chosen), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--address")
    a = p.parse_args()
    if a.address:
        workload(a.address, a.out)
    else:
        hf(a.out)
