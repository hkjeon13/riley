"""Identify the first distinct Riley token and evaluate the same prefix in Transformers."""

import argparse
import json
import os
from pathlib import Path

from agent_workflow import fingerprint, stream
from full_register import body, case

HERE = Path(__file__).resolve().parent
MODEL = Path("/mnt/efs/nlp/psyche/riley-gap-20261102/models/Qwen2.5-7B-Instruct")


def requests():
    for kind, seed, count in json.loads(
        (HERE / "register-reference-registration.json").read_bytes()
    )["cases"]:
        req, records, exp, schema = case(kind, seed, count)
        req = body(req, records, schema, False, False)
        req["return_token_ids"] = True
        yield f"{kind}-{seed}", req, exp


def witness(address, out):
    results = {}
    for name, req, expected in requests():
        m = {}
        answer, calls = stream(address, req, m, timeout_s=600)
        results[name] = {
            "measurement": m,
            "answer": answer,
            "expected": expected,
            "tool_calls": calls,
        }
        out.write_text(json.dumps(results) + "\n")


def reference(out):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("RILEY_GPU_HOLDER_LEASE_DIR") or not os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise ValueError("GPU lease environment is required")
    arms = {
        mode: json.loads((HERE / mode / "witness.json").read_bytes())
        for mode in ("page", "tile")
    }
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        local_files_only=True,
    ).eval()
    eos = model.generation_config.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    result = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "eos": eos,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "name": torch.cuda.get_device_name(),
        "dtype": str(model.dtype),
        "attention": model.config._attn_implementation,
        "cases": {},
    }
    with torch.inference_mode():
        for name, req, expected in requests():
            msgs = json.loads(json.dumps(req["messages"]))
            for msg in msgs:
                for call in msg.get("tool_calls", []):
                    call["function"]["arguments"] = json.loads(
                        call["function"]["arguments"]
                    )
            prompt = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_dict=False
            )
            states = [arms[mode][name]["measurement"] for mode in ("page", "tile")]
            prompt_sha = fingerprint(prompt)
            if any(
                s["prompt_ids_sha256"] != prompt_sha
                or s["usage"]["prompt_tokens"] != len(prompt)
                for s in states
            ):
                raise ValueError(f"reference prompt IDs differ for {name}")
            page, tile = [s["generated_ids"] for s in states]
            first = next(
                (i for i, (a, b) in enumerate(zip(page, tile, strict=False)) if a != b),
                None,
            )
            delta = None
            if first is not None:
                prefix = torch.tensor([prompt + page[:first]], device="cuda")
                logits = (
                    model(input_ids=prefix, use_cache=False, logits_to_keep=1)
                    .logits[0, -1]
                    .float()
                )
                winner = int(logits.argmax())
                delta = {
                    "index": first,
                    "page": page[first],
                    "tile": tile[first],
                    "reference_argmax": winner,
                    "page_gap": float(logits[winner] - logits[page[first]]),
                    "tile_gap": float(logits[winner] - logits[tile[first]]),
                }
                del logits, prefix
            ids = torch.tensor([prompt], device="cuda")
            seq = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=req["max_tokens"],
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
            emitted = seq[0, len(prompt) :].tolist()
            result["cases"][name] = {
                "prompt_tokens": len(prompt),
                "prompt_ids_sha256": prompt_sha,
                "expected": expected,
                "first_difference": delta,
                "generated_ids": emitted,
                "answer": tok.decode(emitted, skip_special_tokens=True),
            }
            out.write_text(json.dumps(result) + "\n")
            print(name, len(emitted), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--address")
    args = p.parse_args()
    if args.address:
        witness(args.address, args.out)
    else:
        reference(args.out)
