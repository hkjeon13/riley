"""Compare identity probes with same-checkpoint BF16 SDPA generation and first fork scores."""

import argparse
import json
import os
from pathlib import Path

from agent_workflow import fingerprint
from fixed_agent_decode import freeze


def read(path):
    return json.loads(path.read_text())


def lcp(left, right):
    return next(
        (i for i, (x, y) in enumerate(zip(left, right)) if x != y),
        min(len(left), len(right)),
    )


def run(root, gpu):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("RILEY_GPU_HOLDER_LEASE_DIR") or not os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise RuntimeError("GPU lease required")
    config = read(root / "model-decode-registration.json")
    checkpoint = root.parents[2] / "models" / config["models"][str(gpu)]
    directory = root / f"gpu{gpu}"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        local_files_only=True,
    ).eval()
    arms = {}
    for i, engine in enumerate(config["order"][str(gpu)][:2]):
        arms[engine] = {
            (e["history"], e["result"]["name"]): e["result"]
            for e in read(directory / f"{i}-{engine}" / "fixed.json")
            if e["phase"] == "identity"
        }
    out = {
        "model": checkpoint.name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": str(model.dtype),
        "attention": model.config._attn_implementation,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "cases": [],
    }
    with torch.inference_mode():
        for history in (0, 200):
            workflows = read(root / "frozen" / f"workflows-{history}.json")
            for name, work in zip(
                ("reader", "long-tool-result"), workflows[-2:], strict=True
            ):
                messages = freeze(work, "m")["messages"]
                for message in messages:
                    for call in message.get("tool_calls", []):
                        call["function"]["arguments"] = json.loads(
                            call["function"]["arguments"]
                        )
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )
                states = [arms[e][history, name] for e in ("riley", "vllm")]
                assert all(
                    s["prompt_ids_sha256"] == fingerprint(prompt)
                    and s["usage"]["prompt_tokens"] == len(prompt)
                    for s in states
                )
                a, b = [s["generated_ids"] for s in states]
                i = lcp(a, b)
                ids = torch.tensor([prompt], device="cuda")
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )[0, len(prompt) :].tolist()
                fork = None
                if i < min(len(a), len(b)):
                    common = torch.tensor([prompt + a[:i]], device="cuda")
                    logits = (
                        model(input_ids=common, use_cache=False, logits_to_keep=1)
                        .logits[0, -1]
                        .float()
                    )
                    values, tokens = logits.topk(5)
                    fork = {
                        "index": i,
                        "riley_id": a[i],
                        "vllm_id": b[i],
                        "top_ids": tokens.tolist(),
                        "top_logits": values.tolist(),
                        "riley_logit": float(logits[a[i]]),
                        "vllm_logit": float(logits[b[i]]),
                    }
                out["cases"].append(
                    {
                        "history": history,
                        "name": name,
                        "prompt_tokens": len(prompt),
                        "prompt_sha256": fingerprint(prompt),
                        "ids": generated,
                        "text": tokenizer.decode(generated),
                        "common_riley_vllm": i,
                        "common_riley_hf": lcp(a, generated),
                        "common_vllm_hf": lcp(b, generated),
                        "full_forward_at_pair_fork": fork,
                    }
                )
                (directory / "transformers.json").write_text(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu", type=int, choices=(6, 7))
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, args.gpu)
