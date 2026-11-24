"""Capture matching HF layer rows during prefill and the original autoregressive step."""

import argparse
import json
import os
from pathlib import Path

from agent_workflow import fingerprint, stream
from register_reference import MODEL, requests

HERE = Path(__file__).resolve().parent


def witness(address, out):
    rows = {}
    targets = json.loads((HERE / "register-layer-registration.json").read_bytes())[
        "cases"
    ]
    for name, req, expected in requests():
        if name not in targets:
            continue
        m = {}
        answer, calls = stream(address, req, m, timeout_s=600)
        rows[name] = {
            "measurement": m,
            "answer": answer,
            "expected": expected,
            "tool_calls": calls,
        }
        out.write_text(json.dumps(rows) + "\n")


def hf(out):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("RILEY_GPU_HOLDER_LEASE_DIR"):
        raise RuntimeError("GPU lease required")
    out.mkdir()
    reg = json.loads((HERE / "register-layer-registration.json").read_bytes())["cases"]
    prior = json.loads((HERE / "prior-hf.json").read_bytes())
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        local_files_only=True,
    ).eval()
    cases = {}
    for name, req, _ in requests():
        if name not in reg:
            continue
        msgs = req["messages"]
        for msg in msgs:
            for call in msg.get("tool_calls", []):
                call["function"]["arguments"] = json.loads(
                    call["function"]["arguments"]
                )
        prompt = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        ref = prior["cases"][name]
        if fingerprint(prompt) != ref["prompt_ids_sha256"]:
            raise ValueError("different prompt")
        n = reg[name]["difference_index"]
        state = {"phase": "prefill"}
        bucket = {}

        def capture(tag, layer, t, *, state=state, bucket=bucket):
            if state["phase"] is not None:
                flat = (
                    t[0, -1]
                    .detach()
                    .to(torch.bfloat16)
                    .contiguous()
                    .cpu()
                    .view(torch.uint16)
                )
                bucket[(state["phase"], tag, layer)] = flat.numpy().tobytes()

        def post(tag, layer, *, capture=capture):
            def hook(_mod, _args, output):
                capture(tag, layer, output)

            return hook

        def pre(tag, layer, *, capture=capture):
            def hook(_mod, args):
                capture(tag, layer, args[0])

            return hook

        handles = [
            model.model.embed_tokens.register_forward_hook(post("embedding", -1)),
            model.model.norm.register_forward_hook(post("final_norm", -1)),
        ]
        for i, layer in enumerate(model.model.layers):
            handles.extend(
                [
                    layer.input_layernorm.register_forward_hook(post("input_norm", i)),
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        pre("attention_context", i)
                    ),
                    layer.self_attn.o_proj.register_forward_hook(
                        post("attention_projection", i)
                    ),
                    layer.register_forward_hook(post("layer_residual", i)),
                ]
            )
        try:
            with torch.inference_mode():
                result = model(
                    input_ids=torch.tensor([prompt], device="cuda"),
                    use_cache=True,
                    logits_to_keep=1,
                )
                capture("logits", -1, result.logits)
                generated = []
                for i in range(n):
                    current = int(result.logits[0, -1].argmax())
                    if current != ref["generated_ids"][i]:
                        raise ValueError(f"cached HF prefix changed at {name}/{i}")
                    generated.append(current)
                    state["phase"] = "decode" if i == n - 1 else None
                    result = model(
                        input_ids=torch.tensor([[current]], device="cuda"),
                        past_key_values=result.past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                capture("logits", -1, result.logits)
                chosen = int(result.logits[0, -1].argmax())
                if chosen != ref["generated_ids"][n]:
                    raise ValueError("HF target differs from prior reproduction")
                cases[name] = {
                    "tokens": len(prompt),
                    "layers": len(model.model.layers),
                    "prompt_ids_sha256": fingerprint(prompt),
                    "prefill_token": prompt[-1],
                    "prefix": generated,
                    "choice": chosen,
                }
                del result
        finally:
            for handle in handles:
                handle.remove()
        for phase in ("prefill", "decode"):
            d = out / f"{name}-{phase}"
            d.mkdir()
            offset = 0
            with (
                (d / "rows.bf16").open("wb") as data,
                (d / "index.tsv").open("w") as index,
            ):
                for (at, tag, layer), raw in bucket.items():
                    if at == phase:
                        index.write(
                            f"{tag}\t{layer if layer >= 0 else '-'}\t{offset}\t{len(raw)}\n"
                        )
                        data.write(raw)
                        offset += len(raw)
        print(name, chosen, flush=True)
    (out / "report.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                "cases": cases,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--address")
    a = p.parse_args()
    if a.address:
        witness(a.address, a.out)
    else:
        hf(a.out)
