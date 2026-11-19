"""Compare actual layer-0 HF and Riley cache rows and replay their attention."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

from replay_attention_mask import attention, bf16, capture, compare


HIDDEN = 3584
HEAD_SIZE = 128
QUERY_HEADS_PER_KV = 7
VOCAB = 152064


def tensor_bytes(tensor):
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError("expected a BF16 capture")
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()


def hf_case(model, ids, positions):
    import torch

    layer = model.model.layers[0].self_attn
    saved = {}

    def rotary(_module, _args, kwargs):
        saved["cosine"], saved["sine"] = kwargs["position_embeddings"]

    def query(_module, _args, result):
        values = result[0, positions].reshape(len(positions), -1, HEAD_SIZE)
        left, right = values.chunk(2, dim=-1)
        flipped = torch.cat((-right, left), dim=-1)
        cosine = saved.pop("cosine")[0, positions, None, :]
        sine = saved.pop("sine")[0, positions, None, :]
        saved["query"] = tensor_bytes((values * cosine + flipped * sine)[:, :QUERY_HEADS_PER_KV])

    def context(_module, args):
        values = args[0][0, positions].reshape(len(positions), -1, HEAD_SIZE)
        saved["context"] = tensor_bytes(values[:, :QUERY_HEADS_PER_KV])

    hooks = (layer.register_forward_pre_hook(rotary, with_kwargs=True),
             layer.q_proj.register_forward_hook(query), layer.o_proj.register_forward_pre_hook(context))
    try:
        with torch.inference_mode():
            output = model(torch.tensor([ids], device="cuda"), use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            first = cache.layers[0]
            expected = (1, model.config.num_key_value_heads, len(ids), HEAD_SIZE)
            if cache.get_seq_length() != len(ids) or first.keys.shape != expected or first.values.shape != expected:
                raise ValueError("HF cache layout differs from the pinned GQA geometry")
            saved["key"] = tensor_bytes(first.keys[0, 0])
            saved["value"] = tensor_bytes(first.values[0, 0])
            logits = output.logits[0, -1]
            selected = int(logits.argmax())
            saved["logits"] = tensor_bytes(logits)
    finally:
        for hook in hooks:
            hook.remove()
    return saved, selected


def kv_stats(riley, hf):
    if riley.shape != hf.shape or not np.all(np.isfinite(riley)) or not np.all(np.isfinite(hf)):
        raise ValueError("invalid KV comparison")
    delta = hf - riley
    different = riley.view(np.uint64) != hf.view(np.uint64)
    return {"elements": int(delta.size), "different_elements": int(different.sum()),
            "tokens_with_difference": int(different.any(axis=1).sum()),
            "rms": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.abs(delta).max())}


def replay_cell(q, k, v, context):
    return {
        f"q_{q_from}_k_{k_from}_v_{v_from}": {
            target: compare(actual, attention(q[q_from], k[k_from], v[v_from]))
            for target, actual in context.items()
        }
        for q_from, k_from, v_from in product(("riley", "hf"), repeat=3)
    }


def compare_case(model, stage, native, reference, fixture, case):
    old = json.loads(fixture.read_bytes())["baseline"][case]
    ids = old["prompt_tokens"]
    native_dir = native / f"native{case}"
    native_receipt = json.loads((native / f"case{case}-replay.json").read_bytes())
    native_output = json.loads((native_dir / "response.json").read_bytes())
    rows, first, last, _, riley_key, riley_value, digest = capture(native_dir, len(ids) - 1)
    if not (native_output["all_original_ids_text_and_finish"]
            and native_output["response"]["prompt_tokens"] == ids
            and native_receipt["page_data_sha256"] == digest
            and native_receipt["prompt_sha256"] == old["prompt_sha256"]):
        raise ValueError("Riley capture does not bind to the pinned prompt and prior replay")
    positions = [first, last]
    hf, hf_id = hf_case(model, ids, positions)
    history_dir = reference / f"hf{case}/hf-rows"
    meta = json.loads((history_dir / "metadata.json").read_bytes())
    past_logits = (history_dir / "rows.bf16").read_bytes()[-VOCAB * 2:]
    if (meta["prompt_sha256"], meta["prompt_tokens"], meta["last_token_id"], meta["selected_id"], meta["backend"]) != (
            old["prompt_sha256"], len(ids), ids[-1], hf_id, "sdpa") or hf["logits"] != past_logits:
        raise ValueError("the unchanged HF SDPA prompt did not reproduce its full historical logits")
    (stage / f"hf-case{case}").mkdir()
    for name, data in hf.items():
        (stage / f"hf-case{case}/{name}.bf16").write_bytes(data)
    hf_key = bf16(hf["key"]).reshape(-1, HEAD_SIZE)
    hf_value = bf16(hf["value"]).reshape(-1, HEAD_SIZE)
    hf_query = bf16(hf["query"]).reshape(len(positions), QUERY_HEADS_PER_KV, HEAD_SIZE)
    hf_context = bf16(hf["context"]).reshape(len(positions), QUERY_HEADS_PER_KV, HEAD_SIZE)
    probes = {}
    for index, (position, suffix) in enumerate(zip(positions, ("_first", ""), strict=True)):
        end = position + 1
        native_q = bf16(rows[(f"query_rope{suffix}", "0")]).reshape(-1, HEAD_SIZE)[:QUERY_HEADS_PER_KV]
        native_ctx = bf16(rows[(f"attention_context{suffix}", "0")]).reshape(-1, HEAD_SIZE)[:QUERY_HEADS_PER_KV]
        probes[str(position)] = replay_cell(
            {"riley": native_q, "hf": hf_query[index]},
            {"riley": riley_key[:end], "hf": hf_key[:end]},
            {"riley": riley_value[:end], "hf": hf_value[:end]},
            {"riley": native_ctx, "hf": hf_context[index]},
        )
    return {
        "case": case, "prompt_sha256": old["prompt_sha256"], "positions": positions,
        "riley_first_id": old["tokens"][0], "hf_first_id": hf_id,
        "riley_capture_sha256": digest, "hf_capture_sha256": {name: sha256(data).hexdigest() for name, data in hf.items()},
        "hf_full_logits_match": True,
        "kv": {name: {"whole_prompt": kv_stats(riley, current),
                      "last_input_token": kv_stats(riley[-1:], current[-1:])}
               for name, riley, current in (("key", riley_key, hf_key), ("value", riley_value, hf_value))},
        "cross_engine_fp64_attention": probes,
    }


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    expected_checkpoint = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
    checkpoint = sha256((args.model / "riley-checkpoint.json").read_bytes()).hexdigest()
    if checkpoint != expected_checkpoint:
        raise ValueError("model checkpoint differs from the pinned W4 candidate")
    fixture = args.reference / "fixtures/legacy.json"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    if model.config.hidden_size != HIDDEN or model.config.hidden_size // model.config.num_attention_heads != HEAD_SIZE:
        raise ValueError("unexpected model geometry")
    result = {"checkpoint_sha256": checkpoint, "fixture_sha256": sha256(fixture.read_bytes()).hexdigest(),
              "torch": str(torch.__version__), "transformers": transformers.__version__, "hf_backend": "sdpa",
              "layer": 0, "kv_head": 0, "query_heads": QUERY_HEADS_PER_KV,
              "cases": [compare_case(model, args.stage, args.native, args.reference, fixture, case) for case in (3, 9)]}
    with (args.stage / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
