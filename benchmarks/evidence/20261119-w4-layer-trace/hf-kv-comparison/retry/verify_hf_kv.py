"""Verify the matched cache captures and replay the four causal attention rows."""

from hashlib import sha256
import json
from pathlib import Path
import sys

from hf_kv_comparison import HEAD_SIZE, QUERY_HEADS_PER_KV, VOCAB, kv_stats, replay_cell
from replay_attention_mask import bf16, capture, compare


def read(path):
    return json.loads(path.read_bytes())


def verify(stage, native, reference):
    if (stage / "campaign.exit").read_text().strip() != "0":
        raise ValueError("HF KV run did not complete")
    for line in (stage / "result.sha256").read_text().splitlines():
        digest, name = line.split()
        if sha256((stage / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"captured digest changed: {name}")
    report = read(stage / "result.json")
    fixture = (reference / "fixtures/legacy.json").read_bytes()
    baselines = json.loads(fixture)["baseline"]
    expected = ("sdpa", 0, 0, QUERY_HEADS_PER_KV)
    actual = tuple(report[key] for key in ("hf_backend", "layer", "kv_head", "query_heads"))
    if actual != expected or sha256(fixture).hexdigest() != report["fixture_sha256"]:
        raise ValueError("reference fixture or attention scope changed")
    before, after = (read(stage / f"idle-{point}.json") for point in ("before", "after"))
    if (any(before[key] != after[key] for key in before if key != "updated_ns")
        or after["phase"] != "idle" or after["physical_index"] != 6 or after["other_processes"]
        or after["allocation_bytes"] != 20 * 1024**3 or after["torch_reserved_bytes"] != 20 * 1024**3):
        raise ValueError("GPU 6 did not restore the original reservation")
    if [item["case"] for item in report["cases"]] != [3, 9]:
        raise ValueError("two captured cases are required")
    rows_by_case = {}
    for case in report["cases"]:
        ordinal = case["case"]
        pinned = baselines[ordinal]
        ids = pinned["prompt_tokens"]
        rows, first, last, _, riley_key, riley_value, native_digest = capture(native / f"native{ordinal}", len(ids) - 1)
        if ([first, last], native_digest, pinned["prompt_sha256"]) != (case["positions"], case["riley_capture_sha256"], case["prompt_sha256"]):
            raise ValueError("Riley source trace or model input changed")
        hf = {name: (stage / f"hf-case{ordinal}/{name}.bf16").read_bytes()
              for name in ("key", "value", "query", "context", "logits")}
        if {name: sha256(data).hexdigest() for name, data in hf.items()} != case["hf_capture_sha256"]:
            raise ValueError("HF capture binding differs")
        history = reference / f"hf{ordinal}/hf-rows"
        if (hf["logits"] != (history / "rows.bf16").read_bytes()[-VOCAB * 2:]
            or case["hf_first_id"] != read(history / "metadata.json")["selected_id"]):
            raise ValueError("HF full historical logits did not reproduce")
        key = bf16(hf["key"]).reshape(-1, HEAD_SIZE)
        value = bf16(hf["value"]).reshape(-1, HEAD_SIZE)
        query = bf16(hf["query"]).reshape(2, QUERY_HEADS_PER_KV, HEAD_SIZE)
        context = bf16(hf["context"]).reshape(2, QUERY_HEADS_PER_KV, HEAD_SIZE)
        for name, left, right in (("key", riley_key, key), ("value", riley_value, value)):
            actual = {"whole_prompt": kv_stats(left, right), "last_input_token": kv_stats(left[-1:], right[-1:])}
            if actual != case["kv"][name]:
                raise ValueError(f"{name} counts differ from the captured vectors")
        probes = {}
        for index, (position, suffix) in enumerate(zip((first, last), ("_first", ""), strict=True)):
            visible = position + 1
            native_q = bf16(rows[(f"query_rope{suffix}", "0")]).reshape(-1, HEAD_SIZE)[:QUERY_HEADS_PER_KV]
            native_ctx = bf16(rows[(f"attention_context{suffix}", "0")]).reshape(-1, HEAD_SIZE)[:QUERY_HEADS_PER_KV]
            variants = replay_cell({"riley": native_q, "hf": query[index]},
                                   {"riley": riley_key[:visible], "hf": key[:visible]},
                                   {"riley": riley_value[:visible], "hf": value[:visible]},
                                   {"riley": native_ctx, "hf": context[index]})
            if variants != case["cross_engine_fp64_attention"][str(position)]:
                raise ValueError("the FP64 causal replay differs from the raw capture")
            probes[str(position)] = {"observed_cross_engine_context": compare(native_ctx, context[index]),
                                     "replay_context_rms_to_hf": {name: values["hf"]["rms"] for name, values in variants.items()},
                                     "replay_context_rms_to_riley": {name: values["riley"]["rms"] for name, values in variants.items()}}
        rows_by_case[str(ordinal)] = probes
    return {"gpu_result_sha256": sha256((stage / "result.json").read_bytes()).hexdigest(),
            "same_historical_hf_logits": True, "gpu_6_reservation_restored": True,
            "scope": {key: report[key] for key in ("hf_backend", "layer", "kv_head", "query_heads")},
            "cases": rows_by_case}


if __name__ == "__main__":
    stage, native, reference = map(Path, sys.argv[1:])
    print(json.dumps(verify(stage, native, reference), indent=2, allow_nan=False))
