"""Verify the layer-1 intervention and independently round its shared BF16 operands."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

from kv_trial import capture as previous_capture
from projection_oracle import compare, file_hash, numeric_words, ref_dot
from runner import (
    BOUNDARY, CHECKPOINT, COMPARISON, CURRENT, GPU, HISTORY, INJECTED, LOGITS, OLD_KEYS,
    PREFIX, RESTORED, capture, differences, digest, read, require, reservation, scores,
)


PARENT = "w4-kv-history-gpu-20261123.G8mePjYa"
DOWN_KEY = "model.layers.1.mlp.down_proj.weight"
LAYER1 = ("context", "attention_residual", "mlp_input", "mlp_gated", "down_native",
          "down_effective", "layer_residual", "layer2_input")
ROLES = (("baseline", "candidate", 6), ("candidate", "baseline", 7))
KEY_WIDTH = 18944


def verify_reservations(stage):
    result = {}
    for role, _, gpu in ROLES:
        lease = reservation(stage / f"gpu{gpu}-before.json", gpu)
        for path in (stage / f"gpu{gpu}-after.json", stage / role / "reservation-before.json",
                     stage / role / "reservation-after.json"):
            require(reservation(path, gpu) == lease, f"reservation drift: {path}")
        result[str(gpu)] = lease
    return result


def verify_server(stage, parent, build, role, binary):
    out = stage / role
    receipt, old = read(out / "receipt.json"), read(parent / role / "receipt.json")
    argv, old_argv = receipt.pop("server_argv"), old.pop("server_argv")
    bind = argv.index("--bind") + 1
    require(argv[0] == str(build / "target/release/riley") and argv[bind].startswith("127.0.0.1:")
            and argv[1:bind] == old_argv[1:bind] and argv[bind+1:] == old_argv[bind+1:], "server options drift")
    for key in ("mode", "switch", "historical_response_sha256", "historical_capture_sha256", "input_ids_sha256"):
        require(receipt[key] == old[key], f"receipt drift in {role}: {key}")
    require(receipt["binary_sha256"] == binary and receipt["model_sha256"] == CHECKPOINT and receipt["gpu"] == GPU[role]
            and receipt["server_exit"] == 0 and receipt["ready"]["ready"] and receipt["ready"]["accepting"]
            and receipt["input_token_count"] == 7836 and (out / "worker.exit").read_text().strip() == "0",
            f"runtime proof drift: {role}")
    shutdown = read(out / "shutdown.json")
    require(all(shutdown[k] == 0 for k in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and not any(shutdown["allocation"].values()), f"resource leak: {role}")
    log = (out / "server.log").read_text()
    plan_used = [line for line in log.splitlines() if line.startswith("W4_DOWN_FP32_PLAN")]
    if role == "candidate":
        require(any("m=1 " in line and "id=21 " in line and "reduction=2" in line for line in plan_used),
                "candidate plan was absent")
    else:
        require(not plan_used, "candidate plan ran in baseline")
    return {"input_ids_sha256": receipt["input_ids_sha256"], "down_plan_log": plan_used}


def trace_all(stage, parent, role):
    out = stage / role
    rows, receipts = {}, {}
    for name, swap in (("calibration", False), ("self", True), ("history", True), ("down", True)):
        target = out / name
        rows[name] = capture(target, swapped=swap, boundary=swap)
        response = read(target / "response.json")
        require(response["first_eight"][:7] == PREFIX and response["generated_tokens"] == 256,
                f"unexpected response: {role}/{name}")
        if name != "down":
            old = previous_capture(parent / role / name, swapped=swap, boundary=swap)
            require(all(rows[name][key] == raw for key, raw in old.items())
                    and response["response_sha256"] == read(parent / role / name / "response.json")["response_sha256"],
                    f"parent trace drift: {role}/{name}")
        chosen = response["first_eight"][-1]
        receipts[name] = {**response, "scores": scores(rows[name][LOGITS], chosen),
                          "trace_sha256": digest(target / "rows.bf16")}
    for name in ("calibration", "self", "history"):
        require(rows[name]["down_effective", "1"] == rows[name]["down_native", "1"], f"down row changed in {role}/{name}")
    require(all(rows["self"][key] == data for key, data in rows["calibration"].items())
            and receipts["self"]["response_sha256"] == receipts["calibration"]["response_sha256"],
            f"self intervention changed serving: {role}")
    return rows, receipts


def verify_intervention(rows, donor):
    prior, changed = rows["history"], rows["down"]
    require(all(changed[name, "1"] == prior[name, "1"] for name in
                HISTORY + CURRENT + INJECTED + RESTORED + LAYER1[:5])
            and all(changed[name, "0"] == prior[name, "0"] for name in BOUNDARY),
            "the intervention changed inputs or its native down row")
    require(changed["down_effective", "1"] == donor["down_native", "1"], "the donor down row was not installed")
    indices = OLD_KEYS[OLD_KEYS.index(("attention_residual", "1")):]
    next_differences = {}
    for name, arm in (("history", prior), ("down", changed)):
        next_differences[name] = next(
            ({"stage": key[0], "layer": key[1], **differences(arm[key], donor[key])}
             for key in indices if arm[key] != donor[key]),
            None,
        )
    return {
        "difference_from_donor": {
            name: {key: differences(arm[key, "1"], donor[key, "1"]) for key in LAYER1}
            for name, arm in (("history", prior), ("down", changed))
        },
        "first_recorded_difference": next_differences,
        "logits_vs_donor_after_down": differences(changed[LOGITS], donor[LOGITS]),
        "logits_vs_history_after_down": differences(changed[LOGITS], prior[LOGITS]),
    }


def oracle(model, runs):
    manifest = model / "riley-checkpoint.json"
    require(file_hash(manifest) == CHECKPOINT, "checkpoint manifest drift")
    files = read(manifest)["files"]
    index_path = model / "model.safetensors.index.json"
    index_entry, = (item for item in files if item["path"] == index_path.name)
    require(file_hash(index_path) == index_entry["sha256"], "model index drift")
    name = read(index_path)["weight_map"][DOWN_KEY]
    shard = model / name
    entry, = (item for item in files if item["path"] == name)
    require(shard.stat().st_size == entry["bytes"] and file_hash(shard) == entry["sha256"], "down weight shard drift")
    torch.set_num_threads(4)
    inputs = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weights = handle.get_tensor(DOWN_KEY)
        require(tuple(weights.shape) == (3584, KEY_WIDTH) and weights.dtype == torch.bfloat16, "down weight shape drift")
        for owner, peer, _ in ROLES:
            observations = {owner: runs[owner]["calibration"], peer: runs[peer]["history"]}
            gated = observations[owner]["mlp_gated", "1"]
            require(observations[peer]["mlp_gated", "1"] == gated, "down inputs are unequal")
            nearest, fp64, fractions, fallbacks, ties = ref_dot(weights, numeric_words(gated))
            inputs[owner] = {
                "operand_sha256": sha256(gated).hexdigest(),
                "nearest_even_bf16_sha256": sha256(nearest.astype("<u2", copy=False).tobytes()).hexdigest(),
                "integer_fallback_rows": fallbacks, "midpoint_ties": ties,
                "output_comparisons": {
                    role: compare(numeric_words(row["down_native", "1"]), nearest, fp64, fractions)
                    for role, row in sorted(observations.items())
                },
            }
    return {"weight_key": DOWN_KEY, "weight_shard_sha256": entry["sha256"],
            "shape_m_n_k": [1, 3584, KEY_WIDTH], "input_owners": inputs}


def verify(stage, build, model):
    binary = (build / "build-result.sha256").read_text().split()[0]
    require(digest(build / "target/release/riley") == binary
            and (stage / "checkpoint-verify.log").read_text().strip() == "checkpoint verify: verified",
            "build or checkpoint proof drift")
    parent = stage.parent / PARENT
    reservations = verify_reservations(stage)
    runs, report = {}, {}
    for role, _, _ in ROLES:
        report[role] = verify_server(stage, parent, build, role, binary)
        runs[role], report[role]["trials"] = trace_all(stage, parent, role)
    for role, donor, _ in ROLES:
        report[role].update(verify_intervention(runs[role], runs[donor]["calibration"]))
    return {"status": "passed", "comparison": COMPARISON, "checkpoint_sha256": CHECKPOINT,
            "remote_stage": str(stage), "remote_build": str(build), "binary_sha256": binary,
            "reservations": reservations, "roles": report, "oracle": oracle(model, runs)}


if __name__ == "__main__":
    stage, build, model = map(Path, sys.argv[1:])
    print(json.dumps(verify(stage, build, model), indent=2, allow_nan=False))
