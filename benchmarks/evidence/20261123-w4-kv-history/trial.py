"""Exchange only layer-1 past K/V after aligning the layer-0 decode boundary."""
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

from boundary_trial import (
    BOUNDARIES, CHECKPOINT, EXTRA, GPU, LOGITS, MODES, OLD_KEYS, PREFIX, ROW_BYTES,
    complete, digest, fingerprint, gpu_processes, identity, read, read_capture,
    read_workload, require, reservation, ready_server, scores, start_server, trace as old_trace,
)


PRIOR = "w4-boundary-swap-gpu-20261123.OhUoSQsC"
MODEL_TRACE = "w4-long-code-gpu-20261122.QAbtnxKV"
COMPARISON = "fp32_down21"
HISTORY_BYTES = 7842 * 4 * 128 * 2
HISTORY = ("past_key", "past_value")
CURRENT = ("query", "current_key", "current_value")
INJECTED = ("injected_key", "injected_value")
RESTORED = ("restored_key", "restored_value")
BOUNDARY = tuple(label for label, _ in BOUNDARIES[COMPARISON][1])
KV_ROWS = dict.fromkeys(HISTORY + INJECTED + RESTORED, HISTORY_BYTES)
KV_ROWS.update({"query": ROW_BYTES, "context": ROW_BYTES, "current_key": 1024, "current_value": 1024})
OLD_ROWS = set(OLD_KEYS) | EXTRA


def capture(path, swapped=False, boundary=False):
    header, *lines = (path / "index.tsv").read_text().splitlines()
    require(header == "# 7842 198 0 1 1", f"wrong token under trace: {path}")
    raw = (path / "rows.bf16").read_bytes()
    expected = dict.fromkeys(OLD_ROWS, ROW_BYTES)
    expected[LOGITS] = 304128
    for label in HISTORY + CURRENT + ("context",) + (INJECTED + RESTORED if swapped else ()):
        expected[label, "1"] = KV_ROWS[label]
    if boundary:
        expected.update({("layer_residual_before", "0"): ROW_BYTES, ("next_input_norm_before", "0"): ROW_BYTES})
    rows = {}
    end = 0
    for line in lines:
        name, layer, offset, length = line.split("\t")
        key = (name, layer)
        offset, length = int(offset), int(length)
        require(key not in rows and expected.get(key) == length and offset == end, f"unexpected layout: {path} {key}")
        end += length
        rows[key] = raw[offset:end]
    require(end == len(raw) and set(rows) == set(expected), f"incomplete trace: {path}")
    return rows


def respond(out, name, address, case, ids, donor=None, swap=False):
    pending = out / "trace/decode-7"
    pending.mkdir()
    if donor:
        (pending / "down.bf16").write_bytes(b"".join(donor[key, "0"] for key in BOUNDARY))
        if swap:
            (pending / "past-kv.bf16").write_bytes(b"".join(donor[key, "1"] for key in HISTORY))
    start = time.monotonic()
    reply = complete(address, "m", case.prompt, diagnostic=True, expected=ids, limit=case.limit)
    require(reply["tokens"][:7] == PREFIX and len(reply["tokens"]) == 256, f"trace request drift: {out}/{name}")
    target = out / name
    pending.rename(target)
    receipt = {"first_eight": reply["tokens"][:8], "generated_tokens": len(reply["tokens"]),
               "response_sha256": fingerprint(reply), "finish": reply["finish"],
               "seconds": time.monotonic() - start}
    (target / "response.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def run(stage, build, role):
    mode, peer_role = ("baseline", "candidate") if role == "baseline" else (COMPARISON, "baseline")
    archive_key, arm, switch, token = MODES[mode]
    archive = stage.parent / archive_key / arm
    trace_build = Path((stage.parent / MODEL_TRACE / "build-path.txt").read_text().strip())
    workload = read_workload(trace_build / "fixtures/20261119-w4-holdout/holdout.json.gz")
    index, = (i for i, row in enumerate(workload.cases) if str(row.label) == "long-code-7836")
    historical = read_capture(archive / "rows/holdout.json", workload)["baseline"][index]
    case, ids = workload.cases[index], historical["prompt_tokens"]
    require(len(ids) == 7836 and ids[-1] == 25 and historical["tokens"][:8] == PREFIX + [token], "input drift")

    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    holder = int(os.environ["RILEY_GPU_HOLDER_PID"])
    ticks = os.environ["RILEY_GPU_HOLDER_START_TICKS"]
    require(gpu == GPU[role] and identity(holder) == ticks and
            {entry["pid"] for entry in gpu_processes(gpu)} == {holder}, "GPU lease drift")
    binary = build / "target/release/riley"
    binary_hash = (build / "build-result.sha256").read_text().split()[0]
    argv = read(archive / "receipt.json")["server_argv"]
    argv[0] = str(binary)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{probe.getsockname()[1]}"
    argv[argv.index("--bind") + 1] = address
    model = Path(argv[argv.index("--model") + 1])
    require(digest(binary) == binary_hash and digest(model / "riley-checkpoint.json") == CHECKPOINT,
            "binary or model drift")

    out, peer = stage / role, stage / peer_role
    (out / "trace").mkdir()
    env = dict(os.environ)
    for name in ("RILEY_W4_ATTENTION_O31", "RILEY_W4_DOWN_FP32", "RILEY_HOST_STEP_TRACE", "RILEY_TIMING"):
        env.pop(name, None)
    env["RILEY_W4_TRACE_DIR"] = str(out / "trace")
    env["RILEY_SHUTDOWN_METRICS_PATH"] = str(out / "shutdown.json")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64:" + env.get("LD_LIBRARY_PATH", "")
    if switch:
        env[switch] = "1"

    server = start_server(out, argv, env)
    try:
        ready = ready_server(server, address, gpu, holder, binary_hash)
        initial = respond(out, "calibration", address, case, ids)
        require(initial["response_sha256"] == fingerprint(historical), f"{role} historical output drift")
        own = capture(out / "calibration")
        (out / "calibrated").touch(exist_ok=False)
        deadline = time.monotonic() + 600
        while not (peer / "calibrated").is_file():
            require(server.poll() is None and not (peer / "worker.exit").exists(), "worker exited before calibration")
            if time.monotonic() >= deadline:
                raise TimeoutError("peer calibration")
            time.sleep(0.5)
        donor = capture(peer / "calibration")
        respond(out, "self", address, case, ids, own, swap=True)
        respond(out, "boundary", address, case, ids, donor)
        respond(out, "history", address, case, ids, donor, swap=True)
        server.send_signal(signal.SIGTERM)
        require(server.wait(timeout=60) == 0, "graceful shutdown failed")
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
    require(identity(holder) == ticks, "GPU lease identity changed")
    receipt = {"mode": mode, "gpu": gpu, "switch": switch, "server_exit": server.returncode,
               "binary_sha256": binary_hash, "model_sha256": CHECKPOINT, "ready": ready,
               "server_argv": argv, "input_token_count": len(ids),
               "input_ids_sha256": digest_ids(ids), "historical_response_sha256": fingerprint(historical),
               "historical_capture_sha256": digest(archive / "rows/holdout.json")}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def digest_ids(ids):
    from hashlib import sha256
    return sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def differences(first, second):
    require(len(first) == len(second) and len(first) % 2 == 0, "incompatible BF16 rows")
    changed = 0
    if first != second:
        changed = sum(a != b for a, b in zip(memoryview(first).cast("H"), memoryview(second).cast("H"), strict=True))
    return {"changed_words": changed, "words": len(first) // 2}


def verify_role(stage, role, gpu, lease, binary_hash, build):
    out = stage / role
    peer = stage / ("candidate" if role == "baseline" else "baseline")
    own = capture(out / "calibration")
    donor = capture(peer / "calibration")
    previous = stage.parent / PRIOR / COMPARISON / role
    receipt = read(out / "receipt.json")
    mode = "baseline" if role == "baseline" else COMPARISON
    archive_key, arm, switch, _ = MODES[mode]
    archive = stage.parent / archive_key / arm
    recorded = read(archive / "receipt.json")["server_argv"]
    argv = receipt["server_argv"]
    bind = recorded.index("--bind") + 1
    require(argv[0] == str(build / "target/release/riley") and argv[bind].startswith("127.0.0.1:")
            and argv[1:bind] == recorded[1:bind] and argv[bind + 1:] == recorded[bind + 1:], "arguments drift")
    require(receipt["mode"] == mode and receipt["gpu"] == GPU[role] and receipt["switch"] == switch
            and receipt["binary_sha256"] == binary_hash and receipt["model_sha256"] == CHECKPOINT
            and receipt["server_exit"] == 0 and receipt["ready"]["ready"] and receipt["ready"]["accepting"]
            and digest(archive / "rows/holdout.json") == receipt["historical_capture_sha256"]
            and receipt["input_token_count"] == 7836, "runtime receipt drift")
    shutdown = read(out / "shutdown.json")
    require(all(shutdown[k] == 0 for k in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and not any(shutdown["allocation"].values()), "resource reclamation failed")
    require(all(reservation(out / f"reservation-{side}.json", gpu) == lease for side in ("before", "after")),
            "GPU lease drift")
    all_results = {}
    cases = {}
    for name, source, prior, swap in (
        ("calibration", None, "calibration", False), ("self", own, "self", True),
        ("boundary", donor, "cross", False), ("history", donor, None, True),
    ):
        rows = capture(out / name, swapped=swap, boundary=source is not None)
        response = read(out / name / "response.json")
        require(response["first_eight"][:7] == PREFIX and response["generated_tokens"] == 256,
                f"wrong step response: {role}/{name}")
        if prior:
            require(response["response_sha256"] == read(previous / prior / "response.json")["response_sha256"]
                    and all(rows[key] == value for key, value in old_trace(previous / prior).items()),
                    f"previous controls drift: {role}/{name}")
        if source is not None:
            require(all(rows[k, "0"] == source[k, "0"] for k in BOUNDARY), f"layer-0 replacement failed: {role}/{name}")
        if swap:
            require(all(rows[test, "1"] == source[truth, "1"] for test, truth in zip(INJECTED, HISTORY, strict=True))
                    and all(rows[test, "1"] == rows[truth, "1"] for test, truth in zip(RESTORED, HISTORY, strict=True)),
                    f"KV replacement or restoration failed: {role}/{name}")
        if name == "self":
            require(all(rows[key] == value for key, value in own.items()), "self replacement changed captured bytes")
        chosen = response["first_eight"][-1]
        all_results[name] = {**response, "scores": scores(rows[LOGITS], chosen),
                             "capture_sha256": digest(out / name / "rows.bf16")}
        cases[name] = rows

    base, swapped = cases["boundary"], cases["history"]
    require(all(swapped[key, "1"] == base[key, "1"] for key in HISTORY + CURRENT), "initial layer-1 inputs drift")
    return {
        "input_ids_sha256": receipt["input_ids_sha256"], "trials": all_results,
        "boundary_vs_donor": {k: differences(base[k, "1"], donor[k, "1"]) for k in HISTORY + CURRENT + ("context",)},
        "history_vs_donor": {k: differences(swapped[k, "1"], donor[k, "1"]) for k in ("context", "attention_residual")},
        "history_vs_boundary": {k: differences(swapped[k, "1"], base[k, "1"]) for k in ("context", "attention_residual")},
        "logits_vs_donor_after_history": differences(swapped[LOGITS], donor[LOGITS]),
        "logits_vs_boundary_after_history": differences(swapped[LOGITS], base[LOGITS]),
    }


def verify(stage, build):
    binary_hash = (build / "build-result.sha256").read_text().split()[0]
    require(digest(build / "target/release/riley") == binary_hash
            and (stage / "checkpoint-verify.log").read_text().strip() == "checkpoint verify: verified",
            "build or model proof drift")
    baseline_leases = {str(i): reservation(stage / f"gpu{i}-before.json", i) for i in (6, 7)}
    require(all(reservation(stage / f"gpu{i}-after.json", int(i)) == lease for i, lease in baseline_leases.items()),
            "GPU lease after campaign drift")
    roles = {role: verify_role(stage, role, gpu, baseline_leases[str(gpu)], binary_hash, build)
             for role, gpu in (("baseline", 6), ("candidate", 7))}
    require(roles["baseline"]["input_ids_sha256"] == roles["candidate"]["input_ids_sha256"],
            "comparison used different tokenization")
    return {"status": "passed", "remote_stage": str(stage), "remote_build": str(build),
            "model_sha256": CHECKPOINT, "binary_sha256": binary_hash,
            "comparison": COMPARISON, "reservations": baseline_leases, "roles": roles}


def stop(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    action, path, binary_build, *options = sys.argv[1:]
    if action == "run":
        run(Path(path), Path(binary_build), *options)
    elif action == "verify" and not options:
        print(json.dumps(verify(Path(path), Path(binary_build)), indent=2, allow_nan=False))
    else:
        raise ValueError("specify run or verify")
