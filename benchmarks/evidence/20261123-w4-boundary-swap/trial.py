"""Swap two BF16 rows in a single long-code decode step and check all controls."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time
from urllib.request import urlopen

from run_case import CHECKPOINT, MODES, PREFIX, digest, gpu_processes, identity, read, require
from long_accuracy import complete, read_capture, read_workload


PRIOR = "w4-long-code-gpu-20261122.QAbtnxKV"
GPU = {
    "baseline": "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7",
    "candidate": "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0",
}
BOUNDARIES = {
    "attention31": ("attention", (("attention_residual", "0"), ("post_attention_norm", "0"))),
    "fp32_down21": ("down", (("layer_residual", "0"), ("next_input_norm", "0"))),
}
ROW_BYTES = 3584 * 2
FINAL = ("final_norm", "-")
LOGITS = ("logits", "-")
OLD_KEYS = [("embedding", "-")]
OLD_KEYS += [(key, str(layer)) for layer in range(28) for key in ("attention_residual", "layer_residual")]
OLD_KEYS += [FINAL, LOGITS]
EXTRA = {("post_attention_norm", "0"), ("next_input_norm", "0")}


def trace(path):
    header, *index = (path / "index.tsv").read_text().splitlines()
    require(header == "# 7842 198 0 1 1", f"wrong decode step in {path}")
    raw = (path / "rows.bf16").read_bytes()
    rows = {}
    end = 0
    for line in index:
        name, layer, offset, length = line.split("\t")
        key = (name, layer)
        offset, length = int(offset), int(length)
        require(key not in rows and offset == end and length == (304128 if key == LOGITS else ROW_BYTES),
                f"invalid trace layout in {path}")
        end += length
        rows[key] = raw[offset:end]
    require(end == len(raw) and [key for key in rows if key in OLD_KEYS] == OLD_KEYS, f"missing trace rows at {path}")
    return rows


def fingerprint(response):
    values = {key: response[key] for key in ("tokens", "text", "finish", "usage")}
    return sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def start_server(out, argv, env):
    with (out / "server.log").open("xb") as log:
        return subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def ready_server(server, address, gpu, holder_pid, binary_hash):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        require(server.poll() is None, "server exited during startup")
        try:
            with urlopen(f"http://{address}/readyz", timeout=1) as response:
                ready = json.load(response)
            if ready["ready"] and ready["accepting"]:
                require({row["pid"] for row in gpu_processes(gpu)} == {holder_pid, server.pid}
                        and digest(Path(f"/proc/{server.pid}/exe")) == binary_hash, "active runtime differs")
                return ready
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("server startup")


def respond(out, trial, address, case, ids, donor=None):
    pending = out / "trace" / "decode-7"
    pending.mkdir()
    if donor:
        filename, payload = donor
        (pending / f"{filename}.bf16").write_bytes(payload)
    start = time.monotonic()
    reply = complete(address, "m", case.prompt, diagnostic=True, expected=ids, limit=case.limit)
    elapsed = time.monotonic() - start
    require(reply["tokens"][:7] == PREFIX, f"{trial} changed the shared prefix")
    target = out / trial
    pending.rename(target)
    receipt = {"first_eight": reply["tokens"][:8], "response_sha256": fingerprint(reply),
               "generated_tokens": len(reply["tokens"]), "finish": reply["finish"], "seconds": elapsed}
    (target / "response.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def run(stage, build, comparison, role):
    mode, peer_role = ("baseline", "candidate") if role == "baseline" else (comparison, "baseline")
    archive_key, arm, switch, token = MODES[mode]
    archive = stage.parent / archive_key / arm
    workload = read_workload(stage.parent / (stage.parent / PRIOR / "build-path.txt").read_text().strip()
                             / "fixtures/20261119-w4-holdout/holdout.json.gz")
    historical = read_capture(archive / "rows/holdout.json", workload)
    index, = [i for i, row in enumerate(workload.cases) if str(row.label) == "long-code-7836"]
    case, expected = workload.cases[index], historical["baseline"][index]
    ids = expected["prompt_tokens"]
    require(len(ids) == 7836 and ids[-1] == 25 and expected["tokens"][:8] == PREFIX + [token], "historical tokens differ")

    holder_pid = int(os.environ["RILEY_GPU_HOLDER_PID"])
    ticks = os.environ["RILEY_GPU_HOLDER_START_TICKS"]
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    require(gpu == GPU[role] and identity(holder_pid) == ticks, "GPU lease differs")
    require({row["pid"] for row in gpu_processes(gpu)} == {holder_pid}, "foreign GPU process present")

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
    out = stage / comparison / role
    peer = out.parent / peer_role
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
        ready = ready_server(server, address, gpu, holder_pid, binary_hash)
        calibration = respond(out, "calibration", address, case, ids)
        require(calibration["response_sha256"] == fingerprint(expected), "calibration changed historical output")
        own = trace(out / "calibration")
        previous = trace(stage.parent / PRIOR / mode / "trace" / "decode-7")
        require(set(own) == set(OLD_KEYS) | EXTRA and all(own[key] == previous[key] for key in OLD_KEYS),
                "calibration did not byte-match previously archived trace")
        (out / "calibrated").touch(exist_ok=False)
        deadline = time.monotonic() + 600
        while not (peer / "calibrated").is_file():
            require(server.poll() is None and not (peer / "worker.exit").exists(), "worker exited before calibration")
            if time.monotonic() >= deadline:
                raise TimeoutError("peer calibration")
            time.sleep(0.5)
        other = trace(peer / "calibration")
        filename, keys = BOUNDARIES[comparison]
        respond(out, "self", address, case, ids, (filename, b"".join(own[key] for key in keys)))
        respond(out, "cross", address, case, ids, (filename, b"".join(other[key] for key in keys)))
        respond(out, "final", address, case, ids, ("final", other[FINAL]))
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
    require(identity(holder_pid) == ticks, "holder changed during trial")
    (out / "receipt.json").write_text(json.dumps({
        "mode": mode, "gpu": gpu, "server_exit": server.returncode, "ready": ready, "switch": switch,
        "server_argv": argv, "binary_sha256": binary_hash, "model_sha256": CHECKPOINT,
        "historical_response_sha256": fingerprint(expected),
        "historical_capture_sha256": digest(archive / "rows/holdout.json"),
    }, indent=2) + "\n")


def scores(payload, chosen):
    values = [struct.unpack("<f", struct.pack("<I", word << 16))[0] for (word,) in struct.iter_unpack("<H", payload)]
    require(all(map(math.isfinite, values)) and max(values) == values[chosen], "greedy ID not on the maximum")
    best = sorted(values, reverse=True)[:2]
    return {"262": values[262], "16019": values[16019], "top_two_gap": best[0] - best[1]}


def delta(first, second):
    differences = sum(a != b for a, b in zip(struct.iter_unpack("<H", first), struct.iter_unpack("<H", second), strict=True))
    return {"changed_words": differences, "words": len(first) // 2}


def reservation(path, gpu):
    status = read(path)
    require(status["phase"] == "idle" and status["physical_index"] == gpu and not status["other_processes"]
            and status["allocation_bytes"] == status["torch_reserved_bytes"] == 20 * 2**30, f"GPU lease changed: {path}")
    return {"pid": status["pid"], "start_ticks": status["start_ticks"], "gib": 20}


def verify(stage, build):
    binary_hash = (build / "build-result.sha256").read_text().split()[0]
    require(digest(build / "target/release/riley") == binary_hash and
            (stage / "checkpoint-verify.log").read_text().strip() == "checkpoint verify: verified", "build or model proof changed")
    leases = {str(gpu): reservation(stage / f"gpu{gpu}-before.json", gpu) for gpu in (6, 7)}
    require(all(reservation(stage / f"gpu{gpu}-after.json", gpu) == leases[str(gpu)] for gpu in (6, 7)), "GPU holder changed")
    comparisons = {}
    for comparison, (filename, keys) in BOUNDARIES.items():
        comparisons[comparison] = {}
        for role, gpu_number in (("baseline", 6), ("candidate", 7)):
            out = stage / comparison / role
            peer = out.parent / ("candidate" if role == "baseline" else "baseline")
            own, donor = trace(out / "calibration"), trace(peer / "calibration")
            recorded = read(out / "receipt.json")
            mode = "baseline" if role == "baseline" else comparison
            archive_dir, arm, switch, token = MODES[mode]
            archive = stage.parent / archive_dir / arm
            expected_argv = read(archive / "receipt.json")["server_argv"]
            actual_argv = recorded["server_argv"]
            bind = actual_argv.index("--bind") + 1
            require(actual_argv[0] == str(build / "target/release/riley") and actual_argv[bind].startswith("127.0.0.1:")
                    and actual_argv[1:bind] == expected_argv[1:bind] and actual_argv[bind+1:] == expected_argv[bind+1:],
                    "server arguments changed")
            require(recorded["mode"] == mode and recorded["gpu"] == GPU[role] and recorded["switch"] == switch
                    and recorded["binary_sha256"] == binary_hash and recorded["model_sha256"] == CHECKPOINT
                    and recorded["server_exit"] == 0 and recorded["ready"]["ready"] and recorded["ready"]["accepting"]
                    and digest(archive / "rows/holdout.json") == recorded["historical_capture_sha256"], "runtime receipt changed")
            shutdown = read(out / "shutdown.json")
            require(all(shutdown[k] == 0 for k in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                    and shutdown["allocation"] and not any(shutdown["allocation"].values()), "resource reclamation failed")
            require(all(reservation(out / f"reservation-{side}.json", gpu_number) == leases[str(gpu_number)]
                        for side in ("before", "after")), "reservation changed")
            old = trace(stage.parent / PRIOR / mode / "trace/decode-7")
            expected_response = fingerprint(read(archive / "rows/holdout.json")["baseline"][9])
            require(set(own) == set(OLD_KEYS) | EXTRA and all(own[key] == old[key] for key in OLD_KEYS), "archive trace changed")
            results = {}
            for trial, location, source, replaced in (
                ("calibration", None, own, ()), ("self", filename, own, keys),
                ("cross", filename, donor, keys), ("final", "final", donor, (FINAL,)),
            ):
                rows = trace(out / trial)
                info = read(out / trial / "response.json")
                chosen = info["first_eight"][7]
                require(info["first_eight"][:7] == PREFIX, "common prefix changed")
                before = {(key[0] + "_before", key[1]): key for key in replaced}
                require(set(rows) == set(own) | set(before), "unexpected trace keys")
                require(all(rows[label] == own[key] for label, key in before.items()), "recipient row drift before replacement")
                require(all(rows[key] == source[key] for key in replaced), "donor did not reach the device")
                if replaced:
                    require((out / trial / f"{location}.bf16").read_bytes() == b"".join(source[key] for key in replaced), "donor payload drift")
                    first = next(key for key in rows if key in replaced)
                    earlier = list(rows)[:list(rows).index(first)]
                    require(all(rows[key] == own[key] for key in earlier if key in own), "trace changed upstream of the intervention")
                if trial in ("calibration", "self"):
                    require(all(rows[key] == own[key] for key in own) and chosen == token
                            and info["response_sha256"] == recorded["historical_response_sha256"] == expected_response,
                            "historical response or trace changed in a control")
                if trial == "final":
                    require(chosen == (16019 if role == "baseline" else 262) and rows[LOGITS] == donor[LOGITS],
                            "final norm control did not reproduce donor logits")
                results[trial] = {**info, "scores": scores(rows[LOGITS], chosen),
                                  "logits_vs_recipient": delta(own[LOGITS], rows[LOGITS]),
                                  "logits_vs_donor": delta(donor[LOGITS], rows[LOGITS]),
                                  "trace_sha256": digest(out / trial / "rows.bf16")}
            comparisons[comparison][role] = {
                "boundary": [key for key, _ in keys], "row_differences": {key: delta(own[key, layer], donor[key, layer]) for key, layer in keys},
                "trials": results,
            }
    return {"status": "passed", "remote_stage": str(stage), "remote_build": str(build),
            "model_sha256": CHECKPOINT, "binary_sha256": binary_hash, "reservations": leases, "comparisons": comparisons}


if __name__ == "__main__":
    def stop(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    action, path, binary_build, *options = sys.argv[1:]
    if action == "run":
        run(Path(path), Path(binary_build), *options)
    elif action == "verify" and not options:
        print(json.dumps(verify(Path(path), Path(binary_build)), indent=2, allow_nan=False))
    else:
        raise ValueError("specify run or verify")
