"""Validate paired Riley traces and summarize the first measured BF16 differences."""

from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import struct
import sys


MODES = ("baseline", "attention31", "fp32_down21")
ARCHIVES = {"baseline": ("w4-fp32-down-serving-20261120.Saj3XXSY", "baseline"),
            "attention31": ("w4-o31-serving-20261120.7W3nr5JH", "candidate"),
            "fp32_down21": ("w4-fp32-down-serving-20261120.Saj3XXSY", "candidate")}
SWITCHES = {"baseline": None, "attention31": "RILEY_W4_ATTENTION_O31", "fp32_down21": "RILEY_W4_DOWN_FP32"}
TARGETS = {"prompt-final": (7835, 25, 576), "decode-7": (7842, 198, None)}
PREFIX = [576, 2701, 2038, 1447, 73594, 12669, 198]
GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
ROWS = [("embedding", "-", 3584)]
ROWS += [(key, str(layer), 3584) for layer in range(28) for key in ("attention_residual", "layer_residual")]
ROWS += [("final_norm", "-", 3584), ("logits", "-", 152064)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def read_trace(path, target):
    header, *index = (path / "index.tsv").read_text().splitlines()
    marker, *numbers = header.split()
    position, token, row, active, dense = map(int, numbers)
    require(marker == "#" and (position, token) == TARGETS[target][:2]
            and 0 <= row < active <= dense, f"trace identity changed: {target}")
    if target == "decode-7":
        require((row, active, dense) == (0, 1, 1), "expected one native decode row")
    raw = (path / "rows.bf16").read_bytes()
    offset = 0
    rows = {}
    for (key, layer, width), line in zip(ROWS, index, strict=True):
        size = width * 2
        require(line.split("\t") == [key, layer, str(offset), str(size)], "trace layout changed")
        rows[key, layer] = raw[offset:offset + size]
        offset += size
    require(len(raw) == offset, "trace extent differs")
    return {"position": position, "input_id": token, "row": row, "active_rows": active, "dense_rows": dense,
            "sha256": sha256(raw).hexdigest()}, rows


def bf16(raw):
    return [struct.unpack("<f", struct.pack("<I", word << 16))[0] for (word,) in struct.iter_unpack("<H", raw)]


def compare(reference, candidate):
    comparisons = []
    for (name, layer, width) in ROWS:
        key = name, layer
        left, right = reference[key], candidate[key]
        words = sum(a != b for a, b in zip(struct.iter_unpack("<H", left), struct.iter_unpack("<H", right), strict=True))
        if words:
            differences = [a - b for a, b in zip(bf16(left), bf16(right), strict=True)]
            require(all(map(math.isfinite, differences)), "nonfinite difference")
            maximum = max(map(abs, differences))
            rms = math.sqrt(math.fsum(x * x for x in differences) / width)
        else:
            maximum = rms = 0
        comparisons.append({"stage": name, "layer": None if layer == "-" else int(layer), "width": width,
                            "changed_words": words, "max_abs": maximum, "rms": rms})
    return {"first_difference": next((row for row in comparisons if row["changed_words"]), None), "rows": comparisons}


def score_row(raw, id):
    values = bf16(raw)
    require(all(map(math.isfinite, values)), "nonfinite logits")
    maximum = max(values)
    top = sorted(range(len(values)), key=lambda i: (-values[i], i))[:5]
    require(values[id] == maximum, f"archived selected ID {id} is below the traced best score")
    return {"selected_id": id, "top": [{"id": i, "score": values[i]} for i in top],
            "selected_pair": {str(i): values[i] for i in (262, 16019)},
            "top_two_gap": maximum - values[top[1]]}


def reservation(path, index):
    row = read(path)
    require(row["phase"] == "idle" and row["physical_index"] == index and not row["other_processes"]
            and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30, f"GPU {index} lease changed")
    return {"pid": row["pid"], "start_ticks": row["start_ticks"], "gib": 20}


def prepared_plans(log, mode):
    signatures = {"attention31": ("W4_O31_PLAN ", {"id": "31", "tile": "15", "stages": "18", "split_k": "1", "reduction": "0"}),
                  "fp32_down21": ("W4_DOWN_FP32_PLAN ", {"id": "21", "tile": "18", "stages": "18", "split_k": "3", "reduction": "2"})}
    found = {}
    for name, (prefix, signature) in signatures.items():
        hits = set()
        for line in log.splitlines():
            if not line.startswith(prefix):
                continue
            row = dict(item.split("=", 1) for item in line[len(prefix):].split())
            m = int(row["m"])
            require(all(row[k] == expected for k, expected in signature.items())
                    and int(row["workspace"]) == (43008 * m if name == "fp32_down21" else 0), "prepared plan changed")
            hits.add(m)
        if hits:
            found[name] = sorted(hits)
    expected = {mode: list(BUCKETS)} if mode != "baseline" else {}
    require(found == expected, "prepared plans do not match the requested arm")
    return found


def verify(stage):
    evidence = stage.parent
    build = Path((stage / "build-path.txt").read_text().strip())
    baseline_hash = (build / "build-result.sha256").read_text().split()[0]
    require(digest(build / "target/release/riley") == baseline_hash, "binary changed")
    require((stage / "checkpoint-verify.log").read_text().strip() == "checkpoint verify: verified", "checkpoint integrity proof missing")
    reserved = {str(index): reservation(stage / f"gpu{index}-before.json", index) for index in (6, 7)}
    require(all(reservation(stage / f"gpu{index}-after.json", index) == reserved[str(index)] for index in (6, 7)),
            "GPU reservation changed")
    all_traces, cases = {}, {}
    for mode in MODES:
        directory = stage / mode
        receipt = read(directory / "receipt.json")
        model, arm = ARCHIVES[mode]
        archive = evidence / model / arm
        expected_options = read(archive / "receipt.json")["server_argv"]
        options = receipt["server_argv"]
        require(options[options.index("--bind") + 1].startswith("127.0.0.1:")
                and options[1:options.index("--bind") + 1] == expected_options[1:expected_options.index("--bind") + 1]
                and options[options.index("--bind") + 2:] == expected_options[expected_options.index("--bind") + 2:],
                "server options changed")
        expected_id = 262 if mode == "baseline" else 16019
        require(receipt["mode"] == mode and receipt["gpu"] == GPU and receipt["binary_sha256"] == baseline_hash
                and receipt["model_sha256"] == CHECKPOINT and receipt["server_exit"] == 0 and receipt["exact_archived_output"]
                and receipt["switch"] == SWITCHES[mode] and receipt["prompt_tokens"] == 7836
                and receipt["first_eight"] == PREFIX + [expected_id]
                and digest(archive / "rows/holdout.json") == receipt["historical_output_sha256"], "case receipt changed")
        old = read(archive / "rows/holdout.json")["baseline"][9]
        expected_hash = sha256(json.dumps({key: old[key] for key in ("tokens","text","finish","usage")},
                                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(expected_hash == receipt["response_sha256"], "full archived output hash differs")
        require(all(reservation(directory / f"reservation-{where}.json", 7) == reserved["7"]
                    for where in ("before", "after")), "GPU 7 arm changed the holder")
        shutdown = read(directory / "shutdown.json")
        require(all(shutdown[k] == 0 for k in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                and shutdown["allocation"] and not any(shutdown["allocation"].values()), "server did not reclaim resources")
        traces, captured = {}, {}
        for target in TARGETS:
            desc, rows = read_trace(directory / "trace" / target, target)
            traces[target] = {**desc, "logits": score_row(rows["logits", "-"], TARGETS[target][2] or expected_id)}
            captured[target] = rows
        cases[mode] = {"first_eight": receipt["first_eight"], "full_response_sha256": expected_hash,
                       "historical_output_sha256": receipt["historical_output_sha256"],
                       "prepared_plans": prepared_plans((directory / "server.log").read_text(), mode), "traces": traces}
        all_traces[mode] = captured
    pairs = {"baseline_vs_attention31": ("baseline", "attention31"), "baseline_vs_fp32_down21": ("baseline", "fp32_down21"),
             "attention31_vs_fp32_down21": ("attention31", "fp32_down21")}
    return {"status": "passed", "remote_stage": str(stage), "remote_build": str(build),
            "binary_sha256": baseline_hash, "model_sha256": CHECKPOINT, "result_sha256": digest(stage / "result.sha256"),
            "reservations": reserved, "cases": cases,
            "comparisons": {name: {label: compare(all_traces[before][label], all_traces[after][label]) for label in TARGETS}
                            for name, (before, after) in pairs.items()}}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
