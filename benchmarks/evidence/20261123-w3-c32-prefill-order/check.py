"""Compare a single-variable prefill-only threshold against the existing W3 rules."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

COUNTS = {"prime": 4, "rag": 64, "short": 96}
METRICS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "output_tok_s")
IDENTITY = ("input_sha256", "prompt_ids_sha256", "token_ids_sha256", "output_sha256",
            "finish", "prompt_tokens", "completion_tokens", "cached_tokens")
SCOPE = ("source_revision", "source_archive_sha256", "binary_sha256", "build_result_sha256",
         "model_sha256", "image_id", "gpu_uuid", "http_workers", "steady_budget", "server_arguments")


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm(folder, selected):
    r = read(folder / "receipt.json")
    require(r["trace_mode"] == "control" and r["stage_trace"] is False
            and r["short_burst_decode_limit"] == selected and r["client_exit"] == r["shutdown_exit"] == 0,
            "different execution mode or unfinished arm")
    for name, expected in r["artifacts_sha256"].items():
        path = folder / name
        require(path.parent == folder and sha256(path.read_bytes()).hexdigest() == expected, "changed arm artifact")
    data = read(folder / "workload.json")
    shutdown = read(folder / "shutdown.json")
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and not any(shutdown["allocation"].values()), "resources remained after shutdown")
    return r, data


def percent(before, after):
    require(all(isfinite(value) and value > 0 for value in (before, after)), "missing or invalid comparison metric")
    return (after / before - 1) * 100


def compare(root):
    results = {}
    for gpu in (6, 7):
        a, b = (arm(root / f"private-short-d{mode}-control-gpu{gpu}", limit)
                for mode, limit in (("0", None), ("prefill-only", 0)))
        require(all(a[0][key] == b[0][key] for key in SCOPE), "comparison serving scope differs")
        proof = {}
        for cell, count in COUNTS.items():
            old, new = a[1][cell], b[1][cell]
            require(all(len(arm["rows"]) == arm["statistics"]["ok"] == count
                        and all(row["status"] == 200 and not row["error"] for row in arm["rows"])
                        for arm in (old, new)), "incomplete measured workload")
            match = sum(all(left[key] == right[key] for key in IDENTITY)
                        for left, right in zip(old["rows"], new["rows"], strict=True))
            proof[cell] = {"requests": count, "matched": match, "change_percent": {
                metric: percent(old["statistics"][metric], new["statistics"][metric]) for metric in METRICS}}
        short, rag = (proof[cell]["change_percent"] for cell in ("short", "rag"))
        proof["checks"] = {
            "exact_outputs": all(proof[cell]["matched"] == count for cell, count in COUNTS.items()),
            "short_ttft_p50": short["ttft_p50_ms"] < 2,
            "short_ttft_p95": short["ttft_p95_ms"] < 2,
            "short_decode_tpot": short["tpot_p50_ms"] < 1,
            "short_throughput": short["output_tok_s"] > -2,
            "rag_ttft_p95_improves": rag["ttft_p95_ms"] < 0,
            "rag_decode_tpot": rag["tpot_p50_ms"] < 1,
        }
        pairs = [row[1]["short"]["rows"] for row in (a, b)]
        origins = [min(item["sent_at"] for item in rows) for rows in pairs]
        errors = sorted(1000 * abs((right["sent_at"] - origins[1]) - (left["sent_at"] - origins[0]))
                        for left, right in zip(*pairs, strict=True))
        proof["paired_write_error_p95_ms"] = errors[int(.95 * len(errors))]
        require(proof["paired_write_error_p95_ms"] <= 5, "replayed arrival times differ")
        before, after = (read(root / f"gpu{gpu}-{name}.json") for name in ("before", "after"))
        require(all(receipt["status"] == "running" and receipt["phase"] == "idle"
                    and receipt["gpu"] == a[0]["gpu_uuid"] and receipt["allocation_bytes"] == 20 * 2**30
                    and not receipt["other_processes"] for receipt in (before, after))
                and all(before[key] == after[key] for key in ("pid", "start_ticks")), "GPU reservation was not restored")
        proof["scope"] = {key: a[0][key] for key in SCOPE}
        results[str(gpu)] = proof
    return {"status": "verified", "remote_evidence": str(root), "gpus": results}


if __name__ == "__main__":
    print(json.dumps(compare(Path(sys.argv[1])), indent=2, allow_nan=False))
