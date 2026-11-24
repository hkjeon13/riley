"""Partition each observed workflow latency difference before aggregating paired starts."""

import json
import math
import tarfile
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def measure(r):
    final = r["turns"][2]
    first, last = final["content_ns"][0], final["content_ns"][-1]
    n = final["usage"]["completion_tokens"]
    if not (
        r["ok"]
        and r["start_ns"] < final["sent_ns"] < first < last <= final["done_ns"]
        and n > 1
    ):
        raise ValueError("incomplete natural output")
    return {
        "upstream_ms": (final["sent_ns"] - r["start_ns"]) / 1e6,
        "final_ttft_ms": (first - final["sent_ns"]) / 1e6,
        "final_stream_ms": (last - first) / 1e6,
        "n": n,
        "tpot": (last - first) / 1e6 / (n - 1),
        "client_e2e_ms": (last - r["start_ns"]) / 1e6,
    }


def compare(a, b):
    if (a["fixture_sha256"], a["seed"], a["history_entries"]) != (
        b["fixture_sha256"],
        b["seed"],
        b["history_entries"],
    ):
        raise ValueError("unpaired tasks")
    x, y = measure(a), measure(b)
    delta = {
        k: x[k] - y[k]
        for k in ("upstream_ms", "final_ttft_ms", "final_stream_ms", "client_e2e_ms")
    }
    # Symmetric product difference is an accounting identity for each observed
    # stream, without assuming that either length could be changed in isolation.
    delta["stream_length_component_ms"] = (
        (x["n"] - y["n"]) * (x["tpot"] + y["tpot"]) / 2
    )
    delta["stream_tpot_component_ms"] = (
        (x["tpot"] - y["tpot"]) * (x["n"] + y["n"] - 2) / 2
    )
    for expected, value in [
        (
            delta["final_stream_ms"],
            delta["stream_length_component_ms"] + delta["stream_tpot_component_ms"],
        ),
        (
            delta["client_e2e_ms"],
            delta["upstream_ms"] + delta["final_ttft_ms"] + delta["final_stream_ms"],
        ),
    ]:
        if not math.isclose(expected, value, abs_tol=1e-7):
            raise ValueError("timestamps disagree")
    return {"riley": x, "vllm": y, "riley_minus_vllm": delta}


def run():
    rows = []
    with tarfile.open(HERE / "promotion-raw.tar.gz") as a:
        for i in range(1, 6):
            for history in (0, 200):
                arms = []
                for engine in ("wide", "vllm"):
                    data = json.load(
                        a.extractfile(
                            f"promotion/pair-{i}-{engine}/workflows-{history}.json"
                        )
                    )
                    records = {r["workflow_id"]: r for r in data}
                    if len(records) != 5:
                        raise ValueError("missing workflows")
                    arms.append(records)
                x, y = arms
                if x.keys() != y.keys():
                    raise ValueError("different workflow inventory")
                for name in x:
                    rows.append(
                        {
                            "pair": i,
                            "history": history,
                            "workflow": name,
                            **compare(x[name], y[name]),
                        }
                    )
    summary = {}
    for h in (0, 200):
        for name in ("cold", "immediate", "delayed", "reader", "long-tool-result"):
            subset = [
                r["riley_minus_vllm"]
                for r in rows
                if r["history"] == h and r["workflow"] == name
            ]
            summary[f"{h}-{name}"] = {
                k: {
                    "median": median(r[k] for r in subset),
                    "range": [min(r[k] for r in subset), max(r[k] for r in subset)],
                }
                for k in subset[0]
            }
    return {
        "paired_restarts": 5,
        "paired_workflows": len(rows),
        "summary": summary,
        "observations": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
