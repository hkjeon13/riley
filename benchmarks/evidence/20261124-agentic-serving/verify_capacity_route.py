"""Validate each registered admission decision using actual visible tokens and stream receipts."""

import argparse
import collections
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    pair = root / "sample"
    registration = read(root / "capacity-route-registration.json")
    data = read(pair / "capacity-route.json")
    if [[r["active"], r["policy"]] for r in data] != registration["order"]:
        raise ValueError("registered order differs")
    trials = []
    for r in data:
        a, b = r["warm_receipt"], r["decision_receipt"]
        route = b["route"]
        if (
            not all(
                x.get("done_ns")
                and x["status"] == 200
                and x["usage"]["completion_tokens"] == 1
                for x in (a, b)
            )
            or a["input_sha256"] != b["input_sha256"]
            or a["usage"]["prompt_tokens"] != b["usage"]["prompt_tokens"]
        ):
            raise ValueError("warm and continuation prompt/token lengths differ")
        trials.append(
            {
                "trial": r["trial"],
                "policy": r["policy"],
                "occupancy": r["active"],
                "actual_overlap": r["peers_active_at_send"],
                "load_at_decision": route["active"],
                "eligible_estimate": route["eligible_estimate"],
                "scores": route["score_ms"],
                "selected_replica": route["replica"],
                "cache_tokens": (b["usage"].get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                ),
                "prompt_tokens": b["usage"]["prompt_tokens"],
                "first_ms": (b["content_ns"][0] - b["sent_ns"]) / 1e6,
            }
        )
    manifest = read(pair / "inventory-replay.json")
    waves = []
    for wave in manifest["waves"]:
        requests = [t for w in wave["records"] for t in w["turns"]]
        waves.append(
            {
                "submitted": wave["submitted"],
                "correct": wave["correct"],
                "slo_completed": wave["slo_completed"],
                "statuses": dict(
                    collections.Counter(str(t.get("status")) for t in requests)
                ),
                "nodes_used": sorted({t["route"]["replica"] for t in requests}),
            }
        )
    for idx in (6, 7):
        d = pair / f"gpu{idx}"
        if read(d / "lease-after.json")["phase"] != "idle" or any(
            read(d / "shutdown.json")["allocation"].values()
        ):
            raise ValueError("GPU not released")
    return {
        "reported_topology": manifest["reported_topology"],
        "inventory_waves": waves,
        "capacity_trials": trials,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(summarize(p.parse_args().root), indent=2))
