"""Verify mixed-fleet stream fidelity, resource equality and registered tail guards."""

import argparse
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

from coder_mixed_fleet import corpus


def read(p):
    return json.loads(p.read_text())


def percentile(x, q=0.95):
    if not x:
        return None
    ordered = sorted(x)
    return ordered[min(math.floor(len(ordered) * q), len(ordered) - 1)]


def expected(archive, prefix):
    with tarfile.open(archive) as t:
        return {
            (short, r["request_index"]): r
            for short, label in (
                (True, "c32-vllm-arrivals"),
                (False, "rag-c8-s1-vllm-arrivals"),
            )
            for r in json.load(t.extractfile(f"{prefix}/{label}.json"))["measured"]
        }


def inspect_arm(rows, reference, manifest):
    if len(rows) != 256 or len({(r["short"], r["index"]) for r in rows}) != 256:
        raise ValueError("incomplete cohort")
    classes = {}
    for ident, short in (("rag", False), ("short", True)):
        subset = [r for r in rows if r["short"] == short]
        good = []
        for r in subset:
            key = (short, r["index"])
            ref = reference[key]
            if (
                r["offset_s"] != manifest[key]
                or r.get("input_sha256") != ref["input_sha256"]
            ):
                raise ValueError("wire body or scheduled offset changed")
            if (
                r.get("status") == 200
                and not r.get("error")
                and all(
                    r[k] == ref[k]
                    for k in (
                        "output_sha256",
                        "prompt_tokens",
                        "completion_tokens",
                        "finish",
                    )
                )
            ):
                good.append(r)
        completed = [r for r in subset if r.get("status") == 200 and not r.get("error")]
        classes[ident] = {
            "n": len(subset),
            "completed": len(completed),
            "matches_reference": len(good),
            "replica_counts": dict(Counter(r["replica"] for r in subset)),
            "arrival_error_p95_ms": percentile(
                [
                    1000 * abs(r["sent_at"] - r["target_s"])
                    for r in subset
                    if "sent_at" in r
                ]
            ),
            "ttft_p95_ms": percentile([r["ttft"] * 1000 for r in completed]),
            "last_p95_ms": percentile([r["e2e"] * 1000 for r in completed]),
            "tpot_p50_ms": percentile([r["tpot"] * 1000 for r in completed], 0.5),
            "request_gap_p95_across_requests_p95_ms": percentile(
                [1000 * percentile(r["gaps"]) for r in completed if r["gaps"]]
            ),
        }
    completed = [r for r in rows if r.get("status") == 200 and not r.get("error")]
    valid = sum(c["matches_reference"] for c in classes.values())
    span = max(r["start"] + r["e2e"] for r in completed) - min(
        r["target_s"] for r in rows
    )
    short0 = min(r["sent_at"] for r in completed if r["short"])
    short1 = max(r["start"] + r["e2e"] for r in completed if r["short"])
    overlapped = sum(
        r["sent_at"] < short1 and r["start"] + r["e2e"] > short0
        for r in completed
        if not r["short"]
    )
    return {
        "classes": classes,
        "long_requests_overlapping_short_window": overlapped,
        "observed_seconds": span,
        "matched_completions_per_gpu_hour": valid / (2 * span / 3600),
    }


def audit_routes(rows, policy, short_home):
    ordered = sorted(rows, key=lambda r: r["start"])
    active_until = []
    checked = 0
    for index, r in enumerate(ordered):
        if policy == "round-robin":
            want = index % 2
        elif policy == "dedicated" or r["short"]:
            want = short_home if r["short"] else 1 - short_home
        elif any(stop >= r["start"] for stop in active_until):
            want = 1 - short_home
        elif not active_until:
            tie = index % 2
            want = min((tie, 1 - tie), key=r["outstanding"].__getitem__)
        else:
            # A final SSE event can arrive after its last visible content.
            # Only assert the decision when a short stream is observably live.
            want = None
        if want is not None:
            if want != r["replica"]:
                raise ValueError("registered routing decision differs")
            checked += 1
        if r["short"]:
            active_until.append(r["start"] + r["e2e"])
    return checked


def ratio(pair, group, metric):
    previous, current = pair
    return current["classes"][group][metric] / previous["classes"][group][metric]


def summarize(root, archive, baseline=False):
    kind = "vllm" if baseline else "riley"
    prefix = "gpu7/coder14-vllm" if baseline else "promotion/protected-a"
    refs = expected(archive, prefix)
    scheduled = corpus(read(root / "plan.json"))
    slots = {(short, idx): offset for offset, idx, body, short in scheduled}
    devices = []
    configs = []
    for gpu in (6, 7):
        d = root / f"gpu{gpu}"
        state = read(d / "lease-after.json")
        if (
            state["physical_index"] != gpu
            or state["phase"] != "idle"
            or state["other_processes"]
        ):
            raise ValueError("lease not released")
        devices.append(state["gpu"])
        if baseline:
            cmd = read(d / "args.json")
            configs.append(
                [v for i, v in enumerate(cmd) if i != cmd.index("--port") + 1]
            )
            if (d / "version.txt").read_text().strip() != "0.27.1+cu129" or not (
                d / "exit-code"
            ).is_file():
                raise ValueError("vLLM lifecycle incomplete")
        else:
            configs.append(
                ((d / "args").read_text(), (d / "binary.sha256").read_text())
            )
            m = read(d / "shutdown.json")
            if any(m["allocation"].values()) or any(
                m[k]
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            ):
                raise ValueError("resource leak")
    if (
        len(set(devices)) != 2
        or configs[0] != configs[1]
        or not (root / "released").exists()
    ):
        raise ValueError("unmatched replicas")
    order = read(root / "coder-fleet-registration.json")["arms"]
    arms = []
    for i, name in enumerate(order):
        rows = read(root / f"arm-{i}-{name}.json")
        arm = inspect_arm(rows, refs, slots)
        arm["independently_resolved_routing_decisions"] = audit_routes(
            rows, name, int(i >= len(order) // 2)
        )
        arms.append(arm)
    result = {"engine": kind, "devices": devices, "order": order, "arms": arms}
    if "dedicated" in order and "elastic" in order:
        candidates = {}
        for policy in ("dedicated", "elastic"):
            comparisons = []
            for i, label in enumerate(order):
                if label != policy:
                    continue
                ref = arms[0 if i < 3 else 5]
                pair = [ref, arms[i]]
                arrival = all(
                    c["arrival_error_p95_ms"] <= 5
                    for a in pair
                    for c in a["classes"].values()
                )
                fidelity = all(
                    c["matches_reference"] == c["n"]
                    for a in pair
                    for c in a["classes"].values()
                )
                gap = max(
                    ratio(pair, c, "request_gap_p95_across_requests_p95_ms")
                    for c in ("rag", "short")
                )
                short_last = ratio(pair, "short", "last_p95_ms")
                rag_first = ratio(pair, "rag", "ttft_p95_ms")
                comparisons.append(
                    {
                        "arrivals_ok": arrival,
                        "exact_responses": fidelity,
                        "short_last_ratio": short_last,
                        "rag_first_ratio": rag_first,
                        "worst_class_gap_ratio": gap,
                        "pass": arrival
                        and fidelity
                        and all(
                            a["long_requests_overlapping_short_window"] for a in pair
                        )
                        and short_last <= 0.9
                        and rag_first <= 1.05
                        and gap <= 1.05,
                    }
                )
            candidates[policy] = comparisons
        result["candidate_comparisons"] = candidates
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("original", type=Path)
    p.add_argument("--vllm", action="store_true")
    a = p.parse_args()
    print(json.dumps(summarize(a.root, a.original, a.vllm), indent=2))
