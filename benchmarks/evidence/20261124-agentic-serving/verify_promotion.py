"""Recompute the registered candidate gates from the completed GPU trials."""

import argparse
import json
from pathlib import Path

from verify_decode import compare


def read(path):
    return json.loads(path.read_text())


def publication(folder):
    pairs = []
    for i in range(1, 6):
        a, b = [folder / f"pair-{i}-{engine}" for engine in ("wide", "vllm")]
        for arm in (a, b):
            if not (arm / "lease-after.json").is_file():
                raise ValueError(f"incomplete launch: {arm.name}")
        comp = compare(a, b)
        comp["worst_ratio"] = max(
            max(r["tpot_ratio"], r["gap_ratio"]) for r in comp["cells"]
        )
        workflows = {}
        for arm in (a, b):
            stats = []
            for history in (0, 200):
                records = read(arm / f"workflows-{history}.json")
                if len(records) != 5:
                    raise ValueError("incomplete natural workflow cohort")
                valid = [r for r in records if r["ok"]]
                stats.append(
                    {
                        "history": history,
                        "valid": len(valid),
                        "submitted": 5,
                        "first_p95_ms": max(
                            (r["first_final_ms"] for r in valid), default=None
                        ),
                        "last_p95_ms": max(
                            (r["last_final_ms"] for r in valid), default=None
                        ),
                    }
                )
            workflows[arm.name] = stats
        comp["natural"] = workflows
        pairs.append(comp)
    bound = max(r["worst_ratio"] for r in pairs)
    return {
        "pairs": pairs,
        "worst_cell_median_upper_bound": bound,
        "strict_decode_screen": bound <= 1.0,
        "scope": "five paired starts on one device, independent-launch assumption needed for the nominal one-sided median bound",
    }


def protected(folder, history):
    rag = "rag-c8-s1-vllm-arrivals"
    short = "c32-vllm-arrivals"
    original = read(history)["gpus"]["7"]["cells"][rag]
    vllm = (
        original["control"]["ttft_p95_ms"]
        / original["frozen_vllm"]["control"]["ttft_p95_ratio"]
    )
    result = []
    for first, second in (("a", "b"), ("d", "c")):
        cells = {}
        for name in (rag, short):
            x, y = [
                read(folder / f"protected-{tag}" / f"{name}.json")
                for tag in (first, second)
            ]
            a, b = (j["statistics"] for j in (x, y))
            if len(x["measured"]) != len(y["measured"]) or [
                (r["input_sha256"], r["scheduled_write_ns"]) for r in x["measured"]
            ] != [(r["input_sha256"], r["scheduled_write_ns"]) for r in y["measured"]]:
                raise ValueError("input identities or arrival schedules differ")
            ratios = {
                k: b[k] / a[k]
                for k in ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "output_tok_s")
            }
            changed = sum(
                i["output_sha256"] != j["output_sha256"]
                for i, j in zip(x["measured"], y["measured"], strict=True)
            )
            valid = all(
                r["relative_write_error_abs_p95_ms"] <= 3 and r["ok"] == r["requests"]
                for r in (a, b)
            )
            cells[name] = {
                "requests": a["requests"],
                "control": a,
                "wide": b,
                "ratios": ratios,
                "changed_outputs": changed,
                "length_mismatches": [x["length_mismatches"], y["length_mismatches"]],
                "arrival_and_http_ok": valid,
            }
            if name == rag:
                cells[name]["frozen_vllm_p95_ms"] = vllm
                cells[name]["wide_vllm_ttft_p95_ratio"] = b["ttft_p95_ms"] / vllm
            else:
                cells[name]["short_thresholds_pass"] = (
                    ratios["ttft_p50_ms"] < 1.02
                    and ratios["tpot_p50_ms"] < 1.01
                    and ratios["output_tok_s"] > 0.98
                )
        result.append(cells)
    return {
        "paired_crossovers": result,
        "timing_gates_pass": all(
            r[short]["short_thresholds_pass"]
            and r[rag]["wide_vllm_ttft_p95_ratio"] <= 1.15
            and all(v["arrival_and_http_ok"] for v in r.values())
            for r in result
        ),
    }


def stats(work):
    target = work["target"]
    if len(target["turns"]) != 3:
        return {"ok": False, "error": target.get("error")}
    final = target["turns"][-1]
    prefix = work["prefix"]
    peer = work.get("peer")
    before = target["turns"][1]

    def cached(row):
        return (row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    return {
        "ok": target["ok"],
        "fixture": target["fixture_sha256"],
        "answer": final.get("answer"),
        "prompt_tokens": final["usage"]["prompt_tokens"],
        "final_cached": cached(final),
        "final_tokens": final["usage"]["completion_tokens"],
        "ready_to_first_ms": (final["content_ns"][0] - before["tool_ready_ns"]) / 1e6,
        "ready_to_last_ms": (final["content_ns"][-1] - before["tool_ready_ns"]) / 1e6,
        "workflow_first_ms": (final["content_ns"][0] - target["start_ns"]) / 1e6,
        "workflow_last_ms": (final["content_ns"][-1] - target["start_ns"]) / 1e6,
        "preview_uncached_tokens": prefix["usage"]["prompt_tokens"] - cached(prefix)
        if "usage" in prefix
        else None,
        "preview_completion_tokens": prefix.get("usage", {}).get("completion_tokens"),
        "preview_finished_before_result": prefix["done_ns"] <= before["tool_ready_ns"]
        if "done_ns" in prefix
        else None,
        "preview_error": prefix.get("error"),
        "peer_ok": peer["ok"] if peer else None,
        "peer_gap_p95_ms": peer.get("final_gap_p95_ms") if peer else None,
        "peer_overlap": peer["turns"][-1]["content_ns"][0]
        <= before["tool_wait_begin_ns"]
        < peer["turns"][-1]["content_ns"][-1]
        if peer
        else None,
    }


def prefix_trial(folder):
    comparisons = []
    for left, right in ((1, 2), (4, 3)):
        ref = read(folder / f"pair-{left}-wide" / "prefix-trial.json")
        trial = read(folder / f"pair-{right}-wide" / "prefix-trial.json")
        if len(ref) != 5 or len(trial) != 5:
            raise ValueError("incomplete prefix cohort")
        rows = []
        for off, on in zip(ref, trial, strict=True):
            a, b = stats(off), stats(on)
            if (
                off["enabled"]
                or not on["enabled"]
                or off["case"] != on["case"]
                or a.get("fixture") != b.get("fixture")
            ):
                raise ValueError("mismatched condition")
            equal = a.pop("answer", None) == b.pop("answer", None)
            rows.append({"case": off["case"], "off": a, "on": b, "answer_equal": equal})
        comparisons.append(rows)
    return {"paired_starts": comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["decode", "protected", "prefix"])
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    if args.kind == "protected":
        prior = (
            Path(__file__).resolve().parent.parent
            / "20261124-w3-prefill-cohort"
            / "gpu-verified.json"
        )
        report = protected(args.directory, prior)
    elif args.kind == "decode":
        report = publication(args.directory)
    else:
        report = prefix_trial(args.directory)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
