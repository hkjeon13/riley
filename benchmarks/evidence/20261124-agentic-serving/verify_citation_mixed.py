"""Reconstruct both tools and every final request; score paired correctness, tails and GPU cost."""

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

from agent_workflow import answer_format, fixture, percentile
from multi_router_trial import with_layout
from verify_agent_citations import with_citations
from verify_defer_mixed import check_work, tails


def read(path):
    return json.loads(path.read_text())


def check_placement(work):
    _, _, docs = fixture(work["seed"], work["document_pairs"])
    text = work["turns"][-1]["answer"].replace("*", "")
    before, after = [d["p95_ms"] for d in docs]
    header = f"Before: {before} ms. After: {after} ms. Reduction: {before - after} ms."
    # The first numeral is an opening, not evidence of citation placement in prose.
    # A document without a paragraph break has no separately verifiable explanation here.
    parts = re.split(r"\n\s*\n", text.strip(), maxsplit=1)

    def placed(section):
        return all(
            re.search(
                rf"\b{doc['p95_ms']}\s*(?:ms|milliseconds)\s*\(\s*{re.escape(doc['id'])}\s*\)",
                section,
                re.IGNORECASE,
            )
            for doc in docs
        )

    explanation = parts[1] if len(parts) == 2 else ""
    explanation = re.sub(r"^\s*Explanation:\s*", "", explanation, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9]+(?:[-’'][A-Za-z0-9]+)*", explanation)
    exact = text.lstrip().startswith(header)
    assert answer_format(work["turns"][-1]["answer"], docs) == {
        "exact_numeric_opening": exact,
        "explanation_words": len(words),
    }
    return {
        "numeric_opening": exact,
        "explanation_90_words": len(words) == 90,
        "anywhere": placed(text),
        "in_separate_explanation": bool(len(parts) == 2 and placed(parts[1])),
    }


def verify(root):
    layout = (root / "citation-layout-registration.json").exists()
    candidate = "layout" if layout else "explicit"
    transform = with_layout if layout else with_citations
    rows = {"original": {}, candidate: {}}
    seconds = Counter()
    blocks = []
    for block in (0, 1):
        pair = root / f"run-{block}"
        allocation = 0
        for gpu in (6, 7):
            path = pair / f"gpu{gpu}"
            digest = (path / "binary.sha256").read_text().split()[0]
            assert (
                digest
                == "506b3e2519dd3c3dbb35c05f966b0fab339ae3326c5ecf21b0e8c118c1a015d6"
            )
            assert read(path / "lease-after.json")["phase"] == "idle"
            assert all(
                read(path / "shutdown.json")[k] == 0
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            )
            receipt = read(pair / f"allocation-{gpu}.json")
            assert receipt["returncode"] == 0
            allocation += receipt["elapsed_s"]
        raw = read(pair / "multi-router.json")
        assert len(raw) == 8
        good = 0
        for wave, group in enumerate(raw):
            assert (group["block"], group["wave"], group["policy"]) == (
                block,
                wave,
                "round-robin",
            )
            mode = candidate if (block + wave) % 2 else "original"
            clients = group["clients"]
            assert len(clients) == len({c["pid"] for c in clients}) == 2
            end = 0
            for j, c in enumerate(clients):
                base = (761 if wave < 6 else 24761 - 240) + 40 * wave + 20 * j
                assert len(c["records"]) == 4
                for i, w in enumerate(c["records"]):
                    seed = base + i
                    assert w["seed"] == seed and seed not in rows[mode]
                    assert w["document_pairs"] == (96 if i % 2 else 2)
                    metric = check_work(
                        w,
                        "round-robin",
                        final_transform=transform if mode == candidate else None,
                    )
                    end = max(end, w["end_ns"])
                    row = w["turns"][-1]
                    assert (
                        row["content_ns"]
                        and row["done_ns"]
                        and row["finish"] == "stop"
                        and row["status"] == 200
                    )
                    good += metric["slo"]
                    rows[mode][seed] = {
                        **metric,
                        "fixture": w["fixture_sha256"],
                        "tokens": row["usage"]["completion_tokens"],
                        "prompt_tokens": row["usage"]["prompt_tokens"],
                        "cached_tokens": (
                            row["usage"].get("prompt_tokens_details") or {}
                        ).get("cached_tokens", 0),
                        "submit_ms": (row["sent_ns"] - w["start_ns"]) / 1e6,
                        "first_after_submit_ms": (row["content_ns"][0] - row["sent_ns"])
                        / 1e6,
                        "stream_ms": (row["content_ns"][-1] - row["content_ns"][0])
                        / 1e6,
                        "strict_placement": check_placement(w),
                        "destinations": [r["route"]["replica"] for r in w["turns"]],
                        "first_all_ms": (row["content_ns"][0] - w["start_ns"]) / 1e6,
                        "last_all_ms": (row["content_ns"][-1] - w["start_ns"]) / 1e6,
                        "gap_all_ms": percentile(
                            [
                                (b - a) / 1e6
                                for a, b in itertools.pairwise(row["content_ns"])
                            ],
                            0.95,
                        ),
                    }
            seconds[mode] += 2 * (
                end / 1e9 - min(c["scheduled_start_s"] for c in clients)
            )
        blocks.append(
            {
                "block": block,
                "allocated_gpu_seconds": allocation,
                "slo_goodput_per_allocated_gpu_hour": good * 3600 / allocation,
            }
        )
    a, b = rows.values()
    assert a.keys() == b.keys() and len(a) == 64 and sum(s >= 24761 for s in a) == 16
    assert all(a[s]["fixture"] == b[s]["fixture"] for s in a)
    common = sorted(s for s in a if a[s]["valid"] and b[s]["valid"])
    stats = {}
    for mode, data in rows.items():
        vals = list(data.values())
        stats[mode] = {
            "submitted": len(vals),
            "http_200_turns": 3 * len(vals),
            "valid": sum(v["valid"] for v in vals),
            "slo": sum(v["slo"] for v in vals),
            "holdout_valid": sum(v["valid"] for s, v in data.items() if s >= 24761),
            "missing": {
                s: v["missing_sources"] for s, v in data.items() if not v["valid"]
            },
            "strict_placement": {
                k: sum(v["strict_placement"][k] for v in vals)
                for k in vals[0]["strict_placement"]
            },
            "warm_gpu_seconds": seconds[mode],
            "median_final_tokens": median(v["tokens"] for v in vals),
            "final_turn_medians": {
                k: median(v[k] for v in vals)
                for k in (
                    "prompt_tokens",
                    "cached_tokens",
                    "submit_ms",
                    "first_after_submit_ms",
                    "stream_ms",
                )
            },
            "common_tails": tails([data[s] for s in common]),
            "all_tails": {
                k: percentile([v[k] for v in vals], 0.95)
                for k in ("first_all_ms", "last_all_ms", "gap_all_ms")
            },
        }
        for key, checks in (
            ("slo_with_opening", ("numeric_opening",)),
            (
                "slo_with_opening_and_placement",
                ("numeric_opening", "in_separate_explanation"),
            ),
        ):
            count = sum(
                v["slo"] and all(v["strict_placement"][k] for k in checks) for v in vals
            )
            stats[mode][key] = {
                "jobs": count,
                "per_warm_gpu_hour": count * 3600 / seconds[mode],
            }
        stats[mode]["slo_goodput"] = stats[mode]["slo"] * 3600 / seconds[mode]
    lost = sorted(s for s in a if a[s]["valid"] and not b[s]["valid"])
    recovered = sorted(s for s in a if not a[s]["valid"] and b[s]["valid"])
    ratios = {
        k: v / stats["original"]["common_tails"][k]
        for k, v in stats[candidate]["common_tails"].items()
    }
    return {
        "groups": stats,
        "blocks": blocks,
        "common_correct": len(common),
        "recovered": recovered,
        "lost": lost,
        "common_tail_ratios": ratios,
        "gate": {
            "all_prior_failures_recovered": {783, 801, 803, 943} <= set(recovered),
            "preserves_correct": not lost,
            "warm_goodput": stats[candidate]["slo_goodput"]
            > stats["original"]["slo_goodput"],
            **(
                {
                    "numeric_opening": stats[candidate]["strict_placement"][
                        "numeric_opening"
                    ]
                    == 64,
                    "in_explanation": stats[candidate]["strict_placement"][
                        "in_separate_explanation"
                    ]
                    == 64,
                }
                if layout
                else {}
            ),
            "paired_tails": max(ratios.values()) <= 1.05,
        },
        "same_destinations": sum(
            a[s]["destinations"] == b[s]["destinations"] for s in a
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
