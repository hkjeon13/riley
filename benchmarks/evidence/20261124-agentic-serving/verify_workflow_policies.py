"""Compare every registered workflow, including wrong forecasts and uncited answers."""

import argparse
import json
from pathlib import Path

from verify_promotion import read, stats


def policy(folder):
    registration = read(folder / "wait-policy-registration.json")
    rows = []
    for launch, order in registration["order"].items():
        data = read(folder / launch / "wait-policy.json")
        expected = [
            (g, state, case[0])
            for g, state in order
            for case in registration["scenarios"]
        ]
        if [
            (r["group"], r["result"]["enabled"], r["result"]["case"]) for r in data
        ] != expected:
            raise ValueError(f"submission or mode does not match {launch} registration")
        for row in data:
            trial = row["result"]
            value = stats(trial)
            value.pop("answer", None)
            rows.append(
                {
                    "launch": launch,
                    "group": row["group"],
                    "mode": trial["enabled"],
                    "case": trial["case"],
                    "decision": trial["decision"],
                    "wait_hint_ms": trial["estimated_wait_ms"],
                    **value,
                }
            )
    pairs = []
    for group in registration["seeds"]:
        for scenario, *_ in registration["scenarios"]:
            compare = [r for r in rows if r["case"] == scenario and r["group"] == group]
            if len({r["fixture"] for r in compare}) != 1:
                raise ValueError("fixture mismatch")
            controls = [r for r in compare if not r["mode"]]
            active = [r for r in compare if r["mode"]]
            for a in controls:
                for b in active:
                    pairs.append(
                        {
                            "group": group,
                            "case": scenario,
                            "control": a["launch"],
                            "candidate": b["launch"],
                            "change_first_ms": b["workflow_first_ms"]
                            - a["workflow_first_ms"],
                            "change_last_ms": b["workflow_last_ms"]
                            - a["workflow_last_ms"],
                            "incremental_cached_tokens": b["final_cached"]
                            - a["final_cached"],
                        }
                    )
    return {
        "observations": rows,
        "within_fixture_comparisons": pairs,
        "scope": "shared launches across comparisons; individual cells are not independent replicates",
    }


def citations(folder):
    reg = read(folder / "citation-registration.json")
    values = []
    for launch, expected in reg["order"].items():
        obs = read(folder / launch / "citations.json")
        if obs["mode"] != expected or len(obs["records"]) != len(reg["cases"]):
            raise ValueError("missing registered cases or mode mismatch")
        for case, row in zip(reg["cases"], obs["records"], strict=True):
            if (row["seed"], row["history_entries"], row["document_pairs"]) != (
                case["seed"],
                case["history"],
                case["pairs"],
            ):
                raise ValueError("case changed")
            final = row["turns"][-1]
            values.append(
                {
                    "launch": launch,
                    "mode": expected,
                    **case,
                    "ok": row["ok"],
                    "error": row.get("error"),
                    "first_ms": (final["content_ns"][0] - row["start_ns"]) / 1e6
                    if final.get("content_ns")
                    else None,
                    "last_ms": (final["content_ns"][-1] - row["start_ns"]) / 1e6
                    if final.get("content_ns")
                    else None,
                    "missing_sources": [
                        n
                        for n in ("report-A", "report-B")
                        if n.lower() not in final.get("answer", "").lower()
                    ],
                }
            )
    return {
        "observations": values,
        "all_explicit_correct": all(r["ok"] for r in values if r["mode"] == "explicit"),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=("citations", "wait"))
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            (citations if args.experiment == "citations" else policy)(args.folder),
            indent=2,
        )
    )
