"""Reconcile the opt-in HTTP field with the explicit-template expression on live servers."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from verify_final_layout import GROUPS, HERE, NAMES, check, inspect, read, tokenize


def verify(root, crossover, exe, tokenizer):
    check(
        read(root / "final-layout-api-registration.json")
        == read(HERE / "final-layout-api-registration.json"),
        "different API experiment",
    )
    for line in (root / "measured-inputs.sha256").read_text().splitlines():
        h, filename = line.split()
        check(sha256((root / filename).read_bytes()).hexdigest() == h, filename)
    for f in ("exit-6", "exit-7", "build.exit"):
        check((root / f).read_text().strip() == "0", f)
    arms = {}
    control = read(crossover)
    binary = (root / "binary.sha256").read_text().split()[0]
    for arm, gpu in (("named", 6), ("duplicate", 7)):
        d = root / arm
        lease = read(root / f"{arm}.lease.json")
        end = read(d / "shutdown.json")
        check(
            lease["phase"] == "idle"
            and lease["physical_index"] == gpu
            and not lease["other_processes"]
            and lease["allocation_bytes"] == 20 * 2**30
            and lease["gpu"] in (d / "device.csv").read_text(),
            "lease",
        )
        check(
            (d / "binary.sha256").read_text().split()[0] == binary, "wrong executable"
        )
        check(
            not any(
                end[k]
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            )
            and not any(end["allocation"].values()),
            "server release",
        )
        data = read(d / "workflows.json")
        check([(r["history"], r["seed"]) for r in data] == list(GROUPS), "groups")
        norm = tokenize(data, "riley", "keep", exe, tokenizer, named=arm == "named")
        check(
            len(norm) == 60
            and all(n["matched"] and n["matched_length"] for n in norm.values()),
            "real prompt tokens",
        )
        rows = {}
        for g in data:
            group = f"{g['history']}-{g['seed']}"
            check(
                [w["workflow_id"] for w in g["records"]] == list(NAMES),
                "incomplete workflows",
            )
            for w in g["records"]:
                cell = inspect(w, norm)
                key = f"{group}-{w['workflow_id']}"
                term = w["turns"][-1]
                old = control["cells"][group][f"gpu{gpu}-riley-keep"][w["workflow_id"]]
                rows[key] = {
                    "prompt_hashes": [n["prompt_ids_sha256"] for n in w["turns"]],
                    "generated": [n.get("generated_ids", []) for n in w["turns"]],
                    "usage_cached": [
                        (n["usage"].get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0
                        )
                        for n in w["turns"]
                    ],
                    "finish": term["finish"],
                    "ok": w["ok"],
                    "unexpected_tool": cell["unexpected_tool"],
                    "original_output_equal": cell["generated_digest"]
                    == old["generated_digest"],
                    "last_final_ms": cell["last_final_ms"],
                }
        arms[arm] = rows
    counts = {
        k: 0 for k in ("prompt_hashes", "generated", "usage_cached", "finish", "ok")
    }
    for key, x in arms["named"].items():
        y = arms["duplicate"][key]
        for k in counts:
            counts[k] += int(x[k] == y[k])
    return {
        "requests": 120,
        "paired_workflows": 20,
        "matched_workflow_count_by_field": counts,
        "valid_by_arm": {
            k: sum(w["ok"] for w in rows.values()) for k, rows in arms.items()
        },
        "unexpected_final_tool": {
            k: sum(w["unexpected_tool"] for w in rows.values())
            for k, rows in arms.items()
        },
        "previous_final_generated_ids_matched": {
            k: sum(w["original_output_equal"] for w in rows.values())
            for k, rows in arms.items()
        },
        "binary": binary,
        "workflow_rows": arms,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "crossover", "exe", "tokenizer"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.crossover, a.exe, a.tokenizer), indent=2))
