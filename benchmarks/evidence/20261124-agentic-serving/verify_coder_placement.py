"""Check the online routing information, both cost denominators and every held-out outcome."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from agent_workflow import fixture, history_messages, percentile
from join_trace import join
from verify_coder_pressure import (
    check_manifest,
    check_server,
    qualifies,
    reconcile,
    work_latency,
)
from verify_coder_scale import aggregate
from verify_model_decode import natural, read


def evaluate(root, mode):
    check_manifest(root)
    assert (root / "client.exit").read_text().strip() == "0"
    sha = (root / "trace-server.sha256").read_text().split()[0]
    assert sha == "8db3d5c634bab3ef5a4d3178e680db0c3c67beb781ecd042067654f2871bd990"
    conf = read(root / "coder-placement-registration.json")
    logs = []
    lifetime = 0
    for g in (6, 7):
        d = root / f"gpu{g}"
        check_server(d, sha, g)
        logs.append((d / "server.log").read_bytes())
        lifetime += (
            int((root / f"gpu{g}.end").read_text())
            - int((root / f"gpu{g}.start").read_text())
        ) / 1e9
    out = []
    summaries = {}
    for w, (size, base, layout) in zip(
        read(root / "waves.json"), conf["waves"], strict=True
    ):
        assert (w["size"], w["base"], w["layout"]) == (size, base, layout)
        offered = (
            list(range(0, size, 4))
            if layout == "regular"
            else random.Random(base).sample(range(size), size // 4)
        )
        swaps = (
            set(offered[:2] + [i for i in range(size) if i not in offered][:2])
            if layout == "imperfect"
            else set()
        )
        stats = []
        sl = []
        for i, log in enumerate(logs):
            m = {k: w[k][i] for k in ("before", "after")}
            stats.append(reconcile(m, log))
            sl.append(
                log[m["before"]["log_offset"] : m["after"]["log_offset"]].decode()
            )
        counts = [Counter(), Counter()]
        ranked = sorted(w["rows"], key=lambda r: r["route"]["rank"])
        assert len(ranked) == size
        for rank, r in enumerate(ranked):
            route = r["route"]
            assert rank == route["rank"]
            k = route["bytes"] >= 4096
            expected = (
                rank % 2
                if mode == "rr"
                else min(
                    range(2), key=lambda i: (counts[i][k], sum(counts[i].values()), i)
                )
            )
            assert route["device"] == expected
            counts[expected][k] += 1
        rows = []
        for i, j in zip(range(size), w["rows"], strict=True):
            q = j["route"]
            gpu = q["device"]
            work = j["work"]
            seed = base + i
            reader = i in offered
            hist = 200 if ((not reader) != (i in swaps)) else 0
            assert j["index"] == i and j["reader"] == reader
            start = w["origin"] + 30_000_000 * i
            assert start == j["scheduled"] and work["end_ns"] <= w["drained"]
            notes = 2 if reader else 96
            prompt = fixture(seed, notes)[1]
            messages = history_messages(seed, hist) + [
                {"role": "user", "content": prompt}
            ]
            assert len(json.dumps(messages, ensure_ascii=False).encode()) == q["bytes"]
            f = natural(work, seed, notes, hist, delay=0 if reader else 0.5)
            t = work_latency(work, start)
            cache = None
            if t.get("completed") and f["verified_tool_calls"] == 2:
                s = join(sl[gpu], [work], details=True)
                assert [r["turn"] for r in s] == [0, 1, 2]
                use = work["turns"][-1]["usage"]
                cache = (use.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                assert use["prompt_tokens"] == cache + s[-1]["prefill_rows"]
            rows.append(
                {
                    "seed": seed,
                    "gpu": 6 + gpu,
                    "rank": q["rank"],
                    "role": "reader" if reader else "target",
                    "history": hist,
                    "first_message_bytes": q["bytes"],
                    "final_cached": cache,
                    "calls": f["verified_tool_calls"],
                    "valid": f["valid"],
                    "slo": qualifies(f["valid"], t),
                    "error": work.get("error"),
                    **t,
                }
            )
        out.append(
            {
                "size": size,
                "base": base,
                "layout": layout,
                "duration_s": (w["drained"] - w["origin"]) / 1e9,
                "steps": stats,
                "rows": rows,
            }
        )
    for layout in ("regular", "shuffled", "imperfect"):
        ws = [w for w in out if w["size"] == 16 and w["layout"] == layout]
        c = aggregate(ws)
        rs = [r for w in ws for r in w["rows"]]
        c["reader_stream_gap_p95_of_p95_ms"] = percentile(
            [
                r["gap_p95_ms"]
                for r in rs
                if r["role"] == "reader" and r.get("gap_p95_ms") is not None
            ],
            0.95,
        )
        c["per_gpu"] = {
            gpu: {
                "jobs": sum(r["gpu"] == gpu for r in rs),
                "computed_prefill_rows": sum(w["steps"][gpu - 6]["p"] for w in ws),
                "executing_gpu_seconds": sum(w["steps"][gpu - 6]["gpu_ns"] for w in ws)
                / 1e9,
                "readers": sum(r["role"] == "reader" and r["gpu"] == gpu for r in rs),
                "slo": sum(r["slo"] for r in rs if r["gpu"] == gpu),
            }
            for gpu in (6, 7)
        }
        summaries[layout] = c
    return {
        "waves": out,
        "groups": summaries,
        "all": aggregate(out),
        "lease_gpu_seconds": lifetime,
    }


def verify(root):
    arguments = {
        (root / mode / gpu / "server-arguments.txt").read_text()
        for mode in ("rr", "history")
        for gpu in ("gpu6", "gpu7")
    }
    assert len(arguments) == 1
    assert (root / "rr/inputs.sha256").read_bytes() == (
        root / "history/inputs.sha256"
    ).read_bytes()
    arms = {mode: evaluate(root / mode, mode) for mode in ("rr", "history")}
    checks = {}
    same_answers = {}
    for layout in ("regular", "shuffled", "imperfect"):
        a, b = [arms[m]["groups"][layout] for m in ("rr", "history")]
        checks[layout] = {
            "preserved_valid": set(b["invalid"]) <= set(a["invalid"]),
            "slo_count": b["slo"] >= a["slo"],
            "slo_gpu_hour": b["slo_per_gpu_hour"] > a["slo_per_gpu_hour"],
            "reader_first": b["reader_first_p95_ms"] <= 1.05 * a["reader_first_p95_ms"],
            "reader_gap": b["reader_stream_gap_p95_of_p95_ms"]
            <= 1.05 * a["reader_stream_gap_p95_of_p95_ms"],
            "overall_tail": b["last_p95_ms"] <= 1.05 * a["last_p95_ms"],
        }
        maps = [
            {
                r["seed"]: r.get("text_hash")
                for w in arms[m]["waves"]
                if w["layout"] == layout and w["size"] == 16
                for r in w["rows"]
            }
            for m in ("rr", "history")
        ]
        assert maps[0].keys() == maps[1].keys()
        same_answers[layout] = sum(
            v == maps[1][k] and v is not None for k, v in maps[0].items()
        )
    return {
        "arms": arms,
        "gates": checks,
        "same_final_text": same_answers,
        "passed": all(all(v.values()) for v in checks.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(verify(p.parse_args().directory), indent=2))
