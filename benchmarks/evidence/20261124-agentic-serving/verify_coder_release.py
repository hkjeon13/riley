"""Check preview intervals against the lifetimes of the foreground workflows that launched them."""

import argparse
from collections import Counter
from itertools import pairwise
import json
from pathlib import Path

from agent_workflow import percentile
from verify_coder_pressure import check_manifest, check_server, qualifies, work_latency
from verify_coder_wait import check_preview
from verify_model_decode import natural, read


def verify(root):
    check_manifest(root)
    configs = list(root.glob("coder-release*-registration.json"))
    assert len(configs) == 1
    cfg = read(configs[0])
    result = {}
    examples = {}
    for gpu, order in cfg["order"].items():
        d = root / f"gpu{gpu}"
        check_server(
            d, "13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f", gpu
        )
        raw = read(d / "waves.json")
        assert len(raw) == 4 and read(d / "warm.json")["ok"]
        for n, wave in enumerate(raw):
            mode = order[n // 2]
            size = cfg["sizes"][n % 2]
            base = cfg["groups"][n // 2][n % 2]
            assert (wave["mode"], wave["size"], wave["base"]) == (mode, size, base)
            rows = []
            spans = []
            choices = Counter()
            for i, job in zip(range(size), wave["rows"], strict=True):
                schedule = wave["origin"] + i * cfg["arrival_ms"] * 1_000_000
                assert schedule == job["scheduled"]
                raw_work = job["record"]
                work = raw_work["target"]
                seed = base + i
                facts = natural(work, seed, 96, 200, delay=0.5)
                t = work_latency(work, schedule)
                choice = raw_work["decision"]
                choices[choice] += 1
                assert work["end_ns"] <= wave["drained"] and raw_work["enabled"]
                if choice == "submit":
                    p = raw_work["prefix"]
                    check_preview(work, "history", p)
                    assert (
                        p["status"] == 200 and p.get("done_ns") and not p.get("error")
                    )
                    assert work["turns"][1]["tool_start_ns"] <= p["sent_ns"]
                    spans.append((p["sent_ns"], p["done_ns"], work["end_ns"], seed))
                else:
                    assert not raw_work["prefix"]
                rows.append(
                    {
                        "seed": seed,
                        "decision": choice,
                        "valid": facts["valid"],
                        "slo": qualifies(facts["valid"], t),
                        "calls": facts["verified_tool_calls"],
                        "error": work.get("error"),
                        **t,
                    }
                )
            spans.sort()
            observed = []
            for old, new in pairwise(spans):
                assert old[1] <= new[0], "two simultaneous speculative requests"
                if new[0] < old[2]:
                    observed.append(
                        {
                            "from_seed": old[3],
                            "next_seed": new[3],
                            "after_preview_done_ms": (new[0] - old[1]) / 1e6,
                            "before_previous_workflow_end_ms": (old[2] - new[0]) / 1e6,
                        }
                    )
            if mode == "legacy":
                assert not observed
            else:
                examples.setdefault(gpu, []).extend(observed)
            dt = (wave["drained"] - wave["origin"]) / 1e9
            result.setdefault(base, {})[mode] = {
                "gpu": gpu,
                "size": size,
                "valid": sum(r["valid"] for r in rows),
                "slo": sum(r["slo"] for r in rows),
                "duration_s": dt,
                "slo_gpu_hour": sum(r["slo"] for r in rows) * 3600 / dt,
                "completion_p95_ms": percentile(
                    [r["last_ms"] for r in rows if "last_ms" in r], 0.95
                ),
                "decisions": dict(choices),
                "overlap_with_previous_workflow": observed,
                "rows": rows,
            }
    return {
        "waves": result,
        "new_reservation_while_previous_answer_pending": {
            g: len(examples.get(g, [])) for g in cfg["order"]
        },
        "lifetime_verified_on_both_gpus": all(examples.get(g) for g in cfg["order"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
