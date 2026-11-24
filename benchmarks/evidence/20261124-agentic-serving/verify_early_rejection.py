"""Distinguish real early HTTP rejections from unclassified socket failures."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(root):
    data = json.loads((root / "arrivals.json").read_text())
    assert len(data) == 8
    rows = []
    for idx, w in enumerate(data):
        assert (w["gpu"], w["arm"], w["limit"]) == (
            6 + idx % 2,
            ["control", "candidate"][(idx // 2 + idx % 2) % 2],
            (16, 16, 64, 64)[idx // 2],
        )
        arrivals = w["arrivals"]
        assert list(range(160)) == [r["index"] for r in arrivals]
        assert all(r["sent_ns"] >= w["release_ns"] for r in arrivals)
        assert len(w["peers"]) == 8 and all(
            r["status"] == 200
            and r["content_ns"][0] <= w["release_ns"] < r["content_ns"][-1]
            and r["done_ns"]
            for r in w["peers"]
        )
        stats = Counter()
        for r in arrivals:
            status = r.get("status")
            if status == 200:
                assert (
                    r["done_ns"]
                    and r["usage"]["completion_tokens"] > 0
                    and "request_written_ns" in r
                    and "error" not in r
                )
                stats["completed"] += 1
            elif status is None:
                assert r["error"]
                stats["transport"] += 1
                stats["transport_" + r["error"]] += 1
            elif r.get("request_write_error"):
                assert (
                    w["arm"] == "candidate"
                    and r["request_write_error"] == "broken_pipe"
                    and status >= 400
                    and "request_written_ns" not in r
                )
                stats["early_http_" + str(status)] += 1
            else:
                assert "request_written_ns" in r and status >= 400
                stats["normal_http_" + str(status)] += 1
        rows.append(
            {
                "gpu": w["gpu"],
                "policy": w["arm"],
                "output_limit": w["limit"],
                "submitted": len(arrivals),
                "counts": dict(stats),
            }
        )
    total = {}
    for arm in ["control", "candidate"]:
        counter = Counter()
        for r in rows:
            if arm == r["policy"]:
                counter.update(r["counts"])
        total[arm] = dict(counter)
    source = root.parent
    hashes = {
        line.split()[1]: line.split()[0]
        for line in (source / "inputs.sha256").read_text().splitlines()
    }
    for i in (6, 7):
        assert (root / f"gpu{i}/binary.sha256").read_text().split()[0] == (
            source / "server.sha256"
        ).read_text().split()[0]
        assert (
            json.loads((root / f"allocation-{i}.json").read_text())["returncode"] == 0
            and json.loads((root / f"gpu{i}/lease-after.json").read_text())["phase"]
            == "idle"
        )
        s = json.loads((root / f"gpu{i}/shutdown.json").read_text())
        assert not any(
            s[k] for k in ["active_requests", "waiting_requests", "kv_allocated_blocks"]
        ) and not any(s["allocation"].values())
    return {
        "waves": rows,
        "totals": total,
        "requests_per_arm": 640,
        "client_hashes": {
            k: hashes[k] for k in ["original_agent_workflow.py", "agent_workflow.py"]
        },
        "early_http_recovered": sum(
            v for k, v in total["candidate"].items() if k.startswith("early_http_")
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    print(json.dumps(audit(p.parse_args().run), indent=2))
