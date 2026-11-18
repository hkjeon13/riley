"""Compare paired workload records with stable input and output identities."""
import hashlib
import json
import math
from pathlib import Path
import sys

out = Path(sys.argv[1])
fields = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")
arms = {}
for arm in ("base1", "cap1", "cap2", "base2"):
    receipt = json.loads((out / arm / "receipt.json").read_text())
    idle = json.loads((out / arm / "idle-after.json").read_text())
    assert receipt["status"] == "passed" and idle["phase"] == "idle"
    assert (idle["pid"], idle["start_ticks"]) == (receipt["holder_pid"], receipt["holder_start_ticks"])
    assert idle["allocation_bytes"] == 20 * 2**30 and not idle["other_processes"]
    cells = {}
    for cell, meta in receipt["cells"].items():
        files = [out / arm / (cell + suffix) for suffix in (".json", ".records.json")]
        for file, key in zip(files, ("summary_sha256", "records_sha256")):
            assert hashlib.sha256(file.read_bytes()).hexdigest() == meta[key], file
        cells[cell] = tuple(json.loads(file.read_text()) for file in files)
    arms[arm] = (receipt["policy"], cells)

comparisons = []
for seed in (1, 2):
    base_cells = arms[f"base{seed}"][1]
    for cell, (base, before) in base_cells.items():
        candidate, after = arms[f"cap{seed}"][1][cell]
        assert [r["input_sha256"] for r in before] == [r["input_sha256"] for r in after]
        assert [r["scheduled_start"] for r in before] == [r["scheduled_start"] for r in after]
        assert all(r["sent_at"] >= r["start"] >= 0 for r in before + after)
        compared = ("output_sha256", "finish", "prompt_tokens", "completion_tokens")
        match = sum(all(a[k] == b[k] for k in compared) for a, b in zip(before, after))
        dispatch_p95 = None
        if cell == "rag-r3":
            delta = lambda rows: sorted((row["sent_at"] - row["scheduled_start"]) * 1000 for row in rows)
            dispatch_p95 = [values[min(math.floor(len(values) * .95), len(values) - 1)]
                             for values in (delta(before), delta(after))]
        row = {
            "seed": seed, "cell": cell, "completed_per_arm": len(before),
            "matched_text_finish_usage": match, "scheduled_to_socket_queue_p95_ms": dispatch_p95,
            "base": {k: base[k] for k in fields},
            "candidate": {k: candidate[k] for k in fields},
            "percent_change": {k: 100 * (candidate[k] / base[k] - 1) for k in fields},
        }
        comparisons.append(row)
        print(seed, cell, "output", match, len(before), {k: round(v, 2) for k, v in row["percent_change"].items()})

report = {"device": 7, "model": "Qwen2.5-0.5B-Instruct", "baseline_policy": arms["base1"][0],
          "candidate_policy": arms["cap1"][0], "comparisons": comparisons}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
