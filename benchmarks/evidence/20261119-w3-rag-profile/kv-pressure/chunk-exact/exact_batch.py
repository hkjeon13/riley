"""Replay pinned greedy prompts at C1, C8 and C32; keep raw IDs on the runner."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from long_accuracy import complete, read_workload

WORKLOAD = Path(__file__).resolve().parents[1] / "holdout.json.gz"
ARMS = {
    "chunk1024-kv2576": (1024, 2576),
    "chunk512-kv768": (512, 768),
    "chunk1024-kv768": (1024, 768),
    "chunk512-kv2576": (512, 2576),
}


def probe(address):
    workload = read_workload(WORKLOAD)

    def send(case):
        row = complete(address, "m", case.prompt, expected=case.expected_ids, limit=case.limit)
        if not row["tokens"]:
            raise ValueError(f"{case.label} returned no generated tokens")
        return {"case": case.label, **row}

    rows = {"c1": [send(case) for case in workload.cases]}
    jobs = workload.cases * 3
    for clients in (8, 32):
        rows[f"c{clients}"] = []
        with ThreadPoolExecutor(clients) as pool:
            for start in range(0, len(jobs), clients):
                wave = jobs[start:start + clients]
                barrier = Barrier(len(wave), timeout=20)

                def concurrent(case):
                    barrier.wait()
                    return send(case)

                rows[f"c{clients}"].extend(pool.map(concurrent, wave))
    return {"workload_sha256": workload.sha256, "rows": rows}
