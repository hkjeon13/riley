"""Exercise the exact patched timing loop with a synthetic local client."""

import asyncio
from hashlib import sha256
import json
from unittest.mock import patch

import profile_workload as workload
from phase_inputs import OFFSETS_MS, ORDER, REQUEST, dispatch_seconds, phase_cells


async def run():
    cells = phase_cells("qwen14", ORDER[0])
    nominal = cells["wave-nominal"]
    assert tuple(phase_cells("qwen14", ORDER[-1])) == tuple(reversed(cells))
    baseline_inputs = None
    for label, config in cells.items():
        observed = {}

        async def fake_request(host, port, body, started_at, record, *, capture_hashes):
            assert capture_hashes and (host, port) == ("127.0.0.1", 1234)
            fingerprint = sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            record.update({"error": False, "start": started_at, "sent_at": started_at, "ttft": 0.1,
                           "e2e": 0.2, "tpot": 0.1, "gaps": [0.1], "prompt_tokens": 64,
                           "completion_tokens": 2, "finish": "stop", "input_sha256": fingerprint,
                           "request_id": fingerprint[:12], "output_sha256": "output"})

        async def instant_wait(_):
            return None

        def offsets(cell, index, first):
            at = dispatch_seconds(cell, index, first)
            baseline = dispatch_seconds(nominal, index, first)
            observed[index] = round((at - baseline) * 1_000)
            return at

        with patch.object(workload, "one", fake_request), patch.object(workload, "dispatch_seconds", offsets), \
             patch.object(workload.asyncio, "sleep", instant_wait):
            result = await workload.load_cell("127.0.0.1:1234", config)
        assert observed == {index: OFFSETS_MS[label] if index == REQUEST else 0 for index in range(48)}
        rows = result["measured"]
        assert result["statistics"]["warmup_ok"] == 8 and len(rows) == 40
        inputs = [row["input_sha256"] for row in rows]
        if baseline_inputs is None:
            baseline_inputs = inputs
        assert baseline_inputs == inputs
    print("48 offsets checked per cell; only request 45 moves; all 40 measured bodies match")


if __name__ == "__main__":
    asyncio.run(run())
