"""Reuse fixed arrival offsets for RAG, C32 and archived waves."""

import ast
import json
from pathlib import Path
import sys

from replay import verify_payloads


FIXED = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals", "c32-riley-arrivals", "c32-vllm-arrivals")
WAVES = ("wave-sync", "wave-50ms")
ORDER = {"6": ("control", "reserve"), "7": ("reserve", "control")}
SERVER_CPUS = {"6": [*range(24, 34), *range(72, 82)], "7": [*range(34, 44), *range(82, 92)]}
CLIENT_CPUS = {"6": 44, "7": 45}
MATCHED_KEYS = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens", "finish")


def schedules(fixed, waves, wave_cells):
    result = {}
    for gpu, source in fixed["schedules"].items():
        entries = {key: source[key] for key in FIXED}
        for name in WAVES:
            raw = waves(gpu, name)["measured"]
            first = min(row["sent_at"] for row in raw)
            rows = [{**{key: row[key] for key in MATCHED_KEYS}, "offset_ns": round((row["sent_at"] - first) * 1e9)}
                    for row in raw]
            entries[name] = {"cell": wave_cells[name], "schedule": rows, "requests": len(rows)}
        result[gpu] = entries
        for plan in entries.values():
            if plan["requests"] != len(plan["schedule"]) or plan["requests"] != plan["cell"]["requests"]:
                raise ValueError(f"inconsistent replay request count on GPU {gpu}")
            verify_payloads(plan["cell"], plan["schedule"])
    return {"schedules": result}


def configure(fixed, waves):
    inputs = json.loads(fixed.read_bytes())
    wave_cells = json.loads((waves / "cells.json").read_bytes())

    def read_wave(gpu, name):
        return json.loads((waves / f"gpu{gpu}/chunk-1024/{name}.json").read_bytes())

    document = schedules(inputs, read_wave, wave_cells)
    Path("schedules.json").write_text(json.dumps(document, indent=2) + "\n")
    placement = {"server_cpus": SERVER_CPUS, "client_cpus": CLIENT_CPUS}
    Path("cpu-placement.json").write_text(json.dumps(placement, indent=2) + "\n")
    arm = Path("arm.py")
    source = arm.read_text()
    for before, after in (
        ('    out = OUT / "coder14-riley"', '    out = OUT / os.environ["W3_PREFILL_ORDER"]'),
        ('sha256:1729f15a677422d552b9a3fd2352dfeb08dd4cccd3558680a016058d27191098', 'sha256:01b5c9d4bedfc93284af6dfc9fbe747002327d2818f7164f8d4e77565eaa1a8c'),
        ('w5-shard-package-20261124.nhBcrYgP/cpu-evidence/report.json', 'w4-uri-package-20261124.73IfvPre/cpu-evidence/report.json'),
        ('0511b9280f62c3ee87c7a1344444423d8b4623c4', '8a7c618c1882effcacd49107a0b41ea3fc29d22d'),
        ('        "--env", "RILEY_W3_EXPORT_OBSERVATIONS=1",\n', ''),
        ('        check_lease(holders)\n        docker("stop",',
         '''        job = [sys.executable, "-B", str(STAGE / "holdout.py"), str(out / "token-id-holdout.json"), f"127.0.0.1:{port}"]
        subprocess.run(job, check=True, timeout=160)
        check_lease(holders)
        require(gpu_pids(GPU) == holders | {pid}, "GPU process changed during holdout")
        docker("stop",'''),
        ('                                   "results": results})',
         '''                                   "holdout_sha256": digest(out / "token-id-holdout.json"), "results": results})'''),
    ):
        if source.count(before) != 1:
            raise ValueError(f"missing pinned arm fragment: {before!r}")
        source = source.replace(before, after)
    ast.parse(source)
    command_anchor = '    name = f"riley-w3-coder14-{uuid4().hex}"'
    if source.count(command_anchor) != 1:
        raise ValueError("missing registered serving arguments")
    source = source.replace(command_anchor, '''    if os.environ["W3_PREFILL_ORDER"] == "reserve":
        server.append("--prefill-aging-reserve-decode")
''' + command_anchor)
    ast.parse(source)
    arm.write_text(source)


if __name__ == "__main__":
    configure(*map(Path, sys.argv[1:]))
