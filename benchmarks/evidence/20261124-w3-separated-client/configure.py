"""Adapt pinned benchmark copies while preserving their server and request settings."""

import json
from pathlib import Path
import sys


def replace(path, edits):
    data = path.read_text()
    for old, new in edits:
        if data.count(old) != 1:
            raise ValueError(f"{path} does not have the pinned fragment {old!r}")
        data = data.replace(old, new)
    path.write_text(data)


def configure(rag, c32):
    sources = [json.loads(path.read_bytes()) for path in (rag, c32)]
    gpu_keys = {"6", "7"}
    if any(source["schedules"].keys() != gpu_keys for source in sources):
        raise ValueError("both frozen schedules must cover GPUs 6 and 7")
    combined = {}
    for gpu in sorted(gpu_keys):
        left, right = (source["schedules"][gpu] for source in sources)
        if len(left) != 4 or len(right) != 2 or left.keys() & right.keys():
            raise ValueError("expected four RAG and two C32 schedules per GPU")
        combined[gpu] = {**left, **right}
    Path("schedules.json").write_text(json.dumps({"schedules": combined}, indent=2) + "\n")

    replace(Path("arm.py"), [
        ('"create", "--pull=never", "--name", name,', '"create", "--pull=never", "--cpuset-cpus", os.environ["W3_SERVER_CPUS"], "--name", name,'),
        ('"docker": command, "cells": CELLS}', '"docker": command, "cells": CELLS,\n                              "server_cpu_affinity": sorted(os.sched_getaffinity(0))}'),
        (
            '''        async def measured(spec):
            job = asyncio.create_task(load_cell(f"127.0.0.1:{port}", spec))
            while not job.done():
                check_lease(holders)
                require(gpu_pids(GPU) == holders | {pid}, "GPU process changed during measurement")
                await asyncio.wait({job}, timeout=2)
            return await job

        results = {}
        for label, spec in CELLS.items():
            data = asyncio.run(measured(spec))
            save(out / f"{label}.json", data)
            require(len(data["measured"]) == spec["requests"], f"missing request records in {label}")
            results[label] = digest(out / f"{label}.json")
            print(label, data["statistics"]["ttft_p95_ms"], flush=True)''',
            '''        require(os.sched_getaffinity(pid) == os.sched_getaffinity(0), "Riley server CPU affinity differs")
        client = [sys.executable, "-B", str(STAGE / "client.py"), str(out / "plan.json"), f"127.0.0.1:{port}"]
        deadline = time.monotonic() + 1000
        with (out / "client.log").open("xb") as log:
            job = subprocess.Popen(client, stdout=log, stderr=subprocess.STDOUT)
            try:
                while job.poll() is None:
                    check_lease(holders)
                    require(gpu_pids(GPU) == holders | {pid}, "GPU process changed during measurement")
                    require(time.monotonic() < deadline, "Riley client timed out")
                    time.sleep(1)
                require(job.returncode == 0, "Riley client failed")
            finally:
                if job.poll() is None:
                    job.terminate()
                    try:
                        job.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        job.kill()
                        job.wait(timeout=5)
        results = {}
        for label, spec in CELLS.items():
            path = out / f"{label}.json"
            require(len(json.loads(path.read_bytes())["measured"]) == spec["requests"], f"missing requests in {label}")
            results[label] = digest(path)''',
        ),
    ])
    replace(Path("tools/profile_workload.py"), [
        ('            for name, cell in plan["cells"].items():',
         '''            server_affinity = sorted(os.sched_getaffinity(server.pid))
            if server_affinity != sorted(os.sched_getaffinity(0)):
                raise RuntimeError("vLLM server CPU affinity differs")
            save(out / "cpu-placement.json", {"server_cpu_affinity": server_affinity})
            for name, cell in plan["cells"].items():'''),
    ])
    replace(Path("replay.py"), [
        ("import json\n", "import json\nimport os\nfrom pathlib import Path\n"),
        ('    host, port = address.split(":")', '''    client_cpu = int(os.environ["W3_CLIENT_CPU"])
    os.sched_setaffinity(0, {client_cpu})
    require(os.sched_getaffinity(0) == {client_cpu}, "client CPU affinity differs")
    host, port = address.split(":")'''),
        ("    epoch = time.perf_counter()\n", '''    schedstat = Path("/proc/thread-self/schedstat")
    before = [int(value) for value in schedstat.read_text().split()]
    epoch = time.perf_counter()
    wake_at = {}
'''),
        ('        pending.append(asyncio.create_task(issue(row["request_index"], measured[row["request_index"]])))',
         '''        index = row["request_index"]
        wake_at[index] = time.perf_counter()
        pending.append(asyncio.create_task(issue(index, measured[index])))'''),
        ("    elapsed = time.perf_counter() - epoch\n", '''    elapsed = time.perf_counter() - epoch
    after = [int(value) for value in schedstat.read_text().split()]
'''),
        ('"scheduled_write_ns": original["offset_ns"], "write_error_ms": error_ms})',
         '''"scheduled_write_ns": original["offset_ns"], "write_error_ms": error_ms,
                        "wake_at": wake_at[index], "content_gaps_s": row["gaps"]})'''),
        ('    return {"statistics": stats, "measured": records, "length_mismatches": length_mismatches}',
         '''    return {"statistics": stats, "measured": records, "length_mismatches": length_mismatches,
            "client_timing": {"cpu_affinity": sorted(os.sched_getaffinity(0)), "epoch": epoch,
                              "schedstat_delta": [end - start for start, end in zip(before, after, strict=True)]}}'''),
    ])


if __name__ == "__main__":
    configure(*map(Path, sys.argv[1:]))
