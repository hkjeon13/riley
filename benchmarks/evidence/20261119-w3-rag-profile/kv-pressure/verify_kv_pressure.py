"""Reconcile the scheduler's KV ledger and matched client requests."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


ARMS = {"riley-first": 2576, "riley-second": 768}
CELLS = ("rag-c8-s2", "wave-50ms")
IDENTITY = ("request_index", "input_sha256", "prompt_tokens")
OUTPUT = ("output_sha256", "completion_tokens", "finish")
DRAIN_GAUGES = ("active_requests", "waiting_requests", "kv_allocated_blocks")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def stripped_server(arguments):
    omit = {arguments.index(flag) + 1 for flag in ("--bind", "--kv-blocks")}
    return [value for i, value in enumerate(arguments) if i not in omit]


def request_trace(log, capacity):
    mapping = {}
    traces = defaultdict(list)
    held = active = now = 0
    waiting = []
    live = {}
    for line in log.splitlines():
        if line.startswith("W3-PHASE ") and " kv_scheduler_id " in line:
            _, request, _, _, native = line.split()
            assert request not in mapping
            mapping[request] = int(native)
        elif line.startswith("W3-KV "):
            _, request_raw, event, *fields = line.split()
            request = int(request_raw)
            stamp, need, logged_held, cap, logged_active, limit, logged_waiting = map(int, fields)
            assert stamp >= now and cap == capacity and limit == 32
            now = stamp
            assert 0 < need <= cap
            if event == "submit_queued":
                assert request not in traces
                assert waiting or held + need > cap or active >= limit
                waiting.append(request)
            elif event in ("submit_active", "admitted"):
                assert request not in live and held + need <= cap and active < limit
                if event == "admitted":
                    assert waiting.pop(0) == request
                else:
                    assert not waiting and request not in traces
                held += need
                active += 1
                live[request] = need
            elif event == "blocked":
                assert waiting[0] == request and held + need > cap and active < limit
            elif event == "released":
                assert live.pop(request) == need
                held -= need
                active -= 1
            else:
                raise ValueError(f"unknown scheduler event {event}")
            assert (held, active, len(waiting)) == (logged_held, logged_active, logged_waiting)
            assert 0 <= held <= cap and 0 <= active <= limit
            traces[request].append((event, stamp, need, held))
    assert mapping and traces and held == active == 0 and not waiting and not live
    return mapping, traces


def cell_summary(rows, mapping, traces):
    intervals = []
    for row in rows:
        trace = traces[mapping[row["request_id"]]]
        counts = Counter(item[0] for item in trace)
        assert counts["released"] == 1 and trace[-1][0] == "released"
        assert counts["submit_queued"] + counts["submit_active"] == 1
        assert counts["admitted"] == counts["submit_queued"]
        if counts["submit_queued"]:
            _, queued_at, _, _ = trace[0]
            _, admitted_at, _, _ = next(item for item in trace if item[0] == "admitted")
            failures = [value for event, value, _, _ in trace if event == "blocked"]
            assert queued_at <= admitted_at and all(queued_at <= value <= admitted_at for value in failures)
            intervals.append({
                "request_index": row["request_index"],
                "queue_ms": (admitted_at - queued_at) / 1e6,
                "kv_blocked_attempts": len(failures),
                "first_kv_block_to_admitted_ms": (admitted_at - failures[0]) / 1e6 if failures else None,
            })
    blocked = [item for item in intervals if item["kv_blocked_attempts"]]
    return {
        "requests": len(rows), "queued": len(intervals), "observed_kv_blocked": len(blocked),
        "scheduler_queue_ms_p50_queued_only": median(item["queue_ms"] for item in intervals) if intervals else None,
        "first_kv_block_to_admitted_ms_p50": median(item["first_kv_block_to_admitted_ms"] for item in blocked) if blocked else None,
        "queued_requests": intervals,
    }


def summarize(root, prior):
    output = {}
    inputs = None
    baseline_results = {}
    binary = (root / "binary.sha256").read_text().split()[0]
    for arm, capacity in ARMS.items():
        stage = root / f"qwen14-{arm}"
        plan, receipt = (read(stage / f"{name}.json") for name in ("plan", "receipt"))
        assert receipt["model"] == "Qwen2.5-14B-Instruct" and not plan["profile"]
        assert tuple(plan["cells"]) == CELLS and receipt["mode"] == arm and receipt["status"] == "passed"
        assert plan["env"] == {"RILEY_W3_REQUEST_TRACE": "1", "RILEY_W3_KV_TRACE": "1"}
        assert receipt["common_kv_blocks"] == capacity and read(stage / "server-exit.json")["exit"] == 0
        hashes = {Path(name).name: value for name, value in receipt["sha256"].items()}
        assert hashes["riley"] == binary
        configuration = (plan["cells"], stripped_server(plan["server"]),
                         {key: value for key, value in hashes.items() if key != "plan.json"})
        if inputs is None:
            inputs = configuration
        else:
            assert inputs == configuration
        before, after = (read(stage / f"{name}.json") for name in ("idle-before", "drain"))
        assert all(value[field] == 0 for value in (before, after) for field in DRAIN_GAUGES)
        assert all(before["counters"][key] == after["counters"][key]
                   for key in ("cancellations", "disconnects", "overloads"))
        mapping, traces = request_trace((stage / "server.log").read_text(), capacity)
        cases = {}
        for cell in CELLS:
            path = stage / f"{cell}.json"
            value = read(path)
            spec = plan["cells"][cell]
            stats, rows = value["statistics"], value["measured"]
            assert digest(path) == receipt["results"][cell]
            assert stats["ok"] == len(rows) == spec["requests"] and stats["warmup_ok"] == spec["warmup"]
            assert stats["errors"] == stats["rejected_429"] == 0
            archived = read(prior / "qwen14-riley-first" / f"{cell}.json")["measured"]
            joined = list(zip(archived, rows, strict=True))
            assert all(all(a[k] == b[k] for k in IDENTITY) for a, b in joined)
            result = cell_summary(rows, mapping, traces)
            if arm == "riley-first":
                assert all(all(a[k] == b[k] for k in OUTPUT) for a, b in joined)
                baseline_results[cell] = rows
                assert result["queued"] == result["observed_kv_blocked"] == 0
            else:
                assert result["observed_kv_blocked"] > 0
            differing = [
                right["request_index"]
                for left, right in zip(baseline_results[cell], rows, strict=True)
                if any(left[key] != right[key] for key in OUTPUT)
            ]
            result.update({
                "matched_outputs_vs_roomy": len(rows) - len(differing),
                "output_mismatch_request_indices": differing,
                "ttft_p95_ms": stats["ttft_p95_ms"], "tpot_p50_ms": stats["tpot_p50_ms"],
            })
            cases[cell] = result
        output[arm] = {
            "capacity_blocks": capacity, "cases": cases, "server_log_sha256": digest(stage / "server.log"),
            "peak_reserved_kv_blocks": max(held for events in traces.values() for _, _, _, held in events),
            "scheduler_kv_blocks_after_last_release": 0,
            "drain": {key: after[key] for key in DRAIN_GAUGES},
            "native_allocation_at_start": before["allocation"], "native_allocation_after_drain": after["allocation"],
        }
    assert (root / "campaign.exit").read_text().strip() == "0"
    reservations = {}
    for arm in ARMS:
        for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
            for moment in ("before", "after"):
                snapshot = read(root / f"gpu{gpu}-{moment}-{arm}.json")
                assert snapshot["phase"] == "idle" and not snapshot["other_processes"]
                assert (snapshot["physical_index"], snapshot["pid"], snapshot["start_ticks"]) == (gpu, pid, ticks)
                assert snapshot["allocation_bytes"] == snapshot["torch_reserved_bytes"] == 20 * 1024**3
            reservations[str(gpu)] = {"pid": pid, "start_ticks": ticks, "allocation_bytes": 20 * 1024**3}
    return {
        "binary_sha256": binary, "physical_gpu": 7, "arms": output, "reservations_restored_each_arm": reservations,
        "execution": {
            "stage": root.name, "exit": 0,
            "boundaries": [line for line in (root / "campaign.log").read_text().splitlines()
                           if line.startswith(("BUILD ", "START ", "END "))],
            "result_sha256_file_digest": digest(root / "result.sha256"),
        },
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("prior", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.prior), indent=2, allow_nan=False))
