"""Compare routing of dependent tool turns on two real, independently cached replicas."""

import argparse
import concurrent.futures
import hashlib
import http.client
import itertools
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from urllib.request import urlopen

from agent_workflow import percentile, replay, stream

SAMPLED_POLICIES = ("observed-capacity", "fresh-capacity", "defer-cold-prefill")
CAPACITY_POLICIES = ("capacity", *SAMPLED_POLICIES)


def history_credit_keys(layout, messages):
    """Match fingerprint([layout, messages[:n]]) using one pass over the history."""
    state = hashlib.sha256()
    state.update(b"[")
    state.update(json.dumps(layout, sort_keys=True, ensure_ascii=False).encode())
    state.update(b", [")
    keys = []
    for message in messages:
        if keys:
            state.update(b", ")
        state.update(json.dumps(message, sort_keys=True, ensure_ascii=False).encode())
        prefix = state.copy()
        prefix.update(b"]]")
        keys.append(prefix.hexdigest())
    return keys


class Router:
    def __init__(self, endpoints, policy, active_limits=None, telemetry=None):
        if policy in CAPACITY_POLICIES and (
            not active_limits
            or len(active_limits) != len(endpoints)
            or any(type(n) is not int or n <= 0 for n in active_limits)
        ):
            raise ValueError(
                "capacity routing needs a positive active limit for every replica"
            )
        if policy in SAMPLED_POLICIES and telemetry is None:
            raise ValueError(f"{policy} needs a readiness sampler")
        self.telemetry = telemetry
        self.active_limits = active_limits
        self.endpoints = endpoints
        self.policy = policy
        self.lock = threading.Lock()
        self.active = [0] * len(endpoints)
        self.credits = [{} for _ in endpoints]
        self.counter = itertools.count()

    def history_credits(self, keys):
        return [
            max((cache.get(key, 0) for key in keys), default=0)
            for cache in self.credits
        ]

    def defer_cold_prefill(self, keys, estimated, row):
        # Wait without reserving a GPU slot. The final placement rechecks both
        # local reservations and the latest server sample under the router lock.
        start = time.monotonic()
        while True:
            with self.lock:
                samples = self.telemetry.snapshot()
                credits = self.history_credits(keys)
                open_replicas = [
                    i
                    for i, s in enumerate(samples)
                    if s is not None
                    and s["available"]
                    and not s["waiting"]
                    and max(s["active"], self.active[i]) < self.active_limits[i]
                ]
                unknown_credit = max(
                    (c for c, s in zip(credits, samples) if s is None), default=0
                )
                protect = bool(open_replicas) and all(
                    max(samples[i]["active"], self.active[i]) > 0
                    and min(estimated, unknown_credit) - credits[i] >= 2048
                    for i in open_replicas
                )
            remaining = 1.5 - (time.monotonic() - start)
            if not protect or remaining <= 0:
                break
            if "initial_readiness" not in row:
                row["initial_readiness"] = samples
            time.sleep(min(0.05, remaining))
        if "initial_readiness" in row:
            row["routing_hold_ms"] = (time.monotonic() - start) * 1000

    def sender(self, session):
        def send(_, request, row, first_visible):
            begin = time.monotonic_ns()
            # Only predict reuse for previously completed requests with exactly
            # matching message history and presentation of tools. The engine's
            # token-level cached_tokens is the independent observation.
            layout = (
                request["model"],
                request["tool_choice"] == "none",
                request["tools"],
            )
            keys = history_credit_keys(layout, request["messages"])
            estimated = len(json.dumps(request).encode()) / 4
            if self.policy == "defer-cold-prefill" and estimated >= 2048:
                self.defer_cold_prefill(keys, estimated, row)
            with self.lock:
                credits = self.history_credits(keys)
                scores = [
                    60 * active + 0.09 * max(0, estimated - credit)
                    for active, credit in zip(self.active, credits)
                ]
                sequence = next(self.counter)
                if self.policy == "round-robin":
                    replica = sequence % len(self.endpoints)
                elif self.policy == "session":
                    replica = session % len(self.endpoints)
                elif self.policy in CAPACITY_POLICIES:
                    eligible = list(range(len(scores)))
                    occupancy = self.active
                    if self.policy in SAMPLED_POLICIES:
                        samples = self.telemetry.snapshot()
                        row["readiness"] = samples
                        eligible = [
                            i
                            for i, sample in enumerate(samples)
                            if sample is None or sample["available"]
                        ]
                        if not eligible:
                            raise ConnectionError("no replica accepting requests")
                        occupancy = [
                            max(
                                local,
                                sample["active"],
                                limit if sample["waiting"] else 0,
                            )
                            if sample is not None
                            else local
                            for local, sample, limit in zip(
                                self.active, samples, self.active_limits
                            )
                        ]
                    open_slots = [
                        i for i in eligible if occupancy[i] < self.active_limits[i]
                    ]
                    if self.policy in ("fresh-capacity", "defer-cold-prefill"):
                        confirmed = [i for i in open_slots if samples[i] is not None]
                        open_slots = confirmed or open_slots
                    replica = min(
                        open_slots or eligible,
                        key=lambda i: (scores[i], (i - sequence) % len(scores)),
                    )
                else:
                    replica = min(
                        range(len(scores)),
                        key=lambda i: (scores[i], (i - sequence) % len(scores)),
                    )
                row["route"] = {
                    "replica": replica,
                    "active": self.active.copy(),
                    "estimated_tokens": estimated,
                    "eligible_estimate": credits,
                    "score_ms": scores,
                    "decision_ms": (time.monotonic_ns() - begin) / 1e6,
                }
                self.active[replica] += 1
            address, bearer = self.endpoints[replica]
            try:
                result = stream(address, request, row, first_visible, bearer=bearer)
                with self.lock:
                    self.credits[replica][keys[-1]] = max(
                        0, row["usage"]["prompt_tokens"] - 16
                    )
                return result
            finally:
                with self.lock:
                    self.active[replica] -= 1

        return send


def check_isolation(endpoints):
    probe = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "Summarize this archived case. " * 200}
        ],
        "temperature": 0,
        "max_tokens": 4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rows = []
    for dest, principal in [(0, 0), (0, 0), (1, 0), (1, 1), (1, 1)]:
        row = {"replica": dest, "principal": principal}
        try:
            stream(endpoints[dest][0], probe, row, bearer=endpoints[principal][1])
        except ValueError:
            if row.get("status") != 401 or dest == principal:
                raise
        rows.append(row)
    if [r["status"] for r in rows] != [200, 200, 401, 200, 200]:
        raise ValueError("principal admission differs from registered cases")
    hits = [
        (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        for r in rows
        if r["status"] == 200
    ]
    if not (hits[0] == hits[2] == 0 and hits[1] > 0 and hits[3] > 0):
        raise ValueError(f"process isolation or same-principal reuse failed: {hits}")
    return rows


def run(
    endpoints,
    policy,
    *,
    model="m",
    waves=(1, 2, 4, 8),
    gpu_count=None,
    active_limits=None,
    dispatch_fixed_tools=False,
    previews=None,
):
    sampler = nullcontext()
    if policy in SAMPLED_POLICIES:
        from replica_load import ReplicaLoad

        sampler = ReplicaLoad(endpoints)
    with sampler as telemetry:
        router = Router(endpoints, policy, active_limits, telemetry)
        return run_waves(
            router,
            model,
            waves,
            gpu_count or len(endpoints),
            dispatch_fixed_tools=dispatch_fixed_tools,
            previews=previews,
        )


def run_waves(
    router, model, waves, gpu_count, *, dispatch_fixed_tools=False, previews=None
):
    result = []
    next_session = 0
    occurrences = {}
    for wave, count in enumerate(waves):
        origin_ns = time.monotonic_ns() + 50_000_000
        first_session = max(wave * 10, next_session)
        next_session = first_session + count
        repetition = occurrences.get(count, 0)
        occurrences[count] = repetition + 1
        label = (
            f"wave-{count}" if not repetition else f"wave-{count}-repeat-{repetition}"
        )

        def workflow(
            idx, origin_ns=origin_ns, first_session=first_session, label=label
        ):
            scheduled_ns = origin_ns + idx * 25_000_000
            time.sleep(max(0, (scheduled_ns - time.monotonic_ns()) / 1e9))
            session = first_session + idx
            if previews is not None:
                record = previews.workflow(
                    router,
                    session,
                    model,
                    761 + session,
                    96 if idx % 2 else 2,
                    f"{label}-{idx}",
                    dispatch_fixed_tools,
                )
            else:
                record = replay(
                    "",
                    model,
                    761 + session,
                    96 if idx % 2 else 2,
                    0.1,
                    f"{label}-{idx}",
                    history=100,
                    send=router.sender(session),
                    dispatch_fixed_tools=dispatch_fixed_tools,
                )
            record["scheduled_ns"] = scheduled_ns
            record["start_lag_ms"] = (record["start_ns"] - scheduled_ns) / 1e6
            return record

        with concurrent.futures.ThreadPoolExecutor(count) as pool:
            records = list(pool.map(workflow, range(count)))
        elapsed = (time.monotonic_ns() - origin_ns) / 1e9
        valid = [r for r in records if r["ok"]]

        def on_time(r, lag_ms):
            return (
                r["first_final_ms"] + lag_ms <= 10000
                and r["last_final_ms"] + lag_ms <= 18000
                and r["final_gap_p95_ms"] <= 150
            )

        good = sum(on_time(r, 0) for r in valid)
        scheduled_good = sum(on_time(r, r["start_lag_ms"]) for r in valid)
        result.append(
            {
                "submitted": count,
                "correct": len(valid),
                "slo_completed": good,
                "window_s": elapsed,
                "warm_goodput_per_gpu_hour": good * 3600 / (gpu_count * elapsed),
                "scheduled_slo_completed": scheduled_good,
                "scheduled_warm_goodput_per_gpu_hour": scheduled_good
                * 3600
                / (gpu_count * elapsed),
                "start_lag_p95_ms": percentile(
                    [r["start_lag_ms"] for r in records], 0.95
                ),
                "scheduled_first_p95_ms": percentile(
                    [r["start_lag_ms"] + r["first_final_ms"] for r in valid], 0.95
                ),
                "scheduled_last_p95_ms": percentile(
                    [r["start_lag_ms"] + r["last_final_ms"] for r in valid], 0.95
                ),
                "first_p95_ms": percentile([r["first_final_ms"] for r in valid], 0.95),
                "last_p95_ms": percentile([r["last_final_ms"] for r in valid], 0.95),
                "records": records,
            }
        )
    return result


def tenant_burst(endpoints):
    """Flood tenant A after tenant B starts reading, then inspect cancellation recovery."""
    (busy, key), (other, other_key) = endpoints
    ready = threading.Event()

    def metrics():
        with urlopen(f"http://{busy}/metrics", timeout=5) as response:
            return json.load(response)

    def tenant_b(_, request, row, first_visible):
        return stream(other, request, row, first_visible, bearer=other_key)

    def flood(idx):
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": f"Case {idx}. "
                    + "Read this latency measurement carefully. " * 300
                    + " Write a detailed long story.",
                }
            ],
            "max_tokens": 96,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        row = {"request": idx}
        if idx:
            try:
                stream(busy, req, row, bearer=key)
            except (ValueError, OSError, http.client.HTTPException) as exc:
                row["error"] = str(exc)
            return row
        # Disconnect the first admitted stream at its first visible chunk.
        host, port = busy.rsplit(":", 1)
        conn = http.client.HTTPConnection(host, int(port), timeout=30)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(req),
                {"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            rsp = conn.getresponse()
            row["status"] = rsp.status
            if rsp.status == 200:
                for line in rsp:
                    if line.startswith(b"data:") and any(
                        c.get("delta", {}).get("content")
                        for c in json.loads(line[5:])["choices"]
                    ):
                        row["client_cancel_ns"] = time.monotonic_ns()
                        break
        finally:
            conn.close()
        return row

    before = metrics()
    with concurrent.futures.ThreadPoolExecutor(18) as pool:
        peer = pool.submit(
            replay, "", "m", 761, 2, 0, "protected", ready, None, 100, tenant_b
        )
        if not ready.wait(60):
            raise ValueError("protected tenant did not begin its final answer")
        begin = time.monotonic_ns()
        drops = [pool.submit(flood, 0)]
        time.sleep(0.02)
        drops.extend(pool.submit(flood, i) for i in range(1, 17))
        rows = [f.result() for f in drops]
        protected = peer.result()
    deadline = time.monotonic() + 10
    after = metrics()
    while time.monotonic() < deadline and any(
        after[k] for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
    ):
        time.sleep(0.1)
        after = metrics()
    return {
        "before": before,
        "after": after,
        "protected": protected,
        "burst_start_ns": begin,
        "submitted": len(rows),
        "requests": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("credentials", type=Path)
    p.add_argument("policy", choices=["session", "round-robin", "cost"])
    a = p.parse_args()
    endpoints = [
        (
            (a.directory / f"gpu{i}" / "address").read_text().strip(),
            (a.credentials / str(i)).read_text().strip(),
        )
        for i in (6, 7)
    ]
    isolation = check_isolation(endpoints)
    (a.directory / "isolation.json").write_text(json.dumps(isolation, indent=2) + "\n")
    results = run(endpoints, a.policy)
    (a.directory / "capacity.json").write_text(json.dumps(results, indent=2) + "\n")

    burst = tenant_burst(endpoints)
    (a.directory / "tenant-burst.json").write_text(json.dumps(burst, indent=2) + "\n")


if __name__ == "__main__":
    main()
