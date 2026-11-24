"""Probe the same real continuation on a warm busy replica and an idle cold one."""

import concurrent.futures
import threading
import time

from agent_workflow import stream


def request(text, n):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": text}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": n,
    }


def run(a, b, key_a, key_b):
    results = []
    for repeat in range(2):
        for occupancy in (0, 4, 8):
            prompt = (
                f"Case {repeat}-{occupancy}. Review this case. "
                + "Observation of routine latency was recorded. " * 450
            )
            probe = request(prompt, 1)
            warm = {}
            stream(a, probe, warm, bearer=key_a)
            with concurrent.futures.ThreadPoolExecutor(10) as pool:
                signals = [threading.Event() for _ in range(occupancy)]
                peer_rows = [{} for _ in signals]
                ongoing = [
                    pool.submit(
                        stream,
                        a,
                        request(
                            f"Task {i}. Write an extensive multi-page novel about a village.",
                            192,
                        ),
                        row,
                        event,
                        bearer=key_a,
                    )
                    for i, (row, event) in enumerate(zip(peer_rows, signals))
                ]
                if not all(s.wait(30) for s in signals):
                    raise ValueError("could not establish ongoing streams")
                gate = threading.Barrier(2, timeout=5)

                def attempt(addr, key, gate=gate, probe=probe):
                    record = {}
                    gate.wait()
                    stream(addr, probe, record, bearer=key)
                    return record

                now = time.monotonic_ns()
                calls = [
                    pool.submit(attempt, addr, key)
                    for addr, key in ((a, key_a), (b, key_b))
                ]
                rows = [c.result() for c in calls]
                for f in ongoing:
                    f.result()
                results.append(
                    {
                        "repeat": repeat,
                        "streams": occupancy,
                        "decision_ns": now,
                        "warm": rows[0],
                        "cold": rows[1],
                        "peers": peer_rows,
                    }
                )
    return results


def overload(a, b, key_a, key_b):
    from agent_workflow import replay

    result = []
    for count in (16, 64, 160):
        ready = threading.Event()
        gate = threading.Barrier(count, timeout=30)

        def protected(host, req, row, signal):
            return stream(host, req, row, signal, bearer=key_b)

        def arrive(idx, gate=gate):
            row = {"index": idx}
            gate.wait()
            try:
                stream(
                    a,
                    request(
                        f"Traffic {idx}. Write a detailed technical explanation of solar energy.",
                        64,
                    ),
                    row,
                    bearer=key_a,
                )
            except (ValueError, OSError) as error:
                row["error"] = str(error)
            return row

        with concurrent.futures.ThreadPoolExecutor(count + 1) as pool:
            peer = pool.submit(
                replay, b, "m", 761, 2, 0, "other-tenant", ready, None, 20, protected
            )
            if not ready.wait(30):
                raise ValueError("protected stream did not start")
            jobs = [pool.submit(arrive, i) for i in range(count)]
            rows = [job.result() for job in jobs]
            reader = peer.result()
        recovered = {}
        stream(
            a,
            request("Name one common source of solar energy.", 8),
            recovered,
            bearer=key_a,
        )
        result.append(
            {
                "submitted": count,
                "requests": rows,
                "protected": reader,
                "post_wave": recovered,
            }
        )
    return result


def main():
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("credentials", type=Path)
    args = p.parse_args()
    a, b = [
        (args.directory / f"gpu{i}" / "address").read_text().strip() for i in (6, 7)
    ]
    ka, kb = [(args.credentials / str(i)).read_text() for i in (6, 7)]
    result = run(a, b, ka, kb)
    (args.directory / "cache-queue-crossover.json").write_text(
        json.dumps(result) + "\n"
    )
    waves = overload(a, b, ka, kb)
    (args.directory / "overload.json").write_text(json.dumps(waves) + "\n")


if __name__ == "__main__":
    main()
