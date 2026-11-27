"""Run the same C16 report comparison requests through the report app HTTP API."""

import argparse
import concurrent.futures
import http.client
import itertools
import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from .agent_workflow import (
    answer_format,
    fingerprint,
    fixture,
    history_messages,
    percentile,
)

WAVES = ((4, 18100), (8, 18200), (16, 18300), (16, 18400))
WARM = 5091
REPORTS = ["report-A", "report-B"]


def cases():
    for size, base in WAVES:
        yield [(base + index, index % 4 == 0) for index in range(size)]


def records(case, reader):
    _, _, docs = fixture(case, 2)
    if reader:
        return docs, docs
    rows = [
        {
            "id": f"aux-{case}-{index:02d}",
            "p95_ms": 100 + (case + index) % 71,
            "statistic": "service response latency",
            "notes": ["daily archive"],
        }
        for index in range(62)
    ]
    rows.insert(3 + case % 19, docs[0])
    rows.insert(38 + case % 23, docs[1])
    return docs, rows


def prepare(database):
    if database.exists():
        raise FileExistsError(database)
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("""CREATE TABLE reports (
            case_id INTEGER NOT NULL, position INTEGER NOT NULL, report_id TEXT NOT NULL,
            document TEXT NOT NULL, PRIMARY KEY (case_id, position), UNIQUE (case_id, report_id)
        )""")
        for case, reader in itertools.chain([(WARM, True)], *cases()):
            _, rows = records(case, reader)
            db.executemany(
                "INSERT INTO reports VALUES (?, ?, ?, ?)",
                [
                    (case, position, row["id"], json.dumps(row))
                    for position, row in enumerate(rows)
                ],
            )


def run_case(address, case, reader, scheduled_ns, optimized):
    docs, rows = records(case, reader)
    body = {"case": case, "history": history_messages(case, 0 if reader else 200)}
    if not reader:
        body["report_ids"] = REPORTS
    selected = docs if optimized and not reader else rows
    owner = "application" if optimized and not reader else "model"
    row = {
        "case": case,
        "reader": reader,
        "scheduled_ns": scheduled_ns,
        "request_sha256": fingerprint(body),
        "events_ns": [],
        "ok": False,
    }
    time.sleep(max(0, (scheduled_ns - time.monotonic_ns()) / 1e9))
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=90)
    fragments = []
    row["sent_ns"] = time.monotonic_ns()
    try:
        connection.request(
            "POST", "/compare", json.dumps(body), {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        row["http_status"] = response.status
        if response.status != 200:
            raise ValueError(f"app HTTP {response.status}")
        event = None
        for line in response:
            if line.startswith(b"event: "):
                event = line[7:].strip()
            elif line.startswith(b"data: "):
                now = time.monotonic_ns()
                data = json.loads(line[6:])
                if event == b"delta":
                    row["events_ns"].append(now)
                    fragments.append(data)
                elif event == b"done":
                    row["receipt"] = data
                    row["done_ns"] = now
                elif event == b"error":
                    raise ValueError(data)
        receipt = row.get("receipt", {})
        events = row["events_ns"]
        if events:
            row.update(
                first_ms=(events[0] - scheduled_ns) / 1e6,
                last_ms=(events[-1] - scheduled_ns) / 1e6,
                gap_ms=percentile(
                    [(b - a) / 1e6 for a, b in itertools.pairwise(events)], 0.95
                ),
            )
        row["answer"] = "".join(fragments)
        plain = row["answer"].replace("*", "")
        before, after = (doc["p95_ms"] for doc in docs)
        values = {"before": before, "after": after, "reduction": before - after}
        row["answer_format"] = answer_format(row["answer"], docs)
        row["receipt_ok"] = (
            receipt.get("dispatch") == [owner, owner]
            and receipt.get("source_count") == len(rows)
            and receipt.get("source_sha256") == fingerprint(rows)
            and receipt.get("forwarded_count") == len(selected)
            and receipt.get("forwarded_sha256") == fingerprint(selected)
        )
        row["ok"] = (
            row["receipt_ok"]
            and len(events) >= 2
            and receipt["usage"]["completion_tokens"] >= 64
            and all(
                re.search(rf"\b{label}\s*:\s*{value}\s*ms\b", plain, re.IGNORECASE)
                for label, value in values.items()
            )
            and all(report.lower() in plain.lower() for report in REPORTS)
        )
    except (OSError, ValueError, KeyError, http.client.HTTPException) as error:
        row["error"] = str(error)
    finally:
        connection.close()
    row["slo"] = (
        row["ok"]
        and row["first_ms"] <= 5000
        and row["last_ms"] <= 8000
        and row["gap_ms"] <= 80
    )
    return row


def summarize(rows):
    usable = [row for row in rows if "last_ms" in row and "done_ns" in row]
    readers = [row for row in usable if row["reader"]]
    targets = [row for row in usable if not row["reader"]]
    return {
        "attempted": len(rows),
        "stream_completed": len(usable),
        "valid": sum(row["ok"] for row in rows),
        "slo": sum(row["slo"] for row in rows),
        "exact_90_words": sum(
            row.get("answer_format", {}).get("explanation_words") == 90 for row in rows
        ),
        "complete_p95_ms": percentile([row["last_ms"] for row in usable], 0.95),
        "reader_first_p95_ms": percentile([row["first_ms"] for row in readers], 0.95),
        "reader_gap_p95_ms": percentile(
            [row["gap_ms"] for row in readers if row["gap_ms"] is not None], 0.95
        ),
        "target_complete_p95_ms": percentile([row["last_ms"] for row in targets], 0.95),
        "target_input_tokens_p50": percentile(
            [row["receipt"]["usage"]["prompt_tokens"] for row in targets], 0.5
        ),
        "target_output_tokens_p50": percentile(
            [row["receipt"]["usage"]["completion_tokens"] for row in targets], 0.5
        ),
        "target_slo": sum(row["slo"] for row in rows if not row["reader"]),
        "send_delay_p95_ms": percentile(
            [(row["sent_ns"] - row["scheduled_ns"]) / 1e6 for row in rows], 0.95
        ),
    }


def measure(url, output, optimized):
    address = urlsplit(url)
    warm = run_case(address, WARM, True, time.monotonic_ns(), optimized)
    waves = []
    results = {"warm": warm, "waves": waves}
    if not warm["ok"]:
        output.write_text(json.dumps(results) + "\n")
        raise RuntimeError("warm-up failed")
    for group in cases():
        with concurrent.futures.ThreadPoolExecutor(len(group)) as pool:
            start = time.monotonic_ns() + 100_000_000
            futures = [
                pool.submit(
                    run_case,
                    address,
                    case,
                    reader,
                    start + index * 30_000_000,
                    optimized,
                )
                for index, (case, reader) in enumerate(group)
            ]
            waves.append([future.result() for future in futures])
        output.write_text(json.dumps(results) + "\n")
    results["c16"] = summarize(list(itertools.chain(*waves[-2:])))
    output.write_text(json.dumps(results) + "\n")
    print(json.dumps(results["c16"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("database", type=Path)
    run = commands.add_parser("measure")
    run.add_argument("address")
    run.add_argument("output", type=Path)
    run.add_argument("--optimized", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.database)
    else:
        measure(args.address, args.output, args.optimized)
