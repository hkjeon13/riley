"""Exercise drain, process loss and explicit recovery against isolated real replicas."""

import argparse
import concurrent.futures
import json
import secrets
import socket
import sqlite3
import tempfile
import threading
import time
from functools import partial
from pathlib import Path

from agent_workflow import fixture_tool_result, replay, stream
from replica_process import serve
from verify_agent_citations import with_citations


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


class Journal:
    def __init__(self, path, workflow_id):
        self.path = path
        self.workflow_id = workflow_id
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS action (workflow TEXT NOT NULL, turn INTEGER NOT NULL, "
                "payload TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY(workflow,turn))"
            )
        self.attempts = []

    def execute(self, index, call, docs):
        action = json.dumps(
            {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            },
            sort_keys=True,
        )
        identity = self.workflow_id, index
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            saved = db.execute(
                "SELECT payload,result FROM action WHERE workflow=? AND turn=?",
                identity,
            ).fetchone()
            if saved is None:
                result = fixture_tool_result(index, docs)
                db.execute(
                    "INSERT INTO action VALUES (?,?,?,?)",
                    (*identity, action, json.dumps(result)),
                )
            else:
                if saved[0] != action:
                    raise ValueError("logical tool action changed across retry")
                result = json.loads(saved[1])
        self.attempts.append({"turn": index, "committed": saved is None})
        return result


def workflow(
    endpoint, key, seed, name, visible=None, journal=None, *, explicit_citations=False
):
    def send(_, request, row, event):
        row["endpoint"] = endpoint
        if explicit_citations and row["turn"] == 2:
            request = with_citations(request)
        return stream(endpoint, request, row, event, bearer=key)

    return replay(
        endpoint,
        "m",
        seed,
        2,
        0,
        name,
        visible,
        history=20,
        send=send,
        execute_tool=journal.execute if journal else None,
    )


def ready(pair, label, job):
    deadline = time.monotonic() + 300
    path = pair / label / "address"
    while not path.is_file():
        if job.done():
            job.result()
            raise RuntimeError(f"{label} exited before readiness")
        if time.monotonic() > deadline:
            raise TimeoutError(label)
        time.sleep(0.1)
    state = json.loads((pair / label / "ready.json").read_text())
    if not state["ready"] or not state["accepting"]:
        raise ValueError("generation did not pass readiness")
    return path.read_text().strip()


def closed(endpoint):
    host, port = endpoint.split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return False
    except ConnectionRefusedError:
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("holder", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--explicit-citations", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    out.mkdir()
    run_workflow = partial(workflow, explicit_citations=args.explicit_citations)
    receipt = {"explicit_citations": args.explicit_citations}
    with (
        tempfile.TemporaryDirectory() as tmp,
        concurrent.futures.ThreadPoolExecutor(6) as pool,
    ):
        keys = Path(tmp)
        for idx in (6, 7):
            (keys / str(idx)).write_text(secrets.token_hex(32))
        key6, key7 = [(keys / str(idx)).read_text() for idx in (6, 7)]

        def launch(idx, label):
            return pool.submit(
                serve, root, args.holder.resolve(), out, idx, keys, label
            )

        old = launch(6, "old")
        survivor = launch(7, "survivor")
        jobs = [old, survivor]
        try:
            a = ready(out, "old", old)
            b = ready(out, "survivor", survivor)
            visible = threading.Event()
            in_flight = pool.submit(
                run_workflow, a, key6, 761, "drain-in-flight", visible
            )
            if not visible.wait(30):
                raise ValueError("expected in-flight final stream")
            receipt["withdraw_ns"] = time.monotonic_ns()
            # All new turns from this point use the surviving endpoint.
            next_work = pool.submit(run_workflow, b, key7, 762, "after-withdraw")
            receipt["drained"] = in_flight.result()
            (out / "old" / "retire").touch()
            old.result(timeout=75)
            receipt["old_now_closed"] = closed(a)
            start = time.monotonic()
            replacement = launch(6, "replacement")
            jobs.append(replacement)
            receipt["newly_routed"] = next_work.result()
            fresh = ready(out, "replacement", replacement)
            receipt["replacement_ready_s"] = time.monotonic() - start
            journal = Journal(out / "actions.sqlite", "logical-1")
            content = threading.Event()
            interrupted = pool.submit(
                run_workflow, b, key7, 761, "logical-1", content, journal
            )
            if not content.wait(30):
                raise ValueError("failed to observe content before process loss")
            (out / "survivor" / "inject-process-failure").touch()
            failure = time.monotonic()
            receipt["partial"] = interrupted.result(timeout=45)
            survivor.result(timeout=75)
            receipt["dead_before_generation"] = closed(b)
            receipt["fault_exit"] = int(
                (out / "survivor" / "fault-exit-code").read_text()
            )
            resumed = Journal(out / "actions.sqlite", "logical-1")
            receipt["recovered"] = run_workflow(
                fresh, key6, 761, "logical-1", journal=resumed
            )
            receipt["successful_recovery_s"] = time.monotonic() - failure
            receipt["tool_attempts"] = journal.attempts + resumed.attempts
            receipt["rollback_workflow"] = run_workflow(
                fresh, key6, 762, "after-rollback"
            )
            drained = receipt["drained"]
            interrupted_turn = receipt["partial"]["turns"][-1]
            receipt["checks"] = {
                "active_stream_drained": drained["ok"]
                and drained["turns"][-1]["content_ns"][0]
                <= receipt["withdraw_ns"]
                < drained["turns"][-1]["content_ns"][-1],
                "new_route_completed": receipt["newly_routed"]["ok"]
                and receipt["old_now_closed"],
                "partial_stream_detected": not receipt["partial"]["ok"]
                and bool(interrupted_turn["content_ns"])
                and "done_ns" not in interrupted_turn
                and receipt["fault_exit"] == 137,
                "recovered_after_loss": receipt["dead_before_generation"]
                and receipt["recovered"]["ok"]
                and receipt["successful_recovery_s"] <= 300,
                "idempotent_tool_effects": [
                    t["committed"] for t in receipt["tool_attempts"]
                ]
                == [True, True, False, False],
                "original_configuration_serves": receipt["rollback_workflow"]["ok"],
            }
        finally:
            write(out / "observations.json", receipt)
            (out / "finished").touch()
            for job in jobs:
                job.result(timeout=90)
    if not all(receipt["checks"].values()):
        raise ValueError(receipt["checks"])


if __name__ == "__main__":
    main()
