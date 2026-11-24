"""Exercise the public fleet CLI with pinned real HTTP listeners and independently encoded requests."""

import argparse
import io
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path

import agent_fleet_replay
from agent_workflow import fingerprint
from budgeted_tool_prefix import PromptProfiler
from test_named_fleet import TestReplica


class Replica(TestReplica):
    def do_POST(self):
        req = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(req)
        if self.headers.get("Authorization") != "Bearer " + self.server.key:
            self.send_error(401)
            return
        ids = self.server.profiler.ids(body)
        key = fingerprint(body)
        with self.server.lock:
            self.server.observed.append(
                {
                    "body": body,
                    "input_sha256": key,
                    "tokens": len(ids),
                    "ids_sha256": fingerprint(ids),
                }
            )
        if body["max_tokens"] == 1:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            usage = {
                "prompt_tokens": len(ids),
                "completion_tokens": 1,
                "total_tokens": len(ids) + 1,
            }
            msg = {
                "id": "preview-" + key[:12],
                "choices": [
                    {"index": 0, "delta": {"content": "."}, "finish_reason": "length"}
                ],
                "usage": usage,
            }
            if self.server.drift == "count":
                usage["prompt_tokens"] += 1
                usage["total_tokens"] += 1
            elif self.server.drift == "ids":
                msg["prompt_token_ids"] = [*ids[:-1], ids[-1] + 1]
            try:
                self.wfile.write(
                    ("data: " + json.dumps(msg) + "\n\ndata: [DONE]\n\n").encode()
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        old = self.rfile
        try:
            self.rfile = io.BytesIO(req)
            super().do_POST()
        finally:
            self.rfile = old


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("profiler", "tokenizer", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True)
    cfg = json.loads(
        Path(__file__).with_name("bounded-fleet-registration.json").read_text()
    )
    caps = [cfg["per_replica_blocks"], cfg["per_workflow_blocks"]]
    summary = {}
    with ExitStack() as stack:
        listener_profiler = stack.enter_context(PromptProfiler(a.profiler, a.tokenizer))
        env = dict(os.environ)
        replicas = []
        members = []
        for i in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Replica)
            server.profiler = listener_profiler
            server.lock = threading.Lock()
            server.received = []
            server.observed = []
            var = f"RILEY_FIXTURE_KEY_{i}"
            server.key = f"ephemeral-loopback-{i}"
            env[var] = server.key
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            stack.callback(server.server_close)
            stack.callback(thread.join)
            stack.callback(server.shutdown)
            replicas.append(server)
            members.append(
                {
                    "node": f"fixture-{i}",
                    "gpu_count": 1,
                    "url": f"http://127.0.0.1:{server.server_port}/v1",
                    "bearer_env": var,
                }
            )
        inv = a.out / "inventory.json"
        inv.write_text(json.dumps({"model": "m", "replicas": members}))
        baseline = {}
        for name, app, bounded, waves in [
            ("control", True, False, cfg["waves"]),
            ("bounded", True, True, cfg["waves"]),
            ("model", False, True, [2]),
            ("drift", True, True, [2, 2]),
        ]:
            for i, s in enumerate(replicas):
                s.drift = ("count", "ids")[i] if name == "drift" else None
                s.observed = []
                s.received = []
            file = a.out / f"{name}.json"
            cmd = [
                sys.executable,
                agent_fleet_replay.__file__,
                str(inv),
                str(file),
                "--waves",
                *map(str, waves),
            ]
            if app:
                cmd.append("--dispatch-fixed-tools")
            if bounded:
                cmd.extend(
                    [
                        "--preview-budget-blocks",
                        *map(str, caps),
                        "--prompt-profiler",
                        str(a.profiler),
                        "--tokenizer-dir",
                        str(a.tokenizer),
                    ]
                )
            else:
                cmd.extend(["--policy", "session"])
            subprocess.run(cmd, env=env, check=True, timeout=100)
            txt = file.read_text()
            assert all(s.key not in txt for s in replicas)
            result = json.loads(txt)
            assert result["policy"] == "session"
            rows = [j for w in result["waves"] for j in w["records"]]
            assert len(rows) == sum(waves) and len({j["seed"] for j in rows}) == len(
                rows
            )
            observed = [s.observed for s in replicas]
            (a.out / f"{name}-requests.json").write_text(json.dumps(observed) + "\n")
            calls = Counter()
            for i, group in enumerate(observed):
                calls.update((i, r["input_sha256"]) for r in group)
            skipped = 0
            for job in rows:
                assert job["ok"], job.get("error")
                owner = (job["seed"] - 761) % 2
                for t in job["turns"]:
                    if t.get("dispatch") == "application":
                        assert app
                        continue
                    assert calls[(owner, t["input_sha256"])] > 0
                    calls[(owner, t["input_sha256"])] -= 1
                final = job["turns"][2]
                rec = next(
                    r
                    for r in observed[owner]
                    if r["input_sha256"] == final["input_sha256"]
                )
                if name == "control":
                    baseline[job["seed"]] = rec["ids_sha256"]
                if name == "bounded":
                    assert rec["ids_sha256"] == baseline[job["seed"]]
                if bounded:
                    preview = job["preview"]
                    assert preview["replica"] == owner
                    if "input_sha256" not in preview:
                        skipped += 1
                        assert (
                            preview["decision"]
                            in (
                                "replica_preview_budget_exhausted",
                                "replica_preview_tokenizer_mismatch",
                            )
                            or preview.get("plan", {}).get("decision") == "cancelled"
                        )
                        continue
                    assert calls[(owner, preview["input_sha256"])] > 0
                    calls[(owner, preview["input_sha256"])] -= 1
                    r = next(
                        r
                        for r in observed[owner]
                        if r["input_sha256"] == preview["input_sha256"]
                    )
                    plan = preview["plan"]
                    assert bool(preview.get("tokenizer_mismatch")) == (name == "drift")
                    assert plan["planned_tokens"] == r["tokens"] <= 16 * caps[1]
                    assert (
                        plan["ids_sha256"] == r["ids_sha256"]
                        and plan["known_shared_blocks"] > 0
                    )
            assert not any(calls.values())
            budgets = result.get("preview_budget", [])
            assert all(
                b["peak_reserved"] <= caps[0] == b["remaining_after_drain"]
                for b in budgets
            )
            if name == "drift":
                assert skipped == 2 and all(b["tokenizer_mismatch"] for b in budgets)
            else:
                assert all(not b["tokenizer_mismatch"] for b in budgets)
            if name == "bounded":
                assert skipped > 0 and all(
                    b["peak_reserved"] == caps[0] for b in budgets
                )
            summary[name] = {
                "finals": len(rows),
                "previews_by_replica": [
                    sum(r["body"]["max_tokens"] == 1 for r in g) for g in observed
                ],
                "all_http": sum(map(len, observed)),
                "skips": skipped,
                "budgets": budgets,
            }
    (a.out / "verified.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
