"""HTTP report comparison app backed by a read-only SQLite report catalog."""

import argparse
import http.client
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from benchmarks.live.agent_tool_plan import resolve_named_step
from benchmarks.live.agent_tool_projection import project_exact_records
from benchmarks.live.agent_workflow import fingerprint, stream, tool, turn_request


def report_ids(value):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"[\w.-]{1,128}", item, re.ASCII)
            for item in value
        )
        or value[0] == value[1]
    ):
        raise ValueError("report_ids must contain two distinct report IDs")
    return value


def comparison_request(body):
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("case"), int)
        or isinstance(body["case"], bool)
    ):
        raise TypeError("case must be an integer")
    if not 0 <= body["case"] <= 2**63 - 1:
        raise ValueError("case is out of range")
    ids = report_ids(body["report_ids"]) if "report_ids" in body else None
    history = body.get("history", [])
    if not isinstance(history, list) or any(
        not isinstance(turn, dict)
        or set(turn) != {"role", "content"}
        or turn["role"] not in ("user", "assistant")
        or not isinstance(turn["content"], str)
        for turn in history
    ):
        raise ValueError("history must contain user or assistant text messages")
    return body["case"], ids, history


class ReportAgent:
    def __init__(
        self,
        db,
        address,
        model,
        *,
        direct=True,
        project=True,
        tool_delay=0,
        bearer=None,
    ):
        self.database = Path(db).resolve().as_uri() + "?mode=ro"
        self.address = address
        self.model = model
        self.direct = direct
        self.project = project
        self.tool_delay = tool_delay
        self.bearer = bearer

    def step(self, messages, tools, name, arguments, direct):
        request = turn_request(self.model, messages, tools, name)
        step = resolve_named_step(tools, request["tool_choice"]) if direct else None
        if step == (name, arguments):
            return {
                "type": "function",
                "id": f"call-{uuid4().hex}",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }, "application"

        row = {}
        _, calls = stream(self.address, request, row, bearer=self.bearer)
        if row["finish"] not in ("tool_calls", "stop") or len(calls) != 1:
            raise ValueError(f"expected one completed {name} call")
        call = calls[0]
        if (
            not call["id"]
            or call["function"]["name"] != name
            or json.loads(call["function"]["arguments"]) != arguments
        ):
            raise ValueError(f"invalid {name} call arguments")
        return call, "model"

    def compare(self, case, ids, history, emit):
        explicit = ids is not None
        with closing(sqlite3.connect(self.database, uri=True)) as db:
            # A read transaction gives both tools the same catalog snapshot.
            db.execute("BEGIN")
            if ids is None:
                ids = report_ids(
                    [
                        row[0]
                        for row in db.execute(
                            "SELECT report_id FROM reports WHERE case_id = ? ORDER BY position LIMIT 3",
                            (case,),
                        )
                    ]
                )
            first, second = ids
            arguments = {"first": first, "second": second}
            tools = [
                tool("list_reports", "Return the two report identifiers.", {}),
                tool(
                    "read_reports",
                    "Read the report pair returned by list_reports.",
                    {
                        field: {"type": "string", "enum": [value]}
                        for field, value in arguments.items()
                    },
                ),
            ]
            prompt = (
                f"Review case {case}. List the reports, then read them. Compare the before ({first}) and "
                f"after ({second}) latency. Start your final answer with 'Before: X ms. After: Y ms. "
                "Reduction: Z ms.' Fill in all three numbers. Then give a 90-word explanation of the "
                "comparison, citing both reports and explaining what p95 means. Do not invent measurements."
            )
            messages = history + [{"role": "user", "content": prompt}]
            direct = self.direct and explicit
            delay = self.tool_delay if explicit else 0
            call, list_dispatch = self.step(messages, tools, "list_reports", {}, direct)
            if delay:
                time.sleep(delay)
            listed = {
                row[0]
                for row in db.execute(
                    "SELECT report_id FROM reports WHERE case_id = ? AND report_id IN (?, ?)",
                    (case, first, second),
                )
            }
            if listed != set(ids):
                raise ValueError("requested reports were not returned by list_reports")
            append_result(messages, call, ids)

            call, read_dispatch = self.step(
                messages, tools, "read_reports", arguments, direct
            )
            if delay:
                time.sleep(delay)
            rows = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT document FROM reports WHERE case_id = ? ORDER BY position",
                    (case,),
                )
            ]

        selected = (
            project_exact_records(rows, key="id", requested_ids=ids)
            if explicit and self.project
            else rows
        )
        append_result(messages, call, selected)
        request = turn_request(self.model, messages, tools, None)
        result = {}
        _, calls = stream(
            self.address, request, result, bearer=self.bearer, on_text=emit
        )
        if result["finish"] != "stop" or calls:
            raise ValueError("final Riley response did not stop normally")
        return {
            "dispatch": [list_dispatch, read_dispatch],
            "source_count": len(rows),
            "source_sha256": fingerprint(rows),
            "forwarded_count": len(selected),
            "forwarded_sha256": fingerprint(selected),
            "usage": result["usage"],
        }


def append_result(messages, call, value):
    messages.extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(value)},
        ]
    )


def serve(agent, host, port):
    class Handler(BaseHTTPRequestHandler):
        def event(self, name, data):
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        def do_POST(self):
            if self.path != "/compare":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 262144:
                    raise ValueError("invalid request length")
                case, ids, history = comparison_request(
                    json.loads(self.rfile.read(size))
                )
            except (TypeError, ValueError) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                receipt = agent.compare(
                    case, ids, history, lambda text: self.event("delta", text)
                )
                self.event("done", receipt)
            except (BrokenPipeError, ConnectionResetError):
                return
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                sqlite3.Error,
                http.client.HTTPException,
            ) as error:
                self.log_error("comparison failed: %s", error)
                self.event("error", "comparison failed")

    with ThreadingHTTPServer((host, port), Handler) as server:
        print(
            f"http://{server.server_address[0]}:{server.server_address[1]}", flush=True
        )
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--riley", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model-tools", action="store_true")
    parser.add_argument("--full-catalog", action="store_true")
    parser.add_argument("--tool-delay-ms", type=int, default=0)
    args = parser.parse_args()
    if args.tool_delay_ms < 0:
        parser.error("tool delay must be nonnegative")
    serve(
        ReportAgent(
            args.database,
            args.riley,
            args.model,
            direct=not args.model_tools,
            project=not args.full_catalog,
            tool_delay=args.tool_delay_ms / 1000,
            bearer=os.getenv("RILEY_API_KEY"),
        ),
        "127.0.0.1",
        args.port,
    )
