#!/usr/bin/env python3
"""Check slow SSE readers against a dedicated Linux loopback server."""

import argparse
import asyncio
from hashlib import sha256
from ipaddress import IPv4Address
import json
import socket
import subprocess
import time

from operational_check import read_json, request, wait_idle
from soak import idle, read_completion, run_request


def send_queue(port, peer):
    selector = f"( sport = :{port} and dport = :{peer} )"
    output = subprocess.check_output(["ss", "-Htn", "state", "established", selector],
                                     text=True, timeout=3)
    if not output.strip():
        return None
    row, = output.splitlines()
    return int(row.split()[1])


async def open_slow_socket(conn, host, port, body):
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    conn.setblocking(False)
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(loop.sock_connect(conn, (host, port)), timeout=5)
    payload = json.dumps(body).encode()
    headers = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
    await asyncio.wait_for(loop.sock_sendall(conn, headers + payload), timeout=5)


async def observe_backlog(conn, port, record):
    peer = conn.getsockname()[1]
    receive_bytes = conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    record.update(receive_buffer_bytes=receive_bytes, send_queue_bytes=[])
    deadline = time.monotonic() + 12
    consecutive = 0
    while time.monotonic() < deadline:
        queued = await asyncio.to_thread(send_queue, port, peer)
        record["send_queue_bytes"].append(queued)
        consecutive = consecutive + 1 if queued is not None and queued > receive_bytes else 0
        if consecutive == 3:
            return
        await asyncio.sleep(.15)
    raise ValueError("did not observe a sustained TCP send queue above the receive buffer")


async def check(args, result):
    host, port_text = args.address.split(":")
    port = int(port_text)
    before = await wait_idle(args.address)
    result["before"] = before
    if not idle(before):
        raise ValueError("requires a dedicated idle server")
    body = request(args.model, "Write a long numbered guide to restoring a historic lighthouse. "
                   "Include hundreds of detailed steps. Start with section one and continue.",
                   stream=True, limit=args.max_tokens)
    body.update(logprobs=True, top_logprobs=20)
    result["request_sha256"] = sha256(json.dumps(body).encode()).hexdigest()

    with socket.socket() as conn:
        started = time.perf_counter()
        await open_slow_socket(conn, host, port, body)
        observation = result["resume"] = {}
        await observe_backlog(conn, port, observation)
        healthy = request(args.model, "Name one common bird.", stream=True, limit=16)
        observation["healthy_during_pause"] = await asyncio.wait_for(
            run_request(host, port, "plain", healthy, time.perf_counter()), timeout=6)
        peer = conn.getsockname()[1]
        observation["send_queue_after_healthy_bytes"] = await asyncio.to_thread(send_queue, port, peer)
        reader, writer = await asyncio.open_connection(sock=conn)
        try:
            observation["resumed"] = await asyncio.wait_for(
                read_completion(reader, "logprobs", args.max_tokens, started), timeout=120)
        finally:
            writer.close()
            await writer.wait_closed()
    between = await wait_idle(args.address)
    result["after_resume"] = between

    with socket.socket() as conn:
        await open_slow_socket(conn, host, port, body)
        observation = result["drop"] = {}
        await observe_backlog(conn, port, observation)
        observation["before_close"] = await asyncio.to_thread(read_json, args.address, "/metrics")
        conn.shutdown(socket.SHUT_RDWR)

    final = await wait_idle(args.address)
    result["after_close"] = final
    counters = before["counters"], between["counters"], final["counters"]
    result["checks"] = {
        "complete_healthy_during_pause": result["resume"]["healthy_during_pause"]["outcome"] == "completed"
        and (result["resume"]["send_queue_after_healthy_bytes"] or 0) > 0,
        "resume_sse_to_finish_and_usage": result["resume"]["resumed"]["outcome"] == "completed"
        and idle(between) and counters[1]["disconnects"] == counters[0]["disconnects"]
        and counters[1]["cancellations"] == counters[0]["cancellations"],
        "cancel_active_stream_after_close": result["drop"]["before_close"]["active_requests"] == 1
        and counters[2]["disconnects"] - counters[1]["disconnects"] == 1
        and counters[2]["cancellations"] - counters[1]["cancellations"] == 1,
        "resources_recovered": idle(final) and final["allocation"] == before["allocation"]
        and counters[2]["overloads"] == counters[0]["overloads"],
    }
    result["status"] = "passed" if all(result["checks"].values()) else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        host, port = args.address.split(":")
        valid = IPv4Address(host).is_loopback and 0 < int(port) < 65536 and args.max_tokens > 0
    except ValueError:
        valid = False
    if not valid:
        parser.error("provide a Linux IPv4 loopback address, its port and a positive output token limit")
    result = {"status": "failed"}
    try:
        asyncio.run(check(args, result))
    except (OSError, RuntimeError, ValueError, asyncio.TimeoutError, subprocess.SubprocessError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": result["status"], "checks": result.get("checks"), "error": result.get("error")}))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
