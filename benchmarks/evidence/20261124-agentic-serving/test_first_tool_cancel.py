"""Observe an unfinished preview disconnect and an independent completed final POST on loopback."""

import io
import json
import socket
import threading
import unittest
from http.server import ThreadingHTTPServer

from agent_workflow import StreamCancellation
from test_named_fleet import TestReplica
from tool_wait_trial import measure


class PreviewReplica(TestReplica):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        req = json.loads(body)
        if req["max_tokens"] == 1:
            self.server.preview = req
            self.connection.settimeout(3)
            self.server.saw_eof = self.connection.recv(1) == b""
            return
        self.server.final = req
        original = self.rfile
        try:
            self.rfile = io.BytesIO(body)
            super().do_POST()
        finally:
            self.rfile = original


class CancelTests(unittest.TestCase):
    def test_first_fixed_preview_disconnected_while_final_completes(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), PreviewReplica) as server:
            server.received = []
            server.saw_eof = False
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                result = measure(
                    f"127.0.0.1:{server.server_port}",
                    True,
                    "first",
                    17101,
                    0.12,
                    include_completed_call=True,
                    first_fixed_tool=True,
                    cancel_on_final=True,
                )
            finally:
                server.shutdown()
                thread.join()
            self.assertTrue(server.saw_eof)
            self.assertEqual(server.received, [(17101, "final")])
            self.assertEqual(
                [
                    m["content"]
                    for m in server.preview["messages"]
                    if m["role"] == "tool"
                ],
                [""],
            )
            self.assertEqual(
                len([m for m in server.final["messages"] if m["role"] == "tool"]), 2
            )
            self.assertTrue(result["target"]["ok"])
            t0, t1, last = result["target"]["turns"]
            self.assertEqual(
                [t0["dispatch"], t1["dispatch"]], ["application", "application"]
            )
            self.assertLessEqual(
                t1["tool_ready_ns"], result["prefix"]["cancel_requested_ns"]
            )
            self.assertLessEqual(
                result["prefix"]["cancel_requested_ns"], last["sent_ns"]
            )
            self.assertNotIn("done_ns", result["prefix"])
            self.assertEqual(last["finish"], "stop")

    def test_cancel_before_socket_registration(self):
        cancel = StreamCancellation()
        cancel.cancel()
        with socket.socket() as sock, self.assertRaises(ConnectionAbortedError):
            cancel.attach(sock)


if __name__ == "__main__":
    unittest.main()
