"""Exercise receipt recovery after a broken POST write; never replay the request."""

import http.client
import unittest
from unittest.mock import patch

from agent_workflow import stream


class EarlyRejectionTest(unittest.TestCase):
    def test_partial_post(self):
        for status in [429, 503, 200, None]:
            with (
                self.subTest(status=status),
                patch("agent_workflow.http.client.HTTPConnection") as factory,
            ):
                con = factory.return_value
                con.request.side_effect = BrokenPipeError("closed during POST")
                if status is None:
                    con.getresponse.side_effect = http.client.RemoteDisconnected()
                else:
                    con.getresponse.return_value.status = status
                    con.getresponse.return_value.read.return_value = b"rejected"
                row = {}
                expected = ValueError if status in (429, 503) else BrokenPipeError
                with self.assertRaises(expected):
                    stream("localhost:8080", {}, row)
                self.assertEqual(
                    row.get("status"), status if expected is ValueError else None
                )
                self.assertEqual(row["request_write_error"], "broken_pipe")
                self.assertNotIn("request_written_ns", row)
                con.request.assert_called_once()
                con.getresponse.assert_called_once()
                con.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
