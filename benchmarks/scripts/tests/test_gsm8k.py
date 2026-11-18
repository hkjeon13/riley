"""GSM8K scoring and response-evidence contracts."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


LIVE = Path(__file__).resolve().parents[2] / "live"
SPEC = importlib.util.spec_from_file_location("benchmark_gsm8k", LIVE / "gsm8k.py")
gsm = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(LIVE), *sys.path]):
    SPEC.loader.exec_module(gsm)


class Gsm8kTests(unittest.TestCase):
    def test_historical_scoring_retains_answer_marker_and_numeric_normalization(self):
        self.assertEqual(gsm.scored("27 + 5. #### 1,200.00", "1200"), ("1200.00", True))
        self.assertEqual(gsm.scored("The answer is -4.", "-4"), ("-4", True))
        self.assertEqual(gsm.scored("I cannot infer the answer.", "0"), (None, False))

    def test_valid_response_binds_prompt_ids_but_accepts_terminal_stop_stripping(self):
        reply = {
            "prompt_token_ids": [42, 1],
            "choices": [{"text": "5", "finish_reason": "stop", "token_ids": [4]}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        row = gsm.inspect_completion(reply, require_ids=True)
        self.assertEqual(row["returned_output_ids"], 1)
        self.assertEqual(row["prompt_ids_sha256"], gsm.id_digest([42, 1]))
        reply["prompt_token_ids"] = [42]
        with self.assertRaisesRegex(ValueError, "incoherent returned token IDs"):
            gsm.inspect_completion(reply, require_ids=True)

    def test_malformed_completion_does_not_score_as_an_incorrect_answer(self):
        reply = {"choices": [{"text": "23", "finish_reason": None}],
                 "usage": {"prompt_tokens": 2, "completion_tokens": 5}}
        with self.assertRaisesRegex(ValueError, "invalid text, finish reason or token usage"):
            gsm.inspect_completion(reply, require_ids=False)
        reply["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "invalid text, finish reason or token usage"):
            gsm.inspect_completion(reply, require_ids=False)
