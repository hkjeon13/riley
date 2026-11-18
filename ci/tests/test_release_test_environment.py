"""Unavailable historical execution prerequisites cannot become gate passes."""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_env", ROOT / "ci/release/release_test_environment.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class ReleaseEnvironmentTests(unittest.TestCase):
    def test_missing_host_is_an_explicit_skip_and_required_lane_fails(self):
        action = mock.Mock(return_value="executed")
        guarded = helper.requires_legacy_host(action)
        with mock.patch.object(helper, "legacy_host_unavailable_reason", return_value="missing reviewed host"):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(unittest.SkipTest, "missing reviewed host"):
                    guarded()
            with mock.patch.dict(os.environ, {"RILEY_REQUIRE_LEGACY_HOST_TESTS": "1"}, clear=True):
                with self.assertRaisesRegex(AssertionError, "missing reviewed host"):
                    guarded()
        action.assert_not_called()
        with mock.patch.object(helper, "legacy_host_unavailable_reason", return_value=None):
            self.assertEqual(guarded(), "executed")

    def test_linux_contract_lane_cannot_skip_an_unavailable_kernel(self):
        with mock.patch.object(helper.sys, "platform", "darwin"):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(unittest.SkipTest):
                    helper.require_linux_proc()
            with mock.patch.dict(os.environ, {"RILEY_REQUIRE_LINUX_CONTRACT_TESTS": "1"}, clear=True):
                with self.assertRaises(AssertionError):
                    helper.require_linux_proc()
