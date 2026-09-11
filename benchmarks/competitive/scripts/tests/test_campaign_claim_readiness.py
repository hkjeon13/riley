"""Independent C01 claim-readiness regression tests.

These cases intentionally rebuild plans and raw evidence around hostile but
well-formed values.  They prove a favorable checker result needs the declared
artifact workspace, a reproducible fully materialized lane, one complete raw
journal chain, and live executable/lock receipts bound to that lane.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
for path in (TESTS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import check_campaign  # noqa: E402
import competitive_common  # noqa: E402
from test_campaign_contract import CampaignFixture, _write_json  # noqa: E402


def _claim_report(
    fixture: CampaignFixture,
    *,
    plan_path: Path | None = None,
    raw_paths: list[Path] | None = None,
) -> dict[str, object]:
    with patch.object(
        check_campaign,
        "current_source_receipt",
        return_value={"git_revision": "d" * 40, "git_dirty": False},
    ):
        return check_campaign.check_campaign(
            plan_path=fixture.plan_path if plan_path is None else plan_path,
            raw_paths=[fixture.raw_path] if raw_paths is None else raw_paths,
            root=fixture.root,
        )


def _write_rebuilt_plan_and_rebind_raw(
    fixture: CampaignFixture,
    plan: dict[str, object],
) -> None:
    """Model a hostile plan rebuild while retaining a valid JSONL chain."""

    fixture.plan = plan
    _write_json(fixture.plan_path, plan)
    fixture.plan_sha256 = hashlib.sha256(fixture.plan_path.read_bytes()).hexdigest()
    workloads = {
        str(receipt["cell_id"]): receipt
        for receipt in plan["workloads"]  # type: ignore[index]
    }
    for row in fixture.rows:
        row["campaign_plan_sha256"] = fixture.plan_sha256
        row["source"] = plan["source"]  # type: ignore[index]
        receipt = workloads[str(row["cell_id"])]
        row["workload_sha256"] = receipt["sha256"]
        row["workload"] = competitive_common.workload_execution_receipt(
            receipt["value"]  # type: ignore[arg-type]
        )
    fixture.write_rows()


class CampaignClaimReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = CampaignFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_legacy_nonmaterialized_lane_cannot_claim_passed_or_partial_win(self) -> None:
        """A forged ready plan cannot revive a placeholder-based legacy lane."""

        legacy = competitive_common.load_json(self.fixture.riley_lane_path)
        template = competitive_common.load_json(
            self.fixture.root / "benchmarks/competitive/lanes/riley.json"
        )
        legacy.pop("materialization")
        legacy["command"] = deepcopy(template["command"])
        legacy["command"]["status"] = "available"
        legacy_path = self.fixture.workspace / "legacy-riley-lane.json"
        _write_json(legacy_path, legacy)

        rebuilt = self.fixture._build_plan(
            riley_lane_path=legacy_path,
            require_executable_lanes=False,
        )
        rebuilt["readiness"] = {"state": "ready", "blocked_reasons": []}
        _write_rebuilt_plan_and_rebind_raw(self.fixture, rebuilt)

        report = _claim_report(self.fixture)
        self.assertEqual(report["status"], "incomparable")
        reasons = report["comparability"]["reasons"]  # type: ignore[index]
        self.assertTrue(any("materialization" in reason for reason in reasons))

    def test_every_claim_row_requires_all_three_journal_fields(self) -> None:
        rows = [deepcopy(row) for row in self.fixture.rows]
        rows[0].pop("adapter_previous_receipt_sha256")
        self.fixture.raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

        report = _claim_report(self.fixture)
        self.assertEqual(report["status"], "incomparable")
        self.assertTrue(
            any(
                "adapter_previous_receipt_sha256" in reason
                for reason in report["comparability"]["reasons"]  # type: ignore[index]
            )
        )

        for row in rows:
            for field in ("adapter_sequence", "adapter_previous_receipt_sha256", "adapter_receipt_sha256"):
                row.pop(field, None)
        self.fixture.raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.assertEqual(_claim_report(self.fixture)["status"], "incomparable")

    def test_claim_rejects_multiple_raw_files_even_when_each_row_is_valid(self) -> None:
        second_journal = self.fixture.workspace / "second-valid-chain.jsonl"
        second_journal.write_bytes(self.fixture.raw_path.read_bytes())

        report = _claim_report(self.fixture, raw_paths=[self.fixture.raw_path, second_journal])
        self.assertEqual(report["status"], "incomparable")
        self.assertTrue(
            any(
                "exactly one JSONL journal" in reason
                for reason in report["comparability"]["reasons"]  # type: ignore[index]
            )
        )

        raw_directory = self.fixture.workspace / "raw-directory"
        raw_directory.mkdir()
        (raw_directory / "copied-chain.jsonl").write_bytes(self.fixture.raw_path.read_bytes())
        report = _claim_report(self.fixture, raw_paths=[raw_directory])
        self.assertEqual(report["status"], "incomparable")
        self.assertTrue(
            any(
                "exactly one regular JSONL journal" in reason
                for reason in report["comparability"]["reasons"]  # type: ignore[index]
            )
        )

    def test_environment_executable_and_lock_hashes_must_match_lane_receipt(self) -> None:
        self.fixture.rows[0]["environment"]["executable_sha256"] = "0" * 64  # type: ignore[index]
        self.fixture.write_rows()
        report = _claim_report(self.fixture)
        self.assertEqual(report["status"], "incomparable")
        self.assertTrue(
            any(
                "executable_sha256 differs from the materialized" in reason
                for reason in report["comparability"]["reasons"]  # type: ignore[index]
            )
        )

        self.fixture.rows[0]["environment"]["executable_sha256"] = "8" * 64  # type: ignore[index]
        self.fixture.rows[0]["environment"]["dependency_lock_sha256"] = "0" * 64  # type: ignore[index]
        self.fixture.write_rows()
        report = _claim_report(self.fixture)
        self.assertEqual(report["status"], "incomparable")
        self.assertTrue(
            any(
                "dependency_lock_sha256 differs from the materialized" in reason
                for reason in report["comparability"]["reasons"]  # type: ignore[index]
            )
        )

    def test_rebuilt_plan_cannot_authorize_hand_edited_executable_argv(self) -> None:
        lane = competitive_common.load_json(self.fixture.riley_lane_path)
        lane["command"]["argv"][0] = "/opt/campaign/forged-riley"
        # An attacker may recompute the local argv hash and rebuild the plan;
        # the immutable input/template re-materialization must still disagree.
        lane["materialization"]["expanded_argv_sha256"] = competitive_common.sha256_bytes(
            competitive_common.canonical_json_bytes(lane["command"]["argv"])
        )
        _write_json(self.fixture.riley_lane_path, lane)

        rebuilt = self.fixture._build_plan(require_executable_lanes=False)
        rebuilt["readiness"] = {"state": "ready", "blocked_reasons": []}
        _write_rebuilt_plan_and_rebind_raw(self.fixture, rebuilt)

        report = _claim_report(self.fixture)
        self.assertEqual(report["status"], "incomparable")
        self.assertTrue(
            any(
                "materialization receipt is not reproducible" in reason
                for reason in report["comparability"]["reasons"]  # type: ignore[index]
            )
        )

    def test_claim_plan_must_stay_in_declared_artifact_workspace(self) -> None:
        outside_plan = self.fixture.root / "campaigns" / "outside-plan.json"
        _write_json(outside_plan, self.fixture.plan)
        report = _claim_report(self.fixture, plan_path=outside_plan)
        self.assertEqual(report["status"], "incomparable")
        self.assertTrue(
            any(
                "claim execution plan path must stay inside" in reason
                for reason in report["comparability"]["reasons"]  # type: ignore[index]
            )
        )

    def test_declared_artifact_workspace_is_gitignored_without_relaxing_source_cleanliness(self) -> None:
        relative_probe = (
            competitive_common.CAMPAIGN_ARTIFACT_WORKSPACE_RELATIVE_PATH
            + "/claim-readiness-probe.json"
        )
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative_probe],
            cwd=SCRIPTS.parents[2],
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
