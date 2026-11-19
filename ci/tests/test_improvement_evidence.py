"""Check that linked reports and published claims match their evidence."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("improvement_evidence", ROOT / "ci/check_improvement_evidence.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ReadmeEvidenceTests(unittest.TestCase):
    def test_readme_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            for filename in ("Cargo.toml", "docs/model-coverage-a100-perf-table.md"):
                (root / filename).write_bytes((ROOT / filename).read_bytes())
            path = root / "README.md"
            original = (ROOT / path.name).read_text(encoding="utf-8")
            path.write_text(original, encoding="utf-8")
            self.assertEqual(checker.check_readme(root), 24)
            for pattern, replacement, message in (
                (r"lower in \d+ of", "lower in 0 of", "summary differs for Time per output token"),
                (r"(higher in all \d+ cells — )\d+\.\d+×", r"\g<1>0.00×", "summary differs for Throughput"),
                (r"> Status: \S+", "> Status: invalid", "Status version must match Cargo"),
            ):
                with self.subTest(message=message):
                    changed, count = re.subn(pattern, replacement, original, count=1)
                    self.assertEqual(count, 1)
                    path.write_text(changed, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        checker.check_readme(root)


class AllocatorTrialTests(unittest.TestCase):
    def setUp(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        self.trial = manifest["current_allocator_trial"]
        self.distribution = manifest["current_distribution_candidate"]

    def test_linked_reports_pass(self):
        self.assertEqual(checker.check_allocator_trial(ROOT, self.trial, self.distribution), 5)

    def test_rehashed_evidence_from_another_arm_or_run_fails(self):
        for kind, replace, message in (
            ("accuracy", lambda report: report.update(combined_report_sha256="0" * 64), "combined GPU report"),
            ("performance", lambda report: report["arms"]["packaged-first"].update(image_id="sha256:" + "0" * 64),
             "different image"),
        ):
            with self.subTest(report=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                trial = deepcopy(self.trial)
                for role, entry in trial["evidence"].items():
                    raw = (ROOT / entry["path"]).read_bytes()
                    destination = root / entry["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if role == kind:
                        report = json.loads(raw)
                        replace(report)
                        raw = json.dumps(report).encode("utf-8")
                        entry["sha256"] = sha256(raw).hexdigest()
                    destination.write_bytes(raw)
                self.assertEqual(checker.check_evidence(root, trial), 5)
                with self.assertRaisesRegex(ValueError, message):
                    checker.check_allocator_trial(root, trial, self.distribution)
