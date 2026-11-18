"""Reject stale and malformed current serving contracts without a GPU."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("serving_contract", ROOT / "ci/check_serving_source_contract.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ServingSourceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for relative in checker.SOURCE_PATHS | {str(checker.CONTRACT_PATH)}:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def test_reviewed_sources_pass(self):
        checker.validate(self.root)

    def test_every_source_rejects_byte_drift_and_missing_files(self):
        for relative in checker.SOURCE_PATHS:
            with self.subTest(relative=relative):
                path = self.root / relative
                raw = path.read_bytes()
                path.write_bytes(raw + b"\n")
                with self.assertRaisesRegex(checker.SourceContractError, "source drift"):
                    checker.validate(self.root)
                path.unlink()
                with self.assertRaises(OSError):
                    checker.validate(self.root)
                path.write_bytes(raw)

    def test_sources_and_parent_directories_cannot_be_links(self):
        source = self.root / "crates/riley-server/src/main.rs"
        source.unlink()
        source.symlink_to(ROOT / "crates/riley-server/src/main.rs")
        with self.assertRaisesRegex(checker.SourceContractError, "not a link"):
            checker.validate(self.root)
        source.unlink()
        shutil.copyfile(ROOT / "crates/riley-server/src/main.rs", source)
        parent = self.root / "deploy/docker"
        parent.rename(parent.with_name("held-docker"))
        parent.symlink_to(parent.with_name("held-docker"), target_is_directory=True)
        with self.assertRaisesRegex(checker.SourceContractError, "not a link"):
            checker.validate(self.root)

    def test_inventory_unknown_fields_and_duplicate_keys_fail(self):
        path = self.root / checker.CONTRACT_PATH
        original = path.read_text()
        value = json.loads(original)
        for mutation in (
            {**value, "extra": True},
            {**value, "sources": {**value["sources"], "../escape": "a" * 64}},
            {**value, "sources": {}},
            {**value, "distribution": "unreviewed"},
        ):
            with self.subTest(mutation=mutation):
                path.write_text(json.dumps(mutation))
                with self.assertRaises(checker.SourceContractError):
                    checker.validate(self.root)
        path.write_text('{"schema_version":"a","schema_version":"b"}')
        with self.assertRaisesRegex(checker.SourceContractError, "duplicate"):
            checker.validate(self.root)
