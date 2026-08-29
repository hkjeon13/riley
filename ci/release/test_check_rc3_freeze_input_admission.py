#!/usr/bin/env python3
"""CPU-only hostile-path tests for RC3 freeze-input structural admission."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import check_rc3_freeze_input_admission as checker  # noqa: E402
import check_rc3_prefreeze as prefreeze  # noqa: E402
import provenance_v2_common as common  # noqa: E402


CANDIDATE_ID = "riley-0.1.0-rc3"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class AdmissionFixture:
    def __init__(self, base: Path) -> None:
        base.mkdir(parents=True, exist_ok=True)
        base.chmod(0o700)
        self.root = base / "checkout"
        self.root.mkdir(parents=True)
        self.defaults = b"fixture reviewed Rust serve defaults\n"
        self._write_source_tree()
        self._git("init", "--quiet")
        self._git("add", "--all")
        self._git(
            "-c",
            "user.name=RC3 fixture",
            "-c",
            "user.email=rc3-fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture source",
        )
        self.evidence = base / "evidence"
        self.evidence.mkdir(mode=0o700)
        self.evidence.chmod(0o700)
        self.request_name = "freeze-input-request.json"
        self.baseline_document = self._baseline_document()
        self.request = self._request_document()
        self.write_request()

    def _write_source_tree(self) -> None:
        (self.root / "Cargo.toml").write_text(
            "[workspace]\n"
            'members = ["crates/riley-server"]\n'
            "\n"
            "[workspace.package]\n"
            'version = "0.1.0"\n'
            'license = "MIT"\n',
            encoding="utf-8",
        )
        (self.root / "Cargo.lock").write_text(
            "# fixture lockfile\nversion = 4\n",
            encoding="utf-8",
        )
        registry = self.root / "deploy/extensions"
        registry.mkdir(parents=True)
        (registry / "registry.json").write_text(
            '{"$schema":"registry.schema.json","schema_version":"riley.extension-registry.v1","extensions":[]}\n',
            encoding="utf-8",
        )
        server = self.root / "crates/riley-server"
        server.mkdir(parents=True)
        (server / "Cargo.toml").write_text(
            "[package]\n"
            'name = "riley-server"\n'
            "license.workspace = true\n",
            encoding="utf-8",
        )
        (server / "src").mkdir()
        (server / "src/main.rs").write_bytes(self.defaults)

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", os.fspath(self.root), *arguments],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @property
    def revision(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.decode("ascii").strip()

    def put(self, relative_path: str, raw: bytes) -> dict[str, object]:
        path = self.evidence / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return common.descriptor_for_bytes(relative_path, raw, relative_path).as_json()

    def synthetic_descriptor(self, path: str, seed: str) -> dict[str, object]:
        raw = seed.encode("ascii")
        return {
            "path": path,
            "sha256": _sha256(raw),
            "byte_length": len(raw),
        }

    def _baseline_document(self) -> dict[str, object]:
        tag_name = "riley-0.1.0-rc2"
        source = {
            "tag_name": tag_name,
            "tag_object": self.synthetic_descriptor("historic/tag-object", "tag-object"),
            "tag_target": self.synthetic_descriptor("historic/tag-target", "tag-target"),
            "archive": self.synthetic_descriptor("historic/source-archive", "source-archive"),
        }

        def pair(name: str) -> dict[str, object]:
            left = self.synthetic_descriptor(f"historic/{name}-a", name)
            right = self.synthetic_descriptor(f"historic/{name}-b", name)
            return {"a": left, "b": right, "sha256": left["sha256"]}

        image = "sha256:" + _sha256(b"historic-image")
        return {
            "schema_version": "riley.reconstructed-prior-baseline.v1",
            "baseline_id": f"reconstructed-{tag_name}",
            "baseline_kind": "reconstructed-tag-baseline",
            "provenance_class": "reconstructed-from-source",
            "historical_distribution": "not-attested",
            "historical_stable_artifact_status": "unavailable",
            "was_previously_shipped": False,
            "source": source,
            "reproductions": {
                "a": self.synthetic_descriptor("historic/reproduction-a", "receipt-a"),
                "b": self.synthetic_descriptor("historic/reproduction-b", "receipt-b"),
            },
            "equality": {
                "bundle": pair("bundle"),
                "oci_archive": pair("oci-archive"),
                "oci_layout": pair("oci-layout"),
                "oci_manifest": pair("oci-manifest"),
                "oci_image": {"a": image, "b": image, "image_id": image},
            },
        }

    def write_baseline(self) -> dict[str, object]:
        return self.put(
            "rollback/reconstructed-baseline.json",
            common.canonical_json_bytes(self.baseline_document),
        )

    def _request_document(self) -> dict[str, object]:
        cargo_lock = (self.root / "Cargo.lock").read_bytes()
        registry = (self.root / "deploy/extensions/registry.json").read_bytes()
        baseline = self.write_baseline()
        return {
            "schema_version": checker.REQUEST_VERSION,
            "candidate_id": CANDIDATE_ID,
            "source": {
                "git_revision": self.revision,
                "archive": self.put("source/archive.tar", b"source archive bytes"),
                "cargo_lock": self.put("source/Cargo.lock", cargo_lock),
                "extension_registry": self.put(
                    "source/extension-registry.json",
                    registry,
                ),
            },
            "release": {
                "elf": self.put("release/riley-server", b"release ELF bytes"),
                "container": {
                    "image_id": "sha256:" + _sha256(b"fixture image id"),
                    "image_digest": "sha256:" + _sha256(b"fixture image digest"),
                    "inspect": self.put(
                        "release/container-inspect.json",
                        b'{"Id":"fixture"}\n',
                    ),
                },
            },
            "toolchain": {
                "probe": self.put("toolchain/probe.txt", b"fixture probe\n"),
                "cuda_c_abi_version": "12",
                "rust_version": "rustc 1.80.0",
                "nvcc_version": "12.4",
                "driver_version": "550.54",
                "cuda_runtime_version": "12.4",
                "cuda_toolkit_version": "12.4",
                "cublas_version": "12.4",
            },
            "models": [
                {
                    "model_id": "fixture-model",
                    "revision": "0123456789abcdef",
                    "tree": self.put("models/tree.txt", b"tree\n"),
                    "config": self.put("models/config.json", b"{}\n"),
                    "tokenizer": self.put("models/tokenizer.json", b"{}\n"),
                    "weights": [
                        self.put("models/weights-00001.safetensors", b"weights\n")
                    ],
                }
            ],
            "launch_profiles": [
                {
                    "profile": "stable-default",
                    "arguments": self.put("launch/stable.args", b""),
                    "environment": self.put(
                        "launch/stable.env",
                        b"PATH=/usr/bin:/bin\n",
                    ),
                },
                {
                    "profile": "max-performance-exact",
                    "arguments": self.put(
                        "launch/max.args",
                        b"--sampling-backend\ngpu-greedy\n",
                    ),
                    "environment": self.put(
                        "launch/max.env",
                        b"PATH=/usr/bin:/bin\nCUDA_VISIBLE_DEVICES=0\n",
                    ),
                },
            ],
            "correctness": {
                "contract": self.put("correctness/contract.json", b"contract\n"),
                "report": self.put("correctness/report.json", b"report\n"),
            },
            "rollback": {
                "reconstructed_baseline_manifest": baseline,
            },
        }

    def write_request(self) -> None:
        (self.evidence / self.request_name).write_bytes(
            common.canonical_json_bytes(self.request)
        )

    def run(
        self,
        *,
        expected_revision: str | None = None,
        candidate_id: str = CANDIDATE_ID,
    ) -> dict[str, object]:
        with mock.patch.object(
            prefreeze,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(self.defaults),
        ):
            return checker.check_rc3_freeze_input_admission(
                self.root,
                expected_revision or self.revision,
                candidate_id,
                self.evidence,
                self.request_name,
            )

    def source_status(self) -> bytes:
        return self._git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=no",
        ).stdout

    def evidence_snapshot(self) -> dict[str, bytes]:
        return {
            path.relative_to(self.evidence).as_posix(): path.read_bytes()
            for path in sorted(self.evidence.rglob("*"))
            if path.is_file()
        }


class CheckRc3FreezeInputAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve(strict=True)
        self.fixture = AdmissionFixture(self.base)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext,
        reason: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def test_valid_request_is_bound_not_frozen_and_read_only(self) -> None:
        before_evidence = self.fixture.evidence_snapshot()
        report = self.fixture.run()

        self.assertEqual(
            set(report),
            {
                "schema_version",
                "scope",
                "status",
                "authority",
                "candidate_status",
                "qualification_status",
                "candidate_id",
                "source_revision",
                "workspace_version",
                "request",
                "source_pre_freeze",
                "bound_inputs",
                "reconstructed_baseline",
                "rollback_scope",
                "checks",
                "reason_codes",
            },
        )
        self.assertEqual(report["schema_version"], checker.REPORT_VERSION)
        self.assertEqual(report["scope"], checker.SCOPE)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["authority"], checker.STRUCTURAL_AUTHORITY)
        self.assertEqual(report["candidate_status"], "not-frozen")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["candidate_id"], CANDIDATE_ID)
        self.assertEqual(report["source_revision"], self.fixture.revision)
        self.assertEqual(
            report["rollback_scope"],
            "reconstructed-baseline-vocabulary-only",
        )
        self.assertEqual(
            report["reconstructed_baseline"],
            {
                "baseline_id": "reconstructed-riley-0.1.0-rc2",
                "tag_name": "riley-0.1.0-rc2",
                "relationship": "immediately-prior-rc-same-semver",
            },
        )
        self.assertEqual(report["reason_codes"], [])
        self.assertNotIn("passed", report)
        self.assertNotIn("freeze_sha256", report)
        self.assertNotIn("gate_e_report_sha256", report)
        self.assertNotIn("configuration_sha256", report)
        self.assertNotIn("base_release_candidate_report_sha256", report)
        self.assertEqual(self.fixture.source_status(), b"")
        self.assertEqual(self.fixture.evidence_snapshot(), before_evidence)

        bound_inputs = report["bound_inputs"]
        self.assertIsInstance(bound_inputs, dict)
        assert isinstance(bound_inputs, dict)
        self.assertEqual(
            bound_inputs["source"]["cargo_lock"],
            self.fixture.request["source"]["cargo_lock"],
        )
        self.assertEqual(
            bound_inputs["source"]["extension_registry"],
            self.fixture.request["source"]["extension_registry"],
        )

    def test_cli_emits_canonical_stdout_only(self) -> None:
        stdout = _CapturedStdout()
        with mock.patch.object(
            prefreeze,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(self.fixture.defaults),
        ), mock.patch.object(checker.sys, "stdout", stdout):
            result = checker.main(
                [
                    "--repository-root",
                    os.fspath(self.fixture.root),
                    "--expected-revision",
                    self.fixture.revision,
                    "--candidate-id",
                    CANDIDATE_ID,
                    "--evidence-root",
                    os.fspath(self.fixture.evidence),
                    "--request",
                    self.fixture.request_name,
                ]
            )
        self.assertEqual(result, 0)
        raw = stdout.buffer.getvalue()
        self.assertTrue(raw.endswith(b"\n"))
        report = json.loads(raw)
        self.assertEqual(raw[:-1], common.canonical_json_bytes(report))
        self.assertEqual(report["status"], "bound")
        self.assertEqual(self.fixture.source_status(), b"")

    def test_candidate_and_revision_cross_bindings_fail_closed(self) -> None:
        self.assertEqual(
            checker._expected_reconstructed_baseline_tag("riley-0.1.0-rc10"),
            "riley-0.1.0-rc9",
        )
        self.assertEqual(
            checker._expected_reconstructed_baseline_tag("riley-0.1.0-rc100"),
            "riley-0.1.0-rc99",
        )
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            checker._expected_reconstructed_baseline_tag("riley-0.1.0-rc1")
        self.assert_reason(raised, "no-prior-rc-baseline")

        self.fixture.request["candidate_id"] = "riley-0.1.0-rc2"
        self.fixture.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "candidate-id-mismatch")

        fresh = AdmissionFixture(self.base / "revision")
        fresh.request["source"]["git_revision"] = "1" * 40
        fresh.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            fresh.run()
        self.assert_reason(raised, "source-revision-mismatch")

    def test_external_source_lock_and_registry_must_match_prefreeze(self) -> None:
        self.fixture.request["source"]["cargo_lock"] = self.fixture.put(
            "source/Cargo.lock",
            b"other Cargo lock\n",
        )
        self.fixture.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "source-input-drift")

        fresh = AdmissionFixture(self.base / "registry")
        fresh.request["source"]["extension_registry"] = fresh.put(
            "source/extension-registry.json",
            b'{"extensions":["drift"]}\n',
        )
        fresh.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            fresh.run()
        self.assert_reason(raised, "source-input-drift")

    def test_total_external_evidence_budget_is_bounded_before_rehash(self) -> None:
        self.fixture.request["source"]["archive"]["byte_length"] = (
            checker.MAX_TOTAL_EXTERNAL_INPUT_BYTES + 1
        )
        self.fixture.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "external-evidence-byte-budget-exceeded")

    def test_launch_profiles_runner_owned_and_self_references_are_rejected(self) -> None:
        self.fixture.request["launch_profiles"][0]["profile"] = "max-performance-exact"
        self.fixture.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "invalid-launch-profile-order")

        fresh = AdmissionFixture(self.base / "arguments")
        fresh.request["launch_profiles"][0]["arguments"] = fresh.put(
            "launch/stable.args",
            b"--model=forbidden\n",
        )
        fresh.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            fresh.run()
        self.assert_reason(raised, "runner-owned-launch-argument")

        env = AdmissionFixture(self.base / "environment")
        env.request["launch_profiles"][1]["environment"] = env.put(
            "launch/max.env",
            b"PATH=/usr/bin:/bin\nRILEY_FREEZE_SHA=forbidden\n",
        )
        env.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            env.run()
        self.assert_reason(raised, "self-referential-launch-environment")

    def test_reconstructed_baseline_is_vocabulary_only_and_keeps_historical_limits(self) -> None:
        report = self.fixture.run()
        self.assertEqual(
            report["checks"][6],
            {"name": "reconstructed-baseline-vocabulary-only", "bound": True},
        )

        self.fixture.baseline_document["historical_distribution"] = "attested"
        self.fixture.request["rollback"]["reconstructed_baseline_manifest"] = (
            self.fixture.write_baseline()
        )
        self.fixture.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "historical-distribution-claim")

        mismatch = AdmissionFixture(self.base / "baseline-mismatch")
        mismatch.baseline_document["source"]["tag_name"] = "riley-0.1.0-rc3"
        mismatch.baseline_document["baseline_id"] = "reconstructed-riley-0.1.0-rc3"
        mismatch.request["rollback"]["reconstructed_baseline_manifest"] = (
            mismatch.write_baseline()
        )
        mismatch.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            mismatch.run()
        self.assert_reason(raised, "reconstructed-baseline-tag-mismatch")

        malformed = AdmissionFixture(self.base / "baseline-malformed")
        malformed.baseline_document["source"]["tag_object"]["sha256"] = "x" * 64
        malformed.request["rollback"]["reconstructed_baseline_manifest"] = (
            malformed.write_baseline()
        )
        malformed.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            malformed.run()
        self.assert_reason(raised, "invalid-descriptor")

    def test_rejects_descriptor_reuse_and_request_leaf_reuse(self) -> None:
        self.fixture.request["release"]["elf"] = self.fixture.request["source"]["archive"]
        self.fixture.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "duplicate-evidence-path")

        fresh = AdmissionFixture(self.base / "request-leaf")
        fresh.request["source"]["archive"] = {
            "path": fresh.request_name,
            "sha256": "1" * 64,
            "byte_length": 1,
        }
        fresh.write_request()
        with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
            fresh.run()
        self.assert_reason(raised, "request-descriptor-path-reused")

    def test_request_mutation_and_exclusive_lock_are_rejected(self) -> None:
        original = checker._verify_opaque_inputs

        def mutate(root_fd: int, request: dict[str, object]) -> None:
            original(root_fd, request)
            self.fixture.request["toolchain"]["driver_version"] = "550.55"
            self.fixture.write_request()

        with mock.patch.object(checker, "_verify_opaque_inputs", mutate), self.assertRaises(
            checker.FreezeInputAdmissionError
        ) as raised:
            self.fixture.run()
        self.assert_reason(raised, "request-changed-during-admission")

        locked = AdmissionFixture(self.base / "locked")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        root_fd = os.open(locked.evidence, flags)
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
                locked.run()
            self.assert_reason(raised, "evidence-root-lock-unavailable")
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def test_bytecode_guard_and_static_surface(self) -> None:
        with mock.patch.object(checker, "_BYTECODE_DISABLED_AT_STARTUP", False):
            with self.assertRaises(checker.FreezeInputAdmissionError) as raised:
                self.fixture.run()
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")

        source = Path(checker.__file__).read_text(encoding="utf-8")
        tree = compile(source, checker.__file__, "exec", flags=0, dont_inherit=True)
        self.assertIsNotNone(tree)
        for forbidden in (
            "import release_common",
            "import check_release_candidate",
            "import write_release_candidate_manifest",
            "import subprocess",
            "import socket",
            "os.O_CREAT",
            "os.rename",
            "--output",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn("_BYTECODE_DISABLED_AT_STARTUP", source)
        self.assertLess(
            source.index("sys.dont_write_bytecode = True"),
            source.index("import provenance_v2_common"),
        )

    def test_schemas_are_closed_and_do_not_offer_semantic_fields(self) -> None:
        schema_root = Path(__file__).resolve().parents[2] / "benchmarks/release/candidates"
        for name, version in (
            ("rc3-freeze-input-request-v1.schema.json", checker.REQUEST_VERSION),
            ("rc3-freeze-input-admission-v1.schema.json", checker.REPORT_VERSION),
        ):
            with self.subTest(name=name):
                document = json.loads((schema_root / name).read_text(encoding="utf-8"))
                self.assertFalse(document["additionalProperties"])
                self.assertEqual(document["properties"]["schema_version"]["const"], version)
                if name == "rc3-freeze-input-admission-v1.schema.json":
                    self.assertIn(
                        "reconstructed_baseline",
                        document["required"],
                    )
                self.assertEqual(
                    len(
                        document["$defs"]["gitRevision"]["allOf"][1]["not"][
                            "const"
                        ]
                    ),
                    40,
                )
                self.assertNotIn("passed", json.dumps(document, sort_keys=True))
                self.assertNotIn("freeze_sha256", json.dumps(document, sort_keys=True))
        request_schema = json.loads(
            (schema_root / "rc3-freeze-input-request-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            request_schema["properties"]["launch_profiles"]["prefixItems"][0]["$ref"],
            "#/$defs/stableProfile",
        )
        self.assertEqual(
            request_schema["properties"]["launch_profiles"]["prefixItems"][1]["$ref"],
            "#/$defs/maxProfile",
        )
        report_schema = json.loads(
            (schema_root / "rc3-freeze-input-admission-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report_schema["properties"]["checks"]["maxItems"], 8)


if __name__ == "__main__":
    unittest.main()
