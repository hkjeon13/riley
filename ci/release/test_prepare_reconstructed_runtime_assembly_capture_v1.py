#!/usr/bin/env python3
"""CPU-only hostile-path tests for runtime assembly capture v1."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import prepare_reconstructed_runtime_assembly_capture_v1 as prepare  # noqa: E402
import provenance_v2_common as common  # noqa: E402
import test_prepare_reconstructed_repro_build_inputs_v1 as repro_inputs_fixture  # noqa: E402
import test_prepare_reconstructed_runtime_oci_inputs_v1 as oci_inputs_fixture  # noqa: E402
import test_reproducible_build as reproducibility_fixture  # noqa: E402
import verify_reconstructed_rc2_pr16_bundle_v1 as reconstructed_rc2_bundle  # noqa: E402
import verify_release_bundle as active_bundle_verifier  # noqa: E402
import verify_reconstructed_runtime_assembly_dockerfile as recipe  # noqa: E402


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_tar(
    entries: list[tuple[str, str, int, bytes]],
    *,
    uid: int = 0,
    gid: int = 0,
) -> bytes:
    """Make a short-name, mtime-zero USTAR using only regular/dir entries."""

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, kind, mode, raw in entries:
            member = tarfile.TarInfo(name)
            member.mode = mode
            member.uid = uid
            member.gid = gid
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            if kind == "directory":
                member.type = tarfile.DIRTYPE
                member.size = 0
                archive.addfile(member)
            else:
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
    return output.getvalue()


class RuntimeAssemblyCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_tempdir = tempfile.tempdir
        tempfile.tempdir = os.fspath(Path(tempfile.gettempdir()).resolve())
        # These hostile fixtures intentionally use a synthetic current-contract
        # bundle.  The fixed historical profile itself is covered separately;
        # keep this test focused on assembly-capture FD and tar boundaries.
        self.bundle_profile_patch = mock.patch.object(
            reconstructed_rc2_bundle,
            "verify_reconstructed_rc2_pr16_bundle",
            side_effect=active_bundle_verifier.verify_bundle,
        )
        self.bundle_profile = self.bundle_profile_patch.start()
        self.repro_fixture = repro_inputs_fixture.ReconstructedReproBuildInputsTests(methodName="runTest")
        self.repro_fixture.setUp()
        self.base = self.repro_fixture.base
        self.repro_root = self.base / "repro-inputs"
        self.repro_receipt = self.repro_fixture._prepare(self.repro_root)
        self.oci_fixture_test = oci_inputs_fixture.RuntimeOciInputsTests(methodName="runTest")
        self.oci_fixture_test.setUp()
        self.oci_fixture = self.oci_fixture_test._write_fixture("capture-oci")
        self._write_image_inspect()
        self.oci_root = self.oci_fixture_test._root("capture-oci-inputs")
        self.oci_receipt = self.oci_fixture_test._prepare(self.oci_fixture, self.oci_root, "a")
        self.raw_capture = self.base / "runtime-assembly-capture.tar"
        self._write_capture(self.raw_capture)

    def tearDown(self) -> None:
        self.oci_fixture_test.tearDown()
        self.repro_fixture.tearDown()
        self.bundle_profile_patch.stop()
        tempfile.tempdir = self.previous_tempdir

    @property
    def expected_source_sha(self) -> str:
        return self.repro_fixture.expected_source_sha256

    @property
    def image_id(self) -> str:
        return self.oci_fixture.image_id

    def _labels(self) -> dict[str, str]:
        selected = self.repro_receipt["builds"]["a"]
        repro_receipt_sha = _sha256((self.repro_root / "reconstructed-repro-build-inputs.json").read_bytes())
        return {
            "org.riley.reconstructed-runtime-assembly.version": "v1",
            "org.riley.reconstructed-runtime-assembly.reconstruction-id": "a",
            "org.riley.reconstructed-runtime-assembly.source-revision": reproducibility_fixture.REVISION,
            "org.riley.reconstructed-runtime-assembly.source-archive-sha256": self.expected_source_sha,
            "org.riley.reconstructed-runtime-assembly.repro-build-inputs-sha256": repro_receipt_sha,
            "org.riley.reconstructed-runtime-assembly.release-binary-sha256": selected["binary"]["sha256"],
            "org.riley.reconstructed-runtime-assembly.release-bundle-sha256": selected["bundle"]["sha256"],
            "org.riley.reconstructed-runtime-assembly.recipe-normalized-instructions-sha256": recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256,
        }

    def _write_image_inspect(self) -> None:
        inspect = [
            {
                "Architecture": "amd64",
                "Config": {
                    "Cmd": ["--help"],
                    "Entrypoint": ["/opt/riley/bin/riley"],
                    "Env": [
                        "PATH=/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                        "NVIDIA_VISIBLE_DEVICES=all",
                        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                    ],
                    "Labels": self._labels(),
                    "User": "65532:65532",
                },
                "Id": self.image_id,
                "Os": "linux",
                "RepoTags": [],
            }
        ]
        self.oci_fixture.inspect.write_bytes(common.canonical_json_bytes(inspect))
        os.chmod(self.oci_fixture.inspect, 0o600)

    def _container_config(self) -> dict[str, object]:
        return {
            "User": prepare.EXPECTED_RUNTIME_USER,
            "Entrypoint": list(prepare.EXPECTED_RUNTIME_ENTRYPOINT),
            "Cmd": list(prepare.EXPECTED_RUNTIME_COMMAND),
            "Env": [
                f"PATH={prepare.EXPECTED_IMAGE_ENVIRONMENT['PATH']}",
                f"NVIDIA_VISIBLE_DEVICES={prepare.EXPECTED_IMAGE_ENVIRONMENT['NVIDIA_VISIBLE_DEVICES']}",
                f"NVIDIA_DRIVER_CAPABILITIES={prepare.EXPECTED_IMAGE_ENVIRONMENT['NVIDIA_DRIVER_CAPABILITIES']}",
            ],
        }

    def _runtime_tree_tar(self) -> bytes:
        bundle = self.repro_root / "repro-builds" / "a" / "riley.tar.gz"
        rows: list[tuple[str, str, int, bytes]] = []
        with tarfile.open(bundle, mode="r:gz") as archive:
            members = archive.getmembers()
            root = members[0].name.rstrip("/").split("/")[0]
            for member in members:
                name = member.name.rstrip("/") if member.isdir() else member.name
                parts = name.split("/")
                if parts[0] != root:
                    raise AssertionError("fixture bundle has more than one root")
                if len(parts) == 1:
                    continue
                relative = "/".join(parts[1:])
                if member.isdir():
                    rows.append((relative, "directory", member.mode, b""))
                else:
                    source = archive.extractfile(member)
                    assert source is not None
                    rows.append((relative, "file", member.mode, source.read()))
        return _canonical_tar(sorted(rows), uid=65532, gid=65532)

    def _context_tar(self) -> bytes:
        binary = (self.repro_root / "repro-builds" / "a" / "riley").read_bytes()
        bundle = (self.repro_root / "repro-builds" / "a" / "riley.tar.gz").read_bytes()
        return _canonical_tar(
            [
                ("Dockerfile", "file", 0o644, recipe.DOCKERFILE.read_bytes()),
                ("input/riley", "file", 0o644, binary),
                ("input/riley.tar.gz", "file", 0o644, bundle),
            ]
        )

    def _build_argv(self, context: bytes) -> list[str]:
        selected = self.repro_receipt["builds"]["a"]
        repro_receipt_sha = _sha256((self.repro_root / "reconstructed-repro-build-inputs.json").read_bytes())
        arguments = (
            ("RILEY_RECONSTRUCTION_ID", "a"),
            ("RILEY_SOURCE_REVISION", reproducibility_fixture.REVISION),
            ("RILEY_SOURCE_ARCHIVE_SHA256", self.expected_source_sha),
            ("RILEY_REPRO_BUILD_INPUTS_SHA256", repro_receipt_sha),
            ("RILEY_RELEASE_BINARY_SHA256", selected["binary"]["sha256"]),
            ("RILEY_RELEASE_BUNDLE_SHA256", selected["bundle"]["sha256"]),
            ("RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256", recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256),
        )
        argv = [
            "docker", "build", "--file", "Dockerfile", "--platform", "linux/amd64", "--network", "none",
            "--pull=false", "--no-cache", "--iidfile", "build.iid",
        ]
        for name, value in arguments:
            argv.extend(("--build-arg", f"{name}={value}"))
        argv.append("-")
        return argv

    def _write_capture(
        self,
        path: Path,
        *,
        started: bool = False,
        extra_runtime_file: bool = False,
        drift_invocation: bool = False,
        iid_trailing_newline: bool = False,
    ) -> None:
        context = self._context_tar()
        runtime_tree = self._runtime_tree_tar()
        if extra_runtime_file:
            with tarfile.open(fileobj=io.BytesIO(runtime_tree), mode="r:") as archive:
                rows = []
                for member in archive.getmembers():
                    source = archive.extractfile(member) if member.isreg() else None
                    rows.append((member.name.rstrip("/"), "file" if member.isreg() else "directory", member.mode, source.read() if source else b""))
            rows.append(("extra", "file", 0o644, b"extra"))
            runtime_tree = _canonical_tar(sorted(rows), uid=65532, gid=65532)
        argv = self._build_argv(context)
        if drift_invocation:
            argv.insert(-1, "--secret=fixture")
        invocation = common.canonical_json_bytes(
            {
                "schema_version": prepare.CAPTURE_INVOCATION_VERSION,
                "argv": argv,
                "stdin": {
                    "member": "context.tar",
                    "format": prepare.CONTEXT_ARCHIVE_FORMAT,
                    "sha256": _sha256(context),
                    "byte_length": len(context),
                },
            }
        )
        state = {
            "Status": "running" if started else "created",
            "Running": started,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "Pid": 1 if started else 0,
            "ExitCode": 0,
            "Error": "",
            "StartedAt": "2026-01-01T00:00:00Z" if started else "0001-01-01T00:00:00Z",
            "FinishedAt": "0001-01-01T00:00:00Z",
        }
        container_id = "c" * 64
        container = common.canonical_json_bytes(
            [
                {
                    "Id": container_id,
                    "Image": self.image_id,
                    "State": state,
                    "Mounts": [],
                    "HostConfig": {"NetworkMode": "none", "Privileged": False},
                    "Config": self._container_config(),
                }
            ]
        )
        raw: dict[str, bytes] = {
            "build.iid": (self.image_id + ("\n" if iid_trailing_newline else "")).encode("ascii"),
            "build.log": b"fixture build log\n",
            "capture-invocation.json": invocation,
            "container-inspect.json": container,
            "container-opt-riley.tar": runtime_tree,
            "context.tar": context,
            "image-inspect.json": self.oci_fixture.inspect.read_bytes(),
            "oci-export-invocation.json": common.canonical_json_bytes(
                {
                    "schema_version": prepare.OCI_EXPORT_INVOCATION_VERSION,
                    "source_image_id": self.image_id,
                    "output_member": "oci-image-layout.tar",
                    "format": prepare.OCI_ARCHIVE_FORMAT,
                    "platform": prepare.PLATFORM,
                }
            ),
            "oci-image-layout.tar": self.oci_fixture.archive.read_bytes(),
        }
        completion_members = {
            name: {"sha256": _sha256(raw[name]), "byte_length": len(raw[name])}
            for name in prepare.CAPTURE_COMPLETION_MEMBER_NAMES
        }
        raw["capture-completion.json"] = common.canonical_json_bytes(
            {
                "schema_version": prepare.CAPTURE_COMPLETION_VERSION,
                "reconstruction_id": "a",
                "image_id": self.image_id,
                "container_id": container_id,
                "container_state": "created",
                "container_started": False,
                "members": completion_members,
            }
        )
        checksums = b"".join(
            f"{_sha256(raw[name])}  {name}\n".encode("ascii") for name in prepare.CAPTURE_MEMBER_NAMES[1:]
        )
        raw["SHA256SUMS"] = checksums
        path.write_bytes(
            _canonical_tar([(name, "file", 0o644, raw[name]) for name in prepare.CAPTURE_MEMBER_NAMES])
        )
        os.chmod(path, 0o600)

    def _prepare(self, root: Path) -> dict[str, object]:
        return prepare.prepare_reconstructed_runtime_assembly_capture(
            root,
            source_input_root=self.repro_fixture.source_root,
            repro_build_input_root=self.repro_root,
            runtime_oci_input_root=self.oci_root,
            expected_source_archive_sha256=self.expected_source_sha,
            expected_build_image_id=reproducibility_fixture.IMAGE_ID,
            reconstruction_id="a",
            assembly_capture=self.raw_capture,
        )

    def _verify(self, root: Path) -> dict[str, object]:
        return prepare.verify_reconstructed_runtime_assembly_capture(
            root,
            source_input_root=self.repro_fixture.source_root,
            repro_build_input_root=self.repro_root,
            runtime_oci_input_root=self.oci_root,
            expected_source_archive_sha256=self.expected_source_sha,
            expected_build_image_id=reproducibility_fixture.IMAGE_ID,
            reconstruction_id="a",
        )

    def _unit_external(self) -> prepare.ExternalFacts:
        descriptor = common.EvidenceDescriptor("fixture", "a" * 64, 1)
        return prepare.ExternalFacts(
            reconstruction_id="a",
            source_inputs={},
            repro_inputs={},
            runtime_oci_inputs={},
            source_revision="b" * 40,
            expected_source_archive_sha256="c" * 64,
            repro_receipt=descriptor,
            binary=descriptor,
            bundle=descriptor,
            oci_image_inspect=descriptor,
            oci_archive=descriptor,
            image_id="sha256:" + "d" * 64,
            repro_root_fd=-1,
        )

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(prepare.RuntimeAssemblyCaptureError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_and_replays_one_closed_arm_capture(self) -> None:
        root = self.base / "runtime-assembly-capture-root"
        receipt = self._prepare(root)
        self.assertEqual(receipt["schema_version"], prepare.RUNTIME_ASSEMBLY_CAPTURE_VERSION)
        self.assertEqual(receipt["status"], "bound")
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["reconstruction_id"], "a")
        self.assertGreaterEqual(self.bundle_profile.call_count, 1)
        self.assertEqual(self._verify(root), receipt)
        self.assertEqual(set(os.listdir(root)), {prepare.RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY, prepare.RUNTIME_ASSEMBLY_CAPTURE_NAME})
        archive = root / prepare.RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY / prepare.RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
        self.assertEqual(archive.stat().st_nlink, 1)

    def test_rejects_never_started_container_drift_before_receipt(self) -> None:
        self._write_capture(self.raw_capture, started=True)
        root = self.base / "started-root"
        self.assert_reason("container-started-or-invalid", lambda: self._prepare(root))
        self.assertFalse((root / prepare.RUNTIME_ASSEMBLY_CAPTURE_NAME).exists())

    def test_rejects_runtime_tree_extra_leaf_before_receipt(self) -> None:
        self._write_capture(self.raw_capture, extra_runtime_file=True)
        root = self.base / "tree-root"
        self.assert_reason("runtime-tree-inventory", lambda: self._prepare(root))
        self.assertFalse((root / prepare.RUNTIME_ASSEMBLY_CAPTURE_NAME).exists())

    def test_rejects_build_invocation_feature_drift_before_receipt(self) -> None:
        self._write_capture(self.raw_capture, drift_invocation=True)
        root = self.base / "invocation-root"
        self.assert_reason("build-invocation-mismatch", lambda: self._prepare(root))
        self.assertFalse((root / prepare.RUNTIME_ASSEMBLY_CAPTURE_NAME).exists())

    def test_rejects_a_non_docker_iidfile_trailing_newline_before_receipt(self) -> None:
        self._write_capture(self.raw_capture, iid_trailing_newline=True)
        root = self.base / "newline-iid-root"
        self.assert_reason("iidfile-image-mismatch", lambda: self._prepare(root))
        self.assertFalse((root / prepare.RUNTIME_ASSEMBLY_CAPTURE_NAME).exists())

    def test_rejects_ustar_extensions_before_any_tarfile_parser(self) -> None:
        raw = bytearray(self.raw_capture.read_bytes())
        raw[prepare.TAR_TYPE_OFFSET] = ord("x")
        raw[prepare.TAR_CHECKSUM_START:prepare.TAR_CHECKSUM_END] = b" " * 8
        checksum = sum(raw[:prepare.TAR_BLOCK_BYTES])
        raw[prepare.TAR_CHECKSUM_START:prepare.TAR_CHECKSUM_END] = f"{checksum:06o}\0 ".encode("ascii")
        self.raw_capture.write_bytes(raw)
        os.chmod(self.raw_capture, 0o600)
        with self.raw_capture.open("rb") as source:
            with mock.patch.object(
                prepare.tarfile,
                "open",
                side_effect=AssertionError("outer PAX header must fail before tarfile is reached"),
            ):
                self.assert_reason(
                    "unsupported-tar-extension",
                    lambda: prepare._parse_capture_archive(source, mock.sentinel.external),
                )

    def test_rejects_oversized_zero_trailer_and_pins_schema_ceiling(self) -> None:
        """Do not snapshot a large all-zero tail before its raw grammar fails."""

        # tarfile may already emit fewer than 20 padding blocks depending on
        # the member total; append a full disallowed trailer rather than
        # assuming its exact padding behavior.
        raw = self.raw_capture.read_bytes() + b"\0" * (
            prepare.MAX_CANONICAL_TAR_TRAILER_BYTES + prepare.TAR_BLOCK_BYTES
        )
        with io.BytesIO(raw) as source:
            self.assert_reason(
                "tar-trailer-size",
                lambda: prepare._parse_capture_archive(source, mock.sentinel.external),
            )

        schema_path = (
            Path(prepare.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/reconstructed-runtime-assembly-capture-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        archive_limit = schema["$defs"]["capture"]["properties"]["archive"]["allOf"][1]["properties"][
            "byte_length"
        ]["maximum"]
        self.assertEqual(archive_limit, prepare.MAX_CAPTURE_ARCHIVE_BYTES)
        self.assertEqual(prepare.MAX_CANONICAL_TAR_TRAILER_BYTES, 20 * prepare.TAR_BLOCK_BYTES)

    def test_rejects_unreviewed_image_environment_and_hidden_tmpfs_in_raw_records(self) -> None:
        external = self._unit_external()
        image_id = external.image_id
        image = common.canonical_json_bytes(
            [
                {
                    "Id": image_id,
                    "Os": "linux",
                    "Architecture": "amd64",
                    "Config": {
                        "Labels": prepare._image_labels(external),
                        "User": "65532:65532",
                        "Entrypoint": ["/opt/riley/bin/riley"],
                        "Cmd": ["--help"],
                        "Env": [
                            "PATH=/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                            "NVIDIA_VISIBLE_DEVICES=all",
                            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                            "LD_LIBRARY_PATH=/tmp/evil-libraries",
                        ],
                    },
                }
            ]
        )
        self.assert_reason(
            "image-environment-mismatch",
            lambda: prepare._validate_image_inspect(image, external, image_id),
        )
        container = common.canonical_json_bytes(
            [
                {
                    "Id": "c" * 64,
                    "Image": image_id,
                    "State": {
                        "Status": "created",
                        "Running": False,
                        "Paused": False,
                        "Restarting": False,
                        "OOMKilled": False,
                        "Dead": False,
                        "Pid": 0,
                        "ExitCode": 0,
                        "Error": "",
                        "StartedAt": "0001-01-01T00:00:00Z",
                        "FinishedAt": "0001-01-01T00:00:00Z",
                    },
                    "Mounts": [],
                    "HostConfig": {
                        "NetworkMode": "none",
                        "Privileged": False,
                        "Tmpfs": {"/opt/riley": "rw,noexec,nosuid"},
                    },
                    "Config": self._container_config(),
                }
            ]
        )
        self.assert_reason(
            "container-host-config-mismatch",
            lambda: prepare._validate_container_inspect(container, image_id),
        )

    def test_rejects_image_healthcheck_and_container_namespace_or_healthcheck_drift(self) -> None:
        external = self._unit_external()
        image_id = external.image_id
        image = json.loads(self.oci_fixture.inspect.read_bytes())
        image[0]["Id"] = image_id
        image[0]["Config"]["Labels"] = prepare._image_labels(external)
        image[0]["Config"]["Healthcheck"] = {"Test": ["CMD", "/tmp/evil-healthcheck"]}
        self.assert_reason(
            "image-config-mismatch",
            lambda: prepare._validate_image_inspect(common.canonical_json_bytes(image), external, image_id),
        )

        container: dict[str, object] = {
            "Id": "c" * 64,
            "Image": image_id,
            "State": {
                "Status": "created",
                "Running": False,
                "Paused": False,
                "Restarting": False,
                "OOMKilled": False,
                "Dead": False,
                "Pid": 0,
                "ExitCode": 0,
                "Error": "",
                "StartedAt": "0001-01-01T00:00:00Z",
                "FinishedAt": "0001-01-01T00:00:00Z",
            },
            "Mounts": [],
            "HostConfig": {"NetworkMode": "none", "Privileged": False, "PidMode": "host"},
            "Config": self._container_config(),
        }
        self.assert_reason(
            "container-host-config-mismatch",
            lambda: prepare._validate_container_inspect(common.canonical_json_bytes([container]), image_id),
        )
        host = container["HostConfig"]
        assert isinstance(host, dict)
        host.pop("PidMode")
        config = container["Config"]
        assert isinstance(config, dict)
        config["Healthcheck"] = {"Test": ["CMD", "/tmp/evil-healthcheck"]}
        self.assert_reason(
            "container-config-mismatch",
            lambda: prepare._validate_container_inspect(common.canonical_json_bytes([container]), image_id),
        )

    def test_rejects_extra_output_and_tampered_snapshot_on_replay(self) -> None:
        root = self.base / "replay-root"
        self._prepare(root)
        extra = root / "extra"
        extra.write_bytes(b"extra")
        os.chmod(extra, 0o600)
        self.assert_reason("unexpected-evidence-entry", lambda: self._verify(root))

    def test_source_file_does_not_import_operational_clients_or_extract(self) -> None:
        source = Path(prepare.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_imports = {"subprocess", "socket", "urllib", "requests"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({alias.name.split(".")[0] for alias in node.names} & forbidden_imports)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden_imports)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {"extract", "extractall"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
