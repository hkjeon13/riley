#!/usr/bin/env python3
"""CPU-only tests for the deterministic raw runtime-assembly archive composer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

import compose_reconstructed_runtime_assembly_capture_v1 as compose  # noqa: E402
import prepare_reconstructed_runtime_assembly_capture_v1 as capture_prepare  # noqa: E402
import test_prepare_reconstructed_runtime_assembly_capture_v1 as capture_fixture  # noqa: E402
import test_reproducible_build as reproducibility_fixture  # noqa: E402
import verify_reconstructed_runtime_assembly_dockerfile as recipe  # noqa: E402


IMAGE_ID = "sha256:" + "a" * 64
CONTAINER_ID = "b" * 64
SOURCE_REVISION = "c" * 40
SOURCE_ARCHIVE_SHA256 = "d" * 64
REPRO_INPUTS_SHA256 = "e" * 64
RECIPE_SHA256 = "f" * 64


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class RuntimeAssemblyCaptureComposerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir(mode=0o700)
        self.raw = self.root / "raw"
        self.raw.mkdir(mode=0o700)
        self.dockerfile = self._write(self.source / "ReconstructedRuntimeAssembly.Dockerfile", b"FROM scratch\n")
        self.binary = self._write(self.source / "riley", b"ELF fixture\n")
        self.bundle = self._write(self.source / "riley.tar.gz", b"bundle fixture\n")
        self.context = self.raw / "context.tar"
        self.runtime_source = self.source / "docker-cp.tar"
        self.runtime_tree = self.raw / "container-opt-riley.tar"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, path: Path, raw: bytes) -> Path:
        path.write_bytes(raw)
        os.chmod(path, 0o600)
        return path

    def _run(self, *arguments: str) -> dict[str, object]:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(compose.__file__).resolve()), *arguments],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def _assert_fails(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(compose.__file__).resolve()), *arguments],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("runtime assembly capture composition failed:", completed.stderr)
        return completed

    def _run_bytes(self, raw: bytes, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-B", str(Path(compose.__file__).resolve()), *arguments],
            input=raw,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def _make_context(self) -> dict[str, object]:
        return self._run(
            "context",
            "--output", str(self.context),
            "--dockerfile", str(self.dockerfile),
            "--dockerfile-sha256", digest(self.dockerfile.read_bytes()),
            "--release-binary", str(self.binary),
            "--release-binary-sha256", digest(self.binary.read_bytes()),
            "--release-bundle", str(self.bundle),
            "--release-bundle-sha256", digest(self.bundle.read_bytes()),
        )

    def _write_docker_cp_tar(self, *, symlink: bool = False) -> None:
        with tarfile.open(self.runtime_source, mode="w", format=tarfile.GNU_FORMAT) as archive:
            root = tarfile.TarInfo("riley")
            root.type = tarfile.DIRTYPE
            root.mode = 0o755
            root.uid = 12
            root.gid = 34
            root.mtime = 987
            archive.addfile(root)
            directory = tarfile.TarInfo("riley/bin")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            directory.uid = 12
            directory.gid = 34
            directory.mtime = 987
            archive.addfile(directory)
            if symlink:
                member = tarfile.TarInfo("riley/bin/riley")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                member.mode = 0o777
                archive.addfile(member)
            else:
                raw = b"ELF fixture\n"
                member = tarfile.TarInfo("riley/bin/riley")
                member.mode = 0o755
                member.uid = 12
                member.gid = 34
                member.mtime = 987
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        os.chmod(self.runtime_source, 0o600)

    def _make_runtime_tree(self) -> dict[str, object]:
        self._write_docker_cp_tar()
        return self._run("runtime-tree", "--input", str(self.runtime_source), "--output", str(self.runtime_tree))

    def _member_bytes(self, archive: tarfile.TarFile, name: str) -> bytes:
        member = archive.getmember(name)
        source = archive.extractfile(member)
        assert source is not None
        return source.read()

    def test_context_is_closed_canonical_ustar_and_create_only(self) -> None:
        report = self._make_context()
        self.assertEqual(report["status"], "composed")
        with tarfile.open(self.context, mode="r:") as archive:
            members = archive.getmembers()
            self.assertEqual([member.name for member in members], list(compose.CONTEXT_MEMBER_NAMES))
            for member in members:
                self.assertTrue(member.isreg())
                self.assertEqual(member.mode, 0o644)
                self.assertEqual((member.uid, member.gid, member.mtime, member.uname, member.gname), (0, 0, 0, "", ""))
            self.assertEqual(self._member_bytes(archive, "Dockerfile"), self.dockerfile.read_bytes())
            self.assertEqual(self._member_bytes(archive, "input/riley"), self.binary.read_bytes())
            self.assertEqual(self._member_bytes(archive, "input/riley.tar.gz"), self.bundle.read_bytes())
        self.assertEqual(stat.S_IMODE(self.context.stat().st_mode), 0o600)
        self._assert_fails(
            "context", "--output", str(self.context), "--dockerfile", str(self.dockerfile),
            "--dockerfile-sha256", digest(self.dockerfile.read_bytes()),
            "--release-binary", str(self.binary), "--release-binary-sha256", digest(self.binary.read_bytes()),
            "--release-bundle", str(self.bundle), "--release-bundle-sha256", digest(self.bundle.read_bytes()),
        )

    def test_context_rejects_dockerfile_digest_drift_before_publishing_output(self) -> None:
        output = self.raw / "bad-context.tar"
        self._assert_fails(
            "context",
            "--output",
            str(output),
            "--dockerfile",
            str(self.dockerfile),
            "--dockerfile-sha256",
            "a" * 64,
            "--release-binary",
            str(self.binary),
            "--release-binary-sha256",
            digest(self.binary.read_bytes()),
            "--release-bundle",
            str(self.bundle),
            "--release-bundle-sha256",
            digest(self.bundle.read_bytes()),
        )
        self.assertFalse(output.exists())

    def test_runtime_tree_strips_docker_root_and_normalizes_ownership(self) -> None:
        report = self._make_runtime_tree()
        self.assertEqual(report["runtime_tree"]["entry_count"], 2)
        with tarfile.open(self.runtime_tree, mode="r:") as archive:
            members = archive.getmembers()
            self.assertEqual([member.name for member in members], ["bin", "bin/riley"])
            self.assertTrue(members[0].isdir())
            self.assertTrue(members[1].isreg())
            for member in members:
                self.assertEqual((member.uid, member.gid, member.mtime), (65532, 65532, 0))
            self.assertEqual(members[0].mode, 0o755)
            self.assertEqual(members[1].mode, 0o755)
            self.assertEqual(self._member_bytes(archive, "bin/riley"), b"ELF fixture\n")

    def test_runtime_tree_rejects_a_link_before_publishing_output(self) -> None:
        self._write_docker_cp_tar(symlink=True)
        self._assert_fails("runtime-tree", "--input", str(self.runtime_source), "--output", str(self.runtime_tree))
        self.assertFalse(self.runtime_tree.exists())

    def test_runtime_tree_rejects_non_ascii_and_no_prefix_ustar_overflow_before_output(self) -> None:
        for name in ("riley/" + "x" * 101, "riley/한글"):
            with self.subTest(name=name):
                self.runtime_tree.unlink(missing_ok=True)
                with tarfile.open(self.runtime_source, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    root = tarfile.TarInfo("riley")
                    root.type = tarfile.DIRTYPE
                    root.mode = 0o755
                    archive.addfile(root)
                    raw = b"fixture\n"
                    member = tarfile.TarInfo(name)
                    member.mode = 0o644
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
                os.chmod(self.runtime_source, 0o600)
                self._assert_fails("runtime-tree", "--input", str(self.runtime_source), "--output", str(self.runtime_tree))
                self.assertFalse(self.runtime_tree.exists())

    def test_runtime_tree_stops_at_the_member_inventory_cap_before_output(self) -> None:
        with tarfile.open(self.runtime_source, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for index in range(compose.MAX_RUNTIME_TREE_MEMBERS + 1):
                member = tarfile.TarInfo(f"riley/empty-{index:05d}")
                member.mode = 0o644
                member.size = 0
                archive.addfile(member, io.BytesIO())
        os.chmod(self.runtime_source, 0o600)
        self._assert_fails("runtime-tree", "--input", str(self.runtime_source), "--output", str(self.runtime_tree))
        self.assertFalse(self.runtime_tree.exists())

    def test_bounded_stream_and_exact_docker_id_readers_are_create_only(self) -> None:
        output = self.raw / "stream.raw"
        streamed = self._run_bytes(
            b"bounded raw bytes\n",
            "stream",
            "--output",
            str(output),
            "--maximum-bytes",
            "32",
        )
        self.assertEqual(streamed.returncode, 0, streamed.stderr.decode("utf-8"))
        report = json.loads(streamed.stdout)
        self.assertEqual(report["output"]["byte_length"], len(b"bounded raw bytes\n"))
        self.assertEqual(output.read_bytes(), b"bounded raw bytes\n")
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        duplicate = self._run_bytes(b"other", "stream", "--output", str(output), "--maximum-bytes", "32")
        self.assertNotEqual(duplicate.returncode, 0)

        iid = self._write(self.raw / "exact.iid", IMAGE_ID.encode("ascii"))
        image = self._run_bytes(b"", "read-id", "--kind", "image", "--input", str(iid))
        self.assertEqual(image.returncode, 0, image.stderr.decode("utf-8"))
        self.assertEqual(image.stdout, IMAGE_ID.encode("ascii"))
        container = self._write(self.raw / "container-id.raw", (CONTAINER_ID + "\n").encode("ascii"))
        container_result = self._run_bytes(b"", "read-id", "--kind", "container", "--input", str(container))
        self.assertEqual(container_result.returncode, 0, container_result.stderr.decode("utf-8"))
        self.assertEqual(container_result.stdout, CONTAINER_ID.encode("ascii"))
        for raw in (
            (IMAGE_ID + "\n").encode("ascii"),
            (IMAGE_ID + "\r").encode("ascii"),
            IMAGE_ID.encode("ascii") + b"\0",
            ("sha256:" + "A" * 64).encode("ascii"),
            ("sha256:" + "0" * 64).encode("ascii"),
        ):
            with self.subTest(raw=raw):
                candidate = self._write(self.raw / f"bad-{digest(raw)[:8]}.iid", raw)
                rejected = self._run_bytes(b"", "read-id", "--kind", "image", "--input", str(candidate))
                self.assertNotEqual(rejected.returncode, 0)

    def test_capture_has_exact_inventory_checksums_and_completion_binding(self) -> None:
        self._make_context()
        self._make_runtime_tree()
        iid = self._write(self.raw / "build.iid", IMAGE_ID.encode("ascii"))
        log = self._write(self.raw / "build.log", b"docker build fixture\n")
        image = self._write(self.raw / "image-inspect.json", b"[{\"Id\":\"fixture\"}]\n")
        oci = self._write(self.raw / "oci-image-layout.tar", b"OCI fixture\n")
        container = self._write(self.raw / "container-inspect.json", b"[{\"Id\":\"fixture\"}]\n")
        output = self.raw / "assembly-capture.tar"
        report = self._run(
            "capture",
            "--output", str(output),
            "--context", str(self.context),
            "--runtime-tree", str(self.runtime_tree),
            "--build-iid", str(iid),
            "--build-log", str(log),
            "--image-inspect", str(image),
            "--oci-archive", str(oci),
            "--container-inspect", str(container),
            "--reconstruction-id", "a",
            "--source-revision", SOURCE_REVISION,
            "--expected-source-archive-sha256", SOURCE_ARCHIVE_SHA256,
            "--repro-build-inputs-sha256", REPRO_INPUTS_SHA256,
            "--release-binary-sha256", digest(self.binary.read_bytes()),
            "--release-bundle-sha256", digest(self.bundle.read_bytes()),
            "--recipe-normalized-instructions-sha256", RECIPE_SHA256,
            "--image-id", IMAGE_ID,
            "--container-id", CONTAINER_ID,
        )
        self.assertEqual(report["image_id"], IMAGE_ID)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        with tarfile.open(output, mode="r:") as archive:
            members = archive.getmembers()
            self.assertEqual([member.name for member in members], list(compose.CAPTURE_MEMBER_NAMES))
            for member in members:
                self.assertTrue(member.isreg())
                self.assertEqual((member.mode, member.uid, member.gid, member.mtime, member.uname, member.gname), (0o644, 0, 0, 0, "", ""))
            invocation = json.loads(self._member_bytes(archive, "capture-invocation.json"))
            self.assertEqual(invocation["stdin"]["member"], "context.tar")
            self.assertEqual(invocation["stdin"]["sha256"], digest(self.context.read_bytes()))
            self.assertEqual(invocation["argv"][0:2], ["docker", "build"])
            self.assertEqual(self._member_bytes(archive, "build.iid"), IMAGE_ID.encode("ascii"))
            completion = json.loads(self._member_bytes(archive, "capture-completion.json"))
            self.assertEqual(completion["image_id"], IMAGE_ID)
            self.assertEqual(completion["container_id"], CONTAINER_ID)
            self.assertFalse(completion["container_started"])
            checksums = self._member_bytes(archive, "SHA256SUMS").decode("ascii").splitlines()
            self.assertEqual(len(checksums), len(compose.CAPTURE_MEMBER_NAMES) - 1)
            for line, name in zip(checksums, compose.CAPTURE_MEMBER_NAMES[1:]):
                recorded, recorded_name = line.split("  ", 1)
                self.assertEqual(recorded_name, name)
                self.assertEqual(recorded, digest(self._member_bytes(archive, name)))
                if name != "capture-completion.json":
                    self.assertEqual(completion["members"][name]["sha256"], recorded)

    def test_bad_iid_is_rejected_before_outer_capture_exists(self) -> None:
        self._make_context()
        self._make_runtime_tree()
        iid = self._write(self.raw / "bad.iid", b"not an image ID\n")
        paths = {
            "build_log": self._write(self.raw / "build.log", b"log\n"),
            "image": self._write(self.raw / "image.json", b"[]\n"),
            "oci": self._write(self.raw / "oci.tar", b"oci\n"),
            "container": self._write(self.raw / "container.json", b"[]\n"),
        }
        output = self.raw / "bad-assembly-capture.tar"
        self._assert_fails(
            "capture", "--output", str(output), "--context", str(self.context), "--runtime-tree", str(self.runtime_tree),
            "--build-iid", str(iid), "--build-log", str(paths["build_log"]),
            "--image-inspect", str(paths["image"]), "--oci-archive", str(paths["oci"]),
            "--container-inspect", str(paths["container"]), "--reconstruction-id", "a",
            "--source-revision", SOURCE_REVISION, "--expected-source-archive-sha256", SOURCE_ARCHIVE_SHA256,
            "--repro-build-inputs-sha256", REPRO_INPUTS_SHA256,
            "--release-binary-sha256", digest(self.binary.read_bytes()),
            "--release-bundle-sha256", digest(self.bundle.read_bytes()),
            "--recipe-normalized-instructions-sha256", RECIPE_SHA256,
            "--image-id", IMAGE_ID, "--container-id", CONTAINER_ID,
        )
        self.assertFalse(output.exists())

    def test_composed_capture_is_accepted_by_the_existing_full_capture_replayer(self) -> None:
        fixture = capture_fixture.RuntimeAssemblyCaptureTests(methodName="runTest")
        fixture.setUp()
        try:
            raw = self.root / "integrated-raw"
            raw.mkdir(mode=0o700)
            extracted: dict[str, Path] = {}
            with tarfile.open(fixture.raw_capture, mode="r:") as archive:
                for name in (
                    "build.iid",
                    "build.log",
                    "container-inspect.json",
                    "context.tar",
                    "image-inspect.json",
                    "oci-image-layout.tar",
                ):
                    extracted[name] = self._write(raw / name, self._member_bytes(archive, name))
                docker_cp = self._write(
                    raw / "container-opt-riley.docker-cp.tar",
                    self._member_bytes(archive, "container-opt-riley.tar"),
                )
            runtime_tree = raw / "container-opt-riley.tar"
            self._run("runtime-tree", "--input", str(docker_cp), "--output", str(runtime_tree))
            container_document = json.loads(extracted["container-inspect.json"].read_text(encoding="utf-8"))
            output = raw / "assembly-capture.tar"
            repro_receipt_sha = digest((fixture.repro_root / "reconstructed-repro-build-inputs.json").read_bytes())
            selected = fixture.repro_receipt["builds"]["a"]
            self._run(
                "capture",
                "--output", str(output),
                "--context", str(extracted["context.tar"]),
                "--runtime-tree", str(runtime_tree),
                "--build-iid", str(extracted["build.iid"]),
                "--build-log", str(extracted["build.log"]),
                "--image-inspect", str(extracted["image-inspect.json"]),
                "--oci-archive", str(extracted["oci-image-layout.tar"]),
                "--container-inspect", str(extracted["container-inspect.json"]),
                "--reconstruction-id", "a",
                "--source-revision", reproducibility_fixture.REVISION,
                "--expected-source-archive-sha256", fixture.expected_source_sha,
                "--repro-build-inputs-sha256", repro_receipt_sha,
                "--release-binary-sha256", selected["binary"]["sha256"],
                "--release-bundle-sha256", selected["bundle"]["sha256"],
                "--recipe-normalized-instructions-sha256", recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256,
                "--image-id", fixture.image_id,
                "--container-id", container_document[0]["Id"],
            )
            replay_root = self.root / "integrated-replay"
            receipt = capture_prepare.prepare_reconstructed_runtime_assembly_capture(
                replay_root,
                source_input_root=fixture.repro_fixture.source_root,
                repro_build_input_root=fixture.repro_root,
                runtime_oci_input_root=fixture.oci_root,
                expected_source_archive_sha256=fixture.expected_source_sha,
                expected_build_image_id=reproducibility_fixture.IMAGE_ID,
                reconstruction_id="a",
                assembly_capture=output,
            )
            self.assertEqual(receipt["capture"]["image_id"], fixture.image_id)
            self.assertEqual(receipt["capture"]["container_id"], container_document[0]["Id"])
        finally:
            fixture.tearDown()

    def test_help_and_source_are_operationally_inert(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(compose.__file__).resolve()), "--help"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = Path(compose.__file__).read_text(encoding="utf-8")
        for forbidden in ("subprocess", "socket", "requests", "urllib", "os.system", "Popen"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
