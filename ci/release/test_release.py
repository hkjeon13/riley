#!/usr/bin/env python3
"""CPU-only unit tests for release packaging and static runtime contracts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import struct
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent))

from build_release_bundle import build_bundle  # noqa: E402
from check_release_preflight import check_preflight  # noqa: E402
from release_common import (  # noqa: E402
    ALLOWED_NATIVE_DEPENDENCIES,
    MIT_LICENSE_BYTES,
    ReleaseContractError,
    SERVER_DEFAULTS_SOURCE_PATH,
    SERVER_DEFAULTS_SOURCE_SHA256,
    release_manifest,
    validate_binary,
)
from verify_release_bundle import verify_bundle  # noqa: E402
from verify_runtime_dockerfile import verify_dockerfile  # noqa: E402

EPOCH = 1_700_000_000
REVISION = "a" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = sorted(
    {
        "libc.so.6",
        "libcublasLt.so.12",
        "libcuda.so.1",
        "libcudart.so.12",
        "libgcc_s.so.1",
    }
)
assert set(DEPENDENCIES) <= ALLOWED_NATIVE_DEPENDENCIES


def install_reviewed_server_defaults_source(repository_root: Path) -> Path:
    """Install the exact reviewed Rust defaults source in a fixture repository."""
    source = REPOSITORY_ROOT / SERVER_DEFAULTS_SOURCE_PATH
    destination = repository_root / SERVER_DEFAULTS_SOURCE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return destination


def fixture_elf(dependencies: list[str] = DEPENDENCIES) -> bytes:
    strings = bytearray(b"\0")
    needed_offsets: list[int] = []
    for dependency in dependencies:
        needed_offsets.append(len(strings))
        strings.extend(dependency.encode("ascii") + b"\0")

    dynamic_offset = 0x200
    string_offset = 0x300
    virtual_base = 0x400000
    dynamic_entries = [(5, virtual_base + string_offset), (10, len(strings))]
    dynamic_entries.extend((1, offset) for offset in needed_offsets)
    dynamic_entries.append((0, 0))
    dynamic = b"".join(struct.pack("<qQ", *entry) for entry in dynamic_entries)
    total_size = string_offset + len(strings)
    binary = bytearray(total_size)
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        binary,
        0,
        ident,
        3,
        62,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        2,
        0,
        0,
        0,
    )
    struct.pack_into(
        "<IIQQQQQQ",
        binary,
        64,
        1,
        5,
        0,
        virtual_base,
        virtual_base,
        total_size,
        total_size,
        0x1000,
    )
    struct.pack_into(
        "<IIQQQQQQ",
        binary,
        120,
        2,
        4,
        dynamic_offset,
        virtual_base + dynamic_offset,
        virtual_base + dynamic_offset,
        len(dynamic),
        len(dynamic),
        8,
    )
    binary[dynamic_offset : dynamic_offset + len(dynamic)] = dynamic
    binary[string_offset : string_offset + len(strings)] = strings
    return bytes(binary)


def rewrite_archive(
    source: Path,
    destination: Path,
    mutate: Callable[[list[tuple[tarfile.TarInfo, bytes | None]]], None],
) -> None:
    with tarfile.open(source, "r:gz") as archive:
        entries = []
        for member in archive.getmembers():
            contents = archive.extractfile(member).read() if member.isreg() else None
            entries.append((member, contents))
    mutate(entries)
    entries.sort(key=lambda entry: entry[0].name)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=EPOCH) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for member, contents in entries:
                    archive.addfile(member, io.BytesIO(contents) if contents is not None else None)


def replace_archive_file(
    entries: list[tuple[tarfile.TarInfo, bytes | None]],
    suffix: str,
    replacement: bytes,
    *,
    update_checksum: bool,
) -> None:
    root = entries[0][0].name.split("/", 1)[0]
    relative_path = suffix.removeprefix("/")
    archive_path = f"{root}/{relative_path}"
    for index, (member, contents) in enumerate(entries):
        if member.name == archive_path:
            if contents is None:
                raise AssertionError(f"fixture member is not a regular file: {archive_path}")
            member.size = len(replacement)
            entries[index] = (member, replacement)
            break
    else:
        raise AssertionError(f"fixture member is missing: {archive_path}")

    if not update_checksum:
        return
    checksum_path = f"{root}/SHA256SUMS"
    for index, (member, contents) in enumerate(entries):
        if member.name != checksum_path:
            continue
        assert contents is not None
        expected_suffix = f"  {relative_path}"
        digest = hashlib.sha256(replacement).hexdigest()
        lines = contents.decode("ascii").splitlines()
        rewritten = [
            f"{digest}{expected_suffix}" if line.endswith(expected_suffix) else line
            for line in lines
        ]
        if rewritten == lines:
            raise AssertionError(f"fixture checksum is missing: {relative_path}")
        checksum_contents = ("\n".join(rewritten) + "\n").encode("ascii")
        member.size = len(checksum_contents)
        entries[index] = (member, checksum_contents)
        return
    raise AssertionError("fixture SHA256SUMS is missing")


class ReleaseBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        member = self.repository / "crates/fixture"
        member.mkdir(parents=True)
        (self.repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["crates/fixture"]\n'
            '[workspace.package]\nversion = "0.1.0"\nlicense = "MIT"\n',
            encoding="utf-8",
        )
        (member / "Cargo.toml").write_text(
            '[package]\nname = "release-license-fixture"\n'
            'version = "0.1.0"\nlicense.workspace = true\n',
            encoding="utf-8",
        )
        (self.repository / "LICENSE").write_bytes(MIT_LICENSE_BYTES)
        self.install_server_defaults_source()
        self.binary = self.root / "riley"
        self.binary.write_bytes(fixture_elf())
        self.binary.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, name: str = "release.tar.gz") -> Path:
        output = self.root / name
        build_bundle(
            binary_path=self.binary,
            output=output,
            repository_root=self.repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        return output

    def install_server_defaults_source(self) -> Path:
        return install_reviewed_server_defaults_source(self.repository)

    def test_bundle_is_deterministic_and_verifies(self) -> None:
        first = self.build("first.tar.gz")
        second = self.build("second.tar.gz")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        verify_bundle(first)

    def test_release_manifest_has_exact_defaults_support_and_scope(self) -> None:
        manifest = release_manifest("0.1.0", REVISION, EPOCH)
        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "artifact",
                "features",
                "defaults",
                "support",
                "unsupported",
                "validation",
                "known_limitations",
                "configuration",
                "rollback",
            },
        )
        self.assertEqual(
            manifest["defaults"],
            {
                "bind": "127.0.0.1:8080",
                "device": 0,
                "max_active_sequences": 8,
                "max_waiting_requests": 64,
                "batch_token_budget": 512,
                "prefill_chunk_tokens": 512,
                "execution_completion": "iteration-batch",
                "residual_rmsnorm": "separate",
                "reduction_profile": "canonical-v1",
                "max_weight_bytes": 2_147_483_648,
                "source_contract": {
                    "path": str(SERVER_DEFAULTS_SOURCE_PATH),
                    "sha256": SERVER_DEFAULTS_SOURCE_SHA256,
                },
            },
        )
        semantic_paths = manifest["features"]["semantic_paths"]
        self.assertEqual(
            [row["feature_id"] for row in semantic_paths],
            [
                "iteration-command-batch",
                "fused-residual-rmsnorm",
                "fixed-contiguous-37-balanced-reductions",
            ],
        )
        self.assertEqual(
            [row["semantic_class"] for row in semantic_paths],
            ["E0", "E0", "E0"],
        )
        self.assertEqual(
            [row["default_enabled"] for row in semantic_paths],
            [True, False, False],
        )
        self.assertEqual(
            [row["release_qualified"] for row in semantic_paths],
            [True, False, False],
        )
        self.assertEqual(semantic_paths[1]["approval_gates"], [])
        self.assertEqual(semantic_paths[1]["release_evidence"], [])
        self.assertEqual(
            semantic_paths[1]["prior_evidence_gates"],
            ["pr15-fused-residual-rmsnorm-exact-v1"],
        )
        self.assertEqual(semantic_paths[2]["approval_gates"], [])
        self.assertEqual(semantic_paths[2]["prior_evidence_gates"], [])
        self.assertEqual(semantic_paths[2]["release_evidence"], [])
        self.assertEqual(
            semantic_paths[2]["availability"],
            "unsupported in the first release candidate",
        )
        self.assertEqual(
            manifest["features"]["approximation_policy"],
            {
                "included_semantic_classes": ["reference", "E0"],
                "excluded_semantic_classes": ["E1", "A1", "M1"],
                "approximation_enabled_by_default": False,
                "error_budget": None,
                "quality_budget": None,
                "exact_fallback_required": True,
            },
        )

        self.assertEqual(
            manifest["support"]["source_families"],
            [
                {
                    "model_type": "llama",
                    "architecture": "LlamaForCausalLM",
                    "scope": "dense causal text decoder",
                    "artifact_profile": "SmolLM2-compatible ByteLevel BPE tokenizer",
                },
                {
                    "model_type": "qwen2",
                    "architecture": "Qwen2ForCausalLM",
                    "scope": "dense Qwen2 causal text decoder",
                    "artifact_profile": (
                        "Qwen2.5-compatible only for the pinned NFC/Split/ByteLevel "
                        "BPE and no-tools tokenizer_config profile"
                    ),
                },
            ],
        )
        self.assertEqual(
            set(manifest["support"]),
            {
                "host_os",
                "architecture",
                "cuda_toolkit",
                "cuda_architectures",
                "gpu_topology",
                "source_families",
                "checkpoint_format",
                "checkpoint_layouts",
                "checkpoint_parser_dtypes",
                "cuda_execution_dtypes",
                "cuda_execution_head_dimension",
                "checkpoint_provenance",
                "model_config_constraints",
                "python_runtime",
                "network_model_download",
                "model_delivery",
            },
        )
        self.assertEqual(manifest["support"]["checkpoint_parser_dtypes"], ["BF16", "FP16"])
        self.assertEqual(manifest["support"]["cuda_execution_dtypes"], ["BF16"])
        self.assertEqual(manifest["support"]["cuda_execution_head_dimension"], 64)
        self.assertIn(
            "production CUDA serving requires head_dim exactly 64",
            " ".join(manifest["support"]["model_config_constraints"]),
        )
        self.assertEqual(
            set(manifest["unsupported"]),
            {
                "model_architectures",
                "checkpoint_and_loading",
                "execution",
                "serving",
                "runtime_fallbacks",
            },
        )
        self.assertIn("quantized weights", manifest["unsupported"]["checkpoint_and_loading"])
        self.assertIn("mixture-of-experts", manifest["unsupported"]["model_architectures"])
        self.assertIn("multimodal and vision-language", manifest["unsupported"]["model_architectures"])
        self.assertIn("FP16 CUDA execution", manifest["unsupported"]["execution"])
        self.assertTrue(
            any(
                "fused residual RMSNorm" in item
                and "not candidate-qualified" in item
                for item in manifest["unsupported"]["execution"]
            )
        )
        self.assertIn(
            "CUDA serving with head_dim values other than 64",
            manifest["unsupported"]["execution"],
        )
        self.assertIn("multi-GPU, tensor-parallel, pipeline-parallel, and distributed execution", manifest["unsupported"]["execution"])
        self.assertEqual(
            manifest["unsupported"]["runtime_fallbacks"],
            ["Python", "PyTorch", "Transformers", "Triton"],
        )
        self.assertEqual(
            set(manifest["validation"]),
            {
                "pr16_release_qualification_lane",
                "prior_pr12_qwen_compatibility_evidence",
            },
        )
        self.assertEqual(
            manifest["validation"]["pr16_release_qualification_lane"]["model_id"],
            "HuggingFaceTB/SmolLM2-135M",
        )
        self.assertEqual(
            manifest["validation"]["prior_pr12_qwen_compatibility_evidence"]["model_id"],
            "Qwen/Qwen2.5-0.5B-Instruct",
        )
        self.assertIn(
            "first stable release candidate has no preceding stable riley bundle",
            " ".join(manifest["known_limitations"]),
        )
        self.assertIn(
            "production continuous-batch CUDA serving executor supports head_dim 64 only",
            " ".join(manifest["known_limitations"]),
        )
        self.assertIn(
            "no current-revision fused parity report",
            " ".join(manifest["known_limitations"]),
        )
        self.assertIn("current checksummed bundle", manifest["rollback"]["validated_scope"])
        self.assertIn("only when one exists", manifest["rollback"]["previous_release_scope"])

    def test_fixed37_cannot_be_promoted_by_manifest_tampering(self) -> None:
        source = self.build("fixed37-source.tar.gz")
        tampered = self.root / "fixed37-tampered.tar.gz"

        def promote_fixed37(
            entries: list[tuple[tarfile.TarInfo, bytes | None]],
        ) -> None:
            manifest_entry = next(
                (
                    entry
                    for entry in entries
                    if entry[0].name.endswith("/manifest/release.json")
                ),
                None,
            )
            if manifest_entry is None or manifest_entry[1] is None:
                raise AssertionError("fixture release manifest is missing")
            manifest = json.loads(manifest_entry[1])
            fixed37 = next(
                row
                for row in manifest["features"]["semantic_paths"]
                if row["feature_id"] == "fixed-contiguous-37-balanced-reductions"
            )
            fixed37["release_qualified"] = True
            fixed37["availability"] = "supported opt-in E0 path"
            fixed37["approval_gates"] = ["unreviewed-fixed37-gate"]
            replacement = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            replace_archive_file(
                entries,
                "manifest/release.json",
                replacement,
                update_checksum=True,
            )

        rewrite_archive(source, tampered, promote_fixed37)
        with self.assertRaisesRegex(ReleaseContractError, "manifest"):
            verify_bundle(tampered)

    def test_safe_application_strings_are_not_runtime_dependencies(self) -> None:
        binary = fixture_elf() + b"transformers_version ExperimentalTriton"
        self.assertEqual(validate_binary(binary), DEPENDENCIES)

    def test_nvcc_process_named_temporary_symbol_is_rejected(self) -> None:
        markers = (
            b"\0tmpxft_000002ab_00000000-6_batch_primitives.cudafe1.cpp\0",
            b"\0/tmp/tmpxft_000002ab_00000000-6_gemm.cudafe1.cpp\0",
            b"\0tmpxft_ABCDEF01_0000000A-12_memory.cudafe1.cpp\0",
            b"\0tmpxft_000002ab_00000000-6_version.cudafe1.stub.c\0",
        )
        for marker in markers:
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(
                    ReleaseContractError,
                    "nondeterministic nvcc temporary symbol",
                ):
                    validate_binary(fixture_elf() + marker)
        self.assertEqual(
            validate_binary(fixture_elf() + b"\0tmpxft files are discussed here\0"),
            DEPENDENCIES,
        )

    def test_missing_owner_selected_license_fails_preflight(self) -> None:
        (self.repository / "LICENSE").unlink()
        with self.assertRaisesRegex(ReleaseContractError, "LICENSE"):
            self.build()

    def test_missing_cargo_license_metadata_fails_preflight(self) -> None:
        (self.repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ReleaseContractError, "workspace.package.license"):
            self.build()

    def test_preflight_binds_release_metadata_to_exact_rust_defaults_source(self) -> None:
        defaults_source = self.install_server_defaults_source()
        check_preflight(self.repository, REVISION, EPOCH)
        defaults_source.write_bytes(defaults_source.read_bytes() + b"\n")
        entrypoints = {
            "preflight": lambda: check_preflight(self.repository, REVISION, EPOCH),
            "standalone-producer": lambda: self.build("drifted-defaults.tar.gz"),
        }
        for name, entrypoint in entrypoints.items():
            with self.subTest(entrypoint=name):
                with self.assertRaisesRegex(
                    ReleaseContractError,
                    "Rust serve defaults changed without a reviewed release-contract update",
                ):
                    entrypoint()

    def test_preflight_rejects_linked_rust_defaults_source(self) -> None:
        defaults_source = self.install_server_defaults_source()
        target = defaults_source.with_name("main.reviewed.rs")
        defaults_source.rename(target)
        defaults_source.symlink_to(target.name)
        entrypoints = {
            "preflight": lambda: check_preflight(self.repository, REVISION, EPOCH),
            "standalone-producer": lambda: self.build("linked-defaults.tar.gz"),
        }
        for name, entrypoint in entrypoints.items():
            with self.subTest(entrypoint=name):
                with self.assertRaisesRegex(
                    ReleaseContractError, "must be a regular file"
                ):
                    entrypoint()

    def test_license_body_and_spdx_must_both_be_canonical_mit(self) -> None:
        cases = {
            "non-mit-body": (
                MIT_LICENSE_BYTES.replace(b"MIT License", b"Other License", 1),
                "MIT",
                "reviewed MIT license bytes",
            ),
            "non-mit-spdx": (
                MIT_LICENSE_BYTES,
                "Apache-2.0",
                "SPDX expression",
            ),
        }
        for name, (license_contents, expression, error_pattern) in cases.items():
            with self.subTest(name=name):
                (self.repository / "LICENSE").write_bytes(license_contents)
                (self.repository / "Cargo.toml").write_text(
                    '[workspace]\nmembers = ["crates/fixture"]\n'
                    '[workspace.package]\nversion = "0.1.0"\n'
                    f'license = "{expression}"\n',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ReleaseContractError, error_pattern):
                    self.build(f"{name}.tar.gz")
                (self.repository / "LICENSE").write_bytes(MIT_LICENSE_BYTES)

    def test_workspace_license_file_is_rejected_by_preflight_and_bundle(self) -> None:
        root_manifest = self.repository / "Cargo.toml"
        root_manifest.write_text(
            root_manifest.read_text(encoding="utf-8").replace(
                'license = "MIT"\n',
                'license = "MIT"\nlicense-file = "LICENSE"\n',
            ),
            encoding="utf-8",
        )
        entrypoints = {
            "preflight": lambda: check_preflight(self.repository, REVISION, EPOCH),
            "bundle": lambda: self.build("workspace-license-file.tar.gz"),
        }
        for name, entrypoint in entrypoints.items():
            with self.subTest(entrypoint=name):
                with self.assertRaisesRegex(ReleaseContractError, "license-file"):
                    entrypoint()

    def test_member_license_file_is_rejected_by_preflight_and_bundle(self) -> None:
        member_manifest = self.repository / "crates/fixture/Cargo.toml"
        member_manifest.write_text(
            member_manifest.read_text(encoding="utf-8")
            + 'license-file = "../../LICENSE"\n',
            encoding="utf-8",
        )
        entrypoints = {
            "preflight": lambda: check_preflight(self.repository, REVISION, EPOCH),
            "bundle": lambda: self.build("member-license-file.tar.gz"),
        }
        for name, entrypoint in entrypoints.items():
            with self.subTest(entrypoint=name):
                with self.assertRaisesRegex(ReleaseContractError, "license-file"):
                    entrypoint()

    def test_every_workspace_member_must_inherit_the_mit_license(self) -> None:
        member_manifest = self.repository / "crates/fixture/Cargo.toml"
        for name, declaration in {
            "missing": "",
            "direct-expression": 'license = "MIT"\n',
            "disabled-inheritance": "license.workspace = false\n",
        }.items():
            with self.subTest(name=name):
                member_manifest.write_text(
                    '[package]\nname = "release-license-fixture"\n'
                    f'version = "0.1.0"\n{declaration}',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ReleaseContractError, "license.workspace = true"
                ):
                    self.build(f"member-{name}.tar.gz")

    def test_path_traversal_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "traversal.tar.gz"

        def add_traversal(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            member = tarfile.TarInfo("../escape")
            member.size = 1
            member.mode = 0o644
            member.mtime = EPOCH
            entries.append((member, b"x"))

        rewrite_archive(valid, invalid, add_traversal)
        with self.assertRaisesRegex(ReleaseContractError, "unsafe archive member path"):
            verify_bundle(invalid)

    def test_python_artifact_path_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "python-artifact.tar.gz"

        def add_python(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = entries[0][0].name.split("/", 1)[0]
            member = tarfile.TarInfo(f"{root}/bootstrap.py")
            member.size = 1
            member.mode = 0o644
            member.mtime = EPOCH
            entries.append((member, b"x"))

        rewrite_archive(valid, invalid, add_python)
        with self.assertRaisesRegex(ReleaseContractError, "forbidden Python runtime artifact"):
            verify_bundle(invalid)

    def test_symlink_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "symlink.tar.gz"

        def replace_binary(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            for index, (member, _) in enumerate(entries):
                if member.name.endswith("/bin/riley"):
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/bin/true"
                    member.size = 0
                    entries[index] = (member, None)

        rewrite_archive(valid, invalid, replace_binary)
        with self.assertRaisesRegex(ReleaseContractError, "links and special files"):
            verify_bundle(invalid)

    def test_extra_file_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "extra.tar.gz"

        def add_extra(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = entries[0][0].name.split("/", 1)[0]
            member = tarfile.TarInfo(f"{root}/extra.txt")
            member.size = 1
            member.mode = 0o644
            member.mtime = EPOCH
            entries.append((member, b"x"))

        rewrite_archive(valid, invalid, add_extra)
        with self.assertRaisesRegex(ReleaseContractError, "unreviewed extra members"):
            verify_bundle(invalid)

    def test_tampered_file_is_rejected_by_checksum(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "tampered.tar.gz"

        def tamper_binary(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            binary = next(
                contents
                for member, contents in entries
                if member.name.endswith("/bin/riley")
            )
            assert binary is not None
            replace_archive_file(
                entries,
                "bin/riley",
                binary + b"tampered\n",
                update_checksum=False,
            )

        rewrite_archive(valid, invalid, tamper_binary)
        with self.assertRaisesRegex(ReleaseContractError, "SHA-256 mismatch"):
            verify_bundle(invalid)

    def test_bundle_rejects_noncanonical_mit_body_with_matching_checksum(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "wrong-license-body.tar.gz"

        def tamper_license(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            replacement = MIT_LICENSE_BYTES.replace(
                b"Riley contributors", b"different contributors", 1
            )
            replace_archive_file(
                entries,
                "LICENSE",
                replacement,
                update_checksum=True,
            )

        rewrite_archive(valid, invalid, tamper_license)
        with self.assertRaisesRegex(ReleaseContractError, "reviewed MIT license bytes"):
            verify_bundle(invalid)

    def test_bundle_rejects_non_mit_spdx_with_matching_checksum(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "wrong-license-spdx.tar.gz"

        def tamper_manifest(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            manifest = next(
                contents
                for member, contents in entries
                if member.name.endswith("/manifest/release.json")
            )
            assert manifest is not None
            replacement = manifest.replace(
                b'"license": "MIT"', b'"license": "Apache-2.0"', 1
            )
            self.assertNotEqual(replacement, manifest)
            replace_archive_file(
                entries,
                "manifest/release.json",
                replacement,
                update_checksum=True,
            )

        rewrite_archive(valid, invalid, tamper_manifest)
        with self.assertRaisesRegex(ReleaseContractError, "canonical contract"):
            verify_bundle(invalid)

    def test_bundle_rejects_overclaimed_cuda_head_dimension_with_matching_checksum(
        self,
    ) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "wrong-cuda-head-dimension.tar.gz"

        def tamper_manifest(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            manifest = next(
                contents
                for member, contents in entries
                if member.name.endswith("/manifest/release.json")
            )
            assert manifest is not None
            replacement = manifest.replace(
                b'"cuda_execution_head_dimension": 64',
                b'"cuda_execution_head_dimension": 128',
                1,
            )
            self.assertNotEqual(replacement, manifest)
            replace_archive_file(
                entries,
                "manifest/release.json",
                replacement,
                update_checksum=True,
            )

        rewrite_archive(valid, invalid, tamper_manifest)
        with self.assertRaisesRegex(ReleaseContractError, "canonical contract"):
            verify_bundle(invalid)

    def test_native_manifest_must_match_elf(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "dependency-mismatch.tar.gz"

        def tamper_manifest(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            for index, (member, contents) in enumerate(entries):
                if member.name.endswith("/manifest/native-dependencies.txt"):
                    replacement = contents.replace(b"dependency=libgcc_s.so.1\n", b"")
                    member.size = len(replacement)
                    entries[index] = (member, replacement)

        rewrite_archive(valid, invalid, tamper_manifest)
        with self.assertRaisesRegex(ReleaseContractError, "does not match"):
            verify_bundle(invalid)


class RuntimeDockerfileTests(unittest.TestCase):
    def test_runtime_dockerfile_contract(self) -> None:
        verify_dockerfile()

    def test_builder_requires_python_310_toml_compatibility(self) -> None:
        contents = (REPOSITORY_ROOT / "ci/release/Dockerfile").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "Dockerfile"
            changed_contents = contents.replace("        python3-tomli \\\n", "", 1)
            changed_contents = changed_contents.replace(
                "FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@",
                "RUN echo python3-tomli\n\n"
                "FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@",
                1,
            )
            changed.write_text(
                changed_contents,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReleaseContractError, "package install"):
                verify_dockerfile(changed)

    def test_builder_release_helpers_require_compatibility_wrapper(self) -> None:
        contents = (REPOSITORY_ROOT / "ci/release/Dockerfile").read_text(
            encoding="utf-8"
        )
        changed_contents = contents.replace(
            "python3 ci/release/run_release_python.py "
            "ci/release/check_release_preflight.py",
            "python3 ci/release/check_release_preflight.py",
            1,
        )
        changed_contents = changed_contents.replace(
            "FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@",
            "RUN echo python3 ci/release/run_release_python.py "
            "ci/release/check_release_preflight.py\n\n"
            "FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "Dockerfile"
            changed.write_text(changed_contents, encoding="utf-8")
            with self.assertRaisesRegex(
                ReleaseContractError, "compatibility wrapper"
            ):
                verify_dockerfile(changed)

    def test_runtime_removes_actual_pinned_base_python_hooks(self) -> None:
        contents = (REPOSITORY_ROOT / "ci/release/Dockerfile").read_text(
            encoding="utf-8"
        )
        changed_contents = contents.replace(
            "        /usr/share/apport/package-hooks/source_shadow.py \\\n",
            "        /usr/share/apport/package-hooks/not-the-base-hook.py \\\n",
            1,
        )
        changed_contents = changed_contents.replace(
            "# The final image receives only verified bundle contents.",
            "RUN echo /usr/share/apport/package-hooks/source_shadow.py\n\n"
            "# The final image receives only verified bundle contents.",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "Dockerfile"
            changed.write_text(changed_contents, encoding="utf-8")
            with self.assertRaisesRegex(ReleaseContractError, "Python hooks"):
                verify_dockerfile(changed)


if __name__ == "__main__":
    unittest.main()
