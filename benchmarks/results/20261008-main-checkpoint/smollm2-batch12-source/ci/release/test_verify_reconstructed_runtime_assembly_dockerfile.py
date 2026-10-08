#!/usr/bin/env python3
"""CPU-only adversarial tests for reconstructed runtime assembly source."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import verify_reconstructed_runtime_assembly_dockerfile as assembly_contract  # noqa: E402
from verify_reconstructed_runtime_assembly_dockerfile import (  # noqa: E402
    DOCKERFILE,
    RuntimeAssemblyContractError,
    verify_reconstructed_runtime_assembly_dockerfile,
)


class ReconstructedRuntimeAssemblyDockerfileTests(unittest.TestCase):
    def test_reviewed_recipe_passes_without_building_an_image(self) -> None:
        self.assertEqual(
            verify_reconstructed_runtime_assembly_dockerfile(),
            hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest(),
        )

    def test_verified_raw_dockerfile_hash_is_available_to_the_held_context_builder(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(assembly_contract.__file__).resolve()), "--print-source-sha256"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest() + "\n")

    def test_recipe_rejects_source_build_context_and_provenance_drift(self) -> None:
        original = DOCKERFILE.read_text(encoding="utf-8")
        mutations = {
            "external-syntax-frontend": "# syntax=docker/dockerfile:latest\n" + original,
            "mutable-runtime-base": original.replace(
                "nvidia/cuda:12.8.1-runtime-ubuntu22.04@"
                "sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd",
                "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
                1,
            ),
            "implicit-builder-platform": original.replace(
                "FROM --platform=linux/amd64 ", "FROM ", 1
            ),
            "wrong-runtime-platform": original.replace(
                "--platform=linux/amd64", "--platform=linux/arm64", 1
            ),
            "source-tree-copy": original.replace(
                "COPY input/riley /assembly-input/riley\n",
                "COPY . /workspace\nCOPY input/riley /assembly-input/riley\n",
                1,
            ),
            "extra-context-leaf": original.replace(
                "COPY input/riley.tar.gz /assembly-input/riley.tar.gz\n",
                "COPY input/riley.tar.gz /assembly-input/riley.tar.gz\n"
                "COPY input/extra /assembly-input/extra\n",
                1,
            ),
            "docker-add": original.replace(
                "COPY input/riley /assembly-input/riley\n",
                "ADD input/riley /assembly-input/riley\n",
                1,
            ),
            "build-secret-mount": original.replace(
                "RUN (test \"${RILEY_RECONSTRUCTION_ID}\"",
                "RUN --mount=type=secret,id=registry (test \"${RILEY_RECONSTRUCTION_ID}\"",
                1,
            ),
            "unverified-binary-digest": original.replace(
                "test \"$(sha256sum /assembly-input/riley | cut -d ' ' -f 1)\" = \"${RILEY_RELEASE_BINARY_SHA256}\"",
                "test \"$(printf %064d 0)\" = \"${RILEY_RELEASE_BINARY_SHA256}\"",
                1,
            ),
            "unverified-bundle-digest": original.replace(
                "test \"$(sha256sum /assembly-input/riley.tar.gz | cut -d ' ' -f 1)\" = \"${RILEY_RELEASE_BUNDLE_SHA256}\"",
                "test \"$(printf %064d 0)\" = \"${RILEY_RELEASE_BUNDLE_SHA256}\"",
                1,
            ),
            "unsafe-extraction": original.replace(
                "--no-overwrite-dir \\\n        ",
                "",
                1,
            ),
            "root-archive-extraction": original.replace(
                "USER 65532:65532\n\n# Verify both selected PR16 artifacts",
                "# Verify both selected PR16 artifacts",
                1,
            ),
            "special-node-accepted": original.replace(
                "-type l -o -type b -o -type c -o -type p -o -type s",
                "-type l",
                1,
            ),
            "bundle-binary-substitution": original.replace(
                "cmp --silent /assembly-input/riley /opt/riley/bin/riley",
                "true",
                1,
            ),
            "raw-input-retained-in-runtime": original.replace(
                "COPY --from=verify-input --chown=65532:65532 /opt/riley/ /opt/riley/",
                "COPY --from=verify-input --chown=65532:65532 /opt/riley/ /opt/riley/\n"
                "COPY --from=verify-input /assembly-input/ /assembly-input/",
                1,
            ),
            "unowned-runtime-tree": original.replace(
                "COPY --from=verify-input --chown=65532:65532 /opt/riley/ /opt/riley/",
                "COPY --from=verify-input /opt/riley/ /opt/riley/",
                1,
            ),
            "missing-repro-receipt-label": original.replace(
                "      org.riley.reconstructed-runtime-assembly.repro-build-inputs-sha256=\"${RILEY_REPRO_BUILD_INPUTS_SHA256}\" \\\n",
                "",
                1,
            ),
            "runtime-package-install": original.replace(
                "RUN /usr/bin/rm \\\n",
                "RUN apt-get update\nRUN /usr/bin/rm \\\n",
                1,
            ),
            "bundle-path-shadowing": original.replace(
                "RUN export PATH=/usr/bin:/bin",
                "RUN export PATH=/opt/riley/bin:/usr/bin:/bin",
                1,
            ),
            "untrusted-elf-ldd": original.replace(
                "    && for command in python python3 pip pip3 cargo rustc nvcc cmake make cc c++; do \\\n",
                "    && ldd /opt/riley/bin/riley \\\n"
                "    && for command in python python3 pip pip3 cargo rustc nvcc cmake make cc c++; do \\\n",
                1,
            ),
            "setid-or-hardlink-accepted": original.replace(
                "-type s -o -perm /6000 \\\n        -o \\( -type f -links +1 \\)",
                "-type s",
                1,
            ),
            "bundle-local-python-ignored": original.replace(
                'if test -e "/opt/riley/bin/${command}" || command -v "${command}"',
                'if command -v "${command}"',
                1,
            ),
            "runtime-python-check-removed": original.replace(
                "    && if /usr/bin/find / -xdev -type f \\( \\\n",
                "    && if false; then \\\n",
                1,
            ),
            "root-runtime": original.replace("USER 65532:65532", "USER 0:0", 1),
            "service-start-command": original.replace('CMD ["--help"]', 'CMD ["serve"]', 1),
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(contents, original)
                normalized = (
                    "\n".join(assembly_contract._instructions(contents)) + "\n"
                ).encode("utf-8")
                candidate_digest = hashlib.sha256(normalized).hexdigest()
                with tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "Dockerfile"
                    candidate.write_text(contents, encoding="utf-8")
                    with mock.patch.object(
                        assembly_contract,
                        "EXPECTED_NORMALIZED_INSTRUCTION_SHA256",
                        candidate_digest,
                    ):
                        with self.assertRaises(RuntimeAssemblyContractError):
                            verify_reconstructed_runtime_assembly_dockerfile(candidate)

    def test_comments_do_not_change_the_normalized_recipe_identity(self) -> None:
        original = DOCKERFILE.read_text(encoding="utf-8")
        commented = "# operator note only\n" + original + "\n# end note\n"
        normalized = "\n".join(assembly_contract._instructions(commented)) + "\n"
        self.assertEqual(
            hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            assembly_contract.EXPECTED_NORMALIZED_INSTRUCTION_SHA256,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
