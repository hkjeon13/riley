#!/usr/bin/env python3
"""Adversarial CPU tests for the native CUDA reproducibility boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))

from check_workspace_boundaries import (  # noqa: E402
    BoundaryError,
    NVCC_REPRODUCIBLE_OBJECT_BLOCK,
    validate_native_build_contract_text,
)


class NativeBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contents = (ROOT / "kernels/CMakeLists.txt").read_text(encoding="utf-8")

    def test_reviewed_contract_passes(self) -> None:
        validate_native_build_contract_text(self.contents)

    def test_flag_scope_and_uniqueness_fail_closed(self) -> None:
        mutations = {
            "deleted": self.contents.replace("--objdir-as-tempdir", "--removed", 1),
            "comment-only": self.contents.replace(
                "$<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>",
                "# --objdir-as-tempdir",
                1,
            ),
            "other-target": self.contents.replace(
                "target_compile_options(riley_cuda_native PRIVATE\n"
                "        $<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>",
                "target_compile_options(riley_cuda_abi_checks PRIVATE\n"
                "        $<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>",
                1,
            ),
            "other-language": self.contents.replace(
                "COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir",
                "COMPILE_LANGUAGE:CXX>:--objdir-as-tempdir",
                1,
            ),
            "outside-nvidia": self.contents.replace(
                'if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")\n',
                "",
                1,
            ),
            "duplicate": self.contents.replace(
                "$<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>",
                "$<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>\n"
                "        $<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>",
                1,
            ),
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(BoundaryError):
                    validate_native_build_contract_text(contents)

    def test_commented_canonical_block_does_not_count(self) -> None:
        commented = "\n".join(
            f"# {line}" for line in NVCC_REPRODUCIBLE_OBJECT_BLOCK.splitlines()
        )
        with self.assertRaises(BoundaryError):
            validate_native_build_contract_text(commented)


if __name__ == "__main__":
    unittest.main()
