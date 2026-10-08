#!/usr/bin/env python3
"""Static CPU guards for offline CUDA build-image toolchain selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verify_runtime_dockerfile import _instructions

ROOT = Path(__file__).resolve().parents[2]
PINNED_RUST_IMAGE = (
    "FROM rust:1.85.0-bookworm@"
    "sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03 "
    "AS rust-toolchain"
)
TOOLCHAIN_ENV = "ENV RUSTUP_TOOLCHAIN=1.85.0-x86_64-unknown-linux-gnu"
CLIPPY_PREPARATION = (
    'RUN rustup component add --toolchain "${RUSTUP_TOOLCHAIN}" clippy '
    '&& rustup component list --installed --toolchain "${RUSTUP_TOOLCHAIN}" '
    "| grep -Fx 'clippy-x86_64-unknown-linux-gnu' "
    "&& cargo clippy --version | grep -E '^clippy 0\\.1\\.85 '"
)


class ToolchainImageTests(unittest.TestCase):
    def assert_clippy_image_contract(self, relative: str) -> None:
        instructions = _instructions((ROOT / relative).read_text(encoding="utf-8"))
        from_indices = [
            index
            for index, instruction in enumerate(instructions)
            if instruction.upper().startswith("FROM ")
        ]
        self.assertEqual(len(from_indices), 2)
        first, second = from_indices
        self.assertEqual(instructions[first], PINNED_RUST_IMAGE)
        toolchain_stage = instructions[first + 1 : second]
        cuda_stage = instructions[second + 1 :]
        self.assertEqual(toolchain_stage.count(TOOLCHAIN_ENV), 1)
        self.assertEqual(toolchain_stage.count(CLIPPY_PREPARATION), 1)
        self.assertEqual(cuda_stage.count(TOOLCHAIN_ENV), 1)
        self.assertFalse(
            any("rustup component add" in instruction for instruction in cuda_stage)
        )
        self.assertIn(
            "COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup",
            cuda_stage,
        )
        self.assertIn(
            "COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo",
            cuda_stage,
        )

    def test_cuda_compile_image_prepares_exact_clippy(self) -> None:
        self.assert_clippy_image_contract("ci/cuda/Dockerfile")

    def test_optimizer_image_prepares_exact_clippy(self) -> None:
        self.assert_clippy_image_contract(
            "ci/release/OptimizationEvidence.Dockerfile"
        )


if __name__ == "__main__":
    unittest.main()
