"""Reuse the verified W3 runner with the fixed RAG cohorts."""

from pathlib import Path
import sys

import run_profile_retry as runner
from fixed_w3 import BINARY_SHA256, ORDER, cell_specs, require


source, mode = Path(sys.argv[1]), sys.argv[2]
require(mode in ORDER, "unrecognized run order")
runner.NATIVE = source / "target/release/riley"
require(runner.file_sha256(runner.NATIVE) == BINARY_SHA256, "unexpected W3 runner binary")
runner.cells = lambda _model, run_mode: cell_specs(source, run_mode)
runner.run("qwen14", mode)
