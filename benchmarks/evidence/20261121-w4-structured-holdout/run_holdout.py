"""Reuse the current package, process, and allocator provenance checks."""

from pathlib import Path
import sys
from unittest.mock import patch

import check_distribution_gpu as checker
from run_shapes import run


stage = Path(sys.argv[1])
name = sys.argv[2]
workload = stage.parent / "w4-holdout-20261119.cgmtjz4x/holdout.json.gz"
original = checker.check_gpu


def check_holdout(cpu, gpu, model, checkpoint, out, tools, *, serving_smoke):
    if not serving_smoke:
        raise ValueError("expected the existing package runner to select its smoke workload")
    return original(cpu, gpu, model, checkpoint, out, tools, structured_holdout=workload)


with (patch.object(checker, "check_gpu", check_holdout),
      patch.object(checker, "STRUCTURED_HOLDOUT_CHECK", stage / "structured_holdout_check.py")):
    raise SystemExit(run(stage, name))
