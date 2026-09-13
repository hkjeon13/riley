"""Compatibility entry point for the build-owned verified prefill overlay."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'kernels/optional'))

runpy.run_path(str(Path(__file__).resolve().parents[2] /
                  'kernels/optional/prepare_flashinfer_prefill_overlay.py'), run_name='__main__')
