"""Prepare a verified build-only header overlay; never modify installed FlashInfer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

source, target = map(Path, sys.argv[1:3])
lock = Path(__file__).resolve().parents[2] / 'kernels/optional/verify_flashinfer.py'
spec = importlib.util.spec_from_file_location('flashinfer_lock', lock)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
base_digest = module.verify(source)
relative = Path('flashinfer/attention/prefill.cuh')
original = (source / 'include' / relative).read_text()
anchor = '''#endif
        }
      }

      uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(
          warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);'''
assert original.count(anchor) == 1, 'expected pinned output-epilogue context exactly once'
patched = original.replace(anchor, anchor.replace(
    '\n      uint32_t o_smem_offset_w',
    '\n      // Riley overlay: publish every lane before the cross-lane shared read.\n'
    '      __syncwarp();\n\n      uint32_t o_smem_offset_w'))
target.mkdir()
shutil.copytree(source / 'include', target / 'include')
(target / 'cccl').symlink_to((source / 'cccl').resolve(), target_is_directory=True)
(target / 'include' / relative).write_text(patched)
receipt = {'upstream_header_tree_sha256': base_digest,
           'original_prefill_sha256': hashlib.sha256(original.encode()).hexdigest(),
           'patched_prefill_sha256': hashlib.sha256(patched.encode()).hexdigest(),
           'change': 'one warp barrier before write_o_reg_gmem cross-lane shared reads',
           'installed_dependency_modified': False}
(target / 'overlay.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
