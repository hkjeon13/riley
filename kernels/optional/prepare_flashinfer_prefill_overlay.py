"""Prepare a verified build-only header overlay; never modify installed FlashInfer."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from verify_flashinfer import verify, header_digest

def prepare(source, target):
    source, target = source.resolve(), target.absolute()
    resolved_target = target.resolve()
    if resolved_target == source or source in resolved_target.parents or resolved_target in source.parents:
        raise ValueError('overlay and installed dependency must be disjoint')
    base_digest = verify(source)
    relative = Path('flashinfer/attention/prefill.cuh')
    original = (source / 'include' / relative).read_text()
    anchor = (
        '#endif\n        }\n      }\n\n'
        '      uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(\n'
        '          warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);')
    if original.count(anchor) != 1:
        raise ValueError('expected pinned output-epilogue context exactly once')
    patched = original.replace(anchor, anchor.replace(
        '\n      uint32_t o_smem_offset_w',
        '\n      // Riley overlay: publish every lane before the cross-lane shared read.\n'
        '      __syncwarp();\n\n      uint32_t o_smem_offset_w'))
    if hashlib.sha256(patched.encode()).hexdigest() != '996253b7c64caaa3ed86540fb5d5ca44482298c9e8c9e3665b91ba5310b0e975':
        raise ValueError('prefill overlay differs from the GPU-verified patch')
    receipt = {'upstream_header_tree_sha256': base_digest,
               'original_prefill_sha256': hashlib.sha256(original.encode()).hexdigest(),
               'patched_prefill_sha256': hashlib.sha256(patched.encode()).hexdigest(),
               'change': 'one warp barrier before write_o_reg_gmem cross-lane shared reads',
               'installed_dependency_modified': False}

    expected = header_digest(source, {'include/' + str(relative): patched.encode()})
    receipt['patched_header_tree_sha256'] = expected
    if target.exists() or target.is_symlink():
        if target.is_symlink() or header_digest(target) != expected:
            raise ValueError('existing generated FlashInfer overlay has changed; remove that build overlay and reconfigure')
        if json.loads((target / 'overlay.json').read_text()) != receipt:
            raise ValueError('existing generated FlashInfer overlay receipt has changed')
        return receipt
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.flashinfer-prefill-', dir=target.parent) as temporary:
        stage = Path(temporary) / 'overlay'
        stage.mkdir()
        shutil.copytree(source / 'include', stage / 'include')
        (stage / 'cccl').symlink_to((source / 'cccl').resolve(), target_is_directory=True)
        (stage / 'include' / relative).write_text(patched)
        if header_digest(stage) != expected:
            raise ValueError('generated FlashInfer overlay digest mismatch')
        (stage / 'overlay.json').write_text(json.dumps(receipt, indent=2)+'\n')
        stage.rename(target)
    return receipt

if __name__ == '__main__':
    try:
        print(json.dumps(prepare(*map(Path, sys.argv[1:3]))))
    except (ValueError, OSError, TypeError) as error:
        sys.exit(str(error))
