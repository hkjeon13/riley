#!/usr/bin/env python3
"""Build-time dependency lock; no package installation or runtime Python use."""
import hashlib
import sys
from pathlib import Path

EXPECTED = '2a7d8ab8f81f6cb7fd259270b63bd71e152bff77a105b3c0ba875108b5567e0c'

def header_digest(root, replacements=None):
    digest = hashlib.sha256()
    for relative in ['include', 'cccl/libcudacxx/include', 'cccl/cub', 'cccl/thrust']:
        base = root / relative
        if not base.is_dir():
            raise ValueError(f'missing FlashInfer dependency directory: {base}')
        for path in sorted(base.rglob('*')):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode() + b'\0')
                data = (replacements or {}).get(str(path.relative_to(root)), None)
                digest.update(hashlib.sha256(path.read_bytes() if data is None else data).digest())
    return digest.hexdigest()

def verify(root):
    actual = header_digest(root)
    if actual != EXPECTED:
        raise ValueError(f'FlashInfer 0.6.16.post3 header lock mismatch: {actual} != {EXPECTED}')
    return actual

if __name__ == '__main__':
    try:
        print(verify(Path(sys.argv[1])))
    except (ValueError, OSError, IndexError) as error:
        sys.exit(str(error))
