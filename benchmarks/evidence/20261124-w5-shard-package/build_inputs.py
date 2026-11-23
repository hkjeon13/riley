"""Reconstruct the exact reviewed build inputs from the existing A100 checkout."""

import gzip
from hashlib import sha1
from io import BytesIO
from pathlib import Path, PurePosixPath
import sys
import tarfile


def build(source, stage):
    modes = {"100644": 0o644, "100755": 0o755}
    entries = {}
    for line in gzip.decompress((stage / "source-objects.tsv.gz").read_bytes()).decode().splitlines():
        identity, name = line.split("\t", 1)
        mode, kind, expected = identity.split()
        path = PurePosixPath(name)
        if kind != "blob" or path.is_absolute() or ".." in path.parts or name in entries:
            raise ValueError(f"invalid source object {name}")
        entries[name] = (modes[mode], expected)
    if len(entries) != 1037:
        raise ValueError("expected all 1,037 pinned build input blobs")
    with tarfile.open(stage / "source.tar", "x") as archive:
        for name, (mode, expected) in sorted(entries.items()):
            path = source / name
            if not path.is_file() or path.resolve() != path or bool(path.stat().st_mode & 0o111) != bool(mode & 0o111):
                raise ValueError(f"source path or executable bit changed for {name}")
            data = path.read_bytes()
            git_sha1 = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if git_sha1 != expected:
                raise ValueError(f"Git source differs at {name}")
            member = tarfile.TarInfo(name)
            member.mode = mode
            member.size = len(data)
            archive.addfile(member, BytesIO(data))
    print(f"Verified {len(entries)} committed input blobs at {(stage / 'source-revision.txt').read_text().strip()}")


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
