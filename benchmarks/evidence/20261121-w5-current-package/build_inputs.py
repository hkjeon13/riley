"""Reconstruct the reviewed distribution inputs and check each committed Git object."""

from hashlib import sha1
from io import BytesIO
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


ENGINE = "crates/riley-server/src/engine.rs"


def build(base_archive, stage):
    modes = {"100644": 0o644, "100755": 0o755}
    entries = {}
    for line in (stage / "source-objects.tsv").read_text().splitlines():
        identity, name = line.split("\t", 1)
        mode, kind, digest = identity.split()
        path = PurePosixPath(name)
        if kind != "blob" or path.is_absolute() or ".." in path.parts or name in entries:
            raise ValueError(f"invalid Git source entry {name}")
        entries[name] = (modes[mode], digest)
    revision = (stage / "source-revision.txt").read_text().strip()
    with tarfile.open(base_archive, "r:") as source, tarfile.open(stage / "source.tar", "x") as output:
        for name, (mode, expected_sha) in sorted(entries.items()):
            member = source.getmember(name)
            if not member.isfile() or bool(member.mode & 0o111) != bool(mode & 0o111):
                raise ValueError(f"unexpected prior source mode {name}")
            with source.extractfile(member) as stream:
                data = stream.read()
            if name == ENGINE:
                with TemporaryDirectory(dir=stage) as folder:
                    engine = Path(folder) / "engine.rs"
                    engine.write_bytes(data)
                    subprocess.run(["patch", "--batch", "--forward", "-s", str(engine),
                                    str(stage / "engine.patch")], check=True)
                    data = engine.read_bytes()
            git_object = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if git_object != expected_sha:
                raise ValueError(f"Git source differs at {name}")
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.size = len(data)
            output.addfile(info, BytesIO(data))
    print(f"Verified {len(entries)} committed input blobs at {revision}")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
