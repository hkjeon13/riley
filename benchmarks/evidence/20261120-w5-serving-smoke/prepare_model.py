"""Stage a readable copy of the pinned, flat model checkpoint for a read-only mount."""

from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
import sys


def prepare(source, output, expected):
    name = "riley-checkpoint.json"
    original = (source / name).read_bytes()
    if sha256(original).hexdigest() != expected:
        raise ValueError("checkpoint manifest changed")
    output.mkdir(mode=0o700)
    for file in json.loads(original)["files"]:
        relative = Path(file["path"])
        payload = source / relative
        if (len(relative.parts) != 1 or relative.name == name
                or payload.is_symlink() or not payload.is_file()):
            raise ValueError("expected an ordinary file in a flat prepared checkpoint")
        target = output / relative
        copyfile(payload, target)
        if target.stat().st_size != file["bytes"]:
            raise ValueError(f"checkpoint payload size changed: {relative.name}")
        target.chmod(0o444)
    (output / name).write_bytes(original)
    (output / name).chmod(0o444)
    output.chmod(0o555)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
