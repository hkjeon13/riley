"""Inventory regular files and symlinks in a pinned source tar."""

from hashlib import file_digest
from pathlib import PurePosixPath
import tarfile

APPROVED_ARCHIVE = "cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5"
def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def inventory(path):
    entries = {}
    with tarfile.open(path, "r|") as archive:
        for item in archive:
            if item.isdir():
                continue
            name = PurePosixPath(item.name)
            if name.is_absolute() or ".." in name.parts or str(name) in entries:
                raise ValueError(f"invalid or duplicate archive member: {item.name}")
            if item.isfile():
                with archive.extractfile(item) as contents:
                    entries[str(name)] = ("file", file_digest(contents, "sha256").hexdigest(), bool(item.mode & 0o100))
            elif item.issym():
                entries[str(name)] = ("symlink", item.linkname)
            else:
                raise ValueError(f"unexpected archive member type: {item.name}")
    return entries
