"""Extract only the captured benchmark data and bind the final build to launch."""

from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tarfile


MANIFESTS = ("build-dispatch.sha256", "campaign-dispatch.sha256", "build-result.sha256",
             "runtime.sha256", "result.sha256")
EXTRA = {
    "build.exit", "build.started", "build.pid", "build.log",
    "campaign.exit", "campaign.started", "campaign.pid", "campaign.log",
    "gpu6-staged-start.json", "gpu7-staged-start.json",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def collect(bundle, launch, output):
    root = bundle.parent / "raw"
    root.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)) or not all(member.isfile() for member in archive):
            raise ValueError("archive contains duplicate names or nonregular files")
        archive.extractall(root, filter="data")
    registered = json.loads(launch.read_bytes())
    if any(digest(root / f"{phase}-dispatch.sha256") != registered[f"{phase}_dispatch_sha256"]
           for phase in ("build", "campaign")):
        raise ValueError("archived dispatch differs from launch receipt")
    if any((root / f"{phase}.exit").read_text().strip() != "0" for phase in ("build", "campaign")):
        raise ValueError("archived experiment has a nonzero exit")
    required = set(EXTRA) | set(MANIFESTS)
    for manifest in MANIFESTS:
        declared = set()
        for line in (root / manifest).read_text().splitlines():
            if (match := re.fullmatch(r"([0-9a-f]{64})  (.+)", line)) is None:
                raise ValueError(f"invalid fingerprint in {manifest}")
            expected, name = match.groups()
            member = Path(name)
            if member.is_absolute() or ".." in member.parts or name in declared:
                raise ValueError(f"invalid captured filename in {manifest}: {name}")
            declared.add(name)
            if (manifest, name) == ("build-result.sha256", "riley-diagnostic"):
                continue
            if digest(root / name) != expected:
                raise ValueError(f"captured data fingerprint differs: {name}")
            required.add(name)
    if set(names) != required:
        raise ValueError("archived files differ from the captured manifests")
    registered["build_result_sha256"] = digest(root / "build-result.sha256")
    output.write_text(json.dumps(registered, indent=2) + "\n")


if __name__ == "__main__":
    collect(*map(Path, sys.argv[1:]))
