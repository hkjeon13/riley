"""Derive the phase runner from the preserved instrumented W3 runner."""

from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
import shutil


PINNED = {
    "holder.py": "446ea0456b699bbca73730caff276bd841553584cdf16d41fa83522f29343309",
    "tools/profile_process.py": "6cff9930273ec2e5240eeb607b5f61cdf171189cec3f669f7b0e448ec154f54f",
    "tools/compare_runner.py": "2181784479235f8ce501994e3344daf7ce586f47a373711424e8c499590fbde0",
    "target/release/riley": "917caae4ff7467c58650ff5cf67004c17893aa862fd38371d2a940952c2ca398",
    "tools/run_profile_retry.py": "424913d431c5c5505f5c77fb520892e9cb99dfb59030751515dfc86cad55b6d1",
    "tools/profile_workload.py": "051efb94522ac713210298c9b2ee1d492ae07200b9bce5ed2b50ae587b4993bf",
    "tools/load.py": "fdfd5389978f6c607c49ef0d0f2bda10885ac9f25701aa1cbec1479127b07ce0",
    "tools/profile_summary.py": "edf89f5ccc1caae92b358f855eb5cfa606dea7b1fcb1e1c1ad51e3262dda5207",
}
COPY = ("load.py", "profile_summary.py", "profile_process.py", "compare_runner.py")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def unique_replacement(source, old, new):
    if source.count(old) != 1:
        raise ValueError("expected a unique patch context in the pinned runner")
    return source.replace(old, new)


def prepare(prior, target):
    for name, expected in PINNED.items():
        if digest(prior / name) != expected:
            raise ValueError(f"the preserved W3 input differs: {name}")
    tools = target / "tools"
    tools.mkdir()
    native = target / "target/release/riley"
    native.parent.mkdir(parents=True)
    native.symlink_to((prior / "target/release/riley").resolve(strict=True))
    for name in COPY:
        shutil.copyfile(prior / "tools" / name, tools / name)
    shutil.copyfile(prior / "holder.py", target / "holder.py")
    shutil.copyfile(target / "phase_inputs.py", tools / "phase_inputs.py")

    retry = (prior / "tools/run_profile_retry.py").read_text()
    retry = unique_replacement(retry, "from compare_runner import file_sha256",
                               "from compare_runner import file_sha256\nfrom phase_inputs import phase_cells")
    retry = unique_replacement(retry, '"cells": cells(model_key, mode)',
                               '"cells": phase_cells(model_key, mode)')
    (tools / "run_profile_retry.py").write_text(retry)

    workload = (prior / "tools/profile_workload.py").read_text()
    workload = unique_replacement(workload, "from load import one, prompt, summarize",
                                  "from load import one, prompt, summarize\nfrom phase_inputs import dispatch_seconds")
    workload = unique_replacement(workload, "base + (index - first) * intervals / 1000",
                                  "base + dispatch_seconds(cell, index, first)")
    (tools / "profile_workload.py").write_text(workload)
    for name in ("run_profile_retry.py", "profile_workload.py"):
        compile((tools / name).read_text(), name, "exec")
    used = ("target/release/riley", "holder.py", *(f"tools/{path.name}" for path in sorted(tools.iterdir())))
    return {name: digest(target / name) for name in used}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    for name, sha in prepare(args.prior, args.target).items():
        print(f"{sha}  {name}")
