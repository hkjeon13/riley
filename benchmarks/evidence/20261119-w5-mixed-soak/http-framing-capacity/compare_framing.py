"""Replay the W5 request generator against the base and diagnostic HTTP framers."""

from argparse import ArgumentParser
from collections import Counter
from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import re
from statistics import median
from struct import pack
import subprocess
import sys
from tempfile import TemporaryDirectory

BASE = "d3d8fe6e5b1c528bff51c0008cdc24544c99f862"
HTTP = Path("crates/riley-server/src/http.rs")
TRIAL = re.compile(
    r"FRAMING requests=(\d+) body_capacity_p50=(\d+) body_capacity_p99=(\d+) "
    r"body_capacity_max=(\d+) time_p50_ns=(\d+) time_p95_ns=(\d+)"
)
METRICS = ("requests", "body_capacity_p50", "body_capacity_p99", "body_capacity_max",
           "time_p50_ns", "time_p95_ns")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def input_fixture(repo, receipt):
    sys.path.insert(0, str(repo / "benchmarks/live"))
    from soak import document, request

    arrival = random.Random(11)
    prefix = random.Random(12)
    prefixes = [document(prefix, 512) for _ in range(8)]
    timestamps = arrival.expovariate(8)
    sizes = []
    counts = Counter()
    with receipt.open("wb") as output:
        while timestamps < 1800:
            kind, body = request("m", arrival, prefixes)
            payload = json.dumps(body).encode()
            output.write(pack("<I", len(payload)))
            output.write(payload)
            sizes.append(len(payload))
            counts[kind] += 1
            timestamps += arrival.expovariate(8)
    sizes.sort()
    return {"counts": dict(counts), "requests": len(sizes), "body_length_p50": sizes[len(sizes) // 2],
            "body_length_p99": sizes[len(sizes) * 99 // 100], "body_length_max": sizes[-1],
            "sha256": digest(receipt)}


def comparison(repo, workspace, here):
    fixture = workspace / "requests.bin"
    inputs = input_fixture(repo, fixture)
    probe = here / "framing_probe.rs"
    patch = here / "reserve-validated-body.patch.gz"
    base = subprocess.check_output(["git", "show", f"{BASE}:{HTTP}"], cwd=repo)
    binaries = {}
    source_hashes = {}
    for name in ("base", "candidate"):
        stage = workspace / name
        source = stage / HTTP
        source.parent.mkdir(parents=True)
        source.write_bytes(base)
        if name == "candidate":
            subprocess.run(["git", "apply", "-"], input=decompress(patch.read_bytes()), cwd=stage, check=True)
        source_hashes[name] = digest(source)
        (stage / "main.rs").write_bytes(probe.read_bytes())
        observer = b"""
impl HttpRequest {
    pub(crate) fn probe_body_capacity(&self) -> usize {
        self.body.capacity()
    }
}
"""
        (stage / "http.rs").write_bytes(source.read_bytes() + observer)
        binary = stage / "framing-probe"
        subprocess.run(["rustc", "--edition", "2024", "--crate-name", "framing_probe", "-C", "opt-level=3",
                        "-A", "dead-code", "--remap-path-prefix", f"{stage}=framing-probe",
                        str(stage / "main.rs"), "-o", str(binary)], cwd=repo, check=True)
        binaries[name] = binary

    trials = []
    for _ in range(5):
        block = {}
        for position, name in enumerate(("base", "candidate", "candidate", "base"), 1):
            output = subprocess.check_output([binaries[name], fixture], text=True)
            observed = TRIAL.fullmatch(output.strip())
            if observed is None:
                raise ValueError(f"unexpected output: {output!r}")
            row = dict(zip(METRICS, map(int, observed.groups()), strict=True))
            if row["requests"] != inputs["requests"]:
                raise ValueError(f"incomplete W5 replay: {name}")
            block[f"{position}-{name}"] = row
        trials.append(block)
    medians = {
        name: {metric: median(row[metric] for block in trials for label, row in block.items()
                              if label.endswith(f"-{name}"))
               for metric in METRICS[1:]}
        for name in binaries
    }
    return {"scope": "CPU replay of W5 HTTP framing with no sockets or inference. Capacity is reserved bytes, not RSS.",
            "base_revision": BASE, "platform": platform.platform(),
            "rustc": subprocess.check_output(["rustc", "--version"], cwd=repo, text=True).strip(),
            "inputs": inputs,
            "generator_sha256": {name: digest(repo / "benchmarks/live" / name) for name in ("soak.py", "load.py")},
            "http_source_sha256": source_hashes, "probe_sha256": digest(probe), "patch_sha256": digest(patch),
            "binary_sha256": {name: digest(binary) for name, binary in binaries.items()},
            "trials": trials, "medians": medians}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    with TemporaryDirectory(prefix="riley-http-framing-") as tmp:
        print(json.dumps(comparison(args.repo_root.resolve(), Path(tmp), Path(__file__).resolve().parent),
                         indent=2, allow_nan=False))
