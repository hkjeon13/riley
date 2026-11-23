"""Test the baseline and diagnostic CLI admission with an absent model and hidden GPUs."""
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys


def checksum(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


stage, build, prior, original_build = map(Path, sys.argv[1:])
receipt_name = "private-s256-fixed-a32-gpu7/receipt.json"
receipt = prior / receipt_name
lines = (prior / "result.sha256").read_text().splitlines()
if f"{checksum(receipt)}  {receipt_name}" not in lines:
    raise ValueError("the source receipt is outside the pinned capture")
argv = json.loads(receipt.read_bytes())["server_arguments"]
placeholder = stage / "absent-checkpoint"
if placeholder.exists():
    raise ValueError("expected an absent checkpoint")
argv[argv.index("--model") + 1] = str(placeholder)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64")
out = {}
for label, location, slots, context, code, needle in (
    ("baseline-32", original_build, 32, 8192, 1, "cannot canonicalize checkpoint root"),
    ("baseline-48", original_build, 48, 8192, 2, "at most 32 active sequences"),
    ("diagnostic-32", build, 32, 8192, 1, "cannot canonicalize checkpoint root"),
    ("diagnostic-48", build, 48, 8192, 1, "cannot canonicalize checkpoint root"),
    ("diagnostic-49", build, 49, 8192, 2, "at most 48 active sequences"),
    ("diagnostic-8193", build, 48, 8193, 2, "explicit context bound <=8192"),
):
    binary = location / "target/release/riley"
    binary_hash = checksum(binary)
    if binary_hash != (location / "build-result.sha256").read_text().split()[0]:
        raise ValueError(f"{label}: unexpected binary")
    args = argv.copy()
    args[args.index("--max-active-sequences") + 1] = str(slots)
    args[args.index("--max-sequence-tokens") + 1] = str(context)
    result = subprocess.run([binary, *args], env=env, capture_output=True, text=True, timeout=30)
    matching = [line for line in (result.stderr + result.stdout).splitlines() if needle in line]
    if result.returncode != code or len(matching) != 1:
        raise ValueError(f"{label}: unexpected exit {result.returncode}, {result.stderr}")
    out[label] = {"binary_sha256": binary_hash, "slots": slots, "max_sequence_tokens": context,
                  "exit": code, "diagnostic": matching[0]}
print(json.dumps(out, indent=2))
