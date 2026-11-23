#!/usr/bin/env bash
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 'set -euo pipefail
cd /mnt/efs/nlp/psyche/riley-gap-20261102/evidence
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
../tools/vllm-venv/bin/python -B -' <<'PY'
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess

prior = Path("w3-c32-prefill-gpu-20261123.K9sfsf0T")
checksums = (prior / "result.sha256").read_bytes()
if sha256(checksums).hexdigest() != "422565c545fcf56213729cc33b4cde907fd5f27a920b3dfb5bc5488bacdc41bf":
    raise ValueError("reference input list changed")
relative = "private-128-w64-control-gpu6/receipt.json"
payload = (prior / relative).read_bytes()
if f"{sha256(payload).hexdigest()}  {relative}\n".encode() not in checksums.splitlines(keepends=True):
    raise ValueError("reference options changed")
args = json.loads(payload)["server_arguments"]
args[args.index("--max-active-sequences") + 1] = "48"
placeholder = Path("/unmounted-riley-c32-cli-probe")
if placeholder.exists():
    raise ValueError("expected an unmounted checkpoint path")
args[args.index("--model") + 1] = str(placeholder)
binary = Path("w3-c32-prefill-build-20261123.nnAtltun/target/release/riley").absolute()
with binary.open("rb") as handle:
    fingerprint = file_digest(handle, "sha256").hexdigest()
if fingerprint != "d9f7e375c4d2434e95e0a3b8ac9412919f777f5f4bd295548590620c2c8e809c":
    raise ValueError("runner binary changed")
outcomes = {}
for mode, expected_exit, expected_detail in (
    ("fixed-tile-unsplit-v1", 2, "at most 32 active sequences"),
    ("page-v1", 1, "cannot canonicalize checkpoint root"),
):
    command = [binary, *args, "--native-attention-mode", mode]
    run = subprocess.run(command, capture_output=True, text=True, timeout=15)
    messages = [line for line in (run.stderr + run.stdout).splitlines() if expected_detail in line]
    if run.returncode != expected_exit or len(messages) != 1:
        raise ValueError(f"unexpected end of CLI probe: {mode}, code {run.returncode}")
    outcomes[mode] = {"exit": run.returncode, "diagnostic": messages[0]}
print(json.dumps({"binary_sha256": fingerprint, "max_active_sequences": 48, "outcomes": outcomes}, indent=2))
PY
