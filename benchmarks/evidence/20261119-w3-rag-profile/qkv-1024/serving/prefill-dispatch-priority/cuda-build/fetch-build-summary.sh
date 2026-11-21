#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-prefill-priority-build-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact prefill priority build directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$dispatch'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
(cd src && sha256sum -c ../runtime-source.sha256 >/dev/null)
grep -q 'ELF 64-bit.*x86-64' binary-type.txt
test -s switch.txt
/usr/bin/python3 -B - "$2" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def entries(path):
    return {path: checksum for checksum, path in
            (line.split(maxsplit=1) for line in Path(path).read_text().splitlines())}

states = [line for line in Path("campaign.log").read_text().splitlines()
          if line.startswith(("BUILD ", "BUILD_COMPLETE "))]
assert len(states) == 2 and states[0].split()[1] < states[1].split()[1]
links = Path("binary-dependencies.txt").read_text()
assert "not found" not in links and "libcuda.so" in links and "libcublasLt.so" in links
switch = "RILEY_W3_EARLY_PREFILL_PRIORITY"
assert switch in Path("switch.txt").read_text()
evidence = {
    "stage": Path.cwd().name, "compiled_source_revision": "d3d8fe6e5b1c528bff51c0008cdc24544c99f862",
    "diagnostic_option": switch, "architecture": "linux-amd64 CUDA 12.4 SM80",
    "status": "compiled", "build_window": states,
    "dispatch_sha256": sys.argv[1], "runtime_inputs_sha256": entries("runtime-source.sha256"),
    "binary_sha256": entries("result.sha256")["target/release/riley"],
    "build_log_sha256": digest("build.log"),
}
print(json.dumps(evidence, indent=2, allow_nan=False))
PY
REMOTE
