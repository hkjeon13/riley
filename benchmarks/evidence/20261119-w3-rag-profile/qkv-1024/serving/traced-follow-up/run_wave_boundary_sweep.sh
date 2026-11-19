#!/usr/bin/env bash
set -euo pipefail
evidence=$(cd -- "$(dirname -- "$0")" && pwd -P)
repository=$(git -C "$evidence" rev-parse --show-toplevel)
workspace=$(mktemp -d "${TMPDIR:-/tmp}/riley-wave-boundary.XXXXXXXX")
cd "$repository"
CARGO_TARGET_DIR="$workspace/target" cargo build --locked --offline -p riley-scheduler --message-format=json \
    >"$workspace/build.jsonl"
python3.12 - "$workspace" "$evidence/replay_wave.rs" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

work, source = map(Path, sys.argv[1:])
libraries = {}
for line in (work / "build.jsonl").read_text().splitlines():
    record = json.loads(line)
    if record["reason"] != "compiler-artifact":
        continue
    name = record["target"]["name"]
    if name in ("riley_runtime", "riley_scheduler"):
        artifact, = [path for path in record["filenames"] if path.endswith(".rlib")]
        libraries[name] = artifact
command = ["rustc", "--edition=2024", "--crate-name", "replay_wave", str(source),
           "-L", f"dependency={work / 'target/debug/deps'}", "-o", str(work / "replay_wave")]
for name in ("riley_runtime", "riley_scheduler"):
    command.extend(("--extern", f"{name}={libraries[name]}"))
subprocess.run(command, check=True)
PY
python3.12 -B "$evidence/sweep_wave_boundary.py" "$evidence/wave-admission.json" \
    "$workspace/replay_wave" "$repository/crates/riley-scheduler/src"
