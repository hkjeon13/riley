#!/usr/bin/env bash
set -euo pipefail
evidence=$(cd -- "$(dirname -- "$0")" && pwd -P)
repository=$(git rev-parse --show-toplevel)
workspace=$(mktemp -d "${TMPDIR:-/tmp}/riley-coder-wave-replay.XXXXXXXX")
if [[ $# != 2 ]]; then
    printf "%s\n" "usage: run_replay.sh PATH_TO_REPLAY_WAVE_RS PATH_TO_REPLAY_INPUTS" >&2
    exit 2
fi
fixture_source=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
fixture_input=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")

cd "$repository"
CARGO_TARGET_DIR="$workspace/target" cargo build --locked --offline -p riley-scheduler --message-format=json \
    >"$workspace/build.jsonl"
python3.12 - "$workspace" "$fixture_source" <<'PY'
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
python3.12 -B "$evidence/check_replay.py" "$fixture_input" \
    "$workspace/replay_wave" "$repository/crates/riley-scheduler/src"
