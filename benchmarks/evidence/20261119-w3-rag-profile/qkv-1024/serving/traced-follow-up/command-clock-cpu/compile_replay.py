"""Link a replay executable against a Cargo build of the selected scheduler."""

import json
from pathlib import Path
import subprocess
import sys


manifest, target, source, out = map(Path, sys.argv[1:])
libraries = {}
for line in manifest.read_text().splitlines():
    event = json.loads(line)
    if event["reason"] != "compiler-artifact":
        continue
    name = event["target"]["name"]
    if name in ("riley_runtime", "riley_scheduler"):
        artifact, = (path for path in event["filenames"] if path.endswith(".rlib"))
        libraries[name] = artifact
command = ["rustc", "--edition=2024", "--crate-name", out.name, str(source),
           "-L", f"dependency={target / 'debug/deps'}", "-o", str(out)]
for name in ("riley_runtime", "riley_scheduler"):
    command.extend(("--extern", f"{name}={libraries[name]}"))
subprocess.run(command, check=True)
