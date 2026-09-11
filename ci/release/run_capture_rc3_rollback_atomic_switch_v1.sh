#!/usr/bin/env bash
# Checked isolated entry point for the FD-only atomic switch raw producer.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
script="$script_dir/capture_rc3_rollback_atomic_switch_v1.py"

exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 -B -I -S -c '
import os
import runpy
import stat
import sys

script = sys.argv[1]
metadata = os.lstat(script)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("atomic switch wrapper: unsafe producer script")
directory = os.path.dirname(script)
if not os.path.isabs(script) or not directory:
    raise SystemExit("atomic switch wrapper: producer path is not absolute")
sys.path.insert(0, directory)
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
' "$script" "$@"
