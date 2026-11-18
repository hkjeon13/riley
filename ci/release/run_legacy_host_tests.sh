#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."

export RILEY_REQUIRE_LEGACY_HOST_TESTS=1
export RILEY_REQUIRE_LINUX_CONTRACT_TESTS=1
/usr/bin/python3.10 -I -B -c '
import sys
sys.path.insert(0, "ci/release")
from release_test_environment import legacy_host_unavailable_reason, require_linux_proc
require_linux_proc()
reason = legacy_host_unavailable_reason()
if reason:
    raise SystemExit(reason)
'

for target in \
    test_rc3_gate_e_private_raw_core_v1.py \
    test_run_remote_rc3_gate_e_session_v2.py \
    test_run_remote_rc3_gate_e_session_v3.py \
    test_verify_rc3_gate_e_execution_anchor_v1.py \
    test_check_reconstructed_runtime_python_prerequisite_v1.py
do
    /usr/bin/python3.10 -B -m unittest discover -s ci/release -p "$target" -v
done
