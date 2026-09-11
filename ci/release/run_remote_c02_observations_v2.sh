#!/usr/bin/env bash
# Run the v2 raw C02 observer in an isolated, deterministic host environment.
#
# This wrapper does not start or stop Riley, CUDA processes, containers, SSH,
# or system services.  The caller must explicitly select an already-running
# local endpoint, server PID, GPU index, private evidence root, and fresh
# capture name; the Python producer rejects omitted or unsafe arguments.

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
observer="${repository_root}/ci/release/capture_c02_observations_v2.py"

if [[ ! -f "${observer}" || -L "${observer}" ]]; then
    printf '%s\n' "C02 v2 raw observer is missing or a symlink: ${observer}" >&2
    exit 1
fi

exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 -B -I -S "${observer}" "$@"
