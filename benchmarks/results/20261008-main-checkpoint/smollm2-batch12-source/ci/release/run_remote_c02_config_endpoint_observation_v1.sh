#!/usr/bin/env bash
# Run the self-contained raw /v1/config bridge observer for an already-live
# loopback server.  This wrapper intentionally does not start/stop a service,
# invoke Docker/SSH, or decide any qualification result.

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
observer="${repository_root}/ci/release/capture_c02_config_endpoint_observation_v1.py"

if [[ ! -f "${observer}" || -L "${observer}" ]]; then
    printf '%s\n' "C02 config bridge observer is missing or a symlink: ${observer}" >&2
    exit 1
fi

exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 -B -I -S "${observer}" "$@"
