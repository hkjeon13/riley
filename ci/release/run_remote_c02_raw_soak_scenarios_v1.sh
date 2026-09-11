#!/usr/bin/env bash
# Local-only wrapper for the C02-P1 raw serial scenario producer.
#
# It intentionally does not start or stop a server, acquire a GPU lock, invoke
# Docker/SSH, or decide qualification.  The future lifecycle supervisor owns
# those actions and calls this wrapper only after it has bound one live server
# process to a fresh private evidence root.
set -euo pipefail

unset CDPATH
readonly SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PRODUCER="$SCRIPT_DIR/capture_c02_raw_soak_scenarios_v1.py"
[[ -f $PRODUCER && ! -L $PRODUCER ]] || {
    printf 'raw C02 scenario producer is missing or unsafe: %s\n' "$PRODUCER" >&2
    exit 2
}

exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 -B -I -S \
    "$PRODUCER" "$@"
