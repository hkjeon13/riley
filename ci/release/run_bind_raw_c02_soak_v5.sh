#!/usr/bin/env bash
# Thin, static entry point for the local-only v5 native-fallback raw binder.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$script_dir" \
    /usr/bin/python3 -B -S "$script_dir/bind_raw_c02_soak_v5.py" "$@"
