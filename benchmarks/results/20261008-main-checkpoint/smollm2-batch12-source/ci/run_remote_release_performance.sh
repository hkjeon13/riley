#!/bin/bash
#
# Retired public entrypoint for the legacy release-performance runner.
#
# This file intentionally contains no GPU lock, container, evidence, child
# process, or source-loading capability.  A non-interactive Bash process may
# evaluate caller-controlled BASH_ENV before this file starts; therefore an
# in-file guard or `exit` cannot be a trust boundary.  Ending naturally after
# an absolute-path status command makes a shadowed shell builtin irrelevant:
# there is no dormant privileged body to reach.
#
# A future raw producer must be a new versioned, source-locked implementation.
# It must not reactivate or append to this retired entrypoint.

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    /usr/bin/printf '%s\n' \
        'release performance: this legacy Bash launcher is retired; no capture can be started from it'
    /usr/bin/true
else
    /usr/bin/printf '%s\n' \
        'release performance: direct legacy Bash launch is disabled; no capture was started' \
        >&2
    /usr/bin/false
fi
