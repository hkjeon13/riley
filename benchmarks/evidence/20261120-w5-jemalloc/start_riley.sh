#!/bin/sh
set -eu
export LD_PRELOAD=/tmp/riley-w5-libjemalloc.so.2
export MALLOC_CONF=abort_conf:true,dirty_decay_ms:0,muzzy_decay_ms:0,oversize_threshold:0
exec /opt/riley/bin/riley-docker "$@"
