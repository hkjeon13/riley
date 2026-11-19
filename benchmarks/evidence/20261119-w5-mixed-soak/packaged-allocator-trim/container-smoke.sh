#!/bin/sh
set -eu
test "$(id -u)" = 65532
getconf GNU_LIBC_VERSION
/tmp/w5-trim-smoke &
pid=$!
until test -f /tmp/w5-trim-smoke.ready; do
    kill -0 "$pid"
    sleep 0.1
done
before=$(awk '$1 == "Anonymous:" {print $2}' "/proc/$pid/smaps_rollup")
test ! -e /tmp/riley-w5-trim.receipt.json
umask 077
: > /tmp/riley-w5-trim.request
until test -f /tmp/riley-w5-trim.receipt.json; do
    kill -0 "$pid"
    sleep 0.1
done
after=$(awk '$1 == "Anonymous:" {print $2}' "/proc/$pid/smaps_rollup")
printf 'pid=%s anonymous_before_kib=%s anonymous_after_kib=%s\n' "$pid" "$before" "$after"
cat /tmp/riley-w5-trim.receipt.json
test "$((before - after))" -ge 4096
: > /tmp/w5-trim-smoke.end
wait "$pid"
