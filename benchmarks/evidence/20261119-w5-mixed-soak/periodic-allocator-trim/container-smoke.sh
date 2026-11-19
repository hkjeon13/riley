#!/bin/sh
set -eu
test "$(id -u)" = 65532
getconf GNU_LIBC_VERSION
/tmp/w5-trim-smoke &
pid=$!
until test -f /tmp/w5-trim-smoke.ready; do
    kill -0 "$pid"
    sleep 0.05
done
log=/tmp/riley-w5-periodic-trim.jsonl
test -f "$log"
test ! -s "$log"
before=$(awk '$1 == "Anonymous:" {print $2}' "/proc/$pid/smaps_rollup")
while test "$(wc -l < "$log")" -lt 4; do
    kill -0 "$pid"
    sleep 0.05
done
after=$(awk '$1 == "Anonymous:" {print $2}' "/proc/$pid/smaps_rollup")
test "$((before - after))" -ge 4096
printf 'pid=%s anonymous_before_kib=%s anonymous_after_kib=%s\n' "$pid" "$before" "$after"
cat "$log"
umask 077
: > /tmp/w5-trim-smoke.end
wait "$pid"
