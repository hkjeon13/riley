#!/bin/bash
# Diagnostic-only alternating single-lane comparison; not a qualifying measurement.
cd /data/riley-serving-261007/kernel-batch14-diagnostic-attempt01
PY=/data/riley-serving-261007/vllm0271-venv/bin/python; B13=/data/riley-serving-261007/kernel-batch13-target-attempt01/release/riley; B14=./riley-b14c-d3c848d9
for r in 0 1; do for c in 1 8 16 32; do for w in fixed natural; do
  if [ $r = 0 ]; then order="b13:$B13 b14c:$B14"; else order="b14c:$B14 b13:$B13"; fi
  for e in $order; do n=${e%%:*}; b=${e#*:}; $PY diag.py --binary $b --workload $w --concurrency $c --label m-c$c$w-$n-r$r 2>&1 | tail -1; done
done; done; done > run_c.log 2>&1
touch run_c.done
