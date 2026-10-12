#!/bin/bash
# Diagnostic-only C32 repeat comparison with rotating order; not a qualifying measurement.
cd /data/riley-serving-261007/kernel-batch14-diagnostic-attempt01
PY=/data/riley-serving-261007/vllm0271-venv/bin/python; B13=/data/riley-serving-261007/kernel-batch13-target-attempt01/release/riley
for r in 0 1 2 3 4 5 6 7; do for w in fixed natural; do
  case $((r%3)) in 0) o="b13:$B13 b14c:./riley-b14c-d3c848d9 b14b:./riley-b14b-787cc968";; 1) o="b14c:./riley-b14c-d3c848d9 b14b:./riley-b14b-787cc968 b13:$B13";; 2) o="b14b:./riley-b14b-787cc968 b13:$B13 b14c:./riley-b14c-d3c848d9";; esac
  for e in $o; do n=${e%%:*}; b=${e#*:}; $PY diag.py --binary $b --workload $w --concurrency 32 --label q-c32$w-$n-r$r 2>&1 | tail -1; done
done; done > run_c32.log 2>&1
touch run_c32.done
