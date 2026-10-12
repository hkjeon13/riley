#!/bin/bash
# Diagnostic-only: b14g (parallel owner lookup) vs b14h (+ fused prefill QKV projection), alternating order.
cd /data/riley-serving-261007/kernel-batch14-diagnostic-attempt01
PY=/data/riley-serving-261007/vllm0271-venv/bin/python
for r in 0 1 2 3 4 5 6 7; do for cw in "32 fixed" "32 natural" "1 fixed" "8 fixed"; do set -- $cw
  if [ $((r%2)) = 0 ]; then o="b14g:./riley-b14g-d6e51362 b14h:./riley-b14h-03b9f306"; else o="b14h:./riley-b14h-03b9f306 b14g:./riley-b14g-d6e51362"; fi
  for x in $o; do n=${x%%:*}; b=${x#*:}; $PY diag.py --binary $b --workload $2 --concurrency $1 --label h-c$1$2-$n-r$r 2>&1 | tail -1; done
done; done > run_h.log 2>&1
touch run_h.done
