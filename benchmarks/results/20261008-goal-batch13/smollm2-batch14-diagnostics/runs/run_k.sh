#!/bin/bash
# Diagnostic-only: b14h (no prefetch) vs b14k (weight prefetch), alternating order.
cd /data/riley-serving-261007/kernel-batch14-diagnostic-attempt01
PY=/data/riley-serving-261007/vllm0271-venv/bin/python
for r in 0 1 2 3 4 5; do for cw in "32 fixed" "32 natural" "8 fixed" "16 natural" "16 fixed" "8 natural"; do set -- $cw
  if [ $((r%2)) = 0 ]; then o="b14h:./riley-b14h-03b9f306 b14k:./riley-b14k-abb8f3c5"; else o="b14k:./riley-b14k-abb8f3c5 b14h:./riley-b14h-03b9f306"; fi
  for x in $o; do n=${x%%:*}; b=${x#*:}; $PY diag.py --binary $b --workload $2 --concurrency $1 --label k-c$1$2-$n-r$r 2>&1 | tail -1; done
done; done > run_k.log 2>&1
touch run_k.done
