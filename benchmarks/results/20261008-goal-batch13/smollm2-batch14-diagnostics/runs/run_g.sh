#!/bin/bash
# Diagnostic-only: Batch14 validated binary (b14f source, 32a06d8e) vs b14g (parallel owner lookup in mixed RoPE), alternating order.
cd /data/riley-serving-261007/kernel-batch14-diagnostic-attempt01
PY=/data/riley-serving-261007/vllm0271-venv/bin/python; F=/data/riley-serving-261007/kernel-batch14-target-attempt01/release/riley
for r in 0 1 2 3 4 5 6 7; do for cw in "32 fixed" "32 natural" "16 fixed"; do set -- $cw
  if [ $((r%2)) = 0 ]; then o="b14f:$F b14g:./riley-b14g-d6e51362"; else o="b14g:./riley-b14g-d6e51362 b14f:$F"; fi
  for x in $o; do n=${x%%:*}; b=${x#*:}; $PY diag.py --binary $b --workload $2 --concurrency $1 --label g-c$1$2-$n-r$r 2>&1 | tail -1; done
done; done > run_g.log 2>&1
touch run_g.done
