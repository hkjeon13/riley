#!/bin/bash
# Diagnostic-only: b14e (fused QKV in one-row capture) vs b14f (fused QKV in every decode capture) vs Batch13, rotating order.
cd /data/riley-serving-261007/kernel-batch14-diagnostic-attempt01
PY=/data/riley-serving-261007/vllm0271-venv/bin/python; B13=/data/riley-serving-261007/kernel-batch13-target-attempt01/release/riley
for r in 0 1 2 3 4 5; do for cw in "8 fixed" "8 natural" "16 fixed" "16 natural" "32 fixed" "32 natural"; do set -- $cw
  case $((r%3)) in 0) o="b13:$B13 b14e:./riley-b14e-b3276218 b14f:./riley-b14f-c079a5b5";; 1) o="b14e:./riley-b14e-b3276218 b14f:./riley-b14f-c079a5b5 b13:$B13";; 2) o="b14f:./riley-b14f-c079a5b5 b13:$B13 b14e:./riley-b14e-b3276218";; esac
  for x in $o; do n=${x%%:*}; b=${x#*:}; $PY diag.py --binary $b --workload $2 --concurrency $1 --label f-c$1$2-$n-r$r 2>&1 | tail -1; done
done; done > run_f.log 2>&1
touch run_f.done
