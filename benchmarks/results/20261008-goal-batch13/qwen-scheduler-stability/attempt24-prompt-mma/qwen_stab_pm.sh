#!/bin/sh
# Stability of the Qwen candidate (KV-write fusion + tensor-core prompt attention), 8 cells x 300 s, attempt 24.
cd /data/riley-serving-261007
for cw in "1 fixed" "1 natural" "8 fixed" "8 natural" "16 fixed" "16 natural" "32 fixed" "32 natural"; do set -- $cw
  python3 qwen_sched_stability_pm_trend.py 24 $1 $2 300 > qwen-pm-stab-c$1$2.log 2>&1; echo "c$1-$2 $?" >> qwen-pm-stab.status
done
touch qwen-pm-stab.finished
