#!/bin/sh
cd /data/riley-serving-261007
if sh ./qwen_rebuild_and_gate.sh 19; then
M="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused"
python3 qwen_matrix_steady_main_flags.py 1 mf-buckets21 $M --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 > qwen-steady-main-mf-buckets21.log 2>&1
fi
touch qwen-buckets21.finished
