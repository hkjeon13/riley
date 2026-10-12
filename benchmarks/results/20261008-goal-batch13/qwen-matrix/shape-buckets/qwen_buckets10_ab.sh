#!/bin/sh
cd /data/riley-serving-261007
while [ ! -f qwen-margin2.finished ]; do sleep 20; done
M="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused"
python3 qwen_matrix_steady_main_flags.py 1 mf-buckets10 $M --batch-shape-buckets=1,8,16,32,64,96,128,160,208,256 > qwen-steady-main-mf-buckets10.log 2>&1
touch qwen-buckets-ab.finished
