#!/bin/sh
# C1 part of the step-budget experiment (the first attempt 65 was stopped to run a microbenchmark).
cd /data/riley-serving-261007
export QRILEY=/data/riley-serving-261007/qwen-riley-promptmma6-candidate
BASE="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1 --prompt-attention=tensor-core-v1"
B128=1,2,4,8,16,24,32,48,64,80,96,112,128
B192=$B128,144,160,176,192
B256=$B192,208,224,240,256
QCONC=1 python3 qwen_matrix_steady_main_env2.py 66 budget256-c1 $BASE --batch-shape-buckets=$B256 > qwen-bs1-66.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 67 budget192-c1 $BASE --batch-shape-buckets=$B192 --batch-token-budget=192 --prefill-chunk-tokens=192 > qwen-bs1-67.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 68 budget128-c1 $BASE --batch-shape-buckets=$B128 --batch-token-budget=128 --prefill-chunk-tokens=128 > qwen-bs1-68.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 69 budget256chunk128-c1 $BASE --batch-shape-buckets=$B256 --prefill-chunk-tokens=128 > qwen-bs1-69.log 2>&1
touch qwen-bs1-c1.finished
