#!/bin/sh
# C1 rows of the step-budget trade-off table on the final binary (qwen-riley-tcprod2-candidate); base = attempt 100 (budget 256).
cd /data/riley-serving-261007
until [ -f qwen-final-hf.finished ]; do sleep 20; done
export QRILEY=/data/riley-serving-261007/qwen-riley-tcprod2-candidate
B128=1,2,4,8,16,24,32,48,64,80,96,112,128
B192=$B128,144,160,176,192
B256=$B192,208,224,240,256
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1 --decode-attention-producer=tensor-core-v1"
QCONC=1 python3 qwen_matrix_steady_main_env2.py 104 tcprod2-budget192-c1 $COMMON --batch-shape-buckets=$B192 --batch-token-budget=192 --prefill-chunk-tokens=192 > qwen-c1-budgets-104.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 105 tcprod2-budget128-c1 $COMMON --batch-shape-buckets=$B128 --batch-token-budget=128 --prefill-chunk-tokens=128 > qwen-c1-budgets-105.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 106 tcprod2-budget256chunk128-c1 $COMMON --batch-shape-buckets=$B256 --prefill-chunk-tokens=128 > qwen-c1-budgets-106.log 2>&1
touch qwen-c1-budgets.finished
