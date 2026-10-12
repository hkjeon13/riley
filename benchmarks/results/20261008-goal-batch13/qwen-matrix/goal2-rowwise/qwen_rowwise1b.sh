#!/bin/sh
# HF token comparison for the row-wise candidate first (attempt 07), then the C1 steady A/B that qwen_rowwise1.sh had queued.
cd /data/riley-serving-261007
python3 qwen_hf_token_ref_rowwise.py 7 > qwen-hf-token-ref-07.log 2>&1
touch qwen-rw-hf.finished
export QRILEY=/data/riley-serving-261007/qwen-riley-rowwise1-candidate
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1"
QCONC=1 python3 qwen_matrix_steady_main_env2.py 81 rowwise1-c1 $COMMON --decode-attention-producer=rowwise-tensor-core-v1 > qwen-rowwise1-81.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 73 rowwise1base-c1 $COMMON --prompt-attention=tensor-core-v1 > qwen-rowwise1-73.log 2>&1
touch qwen-rowwise1.finished
