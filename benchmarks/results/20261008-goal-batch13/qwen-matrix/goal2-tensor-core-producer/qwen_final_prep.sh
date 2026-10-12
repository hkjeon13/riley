#!/bin/sh
# Final candidate (qwen-riley-tcprod2-candidate): HF-exact gate 57 on the same build, then the plateau policy A/B on C8..C32.
cd /data/riley-serving-261007
until [ -f qwen-tcprod2.finished ]; do sleep 10; done
cmp qwen-http-exact-target-attempt01/release/riley qwen-riley-tcprod2-candidate || { touch qwen-final-prep.failed; exit 1; }
python3 run_qwen_http_exact.py 57 > qwen-http-exact-validation-attempt057.log 2>&1
python3 qwen_variant.py hfexact-after-57 --numerics-profile=hf-eager-qwen-p2048-o128-probe-v1 > qwen-variant-hfexact-57.log 2>&1
touch qwen-final-gate.finished
export QRILEY=/data/riley-serving-261007/qwen-riley-tcprod2-candidate
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1 --decode-attention-producer=tensor-core-v1"
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 101 tcprod2-plateau128x192-c8-32 $COMMON --prefill-plateau-rows=128,192 > qwen-final-prep-101.log 2>&1
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 102 tcprod2-plateau128-c8-32 $COMMON --prefill-plateau-rows=128 > qwen-final-prep-102.log 2>&1
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 103 tcprod2-repeat-c8-32 $COMMON > qwen-final-prep-103.log 2>&1
touch qwen-final-prep.finished
