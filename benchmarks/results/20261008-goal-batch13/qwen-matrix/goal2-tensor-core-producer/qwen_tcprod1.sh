#!/bin/sh
# Tensor-core page-state producer (--decode-attention-producer tensor-core-v1): rebuild, gate 56, native tests, HF token comparison 11 + margins, steady A/B on all 8 cells.
cd /data/riley-serving-261007
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=/data/riley-serving-261007/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
sh ./qwen_rebuild_and_gate.sh 56 || { touch qwen-tcprod1.failed; exit 1; }
cp qwen-http-exact-target-attempt01/release/riley qwen-riley-tcprod1-candidate
sha256sum qwen-riley-tcprod1-candidate > qwen-riley-tcprod1-candidate.sha256
( cd qwen-http-exact-source-attempt01 && for t in tensor_core_producer_gpu prompt_mma_attention_gpu; do $E cargo test --locked --release -p riley-cuda --features cuda --test $t -- --ignored --nocapture --test-threads=1; done; $E cargo test --locked --release -p riley-cuda --features cuda --test native_bf16_paged_split_gqa_gpu -- --ignored --test-threads=1 ragged; $E cargo test --locked --release -p riley-scheduler --test prefill_plateau ) > qwen-tcprod1-native-test.log 2>&1
python3 qwen_hf_token_ref_tcprod.py 11 > qwen-hf-token-ref-11.log 2>&1
touch qwen-tcprod1-hf.finished
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 vllm0271-venv/bin/python qwen_hf_margin_n.py 11 > qwen-hf-margin-attempt11.log 2>&1
export QRILEY=/data/riley-serving-261007/qwen-riley-tcprod1-candidate
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1"
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 95 tcprod1-c8-32 $COMMON --decode-attention-producer=tensor-core-v1 > qwen-tcprod1-95.log 2>&1
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 96 tcprod1base-c8-32 $COMMON --prompt-attention=tensor-core-v1 > qwen-tcprod1-96.log 2>&1
touch qwen-tcprod1-c8.finished
QCONC=1 python3 qwen_matrix_steady_main_env2.py 97 tcprod1-c1 $COMMON --decode-attention-producer=tensor-core-v1 > qwen-tcprod1-97.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 98 tcprod1base-c1 $COMMON --prompt-attention=tensor-core-v1 > qwen-tcprod1-98.log 2>&1
touch qwen-tcprod1.finished
