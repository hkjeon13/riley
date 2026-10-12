#!/bin/sh
R=/data/riley-serving-261007
cd $R/qwen-http-exact-source-attempt01
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
$E cargo test --locked --release -p riley-cuda --features cuda --test native_bf16_paged_split_gqa_gpu -- --ignored --test-threads=1 ragged > $R/qwen-sharedkv-native-gpu-test.log 2>&1
echo "exit $?" >> $R/qwen-sharedkv-native-gpu-test.log
cd $R
sh ./qwen_rebuild_and_gate.sh 15 || { echo buildfail > qwen-skv-adopt.done; exit 1; }
python3 qwen_matrix_v3_riley_skv.py 1 > qwen-matrix-rileyonly-sharedkv.log 2>&1
python3 qwen_sched_stability_gg.py 4 32 natural 300 > qwen-sched-stab-skv-c32n.log 2>&1
python3 qwen_sched_stability_gg.py 4 16 fixed 300 > qwen-sched-stab-skv-c16f.log 2>&1
python3 qwen_sched_stability_gg.py 4 1 natural 300 > qwen-sched-stab-skv-c1n.log 2>&1
echo done > qwen-skv-adopt.done
