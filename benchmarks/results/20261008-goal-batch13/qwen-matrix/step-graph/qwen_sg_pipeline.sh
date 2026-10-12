#!/bin/sh
R=/data/riley-serving-261007
cd $R/qwen-http-exact-source-attempt01
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
$E cargo test --locked --release -p riley-cuda --features cuda --test step_capture_gpu -- --ignored --test-threads=1 > $R/qwen-stepgraph-native-gpu-test.log 2>&1
echo "exit $?" >> $R/qwen-stepgraph-native-gpu-test.log
$E cargo test --locked --release -p riley-cuda --features cuda --test native_bf16_paged_split_gqa_gpu -- --ignored --test-threads=1 ragged > $R/qwen-stepgraph-d128-native-gpu-test.log 2>&1
echo "exit $?" >> $R/qwen-stepgraph-d128-native-gpu-test.log
cd $R
if sh ./qwen_rebuild_and_gate.sh 24; then
M="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1"
python3 qwen_matrix_steady_main_flags.py 1 stepgraph-off $M > qwen-steady-main-stepgraph-off.log 2>&1
python3 qwen_matrix_steady_main_flags.py 1 stepgraph-on $M --decode-step-graph=capture-v1 > qwen-steady-main-stepgraph-on.log 2>&1
python3 qwen_sched_stability_sg.py 13 32 natural 300 > qwen-sched-stab-sgm-c32n.log 2>&1
python3 qwen_sched_stability_sg.py 13 32 fixed 300 > qwen-sched-stab-sgm-c32f.log 2>&1
python3 qwen_sched_stability_sg.py 13 16 natural 300 > qwen-sched-stab-sgm-c16n.log 2>&1
python3 qwen_sched_stability_sg.py 13 8 fixed 300 > qwen-sched-stab-sgm-c8f.log 2>&1
python3 qwen_sched_stability_sg.py 13 1 natural 300 > qwen-sched-stab-sgm-c1n.log 2>&1
fi
touch qwen-sg-pipeline.finished
