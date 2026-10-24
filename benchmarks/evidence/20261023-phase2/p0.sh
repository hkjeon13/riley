#!/bin/bash
# P0/P1 diagnostics on GPU7 after the hybrid qualification: Riley attention
# bench (Rust) and the FlashInfer/FA2 probe (Python).
set -uo pipefail
R=/mnt/efs/nlp/psyche/riley-phase1-20261022/build-fused-v1
P2=/mnt/efs/nlp/psyche/riley-phase2-20261023
W0=/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019
export RUSTUP_HOME=/home/nlp/riley-261013/rustup CARGO_HOME=/home/nlp/riley-261013/cargo RUSTUP_TOOLCHAIN=1.85.0
export PATH="$CARGO_HOME/bin:/usr/local/cuda-12.4/bin:/usr/local/bin:/usr/bin:/bin"
export CMAKE=/usr/local/bin/cmake CARGO_BUILD_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4 CUDACXX=/usr/local/cuda-12.4/bin/nvcc RILEY_CUDA_ARCHITECTURES=80 CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 CARGO_TARGET_DIR="$R/target" TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/cache"
cd $R/source
BENCH=$(taskset -c 48-55 cargo test --locked --offline --release -p riley-cuda --features cuda,cuda-cublas-gemm-probe --test attention_bench_gpu --no-run --message-format=json 2>$P2/bench-build.log | python3 -c "import json,sys
for l in sys.stdin:
  d=json.loads(l)
  if d.get('reason')=='compiler-artifact' and d.get('executable'): print(d['executable'])")
echo "bench=$BENCH" > $P2/bench-binary.txt
while [ ! -f /mnt/efs/nlp/psyche/riley-phase1-20261022/runs/qual-hybrid.finished ]; do sleep 30; done
mkdir -p $P2/cache
python3 $W0/control.py --lease-dir $W0/leases-v2 --index 7 run -- bash -c "export CUDA_DEVICE_ORDER=PCI_BUS_ID; $BENCH --ignored --nocapture > $P2/riley-bench.log 2>&1; echo \$? > $P2/riley-bench.exit; cd $P2 && HOME=$P2 XDG_CACHE_HOME=$P2/cache FLASHINFER_WORKSPACE_BASE=$P2/cache PATH=/home/nlp/riley-261013/vllm0271-venv/bin:/usr/local/cuda-12.4/bin:/usr/bin:/bin /home/nlp/riley-261013/vllm0271-venv/bin/python flashinfer_probe.py $P2/flashinfer-probe.json > $P2/flashinfer-probe.log 2>&1; echo \$? > $P2/flashinfer-probe.exit" < /dev/null
touch $P2/p0.finished
