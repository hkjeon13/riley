#!/bin/bash
# lp11_checks.sh: build lp11 (vector RMSNorm above 256 rows), GPU tests, parity and rag TTFT vs lp10 on GPU 6.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
bash build2.sh lp11 $W/lp11.patch > logs/build-lp11.log 2>&1; grep -q BUILD_OK logs/build-lp11.log || { echo BUILD_FAILED; tail -5 logs/build-lp11.log; exit 1; }
export RUSTUP_HOME=$W/tools/rustup CARGO_HOME=$W/tools/cargo PATH=$W/tools/cargo/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 RILEY_CUDA_ARCHITECTURES=80 CARGO_TARGET_DIR=$W/target
(cd src && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 nice -n 19 cargo test --locked --release -p riley-cuda --features cuda,cuda-cublas-gemm-probe --test primitives_gpu -- --ignored --test-threads 1 rms_norm 2>&1 | grep -E "test result|FAILED|panicked" | head -5)
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
mkdir -p evidence/lp11
for M in Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct; do
  for b in lp10 lp11; do
    P=$(MAXOUT=1024 bash serve.sh $b 6 18342 $M m logs/l11.log --max-sequence-tokens 8192 $B | tail -1 | awk '{print $2}')
    python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18342 m evidence/lp11/$M-$b.json $( [ $b = lp11 ] && echo --against evidence/lp11/$M-lp10.json ) | tail -1
    python3 src/benchmarks/live/long_check.py 2>/dev/null >/dev/null; kill $P; sleep 15
  done
  for rep in 1 2; do for b in lp11 lp10; do
    python3 src/benchmarks/live/compare.py $W lp11/$M/$b-$rep $M 6 $b --order r --scenarios rag-c1,rag6k-c1,rag-c8 --port 18343 --riley-long "$B" | tail -1
  done; done
done
echo LP11_DONE
