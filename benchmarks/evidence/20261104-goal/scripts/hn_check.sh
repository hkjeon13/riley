#!/bin/bash
# hn_check.sh: after models7, GPU tests for the head-norm split, then Qwen3 output parity and rag cells of lp8 vs lp7.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
true
export RUSTUP_HOME=$W/tools/rustup CARGO_HOME=$W/tools/cargo PATH=$W/tools/cargo/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 RILEY_CUDA_ARCHITECTURES=80 CARGO_TARGET_DIR=$W/target

B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
mkdir -p evidence/hn
for M in Qwen3-4B; do
  P=$(MAXOUT=1024 bash serve.sh lp7 7 18331 $M m logs/hn-a.log --max-sequence-tokens 8192 $B --kv-blocks 10050 | tail -1 | awk '{print $2}')
  python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18331 m evidence/hn/$M-lp7.json | tail -1
  kill $P; sleep 15
  P=$(MAXOUT=1024 bash serve.sh lp8 7 18331 $M m logs/hn-b.log --max-sequence-tokens 8192 $B --kv-blocks 10050 | tail -1 | awk '{print $2}')
  python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18331 m evidence/hn/$M-lp8.json --against evidence/hn/$M-lp7.json | tail -1
  kill $P; sleep 15
  for rep in 1 2; do for b in lp8 lp7; do
    python3 src/benchmarks/live/compare.py $W hn/$M/$b-$rep $M 7 $b --order r --scenarios rag-c1,rag6k-c1 --port 18330 --riley-long "$B --kv-blocks 10050" | tail -1
  done; done
done
echo HN_DONE
