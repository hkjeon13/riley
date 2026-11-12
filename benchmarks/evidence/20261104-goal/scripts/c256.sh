#!/bin/bash
# c256.sh: chunk 256 speed (GPU 7) and long-prompt correctness (GPU 6) on Llama-3.1-8B and Qwen3-4B, lp11.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
(
for M in Meta-Llama-3.1-8B-Instruct Qwen3-4B; do
  K="--kv-blocks 10050"; [ $M = Meta-Llama-3.1-8B-Instruct ] && K="--kv-blocks 7484"
  P=$(MAXOUT=2048 bash serve.sh lp11 6 18346 $M m logs/l256.log --max-sequence-tokens 8192 --attention-kv-chunk-tokens 256 $K | tail -1 | awk '{print $2}')
  echo "$M chunk 256 $(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18346 m models/$M evidence/long512/$M-256.json $P 2>&1 | tail -1 | cut -c1-160)"
  sleep 10
done ) &
for M in Meta-Llama-3.1-8B-Instruct Qwen3-4B; do
  K="--kv-blocks 10050"; [ $M = Meta-Llama-3.1-8B-Instruct ] && K="--kv-blocks 7484"
  for rep in 1 2; do
    python3 src/benchmarks/live/compare.py $W chunk/$M/c256-$rep $M 7 lp11 --order r --scenarios reason-c8,reason-c32,rag-c1 --port 18344 --riley-long "--attention-kv-chunk-tokens 256 $K" | tail -1
  done
done
wait
echo C256_DONE
