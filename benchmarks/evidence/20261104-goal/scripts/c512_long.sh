#!/bin/bash
# c512_long.sh: GPU 6, long-prompt correctness with KV chunk 1024 vs 512 on more models (lp11).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/long512
for M in Qwen2.5-0.5B-Instruct Qwen3-4B Meta-Llama-3.1-8B-Instruct; do
  K=""; [ $M = Qwen3-4B ] && K="--kv-blocks 10050"; [ $M = Meta-Llama-3.1-8B-Instruct ] && K="--kv-blocks 7484"
  for c in 1024 512; do
    P=$(MAXOUT=2048 bash serve.sh lp11 6 18346 $M m logs/l512-$c.log --max-sequence-tokens 8192 --attention-kv-chunk-tokens $c $K | tail -1 | awk '{print $2}')
    echo "$M chunk $c $(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18346 m models/$M evidence/long512/$M-$c.json $P 2>&1 | tail -1 | cut -c1-160)"
    sleep 10
  done
done
echo C512L_DONE
