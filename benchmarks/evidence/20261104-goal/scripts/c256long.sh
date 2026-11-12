#!/bin/bash
# c256long.sh: long_check (invariance at 8/32 clients, HF agreement) for chunk 256 on the two-KV-head 3B models, lp13, GPU 7.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=7; mkdir -p evidence/c256long
for M in Qwen2.5-3B-Instruct Qwen2.5-Coder-3B-Instruct; do
  K=$(python3 kvb.py models/$M)
  P=$(MAXOUT=2048 bash serve.sh lp13 $G 1836$G $M m logs/c256long.log --max-sequence-tokens 8192 --attention-kv-chunk-tokens 256 $K | tail -1 | awk "{print \$2}")
  echo "$M $(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:1836$G m models/$M evidence/c256long/long-$M.json $P 2>&1 | tail -1 | cut -c1-200)"
  sleep 10
done
echo C256LONG_DONE
