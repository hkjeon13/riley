#!/bin/bash
# long512.sh: long-prompt correctness with KV chunk 512 vs 1024 (lp11, Qwen2.5-7B, GPU 6) after the rechecks.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-7B-Instruct; mkdir -p evidence/long512
until grep -q RECHECK_DONE logs/recheck6.log; do sleep 30; done; sleep 10
for c in 1024 512; do
  P=$(MAXOUT=2048 bash serve.sh lp11 6 18346 $M m logs/l512-$c.log --max-sequence-tokens 8192 --attention-kv-chunk-tokens $c | tail -1 | awk '{print $2}')
  echo "chunk $c $(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18346 m models/$M evidence/long512/long-$c.json $P 2>&1 | tail -1 | cut -c1-300)"
  sleep 10
done
echo LONG512_DONE
