#!/bin/bash
# long2.sh: long-prompt correctness on GPU 7 for the default steps and the 1024-row burst (server stopped before HF loads).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-7B-Instruct; mkdir -p evidence/long2
until grep -q BURST_DONE logs/burst.log; do sleep 10; done; sleep 15
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
for v in base burst; do
  X=""; [ $v = burst ] && X="$B"
  P=$(MAXOUT=2048 bash serve.sh lp4 7 18331 $M m logs/l2-$v.log --max-sequence-tokens 8192 $X | tail -1 | awk '{print $2}')
  echo $v server $P
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18331 m models/$M evidence/long2/long-$v.json $P 2>&1 | tail -1 | cut -c1-400
  sleep 10
done
python3 -c "
import json; a=json.load(open('evidence/long2/long-base.json'))['riley_tokens']; b=json.load(open('evidence/long2/long-burst.json'))['riley_tokens']
print('base vs burst identical', sum(x==y for x,y in zip(a,b)), 'of', len(a))"
echo LONG2_DONE
