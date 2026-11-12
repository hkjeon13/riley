#!/bin/bash
# hosttrace.sh MODEL: host step trace of rag-c1 for the default and the burst settings on GPU 7.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=$1; mkdir -p evidence/hosttrace
until grep -q PROFREASON_DONE logs/profreason.log; do sleep 15; done; sleep 10
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
for v in base burst; do
  X=""; [ $v = burst ] && X="$B"
  P=$(RILEY_HOST_STEP_TRACE=1 MAXOUT=2048 bash serve.sh lp6 7 18332 $M m evidence/hosttrace/$M-$v.log --max-sequence-tokens 8192 $X | tail -1 | awk '{print $2}')
  python3 src/benchmarks/live/load.py 127.0.0.1:18332 m --prompt-words 1500,3000 --max-tokens 128 --concurrency 1 --requests 16 --warmup 4 --out evidence/hosttrace/$M-$v-rag-c1.json > /dev/null
  python3 src/benchmarks/live/load.py 127.0.0.1:18332 m --prompt-words 30 --max-tokens 1024 --concurrency 1 --requests 4 --warmup 1 --out evidence/hosttrace/$M-$v-gen-c1.json > /dev/null
  kill $P; sleep 15
done
echo HOSTTRACE_DONE
