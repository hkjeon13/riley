#!/bin/bash
# ab14.sh GPU MODELS...: lp13 vs lp14 (burst kept open for a started long prompt's last chunk), standard burst flags,
# rag-c1, rag6k-c1, rag-c8, rag-r3 twice each alternating; byte check of 16 long prompts.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; shift; O=evidence/ab14; mkdir -p $O
X="--max-sequence-tokens 8192 --batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
L="python3 src/benchmarks/live/load.py 127.0.0.1:1838$G m --max-tokens 128"
for M in "$@"; do
 K=$(python3 kvb.py models/$M)
 for rep in 1 2; do for v in lp13 lp14; do
  P=$(MAXOUT=2048 bash serve.sh $v $G 1838$G $M m $O/$M-$v-$rep.log $K $X | tail -1 | awk '{print $2}')
  [ $rep = 1 ] && python3 bytes.py 127.0.0.1:1838$G $O/$M-$v-bytes.json > /dev/null
  $L --prompt-words 1500,3000 --concurrency 1 --requests 16 --warmup 4 --out $O/$M-$v-$rep-rag-c1.json > /dev/null
  $L --prompt-words 3000,5000 --concurrency 1 --requests 16 --warmup 4 --out $O/$M-$v-$rep-rag6k-c1.json > /dev/null
  $L --prompt-words 1500,3000 --concurrency 8 --requests 64 --warmup 8 --out $O/$M-$v-$rep-rag-c8.json > /dev/null
  $L --prompt-words 1500,3000 --rate 3 --requests 64 --warmup 8 --out $O/$M-$v-$rep-rag-r3.json > /dev/null
  kill $P; sleep 12
 done; done
 cmp -s $O/$M-lp13-bytes.json $O/$M-lp14-bytes.json && echo "$M bytes same" || echo "$M bytes DIFFER"
done
echo AB14_${G}_DONE
