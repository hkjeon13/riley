#!/bin/bash
# fineb.sh GPU MODELS...: burst buckets as today vs. with 384/640/768/896 added; rag-c1, rag6k-c1, rag-c8 twice each, Riley lp12 only; byte check.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; shift; O=evidence/fineb; mkdir -p $O
C="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128"
for M in "$@"; do
 K=$(python3 kvb.py models/$M)
 for rep in 1 2; do for v in base fine; do
  X="$C,144,160,176,192,208,224,240,256,512,1024"; [ $v = fine ] && X="$C,160,192,224,256,384,512,640,768,896,1024"
  P=$(MAXOUT=2048 bash serve.sh lp12 $G 1838$G $M m $O/$M-$v-$rep.log --max-sequence-tokens 8192 $K $X | tail -1 | awk '{print $2}')
  [ $rep = 1 ] && python3 bytes.py 127.0.0.1:1838$G $O/$M-$v-bytes.json
  python3 src/benchmarks/live/load.py 127.0.0.1:1838$G m --prompt-words 1500,3000 --max-tokens 128 --concurrency 1 --requests 16 --warmup 4 --out $O/$M-$v-$rep-rag-c1.json > /dev/null
  python3 src/benchmarks/live/load.py 127.0.0.1:1838$G m --prompt-words 3000,5000 --max-tokens 128 --concurrency 1 --requests 16 --warmup 4 --out $O/$M-$v-$rep-rag6k-c1.json > /dev/null
  python3 src/benchmarks/live/load.py 127.0.0.1:1838$G m --prompt-words 1500,3000 --max-tokens 128 --concurrency 8 --requests 64 --warmup 8 --out $O/$M-$v-$rep-rag-c8.json > /dev/null
  kill $P; sleep 12
 done; done
 cmp -s $O/$M-base-bytes.json $O/$M-fine-bytes.json && echo "$M bytes same" || echo "$M bytes DIFFER"
done
echo FINEB_${G}_DONE
