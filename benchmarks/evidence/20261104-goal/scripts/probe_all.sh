#!/bin/bash
# probe_all.sh GPU MODELS...: which fine buckets each model prepares (lp13, burst, fine bucket list).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; shift; mkdir -p evidence/probe
B="--max-sequence-tokens 8192 --batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,384,512,640,768,896,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 0"
for M in "$@"; do
  P=$(MAXOUT=2048 bash serve.sh lp13 $G 1837$G $M m evidence/probe/$M.log $(python3 kvb.py models/$M) $B | tail -1 | awk '{print $2}')
  echo "$M $(python3 bucket_probe.py 127.0.0.1:1837$G)"; kill $P; sleep 12
done
echo PROBE_${G}_DONE
