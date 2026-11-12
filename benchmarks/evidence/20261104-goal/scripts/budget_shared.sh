#!/bin/bash
# budget_shared.sh: Riley lp8 on GPU 7, short server with steps of 256 / 512 / 1024, shared and closed cells.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-7B-Instruct
BK="1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256"
for rep in 1 2; do for b in 256 512 1024; do
  X=""; [ $b != 256 ] && X="--batch-token-budget $b --prefill-chunk-tokens $b --batch-shape-buckets $BK,512$([ $b = 1024 ] && echo ,1024)"
  python3 src/benchmarks/live/compare.py $W bshared/b$b-$rep $M 7 lp8 --order r --scenarios shared-c8,shared-c32,closed-c8,closed-c32,mid-r16 --port 18330 --prefix-cache 4096 --riley-short "$X" | tail -1
done; done
echo BSHARED_DONE
