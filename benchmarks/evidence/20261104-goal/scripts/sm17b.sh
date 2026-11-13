#!/bin/bash
# sm17b.sh GPU: SmolLM2-1.7B reason-c32 Riley-only (rr) with step budget 512 and 128 (byte-preserving lever).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; M=SmolLM2-1.7B-Instruct; KVB="$(python3 kvb.py models/$M)"
UUID=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader); q=0; while [ $q -lt 4 ]; do if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then q=0; else q=$((q+1)); fi; [ $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits) -gt 1000 ] && q=0; sleep 30; done
B5="--batch-token-budget 512 --prefill-chunk-tokens 512 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512"
B1="--batch-token-budget 192 --prefill-chunk-tokens 192 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192 --prefill-plateau-rows 128"
python3 src/benchmarks/live/compare.py $W sm17/g7-base $M $G lp16 --order rr --scenarios reason-c32 --port 1848$G --riley-long "$KVB" | tail -n 1
python3 src/benchmarks/live/compare.py $W sm17/g7-b512 $M $G lp16 --order rr --scenarios reason-c32 --port 1848$G --riley-long "$KVB $B5" | tail -n 1
python3 src/benchmarks/live/compare.py $W sm17/g7-b192 $M $G lp16 --order rr --scenarios reason-c32 --port 1848$G --riley-long "$KVB $B1" | tail -n 1
echo SM17B_DONE
