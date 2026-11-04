#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/b0 evidence/curve
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk "{print \$1}" | xargs -r kill; sleep 10; }
SC=closed-c8,closed-c32,mid-r16,rag-c1,rag-c8,rag-r3,rag6k-c1,reason-c8,reason-c32
F="--batch-token-budget 512 --prefill-chunk-tokens 512 --batch-shape-buckets 1,2,4,8,16,32,64,128,256,512 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 0"
( python3 src/benchmarks/live/compare.py $W b0/ab-g6-base Qwen2.5-7B-Instruct 6 vec1 --order r --scenarios $SC --port 18301 > logs/b6a.log 2>&1
  python3 src/benchmarks/live/compare.py $W b0/ab-g6-b0 Qwen2.5-7B-Instruct 6 vec1 --order r --scenarios $SC --port 18301 --riley-short "$F" --riley-long "$F" > logs/b6b.log 2>&1; echo AB6_DONE ) &
( python3 src/benchmarks/live/compare.py $W b0/ab-g7-b0 Qwen2.5-7B-Instruct 7 vec1 --order r --scenarios $SC --port 18302 --riley-short "$F" --riley-long "$F" > logs/b7a.log 2>&1
  python3 src/benchmarks/live/compare.py $W b0/ab-g7-base Qwen2.5-7B-Instruct 7 vec1 --order r --scenarios $SC --port 18302 > logs/b7b.log 2>&1; echo AB7_DONE ) &
( MAXOUT=2048 bash serve.sh vec1 5 18303 Qwen2.5-7B-Instruct m logs/bc.log --max-sequence-tokens 8192 $F | tail -1
  python3 ttft_curve.py 127.0.0.1:18303 m evidence/curve/riley-vec1-b0.json | cut -c1-600
  kp 18303; echo CURVE_DONE ) &
( MAXOUT=1024 bash serve.sh vec1 4 18304 Qwen2.5-7B-Instruct m logs/bp.log --max-sequence-tokens 8192 $F | tail -1
  python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18304 m evidence/b0/parity-b0.json --against evidence/merge/parity-kv1.json | tail -1 | cut -c1-200
  kp 18304; echo PARITY_DONE ) &
wait
echo B0_DONE
