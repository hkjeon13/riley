#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/lm evidence/curve
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk "{print \$1}" | xargs -r kill; sleep 10; }
SC=closed-c8,closed-c32,mid-r16,rag-c1,rag-c8,rag-r3,rag6k-c1,reason-c8,reason-c32
( for b in bur1 lm1; do python3 src/benchmarks/live/compare.py $W lm/ab-g6-$b Qwen2.5-7B-Instruct 6 $b --order r --scenarios $SC --port 18301 > logs/l6-$b.log 2>&1; done; echo AB6_DONE
  sed -e s/merge1/lm1/g prof_m.sh > prof_lm1.sh; bash prof_lm1.sh > logs/prof-lm1.log 2>&1; echo PROF_DONE ) &
( for b in lm1 bur1; do python3 src/benchmarks/live/compare.py $W lm/ab-g7-$b Qwen2.5-7B-Instruct 7 $b --order r --scenarios $SC --port 18302 > logs/l7-$b.log 2>&1; done; echo AB7_DONE ) &
( MAXOUT=2048 bash serve.sh lm1 5 18303 Qwen2.5-7B-Instruct m logs/lc.log --max-sequence-tokens 8192 | tail -1
  python3 ttft_curve.py 127.0.0.1:18303 m evidence/curve/riley-lm1.json | cut -c1-700
  kp 18303; echo CURVE_DONE ) &
CURVE=$!
( MAXOUT=1024 bash serve.sh lm1 4 18304 Qwen2.5-7B-Instruct m logs/lp.log --max-sequence-tokens 8192 | tail -1
  python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18304 m evidence/lm/parity-lm1.json --against evidence/merge/parity-kv1.json | tail -1 | cut -c1-200
  python3 src/benchmarks/live/api_check.py 127.0.0.1:18304 m --tools --out evidence/lm/check.json | grep -E "FAIL|checks passed" | cut -c1-200
  kp 18304
  while kill -0 $CURVE 2>/dev/null; do sleep 10; done
  MAXOUT=2048 bash serve.sh lm1 4 18304 Qwen2.5-7B-Instruct m logs/lq.log --max-sequence-tokens 8192 | tail -1
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18304 m models/Qwen2.5-7B-Instruct evidence/lm/long-q7.json 2>&1 | tail -1 | cut -c1-300
  kp 18304; echo CHECK_DONE ) &
wait
echo LM_DONE
