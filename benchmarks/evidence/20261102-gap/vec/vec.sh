#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/vec evidence/curve evidence/prof
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk '{print $1}' | xargs -r kill; sleep 10; }
SC=closed-c8,closed-c32,mid-r16,rag-c1,rag-c8,rag-r3,rag6k-c1,reason-c8,reason-c32
# Riley A/B, two orders on two GPUs.
( python3 src/benchmarks/live/compare.py $W vec/ab-g6-merge1 Qwen2.5-7B-Instruct 6 merge1 --order r --scenarios $SC --port 18301 > logs/g6a.log 2>&1
  python3 src/benchmarks/live/compare.py $W vec/ab-g6-vec1 Qwen2.5-7B-Instruct 6 vec1 --order r --scenarios $SC --port 18301 > logs/g6b.log 2>&1; echo AB6_DONE; sed -e s/merge1/vec1/g prof_m.sh > prof_v.sh; bash prof_v.sh > logs/prof-v.log 2>&1; echo PROF_DONE ) &
( python3 src/benchmarks/live/compare.py $W vec/ab-g7-vec1 Qwen2.5-7B-Instruct 7 vec1 --order r --scenarios $SC --port 18302 > logs/g7a.log 2>&1
  python3 src/benchmarks/live/compare.py $W vec/ab-g7-merge1 Qwen2.5-7B-Instruct 7 merge1 --order r --scenarios $SC --port 18302 > logs/g7b.log 2>&1; echo AB7_DONE ) &
# TTFT by length, GPU 5.
( MAXOUT=2048 bash serve.sh vec1 5 18303 Qwen2.5-7B-Instruct m logs/gc.log --max-sequence-tokens 8192 | tail -1
  python3 ttft_curve.py 127.0.0.1:18303 m evidence/curve/riley-vec1.json | cut -c1-600
  kp 18303; echo CURVE_DONE ) &
CURVE=$!
# Output parity with kv1 and API checks, GPU 4; then the long check with HF on GPU 5.
( MAXOUT=1024 bash serve.sh kv1 4 18304 Qwen2.5-7B-Instruct m logs/gp1.log --max-sequence-tokens 8192 | tail -1
  python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18304 m evidence/merge/parity-kv1.json | tail -1
  kp 18304
  MAXOUT=1024 bash serve.sh vec1 4 18304 Qwen2.5-7B-Instruct m logs/gp2.log --max-sequence-tokens 8192 | tail -1
  python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18304 m evidence/vec/parity-vec1.json --against evidence/merge/parity-kv1.json | tail -1 | cut -c1-200
  python3 src/benchmarks/live/api_check.py 127.0.0.1:18304 m --tools --out evidence/vec/check.json | grep -E "FAIL|checks passed" | cut -c1-200
  kp 18304
  while kill -0 $CURVE 2>/dev/null; do sleep 10; done
  MAXOUT=2048 bash serve.sh vec1 4 18304 Qwen2.5-7B-Instruct m logs/gp3.log --max-sequence-tokens 8192 | tail -1
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18304 m models/Qwen2.5-7B-Instruct evidence/vec/long-q7.json 2>&1 | tail -1 | cut -c1-300
  kp 18304; echo CHECK_DONE ) &
wait
echo VEC_DONE
