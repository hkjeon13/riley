#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/gemm evidence/curve
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk "{print \$1}" | xargs -r kill; sleep 10; }
SC=closed-c8,closed-c32,mid-r16,rag-c1,rag-c8,rag-r3,rag6k-c1,reason-c8,reason-c32
( for b in bur1 gA gB; do python3 src/benchmarks/live/compare.py $W gemm/ab-g6-$b Qwen2.5-7B-Instruct 6 $b --order r --scenarios $SC --port 18301 > logs/m6-$b.log 2>&1; done; echo AB6_DONE ) &
( for b in gB gA bur1; do python3 src/benchmarks/live/compare.py $W gemm/ab-g7-$b Qwen2.5-7B-Instruct 7 $b --order r --scenarios $SC --port 18302 > logs/m7-$b.log 2>&1; done; echo AB7_DONE ) &
( for b in gA gB; do MAXOUT=2048 bash serve.sh $b 5 18303 Qwen2.5-7B-Instruct m logs/mc-$b.log --max-sequence-tokens 8192 | tail -1
  python3 ttft_curve.py 127.0.0.1:18303 m evidence/curve/riley-$b.json | cut -c1-700
  kp 18303; done; echo CURVE_DONE ) &
CURVE=$!
( for b in gA gB; do RILEY_GEMM_EQUIVALENT_LOG=1 MAXOUT=1024 bash serve.sh $b 4 18304 Qwen2.5-7B-Instruct m logs/mp-$b.log --max-sequence-tokens 8192 | tail -1
  grep GEMMEQUIV logs/mp-$b.log | grep -E "n=(37888|3584|4608) " | grep -E "m=(1|64|128|256) " | sort | uniq | head -40
  python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18304 m evidence/gemm/parity-$b.json --against evidence/merge/parity-kv1.json | tail -1 | cut -c1-200
  python3 src/benchmarks/live/api_check.py 127.0.0.1:18304 m --tools --out evidence/gemm/check-$b.json | grep -E "FAIL|checks passed" | cut -c1-200
  kp 18304; done
  while kill -0 $CURVE 2>/dev/null; do sleep 10; done
  MAXOUT=2048 bash serve.sh gB 4 18304 Qwen2.5-7B-Instruct m logs/mq-gB.log --max-sequence-tokens 8192 | tail -1
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18304 m models/Qwen2.5-7B-Instruct evidence/gemm/long-gB.json 2>&1 | tail -1 | cut -c1-300
  kp 18304; echo CHECK_DONE ) &
wait
echo GM_DONE
