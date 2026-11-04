#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/kc evidence/curve
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk "{print \$1}" | xargs -r kill; sleep 10; }
run() { G=$1 P=$2 C=$3
  MAXOUT=2048 bash serve.sh lm1 $G $P Qwen2.5-7B-Instruct m logs/kc-$C.log --max-sequence-tokens 8192 --attention-kv-chunk-tokens $C | tail -1
  python3 ttft_curve.py 127.0.0.1:$P m evidence/curve/riley-lm1-kc$C.json | cut -c1-700
  kp $P; }
( run 4 18311 256; echo C256_DONE ) &
( run 5 18312 512; echo C512_DONE ) &
( run 6 18313 1024; echo C1024_DONE ) &
wait
# HF agreement and invariance for the smaller chunk
MAXOUT=2048 bash serve.sh lm1 4 18311 Qwen2.5-7B-Instruct m logs/kq.log --max-sequence-tokens 8192 --attention-kv-chunk-tokens 256 | tail -1
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18311 m models/Qwen2.5-7B-Instruct evidence/kc/long-kc256.json 2>&1 | tail -1 | cut -c1-300
kp 18311
echo KC_DONE
