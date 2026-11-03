#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/kv2
kp() { ps -eo pid,cmd | grep "[r]iley-$1 serve .*$2" | awk "{print \$1}" | xargs -r kill; sleep 6; }
PY5="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 tools/vllm-venv/bin/python"
# Parity with the unsplit build, GPU 6: short context and long context (rows within one chunk).
( MAXOUT=1024 bash serve.sh api9 6 18251 Qwen2.5-7B-Instruct m logs/k6a.log --max-sequence-tokens 8192 | tail -1
python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18251 m evidence/kv2/parity-api9.json | tail -1
kp api9 18251
MAXOUT=1024 bash serve.sh kv1 6 18251 Qwen2.5-7B-Instruct m logs/k6b.log --max-sequence-tokens 8192 | tail -1
python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18251 m evidence/kv2/parity-kv1.json --against evidence/kv2/parity-api9.json | tail -1 | cut -c1-200
python3 src/benchmarks/live/api_check.py 127.0.0.1:18251 m --tools --out evidence/kv2/check-q7.json | grep -E "FAIL|checks passed" | cut -c1-200
kp kv1 18251
echo PARITY_DONE ) &
# Long-context correctness, GPU 5: Qwen2.5-7B then Qwen3-4B.
( MAXOUT=2048 bash serve.sh kv1 5 18252 Qwen2.5-7B-Instruct m logs/k5a.log --max-sequence-tokens 8192 | tail -1
$PY5 src/benchmarks/live/long_check.py 127.0.0.1:18252 m models/Qwen2.5-7B-Instruct evidence/kv2/long-q7.json 2>&1 | tail -1 | cut -c1-300
kp kv1 18252
MAXOUT=2048 bash serve.sh kv1 5 18252 Qwen3-4B m logs/k5b.log --max-sequence-tokens 8192 --kv-blocks 6000 | tail -1
$PY5 src/benchmarks/live/long_check.py 127.0.0.1:18252 m models/Qwen3-4B evidence/kv2/long-q3.json 2>&1 | tail -1 | cut -c1-300
kp kv1 18252
echo LONG_DONE ) &
# Riley and vLLM alternating on GPU 4.
python3 src/benchmarks/live/compare.py $W kv2/cmp Qwen2.5-7B-Instruct 4 kv1 --order rvvr --scenarios closed-c8,closed-c32,mid-r16,rag-c1,rag-c8,rag-r1,rag-r3,rag6k-c1,reason-c8,reason-c32 --port 18253 > logs/kv2-cmp.log 2>&1 &
wait
echo KV2_DONE
