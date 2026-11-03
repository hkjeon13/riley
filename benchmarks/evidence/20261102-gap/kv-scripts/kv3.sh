#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/kv3
kp() { ps -eo pid,cmd | grep "[r]iley-$1 serve .*$2" | awk "{print \$1}" | xargs -r kill; sleep 6; }
# Long-context correctness on Qwen2.5-7B: Riley on GPU 5, the HF reference on GPU 4.
( MAXOUT=2048 bash serve.sh kv1 5 18262 Qwen2.5-7B-Instruct m logs/k5c.log --max-sequence-tokens 8192 | tail -1
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18262 m models/Qwen2.5-7B-Instruct evidence/kv3/long-q7.json 2>&1 | tail -1 | cut -c1-300
kp kv1 18262
echo LONG_DONE ) &
# Riley and vLLM alternating on GPU 7, where vLLM is healthy.
python3 src/benchmarks/live/compare.py $W kv3/cmp Qwen2.5-7B-Instruct 7 kv1 --order rvvr --scenarios closed-c8,closed-c32,mid-r16,rag-c1,rag-c8,rag-r1,rag-r3,rag6k-c1,reason-c8,reason-c32 --port 18263 > logs/kv3-cmp.log 2>&1 &
wait
echo KV3_DONE
