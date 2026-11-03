#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/kv4
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk '{print $1}' | xargs -r kill; sleep 8; }
# Ten minutes of mixed traffic on the chunked build, GPU 7.
( R=$(MAXOUT=2048 bash serve.sh kv1 7 18291 Qwen2.5-7B-Instruct m logs/k4s.log --max-sequence-tokens 8192 | tail -1); echo $R
P=$(echo $R | awk '{print $2}')
python3 src/benchmarks/live/soak.py 127.0.0.1:18291 m --seconds 600 --rate 8 --pid $P --out evidence/kv4/soak.json | tail -1 | cut -c1-700
kp 18291
echo SOAK_DONE ) &
# Long context on two GPUs: Riley on 4,5, the HF reference on 6.
( MAXOUT=2048 bash serve.sh kv1 4,5 18292 Qwen2.5-7B-Instruct m logs/k4t.log --max-sequence-tokens 8192 --tensor-parallel-devices 0,1 | tail -1
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18292 m models/Qwen2.5-7B-Instruct evidence/kv4/long-tp.json 2>&1 | tail -1 | cut -c1-300
kp 18292
echo TP_DONE ) &
wait
echo KV4_DONE
