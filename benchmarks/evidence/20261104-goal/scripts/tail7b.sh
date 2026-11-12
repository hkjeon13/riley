#!/bin/bash
# tail7b.sh: per-request TTFT records for rag-c8 and rag-r3 on Qwen2.5-7B, Riley lp13 + burst and vLLM, GPU 6.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; O=evidence/tail; mkdir -p $O; M=Qwen2.5-7B-Instruct
B="--max-sequence-tokens 8192 --batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
L="python3 src/benchmarks/live/load.py 127.0.0.1:18369 m --prompt-words 1500,3000 --max-tokens 128 --requests 64 --warmup 8"
P=$(MAXOUT=2048 bash serve.sh lp13 6 18369 $M m $O/r.log $B | tail -1 | awk '{print $2}')
$L --concurrency 8 --out $O/r-c8.json --records $O/r-c8-rec.json; $L --rate 3 --out $O/r-r3.json --records $O/r-r3-rec.json
kill $P; sleep 15
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PATH=$W/tools/vllm-venv/bin:$PATH
setsid nohup vllm serve models/$M --served-model-name m --port 18369 --host 127.0.0.1 --max-model-len 8192 --gpu-memory-utilization 0.8 --generation-config vllm --max-num-seqs 32 --disable-log-stats --no-enable-prefix-caching > $O/v.log 2>&1 < /dev/null &
V=$!; for i in $(seq 1 600); do curl -sf -m 2 http://127.0.0.1:18369/health >/dev/null && break; sleep 1; done
$L --concurrency 8 --out $O/v-c8.json --records $O/v-c8-rec.json; $L --rate 3 --out $O/v-r3.json --records $O/v-r3-rec.json
kill $V; sleep 20; echo TAIL_DONE
