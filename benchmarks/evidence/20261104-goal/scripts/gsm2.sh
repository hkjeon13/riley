#!/bin/bash
# gsm2.sh: GSM8K Riley lp8 vs vLLM on GPU 7 for three models, skipping results already written.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/gsm8k
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 PATH=$W/tools/vllm-venv/bin:$PATH
while pgrep -f "[g]sm8k.py 127.0.0.1:18338" >/dev/null; do sleep 10; done
stop() { for p in $(ps -eo pid,cmd | grep -E "[b]ind 127.0.0.1:18338|[p]ort 18338 " | awk '{print $1}'); do kill $p; done; sleep 15
         for p in $(ps -eo pid,cmd | grep -E "[b]ind 127.0.0.1:18338|[p]ort 18338 " | awk '{print $1}'); do kill -9 $p; done; sleep 5; }
stop
for M in Qwen2.5-7B-Instruct Llama-3.2-3B-Instruct Qwen3-4B; do
  if [ ! -s evidence/gsm8k/$M-riley.json ]; then
    MAXOUT=512 bash serve.sh lp8 7 18338 $M m logs/gsm-r.log --max-sequence-tokens 4096 | tail -1
    python3 src/benchmarks/live/gsm8k.py 127.0.0.1:18338 m data/gsm8k evidence/gsm8k/$M-riley.json > /dev/null 2>&1
    stop
  fi
  if [ ! -s evidence/gsm8k/$M-vllm.json ]; then
    setsid nohup vllm serve models/$M --served-model-name m --port 18338 --host 127.0.0.1 --max-model-len 4096 --gpu-memory-utilization 0.8 --generation-config vllm --no-enable-prefix-caching --max-num-seqs 32 --disable-log-stats > logs/gsm-v.log 2>&1 < /dev/null &
    for i in $(seq 1 900); do curl -sf -m 2 127.0.0.1:18338/health >/dev/null && break; sleep 1; done
    python3 src/benchmarks/live/gsm8k.py 127.0.0.1:18338 m data/gsm8k evidence/gsm8k/$M-vllm.json > /dev/null 2>&1
    stop
  fi
  for e in riley vllm; do python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[2], sys.argv[3], round(100*d['accuracy'],2))" evidence/gsm8k/$M-$e.json $M $e 2>/dev/null || echo "$M $e missing"; done
done
echo GSM_DONE
