#!/bin/bash
# val256.sh GPU MODE MODELS...: lp15 (KV chunk 256 default for <=2 KV heads). MODE long = long_check; MODE gsm = GSM8K Riley lp15 and vLLM.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; MODE=$2; shift 2; mkdir -p evidence/val256
export CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$W/tools/vllm-venv/bin:$PATH
for M in "$@"; do
  K=$(python3 kvfree.py models/$M $G)
  if [ $MODE = long ]; then
    [ -s evidence/val256/long-$M.json ] && continue
    P=$(MAXOUT=2048 bash serve.sh lp15 $G 1836$G $M m logs/v256-$G.log --max-sequence-tokens 8192 $K | tail -1 | awk "{print \$2}")
    echo "$M $K $(env CUDA_VISIBLE_DEVICES=$G tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:1836$G m models/$M evidence/val256/long-$M.json $P 2>&1 | tail -1 | cut -c1-200)"
    sleep 10
  else
    if [ ! -s evidence/val256/gsm-$M-riley.json ]; then
    P=$(MAXOUT=512 bash serve.sh lp15 $G 1836$G $M m logs/g256-$G.log --max-sequence-tokens 4096 $K | tail -1 | awk "{print \$2}")
    echo "$M riley $(python3 src/benchmarks/live/gsm8k.py 127.0.0.1:1836$G m data/gsm8k evidence/val256/gsm-$M-riley.json 2>&1 | tail -1)"
    kill $P; sleep 15; fi
    [ -s evidence/val256/gsm-$M-vllm.json ] && continue
    U=$(nvidia-smi -i $G --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | awk -F, "{printf \"%.2f\", (\$1-2500)/\$2}")
    CUDA_VISIBLE_DEVICES=$G setsid nohup tools/vllm-venv/bin/vllm serve models/$M --served-model-name m --port 1836$G --host 127.0.0.1 --max-model-len 4096 --gpu-memory-utilization $U --generation-config vllm --no-enable-prefix-caching --max-num-seqs 32 --disable-log-stats > logs/g256v-$G.log 2>&1 < /dev/null &
    V=$!
    for i in $(seq 1 900); do curl -sf -m 2 127.0.0.1:1836$G/health >/dev/null && break; kill -0 $V 2>/dev/null || break; sleep 1; done
    echo "$M vllm(util $U) $(python3 src/benchmarks/live/gsm8k.py 127.0.0.1:1836$G m data/gsm8k evidence/val256/gsm-$M-vllm.json 2>&1 | tail -1)"
    kill -TERM -- -$V 2>/dev/null || kill $V; sleep 20
    for p in $(ps -eo pid,cmd | grep -E "[p]ort 1836$G " | awk "{print \$1}"); do kill -9 $p; done; sleep 5
  fi
done
echo "VAL256_${G}_${MODE}_DONE"
