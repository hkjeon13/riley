#!/bin/bash
# gsm256.sh GPU MODELS...: GSM8K for Riley lp15 and vLLM per model; retries start-up while other tenants change their GPU memory.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; shift; mkdir -p evidence/val256
export CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$W/tools/vllm-venv/bin:$PATH
stop() { kill -TERM -- -$1 2>/dev/null; kill $1 2>/dev/null; sleep 20; for p in $(ps -u $(whoami) -o pid,args | grep -E "[p]ort 1836$G |[r]iley-lp15 serve.*1836$G" | awk "{print \$1}"); do kill -9 $p; done; sleep 5; }
for M in "$@"; do
  for t in 1 2 3 4 5 6; do
    [ -s evidence/val256/gsm-$M-riley.json ] && break
    K="--kv-blocks $(python3 kvfree.py models/$M $G | awk "{print int(\$2*0.6)}")"
    R=$(MAXOUT=512 bash serve.sh lp15 $G 1836$G $M m logs/g256-$G.log --max-sequence-tokens 4096 $K | tail -n 1)
    case "$R" in READY*) P=$(echo $R | awk "{print \$2}")
      echo "$M riley ($K, try $t) $(python3 src/benchmarks/live/gsm8k.py 127.0.0.1:1836$G m data/gsm8k evidence/val256/gsm-$M-riley.json 2>&1 | tail -n 1)"; stop $P;;
    *) echo "$M riley start failed (try $t): $(grep -oE "failed: .{0,120}" logs/g256-$G.log | tail -n 1)"; sleep 60;; esac
  done
  for t in 1 2 3 4 5 6; do
    [ -s evidence/val256/gsm-$M-vllm.json ] && break
    U=$(nvidia-smi -i $G --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | awk -F, "{printf \"%.2f\", (\$1-4000)/\$2}")
    CUDA_VISIBLE_DEVICES=$G setsid nohup tools/vllm-venv/bin/vllm serve models/$M --served-model-name m --port 1836$G --host 127.0.0.1 --max-model-len 4096 --gpu-memory-utilization $U --generation-config vllm --no-enable-prefix-caching --max-num-seqs 32 --disable-log-stats > logs/g256v-$G.log 2>&1 < /dev/null &
    V=$!; ok=0
    for i in $(seq 1 900); do curl -sf -m 2 127.0.0.1:1836$G/health >/dev/null && { ok=1; break; }; kill -0 $V 2>/dev/null || break; sleep 1; done
    [ $ok = 1 ] && echo "$M vllm(util $U, try $t) $(python3 src/benchmarks/live/gsm8k.py 127.0.0.1:1836$G m data/gsm8k evidence/val256/gsm-$M-vllm.json 2>&1 | tail -n 1)"
    stop $V
    [ $ok = 1 ] || { echo "$M vllm start failed (try $t): $(grep -oE "(Assertion|Runtime|OutOfMemory)Error: .{0,80}" logs/g256v-$G.log | tail -n 1)"; sleep 60; }
  done
done
echo GSM256_${G}_DONE
