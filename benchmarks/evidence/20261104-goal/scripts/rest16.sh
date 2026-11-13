#!/bin/bash
# rest16.sh GPU: 0.1.0 (lp16) re-measure of the remaining goal rows. Models are taken from a shared queue (mkdir lock), so GPU 6 and 7 split the work.
# Per model, once the GPU has no process and <1 GB used for 2 minutes: closed-c1, logprobs-c8, logprobs-c32 vs vLLM (rvvr) -> evidence/rest16/<model>.
# Then (GPU 7 only, after both queues drain): shared-c8/c32 Qwen2.5-7B with --prefix-cache 4096 (rest16/shared), 32B TP=2 closed c1-c32 on GPUs 6,7 (rest16/tp32).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; mkdir -p evidence/rest16 logs/rest16q
quiet() { for g in "$@"; do local U=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader) q=0
  while [ $q -lt 4 ]; do q=$((q+1)); nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$U" && q=0
    [ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -gt 1000 ] && q=0; sleep 30; done; done; }
for M in $(ls models | grep -v -- "-32B"); do
  [ -d logs/rest16q/$M ] && continue
  quiet $G; mkdir logs/rest16q/$M 2>/dev/null || continue
  W_BYTES=$(du -bc models/$M/*.safetensors | tail -1 | awk '{print $1}'); VM=0.8; [ $W_BYTES -gt 21474836480 ] && VM=0.9
  echo "$(date +%m-%d\ %H:%M) $M"
  python3 src/benchmarks/live/compare.py $W rest16/$M $M $G lp16 --order rvvr --scenarios closed-c1,logprobs-c8,logprobs-c32 --port 1850$G --vllm-memory $VM | tail -n 1
  touch logs/rest16q/$M/done
done
echo REST16_${G}_QUEUE_DONE
if [ $G = 7 ]; then
  while [ $(ls -d logs/rest16q/*/ | wc -l) -ne $(ls -d logs/rest16q/*/done 2>/dev/null | wc -l) ]; do sleep 60; done
  quiet 7; python3 src/benchmarks/live/compare.py $W rest16/shared/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct 7 lp16 --order rvvr --scenarios shared-c8,shared-c32 --port 18517 --prefix-cache 4096 | tail -n 1
  quiet 6 7; python3 src/benchmarks/live/compare.py $W rest16/tp32/Qwen2.5-32B-Instruct Qwen2.5-32B-Instruct 6,7 lp16 --tp 2 --order rvvr --scenarios closed-c1,closed-c8,closed-c16,closed-c32 --port 18550 --vllm-memory 0.9 | tail -n 1
  echo REST16_ALL_DONE
fi
