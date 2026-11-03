#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/curve
G=6 P=18281; M=$W/models/Qwen2.5-7B-Instruct
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$P\|[-]-port $P" | awk '{print $1}' | xargs -r kill; sleep 12; }
for arm in "256 " "512 --batch-token-budget 512 --batch-shape-buckets 1,2,4,8,16,32,64,128,256,512"; do
  set -- $arm; B=$1; shift
  MAXOUT=2048 bash serve.sh kv1 $G $P Qwen2.5-7B-Instruct m logs/c$B.log --max-sequence-tokens 8192 "$@" | tail -1
  python3 ttft_curve.py 127.0.0.1:$P m evidence/curve/riley-$B.json
  kp
done
PATH=$W/tools/vllm-venv/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G setsid nohup $W/tools/vllm-venv/bin/vllm serve $M --served-model-name m --port $P --host 127.0.0.1 --max-model-len 8192 --gpu-memory-utilization 0.8 --generation-config vllm --no-enable-prefix-caching --max-num-seqs 32 --disable-log-stats > logs/cv.log 2>&1 < /dev/null &
for i in $(seq 1 600); do curl -sf -m 2 http://127.0.0.1:$P/health >/dev/null && break; sleep 1; done
echo VLLM_READY $i
python3 ttft_curve.py 127.0.0.1:$P m evidence/curve/vllm.json
kp
echo CURVE_DONE
