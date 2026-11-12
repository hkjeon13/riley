#!/bin/bash
# prof14.sh GPU LABEL [flags...]: nsys profile of Coder-14B closed-c32 (30 words, 64 tokens) on lp16, once the GPU is quiet for 10 minutes.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; L=$2; shift 2; M=$W/models/Qwen2.5-Coder-14B-Instruct; P=1829$G
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
UUID=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader); quiet=0
while [ $quiet -lt 20 ]; do
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then quiet=0; else quiet=$((quiet+1)); fi
  sleep 30
done
echo "$(date +%m-%d\ %H:%M) GPU $G quiet"
export TMPDIR=$W/tmp-prof14-$G CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
mkdir -p $TMPDIR evidence/prof14; rm -f evidence/prof14/$L.*
setsid nohup $NS profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/prof14/$L $W/bin/riley-lp16 serve --model $M --model-id m --bind 127.0.0.1:$P --max-active-sequences 32 --max-weight-bytes 68719476736 --inference-profile native-bf16-varlen-v1 --max-output-tokens 128 "$@" > logs/prof14-$L.log 2>&1 < /dev/null &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
sleep 5
python3 src/benchmarks/live/load.py 127.0.0.1:$P m --prompt-words 30 --max-tokens 64 --concurrency 32 --requests 128 --warmup 32 --out evidence/prof14/$L-load.json | tail -2
curl -s http://127.0.0.1:$P/metrics > evidence/prof14/$L-metrics.json
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-lp16 serve .*$P" | awk '{print $1}' | head -1); kill -INT $V
for i in $(seq 1 300); do [ -s evidence/prof14/$L.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 20
$NS export --type=sqlite --force-overwrite=true -o evidence/prof14/$L.sqlite evidence/prof14/$L.nsys-rep > /dev/null 2>&1; echo EXPORT $?
rm -rf $TMPDIR
echo PROF14_DONE
