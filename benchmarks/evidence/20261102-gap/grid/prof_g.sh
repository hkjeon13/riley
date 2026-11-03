#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=$W/models/Qwen2.5-7B-Instruct; P=18281
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
mkdir -p tmp-prof; rm -f evidence/prof/grid1.*
setsid nohup $NS profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/prof/grid1 $W/bin/riley-grid1 serve --model $M --model-id m --bind 127.0.0.1:$P --inference-profile native-bf16-varlen-v1 --max-active-sequences 32 --max-output-tokens 2048 --max-weight-bytes 68719476736 --max-sequence-tokens 8192 > logs/prof-grid1.log 2>&1 < /dev/null &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
python3 prof_load.py 127.0.0.1:$P m 10 | tail -2
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-grid1 serve" | awk '{print $1}' | head -1); echo stop $V; kill -INT $V
for i in $(seq 1 300); do [ -s evidence/prof/grid1.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 30
$NS export --type=sqlite --force-overwrite=true -o evidence/prof/grid1.sqlite evidence/prof/grid1.nsys-rep > /dev/null 2>&1; echo EXPORT $?
python3 prof_an.py evidence/prof/grid1.sqlite | head -14; python3 prof_steps.py evidence/prof/grid1.sqlite
rm -rf tmp-prof; echo RUN_DONE
