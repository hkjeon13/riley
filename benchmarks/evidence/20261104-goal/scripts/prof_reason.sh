#!/bin/bash
# prof_reason.sh: nsys of 8 concurrent long generations (reason-c8 shape) on GPU 7, Riley lp6 and vLLM, same prompts.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=$W/models/Qwen2.5-7B-Instruct; P=18283
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
mkdir -p tmp-prof evidence/profreason; rm -f evidence/profreason/*
setsid nohup $NS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/profreason/riley $W/bin/riley-lp6 serve --model $M --model-id m --bind 127.0.0.1:$P --inference-profile native-bf16-varlen-v1 --max-active-sequences 32 --max-output-tokens 2048 --max-weight-bytes 68719476736 --max-sequence-tokens 8192 > logs/profreason-r.log 2>&1 < /dev/null &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
sleep 8
python3 reason_client.py 127.0.0.1:$P
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-lp6 serve .*$P" | awk '{print $1}' | head -1); kill -INT $V
for i in $(seq 1 300); do [ -s evidence/profreason/riley.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 20
$NS export --type=sqlite --force-overwrite=true -o evidence/profreason/riley.sqlite evidence/profreason/riley.nsys-rep > /dev/null 2>&1; echo EXPORT_R $?
VLLM_ENABLE_V1_MULTIPROCESSING=0 $NS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/profreason/vllm tools/vllm-venv/bin/python vllm_reason_prof.py $M > logs/profreason-v.log 2>&1
$NS export --type=sqlite --force-overwrite=true -o evidence/profreason/vllm.sqlite evidence/profreason/vllm.nsys-rep > /dev/null 2>&1; echo EXPORT_V $?
python3 cat_an.py evidence/profreason/riley.sqlite evidence/profreason/vllm.sqlite
rm -rf tmp-prof; echo PROFREASON_DONE
