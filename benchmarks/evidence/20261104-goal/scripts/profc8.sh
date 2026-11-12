#!/bin/bash
# profc8.sh MODEL: nsys of rag-c8 (24 requests after 8 warm-up), Riley lp13 + burst, GPU 6.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=6; P=18396; MN=$1; M=$W/models/$MN; O=evidence/profc8/$MN
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof$G CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
mkdir -p tmp-prof$G $O; rm -f $O/*
setsid nohup $NS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $O/riley $W/bin/riley-lp13 serve --model $M --model-id m --bind 127.0.0.1:$P --inference-profile native-bf16-varlen-v1 --max-active-sequences 32 --max-output-tokens 2048 --max-weight-bytes 68719476736 --max-sequence-tokens 8192 $(python3 kvb.py $M) $B > $O/server.log 2>&1 < /dev/null &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
sleep 8
python3 src/benchmarks/live/load.py 127.0.0.1:$P m --prompt-words 1500,3000 --max-tokens 128 --concurrency 8 --requests 24 --warmup 8 --out $O/load.json --records $O/rec.json
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-lp13 serve .*$P" | awk '{print $1}' | head -1); kill -INT $V
for i in $(seq 1 300); do [ -s $O/riley.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 20
$NS export --type=sqlite --force-overwrite=true -o $O/riley.sqlite $O/riley.nsys-rep > /dev/null 2>&1; echo EXPORT_R $?
rm -rf tmp-prof$G; echo PROFC8_DONE
