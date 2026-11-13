#!/bin/bash
# sm17.sh GPU: SmolLM2-1.7B reason-c32 diagnosis on lp16. Riley-only (rr) reason-c32: default, kv chunk 1024 and 256 (diagnostic, output-changing),
# then an nsys profile of the default. Evidence: evidence/sm17/.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; M=SmolLM2-1.7B-Instruct; mkdir -p evidence/sm17
UUID=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader)
quiet() { local q=0; while [ $q -lt 4 ]; do if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then q=0; else q=$((q+1)); fi; sleep 30; done; }
KVB="$(python3 kvb.py models/$M)"
for v in "base:" "c1024:--attention-kv-chunk-tokens 1024" "c256:--attention-kv-chunk-tokens 256" "c2048:--attention-kv-chunk-tokens 2048"; do
  L=${v%%:*}; F=${v#*:}; quiet; echo "$(date +%H:%M) $L"
  python3 src/benchmarks/live/compare.py $W sm17/$L $M $G lp16 --order rr --scenarios reason-c32 --port 1842$G --riley-long "$KVB $F" | tail -n 1
done
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys; P=1843$G; L=prof
quiet; echo "$(date +%H:%M) prof"
export TMPDIR=$W/tmp-sm17-$G CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
mkdir -p $TMPDIR
setsid nohup $NS profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/sm17/$L $W/bin/riley-lp16 serve --model $W/models/$M --model-id m --bind 127.0.0.1:$P --max-active-sequences 32 --max-weight-bytes 68719476736 --inference-profile native-bf16-varlen-v1 --max-output-tokens 2048 $KVB > logs/sm17-$L.log 2>&1 < /dev/null &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
sleep 5
python3 src/benchmarks/live/load.py 127.0.0.1:$P m --prompt-words 30 --max-tokens 1024 --concurrency 32 --requests 32 --warmup 0 --out evidence/sm17/$L-load.json | tail -2
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-lp16 serve .*$P" | awk '{print $1}' | head -1); kill -INT $V
for i in $(seq 1 300); do [ -s evidence/sm17/$L.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 20
$NS export --type=sqlite --force-overwrite=true -o evidence/sm17/$L.sqlite evidence/sm17/$L.nsys-rep > /dev/null 2>&1; echo EXPORT $?
rm -rf $TMPDIR
echo SM17_DONE
