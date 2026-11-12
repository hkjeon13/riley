#!/bin/bash
# prof_dec.sh BIN LABEL [flags...]: nsys profile of long-context decode (6 x ~3,500-token prompt, 200 tokens) on GPU 7.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; B=$1; L=$2; shift 2; M=$W/models/Qwen2.5-7B-Instruct; P=18282
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
mkdir -p tmp-prof evidence/profdec; rm -f evidence/profdec/$L.*
setsid nohup $NS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/profdec/$L $W/bin/riley-$B serve --model $M --model-id m --bind 127.0.0.1:$P --inference-profile native-bf16-varlen-v1 --max-active-sequences 32 --max-output-tokens 2048 --max-weight-bytes 68719476736 --max-sequence-tokens 8192 "$@" > logs/profdec-$L.log 2>&1 < /dev/null &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
sleep 8
python3 - <<PY
import json, random, urllib.request
words = "the river market station garden winter engine letter mountain copper signal harbor forest planet".split()
rng = random.Random(3)
for i in range(6):
    prompt = " ".join(rng.choice(words) for _ in range(3000)) + "\n\nNow write a long story, at least 500 words, about the river and the harbor:\n"
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": 200, "temperature": 0}).encode()
    reply = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$P/v1/completions", body, {"Content-Type": "application/json"}), timeout=600))
    print(reply["usage"])
PY
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-$B serve .*$P" | awk '{print $1}' | head -1); kill -INT $V
for i in $(seq 1 300); do [ -s evidence/profdec/$L.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 20
$NS export --type=sqlite --force-overwrite=true -o evidence/profdec/$L.sqlite evidence/profdec/$L.nsys-rep > /dev/null 2>&1; echo EXPORT $?
rm -rf tmp-prof
