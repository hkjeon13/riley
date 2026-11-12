#!/bin/bash
# prof_prefill.sh MODEL: nsys of ten ~3,500-token prompts (1 output token) on GPU 7: Riley lp6 with the 1024-row burst, and vLLM.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; MN=$1; M=$W/models/$MN; P=18284; O=evidence/profprefill/$MN
until grep -q HOSTTRACE_DONE logs/hosttrace.log; do sleep 15; done; sleep 10
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
mkdir -p tmp-prof $O; rm -f $O/*
setsid nohup $NS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $O/riley $W/bin/riley-lp6 serve --model $M --model-id m --bind 127.0.0.1:$P --inference-profile native-bf16-varlen-v1 --max-active-sequences 32 --max-output-tokens 2048 --max-weight-bytes 68719476736 --max-sequence-tokens 8192 $B > logs/profprefill-r.log 2>&1 < /dev/null &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
sleep 8
python3 - <<PY
import json, random, urllib.request
words = "the of model server request token memory context cache vector search document answer query system data result value time long short".split()
for seed in range(10):
    rng = random.Random(seed)
    text = " ".join(rng.choice(words) for _ in range(2600))
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0}).encode()
    reply = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$P/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600))
    print(reply["usage"]["prompt_tokens"], end=" ")
PY
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-lp6 serve .*$P" | awk '{print $1}' | head -1); kill -INT $V
for i in $(seq 1 300); do [ -s $O/riley.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 20
$NS export --type=sqlite --force-overwrite=true -o $O/riley.sqlite $O/riley.nsys-rep > /dev/null 2>&1; echo EXPORT_R $?
sed "s/max_tokens=4/max_tokens=1/" vllm_prof.py > vllm_prof1.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 $NS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $O/vllm tools/vllm-venv/bin/python vllm_prof1.py $M > logs/profprefill-v.log 2>&1
$NS export --type=sqlite --force-overwrite=true -o $O/vllm.sqlite $O/vllm.nsys-rep > /dev/null 2>&1; echo EXPORT_V $?
python3 cat_an.py $O/riley.sqlite $O/vllm.sqlite
rm -rf tmp-prof; echo PROFPREFILL_DONE
