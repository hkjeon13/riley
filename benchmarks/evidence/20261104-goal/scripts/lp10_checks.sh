#!/bin/bash
# lp10_checks.sh: build lp10; at the short matrix's next model boundary pause it, check lp10, resume it on lp10.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
bash build2.sh lp10 $W/lp10.patch > logs/build-lp10.log 2>&1; grep -q BUILD_OK logs/build-lp10.log || { echo BUILD_FAILED; exit 1; }
echo built
# wait for the model in progress to finish (a new "date model" line appears), then stop the runner
n=$(grep -c "^[0-9][0-9]:[0-9][0-9] " logs/short.log)
until [ $(grep -c "^[0-9][0-9]:[0-9][0-9] " logs/short.log) -gt $n ]; do sleep 10; done
for p in $(ps -eo pid,cmd | grep "bash short_matrix.s[h]" | awk '{print $1}'); do kill $p; done; sleep 1
for p in $(ps -eo pid,cmd | grep -E "[c]ompare.py .* short/" | awk '{print $1}'); do kill $p; done; sleep 2
for p in $(ps -eo pid,cmd | grep -E "[b]ind 127.0.0.1:18339|[p]ort 18339 " | awk '{print $1}'); do kill -TERM -- -$p 2>/dev/null || kill $p; done; sleep 20
for p in $(ps -eo pid,cmd | grep -E "[b]ind 127.0.0.1:18339|[p]ort 18339 " | awk '{print $1}'); do kill -9 $p; done
M=$(grep "^[0-9][0-9]:[0-9][0-9] " logs/short.log | tail -1 | awk '{print $2}'); rm -rf evidence/short/$M; echo "paused before $M"
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
mkdir -p evidence/lp10
for M in Qwen2.5-14B-Instruct Qwen3-14B DeepSeek-R1-Distill-Qwen-14B Qwen2.5-Coder-14B-Instruct; do
  K=$(python3 - models/$M <<'PY'
import json, os, sys
p = sys.argv[1]; c = json.load(open(f"{p}/config.json")); h = c["num_attention_heads"]
d = c.get("head_dim") or c["hidden_size"] // h
per = c["num_hidden_layers"] * c.get("num_key_value_heads", h) * d * 4
w = sum(os.path.getsize(f"{p}/{f}") for f in os.listdir(p) if f.endswith(".safetensors"))
print(int((0.9 * 39.4 * 2**30 - w - 2.5 * 2**30) // (16 * per)))
PY
)
  R=$(MAXOUT=1024 bash serve.sh lp10 7 18341 $M m logs/l10-$M.log --max-sequence-tokens 8192 $B --kv-blocks $K | tail -1); P=$(echo $R | awk '{print $2}')
  echo "$M kv $K | $R | $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 7)"
  [ -n "$P" ] && python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18341 m evidence/lp10/$M-burst.json | tail -1
  [ -n "$P" ] && kill $P; sleep 15
done
for M in Qwen2.5-0.5B-Instruct Qwen3-4B Qwen2.5-7B-Instruct; do
  for b in lp9 lp10; do
    X="$B"; [ $M = Qwen3-4B ] && X="$B --kv-blocks 10050"
    P=$(MAXOUT=1024 bash serve.sh $b 7 18341 $M m logs/l10-$M.log --max-sequence-tokens 8192 $X | tail -1 | awk '{print $2}')
    echo "$M $b mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 7)"
    python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18341 m evidence/lp10/$M-$b.json $( [ $b = lp10 ] && echo --against evidence/lp10/$M-lp9.json ) | tail -1
    kill $P; sleep 15
  done
done
sed -i 's/ 7 lp9 / 7 lp10 /; s/until grep -q BOOT14_DONE logs\/boot14.log; do sleep 30; done; sleep 10/true/' short_matrix.sh
setsid nohup bash short_matrix.sh >> logs/short.log 2>&1 < /dev/null &
echo LP10_DONE
