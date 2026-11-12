#!/bin/bash
# boot14.sh: after the boot test, start each 14B model with the burst and the 4.5 GiB reserve; then release the short matrix.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until grep -q BOOT_DONE logs/boot.log; do sleep 20; done; sleep 15
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
for M in Qwen2.5-14B-Instruct Qwen2.5-Coder-14B-Instruct DeepSeek-R1-Distill-Qwen-14B Qwen3-14B; do
  K=$(python3 - models/$M <<'PY'
import json, os, sys
p = sys.argv[1]; c = json.load(open(f"{p}/config.json")); h = c["num_attention_heads"]
d = c.get("head_dim") or c["hidden_size"] // h
per = c["num_hidden_layers"] * c.get("num_key_value_heads", h) * d * 4
w = sum(os.path.getsize(f"{p}/{f}") for f in os.listdir(p) if f.endswith(".safetensors"))
print(int((0.9 * 39.4 * 2**30 - w - 4.5 * 2**30) // (16 * per)))
PY
)
  R=$(MAXOUT=64 bash serve.sh lp9 7 18340 $M m logs/boot14-$M.log --max-sequence-tokens 8192 $B --kv-blocks $K | tail -1)
  P=$(echo $R | awk '{print $2}')
  OUT=$(curl -s -m 120 127.0.0.1:18340/v1/completions -H 'Content-Type: application/json' -d '{"model":"m","prompt":"The capital of France is","max_tokens":8,"temperature":0}' | head -c 160)
  echo "$M kv $K | $R | $OUT"
  [ -n "$P" ] && kill $P 2>/dev/null; sleep 15
done
echo BOOT14_DONE
