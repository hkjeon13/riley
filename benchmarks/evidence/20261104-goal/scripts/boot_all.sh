#!/bin/bash
# boot_all.sh: start every model with lp9 and the 1024-row burst on GPU 7 (warm-up runs a wide step), one request, stop.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until grep -q GSM_DONE logs/gsm.log; do sleep 30; done; sleep 10
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
for M in $(ls models); do
  K=$(python3 - models/$M <<'PY'
import json, os, sys
p = sys.argv[1]; c = json.load(open(f"{p}/config.json")); h = c["num_attention_heads"]
d = c.get("head_dim") or c["hidden_size"] // h
per = c["num_hidden_layers"] * c.get("num_key_value_heads", h) * d * 4
w = sum(os.path.getsize(f"{p}/{f}") for f in os.listdir(p) if f.endswith(".safetensors"))
if w > 40 * 2**30: print("SKIP"); sys.exit()
b = int(((0.9 * 39.4 * 2**30 - w - 2.5 * 2**30) if w > 20 * 2**30 else (0.85 * 39.5 * 2**30 - w - 4 * 2**30)) // (16 * per))
print("" if b >= 16384 else f"--kv-blocks {b}")
PY
)
  [ "$K" = SKIP ] && continue
  R=$(MAXOUT=64 bash serve.sh lp9 7 18340 $M m logs/boot-$M.log --max-sequence-tokens 8192 $B $K | tail -1)
  P=$(echo $R | awk '{print $2}')
  OUT=$(curl -s -m 120 127.0.0.1:18340/v1/completions -H 'Content-Type: application/json' -d '{"model":"m","prompt":"The capital of France is","max_tokens":8,"temperature":0}' | head -c 160)
  echo "$M | $R | $OUT"
  [ -n "$P" ] && kill $P 2>/dev/null; sleep 12
done
echo BOOT_DONE
