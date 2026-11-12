#!/bin/bash
# exp_models.sh: per model on GPU 7, Riley lp7 default vs 1024-row burst on long-prompt cells, plus output parity between the two.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
SC=rag-c1,rag6k-c1,rag-c8,rag-r3
kv() { python3 - "$1" <<'PY'
import json, os, sys
path = sys.argv[1]; c = json.load(open(f"{path}/config.json")); h = c["num_attention_heads"]
d = c.get("head_dim") or c["hidden_size"] // h
per = c["num_hidden_layers"] * c.get("num_key_value_heads", h) * d * 4
w = sum(os.path.getsize(f"{path}/{f}") for f in os.listdir(path) if f.endswith(".safetensors"))
b = min(16384, int((0.85 * 39.5 * 2**30 - w - 4 * 2**30) // (16 * per)))
print("" if b >= 16384 else f"--kv-blocks {b}")
PY
}
for M in "$@"; do
  K=$(kv models/$M)
  for rep in 1 2; do
    python3 src/benchmarks/live/compare.py $W models7/$M/burst-$rep $M 7 lp7 --order r --scenarios $SC --port 18330 --riley-long "$B $K" | tail -1
    python3 src/benchmarks/live/compare.py $W models7/$M/base-$rep $M 7 lp7 --order r --scenarios $SC --port 18330 --riley-long "$K" | tail -1
  done
  for v in base burst; do
    X="$K"; [ $v = burst ] && X="$B $K"
    P=$(MAXOUT=1024 bash serve.sh lp7 7 18331 $M m logs/m6-$v.log --max-sequence-tokens 8192 $X | tail -1 | awk '{print $2}')
    python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18331 m evidence/models7/$M/parity-$v.json $( [ $v = burst ] && echo --against evidence/models7/$M/parity-base.json ) | tail -1
    kill $P; sleep 15
  done
done
echo MODELS7_DONE
