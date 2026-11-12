#!/bin/bash
# stage3.sh: API check (default and burst) and a 30-minute mixed soak with the burst, lp8, Qwen2.5-7B, GPU 7.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-7B-Instruct; mkdir -p evidence/stage3
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
for v in default burst; do
  X=""; [ $v = burst ] && X="$B"
  P=$(MAXOUT=1024 bash serve.sh lp8 7 18337 $M m logs/s3-$v.log --max-sequence-tokens 8192 $X | tail -1 | awk '{print $2}')
  python3 src/benchmarks/live/api_check.py 127.0.0.1:18337 m --tools --out evidence/stage3/check-$v.json | grep -E "FAIL|checks passed"
  if [ $v = burst ]; then
    python3 src/benchmarks/live/soak.py 127.0.0.1:18337 m --seconds 1800 --rate 8 --pid $P --out evidence/stage3/soak-burst.json 2>&1 | tail -3
  fi
  kill $P; sleep 15
done
echo STAGE3_DONE
