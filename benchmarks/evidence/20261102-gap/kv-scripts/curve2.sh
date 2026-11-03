#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
G=6 P=18281
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$P" | awk '{print $1}' | xargs -r kill; sleep 12; }
for arm in "c128 --prefill-chunk-tokens 128" "c512 --batch-token-budget 512 --prefill-chunk-tokens 512 --batch-shape-buckets 1,2,4,8,16,32,64,128,256,512"; do
  set -- $arm; B=$1; shift
  MAXOUT=2048 bash serve.sh kv1 $G $P Qwen2.5-7B-Instruct m logs/c$B.log --max-sequence-tokens 8192 "$@" | tail -1
  python3 ttft_curve.py 127.0.0.1:$P m evidence/curve/riley-$B.json
  kp
done
echo CURVE2_DONE
