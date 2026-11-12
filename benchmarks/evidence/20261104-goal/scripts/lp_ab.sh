#!/bin/bash
# lp_ab.sh GPU MODEL: logprob byte parity and speed, h6e7a vs lp1 (order r-new-new-r is not needed for parity)
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; M=$2; O=evidence/lp/$M; mkdir -p $O
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk "{print \$1}" | xargs -r kill; sleep 10; }
L="python3 src/benchmarks/live/load.py 127.0.0.1:18320 m --prompt-words 30 --max-tokens 64 --requests 192"
for b in h6e7a lp1 lp1 h6e7a; do
  n=$((${n:-0}+1))
  bash serve.sh $b $G 18320 $M m logs/lp-$b.log | tail -1
  [ $n -le 2 ] && python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18320 m $O/parity-$b.json ${AG:+--against $AG} | tail -1
  AG=$O/parity-h6e7a.json
  $L --concurrency 8 --out $O/$b-$n-plain-c8.json > /dev/null
  $L --concurrency 8 --extra '{"logprobs":true,"top_logprobs":5}' --out $O/$b-$n-logprobs-c8.json > /dev/null
  $L --concurrency 32 --extra '{"logprobs":true,"top_logprobs":5}' --out $O/$b-$n-logprobs-c32.json > /dev/null
  kp 18320
done
echo LP_DONE
