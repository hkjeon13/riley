#!/bin/bash
# t17.sh GPU: SmolLM2-1.7B QKV table rows (lp17) vs lp16: output bytes (bytes.py long prompts at 8, bytes32.py 32-row decode), then speed vs vLLM (rvvr).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; M=SmolLM2-1.7B-Instruct; E=evidence/sm17; P=1844$G
KVB="$(python3 kvb.py models/$M)"
for B in lp16 lp17; do
  R=$(MAXOUT=1024 ./serve.sh $B $G $P $M m logs/t17-$B.log --max-sequence-tokens 8192 $KVB); echo "$B $R"
  python3 bytes.py 127.0.0.1:$P $E/bytes-$B.json; python3 bytes32.py 127.0.0.1:$P $E/bytes32-$B.json
  grep -h "RILEY_GEMM\|equivalent" logs/t17-$B.log | head -5
  kill ${R#READY }; R=${R#READY }; kill ${R%% *} 2>/dev/null; sleep 15
done
cmp $E/bytes-lp16.json $E/bytes-lp17.json && echo BYTES_SAME; cmp $E/bytes32-lp16.json $E/bytes32-lp17.json && echo BYTES32_SAME
python3 src/benchmarks/live/compare.py $W sm17/lp17 $M $G lp17 --order rvvr --scenarios closed-c8,closed-c16,closed-c32,reason-c8,reason-c32 --port 1845$G --riley-long "$KVB" | tail -n 1
echo T17_DONE
