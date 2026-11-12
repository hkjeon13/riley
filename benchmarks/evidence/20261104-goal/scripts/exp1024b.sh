#!/bin/bash
# exp1024b.sh: Riley-only A/B on GPU 7, prompt steps of 2048 vs the default 256, plus output parity.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-7B-Instruct
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024"
kp() { ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$1" | awk "{print \$1}" | xargs -r kill; sleep 10; }
SC=rag-c1,rag6k-c1,rag-c8,rag-r3,reason-c8
for rep in 1 2; do
  python3 src/benchmarks/live/compare.py $W exp1024b/b1024-$rep $M 7 lp4 --order r --scenarios $SC --port 18330 --riley-long "$B" | tail -1
  python3 src/benchmarks/live/compare.py $W exp1024b/b256-$rep $M 7 lp4 --order r --scenarios $SC --port 18330 | tail -1
done
MAXOUT=1024 bash serve.sh lp4 7 18331 $M m logs/e2.log --max-sequence-tokens 8192 $B | tail -1
python3 src/benchmarks/live/structured_parity.py 127.0.0.1:18331 m evidence/exp1024b/parity-b1024.json --against evidence/lm/parity-lm1.json | tail -1
python3 src/benchmarks/live/logprob_parity.py 127.0.0.1:18331 m evidence/exp1024b/lp-b1024.json --against evidence/lp/$M/parity-h6e7a.json | tail -1
kp 18331
MAXOUT=2048 bash serve.sh lp4 7 18331 $M m logs/e3.log --max-sequence-tokens 8192 $B | tail -1
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 tools/vllm-venv/bin/python src/benchmarks/live/long_check.py 127.0.0.1:18331 m models/$M evidence/exp1024b/long-b1024.json 2>&1 | tail -1 | cut -c1-300
kp 18331
echo EXP_DONE
