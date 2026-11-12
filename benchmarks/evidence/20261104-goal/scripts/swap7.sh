#!/bin/bash
# swap7.sh: after Qwen3-4B's baseline finishes on GPU 7, pause the GPU 7 runner, run the logprob A/B, resume.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until [ -f evidence/base0/Qwen3-4B/r2-rag6k-c1.json ]; do sleep 5; done
R=$(ps -eo pid,cmd | grep "[b]ase0.py $W h6e7a 7 " | awk '{print $1}')
kill $R; sleep 1
for p in $(ps -eo pid,cmd | grep -E "[c]ompare.py .* 7 h6e7a" | awk '{print $1}'); do kill $p; done
sleep 2
for p in $(ps -eo pid,cmd | grep -E "[b]ind 127.0.0.1:18407|[p]ort 18407 " | awk '{print $1}'); do kill -TERM -- -$p 2>/dev/null || kill $p; done
sleep 20; nvidia-smi --query-gpu=index,memory.used --format=csv | sed -n 8,9p
rm -rf evidence/base0/Llama-3.2-1B-Instruct
bash lp_ab.sh 7 Qwen2.5-7B-Instruct
setsid nohup python3 base0.py $W h6e7a 7 Llama-3.2-1B-Instruct,SmolLM2-360M-Instruct,Qwen2.5-1.5B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B,SmolLM2-1.7B-Instruct,Qwen2.5-Coder-3B-Instruct,Qwen2.5-Coder-7B-Instruct,Qwen3-8B,DeepSeek-R1-Distill-Llama-8B,Qwen2.5-Coder-14B-Instruct,Qwen3-14B > logs/base0-g7b.log 2>&1 < /dev/null &
echo SWAP7_DONE
