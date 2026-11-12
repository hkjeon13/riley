#!/bin/bash
# chain6.sh: once swap7 resumed the GPU 7 runner, stop it and queue its models after the GPU 6 runner instead.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until grep -q SWAP7_DONE logs/swap7.log; do sleep 5; done
sleep 3
R=$(ps -eo pid,cmd | grep "[b]ase0.py $W h6e7a 7 " | awk '{print $1}'); [ -n "$R" ] && kill $R; sleep 1
for p in $(ps -eo pid,cmd | grep -E "[c]ompare.py .* 7 h6e7a" | awk '{print $1}'); do kill $p; done
sleep 2
for p in $(ps -eo pid,cmd | grep -E "[b]ind 127.0.0.1:18407|[p]ort 18407 " | awk '{print $1}'); do kill -TERM -- -$p 2>/dev/null || kill $p; done
rm -rf evidence/base0/Llama-3.2-1B-Instruct
echo GPU7_FREE
R6=$(ps -eo pid,cmd | grep "[b]ase0.py $W h6e7a 6 " | awk '{print $1}')
while kill -0 $R6 2>/dev/null; do sleep 30; done
python3 base0.py $W h6e7a 6 Llama-3.2-1B-Instruct,SmolLM2-360M-Instruct,Qwen2.5-1.5B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B,SmolLM2-1.7B-Instruct,Qwen2.5-Coder-3B-Instruct,Qwen2.5-Coder-7B-Instruct,Qwen3-8B,DeepSeek-R1-Distill-Llama-8B,Qwen2.5-Coder-14B-Instruct,Qwen3-14B
echo CHAIN6_DONE
