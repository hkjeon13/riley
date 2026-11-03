#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=$W/models/Qwen2.5-7B-Instruct
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PATH=$W/tools/vllm-venv/bin:$PATH VLLM_ENABLE_V1_MULTIPROCESSING=0
rm -f evidence/prof/vllm.*
$NS profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/prof/vllm $W/tools/vllm-venv/bin/python vllm_prof.py $M > logs/prof-vllm.log 2>&1 < /dev/null
grep -E "^[0-9] " logs/prof-vllm.log
$NS export --type=sqlite --force-overwrite=true -o evidence/prof/vllm.sqlite evidence/prof/vllm.nsys-rep > /dev/null 2>&1; echo EXPORT $?
python3 prof_an.py evidence/prof/vllm.sqlite
echo RUN_DONE
