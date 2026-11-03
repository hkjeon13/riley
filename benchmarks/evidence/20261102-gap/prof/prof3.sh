#!/bin/bash
# prof3.sh LABEL GPU PORT READYPATH -- server command
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; mkdir -p evidence/prof tmp-prof
L=$1 G=$2 P=$3 RP=$4; shift 5
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PATH=$W/tools/vllm-venv/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
setsid nohup $NS profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o evidence/prof/$L "$@" > logs/prof-$L.log 2>&1 < /dev/null &
N=$!
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:$P$RP >/dev/null && break; sleep 1; done
echo READY after $i
python3 prof_load.py 127.0.0.1:$P m 10
ps -eo pid,cmd | grep "[b]ind 127.0.0.1:$P\|[-]-port $P" | grep -v "nsys profile" | awk '{print $1}' | head -1 | xargs -r kill -INT
for i in $(seq 1 1500); do kill -0 $N 2>/dev/null || break; sleep 1; done
ls -la evidence/prof | grep $L
$NS export --type=sqlite --force-overwrite=true -o evidence/prof/$L.sqlite evidence/prof/$L.nsys-rep > /dev/null 2>&1; echo EXPORT $?
echo PROF_DONE $L
