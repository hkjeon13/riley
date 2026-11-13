#!/bin/bash
# run17.sh: gemm_anchor_search on the SmolLM2-1.7B shapes (GPU 7).
cd /mnt/efs/nlp/psyche/riley-gap-20261102/search; mkdir -p ../evidence/search17
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
for s in "2048 6144 qkv" "2048 2048 o" "2048 16384 gateup" "8192 2048 down" "2048 49152 lmhead"; do set -- $s; echo "== $3 K=$1 N=$2"; ./gemm_anchor_search_v2 $1 $2 12 > ../evidence/search17/$3.txt 2>&1; echo rc $?; done
echo SEARCH17_DONE
