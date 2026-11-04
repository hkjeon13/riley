#!/bin/bash
cd /mnt/efs/nlp/psyche/riley-gap-20261102/search
export CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=4 ./gemm_anchor_search 3584 37888 > gateup.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 ./gemm_anchor_search 18944 3584 > down.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 ./gemm_anchor_search 3584 4608 > qkv.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 ./gemm_anchor_search 3584 3584 > oproj.txt 2>&1 &
wait; echo SEARCH_DONE
