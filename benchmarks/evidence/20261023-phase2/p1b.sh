#!/bin/bash
set -uo pipefail
P2=/mnt/efs/nlp/psyche/riley-phase2-20261023
W0=/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019
mkdir -p $P2/cache-tile
python3 $W0/control.py --lease-dir $W0/leases-v2 --index 7 run -- bash -c "export CUDA_DEVICE_ORDER=PCI_BUS_ID; cd $P2 && PYTHONPATH=$P2/fi-patched HOME=$P2/cache-tile XDG_CACHE_HOME=$P2/cache-tile FLASHINFER_WORKSPACE_BASE=$P2/cache-tile PATH=/home/nlp/riley-261013/vllm0271-venv/bin:/usr/local/cuda-12.4/bin:/usr/bin:/bin /home/nlp/riley-261013/vllm0271-venv/bin/python flashinfer_probe_tile.py $P2/flashinfer-probe-tile.json > $P2/flashinfer-probe-tile.log 2>&1; echo \$? > $P2/flashinfer-probe-tile.exit" < /dev/null
touch $P2/p1b.finished
