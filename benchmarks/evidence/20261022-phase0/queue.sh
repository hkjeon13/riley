#!/bin/bash
# Phase 0 queue on GPU7: A/A calibration, then contract cost (fixed, natural).
P=/mnt/efs/nlp/psyche/riley-phase0-20261022
cd $P
./run-campaign.sh aa-c16-c32 16,32 10 64 192 owned-before,owned-after fixed
./run-campaign.sh cost-fixed 1,8,16,32 3 64 192 owned-before,native,vllm fixed
./run-campaign.sh cost-natural 8,32 3 64 192 owned-before,native,vllm natural
date +%s > $P/runs/queue.finished
