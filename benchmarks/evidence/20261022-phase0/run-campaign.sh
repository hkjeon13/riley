#!/bin/bash
# Usage: run-campaign.sh LABEL CONCURRENCY REPEATS WARM KEEP MODES WORKLOAD
# Runs one Phase 0 campaign on GPU7 through the persistent-holder controller.
set -u
P=/mnt/efs/nlp/psyche/riley-phase0-20261022
W0=/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019
V15C=/mnt/efs/nlp/psyche/riley-prefill-host-v15c-20261021/riley-prefill-host-v15c
LABEL=$1; CONC=$2; REPEATS=$3; WARM=$4; KEEP=$5; MODES=$6; WORKLOAD=$7
STRICT=$(cat $P/strict-extra.json); NATIVE=$(cat $P/native-extra.json)
cd $P
python3 $W0/control.py --lease-dir $W0/leases-v2 --index 7 run -- \
  python3 $P/measure-phase0.py --out $P/runs/$LABEL --binary $W0/riley-baseline \
  --baseline-server $V15C --candidate-binary $V15C --native-binary $V15C \
  --native-reference $W0/runs/qwen7-bootstrap/native-reference.json \
  --model /mnt/efs/nlp/psyche/riley-261013/models/Qwen3-4B \
  --reference $W0/qwen4b-serving-HF-GPU7-clean \
  --gpu GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0 --workload $WORKLOAD \
  --concurrency $CONC --repeats $REPEATS --keep $KEEP --warm $WARM --modes $MODES \
  --batch-policy bounded-projections-sm80-v2 --candidate-mode native-prefill-sm80-v5 \
  --candidate-extra "$STRICT" --baseline-extra "$STRICT" --native-extra "$NATIVE" \
  > $P/runs/$LABEL.log 2>&1 < /dev/null
echo $? > $P/runs/$LABEL.exit
