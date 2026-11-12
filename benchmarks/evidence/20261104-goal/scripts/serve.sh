#!/bin/bash
# serve.sh BIN GPU PORT MODEL_DIR MODEL_ID LOG [extra flags...]: run in background, print PID when ready.
W=/mnt/efs/nlp/psyche/riley-gap-20261102
BIN=$1 GPU=$2 PORT=$3 M=$4 ID=$5 LOG=$6; shift 6
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
setsid nohup $W/bin/riley-$BIN serve --model $W/models/$M --model-id $ID --bind 127.0.0.1:$PORT --inference-profile ${PROFILE:-native-bf16-varlen-v1} --max-active-sequences 32 --max-output-tokens ${MAXOUT:-128} --max-weight-bytes 68719476736 "$@" > $LOG 2>&1 < /dev/null &
P=$!
for i in $(seq 1 900); do
  curl -sf -m 2 http://127.0.0.1:$PORT/readyz >/dev/null && { echo "READY $P after ${i}s"; exit 0; }
  kill -0 $P 2>/dev/null || { echo "DIED"; tail -5 $LOG; exit 1; }
  sleep 1
done
echo TIMEOUT; kill $P; exit 1
