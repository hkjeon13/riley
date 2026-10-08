#!/bin/bash
set -eu
TASK=/data/riley-serving-261007
PY=/data/riley-serving-260913-recovery/vllm029-venv/bin/python
test ! -e "$TASK/vllm0271-venv"
"$PY" -m venv --without-pip "$TASK/vllm0271-venv"
set +e
"$PY" -m pip --python "$TASK/vllm0271-venv/bin/python" install \
  'vllm==0.27.1' 'torch==2.13.0' 'transformers==5.15.1' \
  'flashinfer-python==0.6.16.post3' > "$TASK/vllm0271-install.log" 2>&1
result=$?
printf '%s\n' "$result" > "$TASK/vllm0271-install.exit"
if [ "$result" = 0 ]; then
  "$PY" -m pip --python "$TASK/vllm0271-venv/bin/python" freeze > "$TASK/vllm0271-packages.txt"
  "$PY" -m pip --python "$TASK/vllm0271-venv/bin/python" check > "$TASK/vllm0271-pip-check.txt" 2>&1
fi
exit "$result"
