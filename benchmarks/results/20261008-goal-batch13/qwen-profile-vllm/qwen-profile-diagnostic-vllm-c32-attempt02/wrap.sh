#!/bin/sh
exec /data/riley-serving-261007/nsight2025-assets/extracted/opt/nvidia/nsight-systems/2025.5.2/bin/nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --output=/data/riley-serving-261007/qwen-profile-diagnostic-vllm-c32-attempt02/trace --force-overwrite=true /data/riley-serving-261007/vllm0271-venv/bin/vllm "$@"
