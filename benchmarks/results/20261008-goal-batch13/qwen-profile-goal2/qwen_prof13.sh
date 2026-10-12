#!/bin/sh
cd /data/riley-serving-261007
python3 qwen_matrix_prof13c32.py 1 > qwen-matrix-prof13c32.log 2>&1
sleep 20
/data/riley-serving-261007/nsight2025-assets/extracted/opt/nvidia/nsight-systems/2025.5.2/bin/nsys export --type=sqlite --output=qwen-profile-diagnostic-c32-attempt13/trace.sqlite qwen-profile-diagnostic-c32-attempt13/trace.nsys-rep > qwen-profile-diagnostic-c32-attempt13/export.log 2>&1
python3 qwen_matrix_prof14c16f.py 1 > qwen-matrix-prof14c16f.log 2>&1
sleep 20
/data/riley-serving-261007/nsight2025-assets/extracted/opt/nvidia/nsight-systems/2025.5.2/bin/nsys export --type=sqlite --output=qwen-profile-diagnostic-c16f-attempt14/trace.sqlite qwen-profile-diagnostic-c16f-attempt14/trace.nsys-rep > qwen-profile-diagnostic-c16f-attempt14/export.log 2>&1
touch qwen-prof13.finished
