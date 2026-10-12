#!/bin/sh
cd /data/riley-serving-261007
if sh ./qwen_rebuild_and_gate.sh 18; then
python3 qwen_matrix_steady_main_flags.py 1 mergedfused --projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused > qwen-steady-main-mergedfused.log 2>&1
python3 qwen_hf_token_ref_mf.py 2 > qwen-hf-token-ref-2.log 2>&1
python3 qwen_sched_stability_mf.py 6 32 natural 300 > qwen-sched-stab-mf-c32n.log 2>&1
python3 qwen_sched_stability_mf.py 6 16 fixed 300 > qwen-sched-stab-mf-c16f.log 2>&1
python3 qwen_sched_stability_mf.py 6 8 natural 300 > qwen-sched-stab-mf-c8n.log 2>&1
python3 qwen_sched_stability_mf.py 6 1 fixed 300 > qwen-sched-stab-mf-c1f.log 2>&1
fi
touch qwen-mf-adopt.finished
