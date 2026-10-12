#!/bin/sh
# Final candidate (qwen-riley-tcprod2-candidate + --prefill-plateau-rows 128,192): same-run comparison with vLLM 0.27.1 (final10),
# then stability on 8 cells (attempt 26), then the HF token comparison with the final flag set (attempt 13).
cd /data/riley-serving-261007
python3 qwen_matrix_steady_final10.py 1 > qwen-matrix-steady-final10-1.log 2>&1
touch qwen-final10.finished
rm -f qwen-tc-stab.status
for cw in "1 fixed" "1 natural" "8 fixed" "8 natural" "16 fixed" "16 natural" "32 fixed" "32 natural"; do set -- $cw
  python3 qwen_sched_stability_tc_trend.py 26 $1 $2 300 > qwen-tc-stab-c$1$2.log 2>&1; echo "c$1-$2 $?" >> qwen-tc-stab.status
done
touch qwen-tc-stab.finished
python3 qwen_hf_token_ref_final.py 13 > qwen-hf-token-ref-13.log 2>&1
touch qwen-final-hf.finished
