from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
p=(r/'decode-mask-v55/variant2.cuh').read_text().replace('riley_decode_mask_v55_2','riley_gqa50_attention')
(s/'kernels/src/decode_gqa_attention_v50.cuh').write_text(p)
for n in ['qualify_attention_v54.py','qualify_serving_v54.py','run_v7_http_v54.py','run_v7_fallback_v54.py']:
 (r/n.replace('v54','v55')).write_text((r/n).read_text().replace('v54','v55'))
