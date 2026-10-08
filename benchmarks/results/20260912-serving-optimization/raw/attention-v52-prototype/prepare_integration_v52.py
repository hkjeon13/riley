from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'attention-state-v52'
assert 'ERROR SUMMARY: 0 errors' in (d/'boundary-memcheck.log').read_text()
assert '0 errors, 0 warnings' in (d/'racecheck.log').read_text()
s=(d/'variant3.cuh').read_text().replace('namespace riley_attention_v52_3','namespace riley_mixed_attention')
(r/'prefill-shapes-source-v11/kernels/src/mixed_attention_v49.cuh').write_text(s)
s=(r/'qualify_prefill_v51.py').read_text().replace('prefill-v51','attention-v52').replace('v51','v52');(r/'qualify_attention_v52.py').write_text(s)
for kind in ['http','fallback']:(r/f'run_v7_{kind}_v52.py').write_text((r/f'run_v7_{kind}_v51.py').read_text().replace('v51','v52'))
s=(r/'qualify_serving_v51.py').read_text().replace('prefill-v51','attention-v52').replace('v51','v52');(r/'qualify_serving_v52.py').write_text(s)
