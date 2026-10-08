from pathlib import Path
import json,statistics
r=Path('/tmp/riley-opt-260912');d=r/'packed-value-v54'
rows=[json.loads(x) for x in (d/'decode-timing.jsonl').read_text().splitlines()];assert len(rows)==160
out=[]
for ctx in [128,398,1024,4096]:
 for count in [1,4,8,16,32]:
  m=[statistics.median(x['us'] for x in rows if x['context']==ctx and x['rows']==count and x['variant']==v) for v in [0,1]]
  out.append({'context':ctx,'rows':count,'baseline_us':m[0],'packed_us':m[1],'change_percent':100*(m[1]/m[0]-1)})
(d/'decode-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
for name in ['serving_screen_round60.py','analyze_serving_round60.py','export_round60.py']:
 p=r/name;t=p.read_text().replace('v53','v54').replace('V53 column attention','V54 packed V').replace('V53 new','V54 new').replace('column parallel attention','packed V writes and vector reads').replace('both compact','all compact');p.write_text(t)
p=(r/'freeze_attention_v52.py').read_text().replace('v52','v54')
p=p.replace("paths=['kernels/src/mixed_attention_v49.cuh']","paths=['crates/riley-cuda/build.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','kernels/src/decode_gqa_attention_v50.cuh','kernels/src/decode_shared32_model.cuh','kernels/src/mixed_attention_v49.cuh','kernels/src/mixed_model_v49.cuh','kernels/src/mixed_rope_v49.cuh','kernels/src/packed_value_v54.cuh']")
p=p.replace("assert set(subprocess.check_output(['git','diff','--name-only'],cwd=s,text=True).splitlines())==set(paths)","assert {line[3:] for line in subprocess.check_output(['git','status','--porcelain'],cwd=s,text=True).splitlines()}==set(paths)")
p=p.replace('Reduce mixed attention shared state and reuse softmax storage','Pack V7 value cache for paired attention loads across mixed and decode')
p=p.replace('V7 mixed rows with compact shared attention state, fused prefill and grouped decode','V7 packed V direct writes and paired reads, compact shared attention state, fused prefill and grouped decode')
needle="subprocess.run(['git','diff','--check'],cwd=s,check=True)"
extra="""for name in ['memcheck','boundary-memcheck','writer-memcheck','decode-memcheck']:
 assert 'ERROR SUMMARY: 0 errors' in (r/'packed-value-v54'/f'{name}.log').read_text()
for name in ['racecheck','writer-racecheck','decode-racecheck']:
 assert '0 errors, 0 warnings' in (r/'packed-value-v54'/f'{name}.log').read_text()
assert 'writer_cases=84' in (r/'packed-value-v54/writer-correctness.log').read_text()
assert 'cases=168' in (r/'packed-value-v54/decode-correctness.log').read_text()
"""
p=p.replace(needle,extra+needle);(r/'freeze_attention_v54.py').write_text(p)
p=(r/'export_integration_v52.py').read_text().replace('v52','v54').replace('round59','round60');(r/'export_integration_v54.py').write_text(p)
