from pathlib import Path
r=Path('/tmp/riley-opt-260912')
for name in ['serving_screen_round60.py','analyze_serving_round60.py','export_round60.py']:
 s=(r/name).read_text().replace('round60','round62').replace('v54','v56').replace('V54 packed V','V56 FFN fusion with token-major V').replace('V54 new','V56 new').replace('packed V writes and vector reads','decode FFN and projection normalization fusion').replace('serving-round62-execution-receipt.json','serving-round62-controller.log')
 (r/name.replace('round60','round62')).write_text(s)
s=(r/'freeze_attention_v55.py').read_text().replace('v55','v56').replace('attention-v56','ffn-v56')
a=s.index("assert 'cases=2688'");b=s.index("subprocess.run(['git','diff','--check']",a)
s=s[:a]+'''for directory,cases in [('decode-gate-layout-v56',1632),('merge-norm-v56',816)]:
 assert f'cases={cases}' in (r/directory/'correctness.log').read_text()
 assert 'ERROR SUMMARY: 0 errors' in (r/directory/'memcheck.log').read_text()
 assert '0 errors, 0 warnings' in (r/directory/'racecheck.log').read_text()
'''+s[b:]
s=s.replace("paths=['kernels/src/decode_gqa_attention_v50.cuh']","paths=['kernels/src/decode_gate_v56.cuh','kernels/src/decode_merge_norm_v56.cuh','kernels/src/decode_shared32_model.cuh','crates/riley-cuda/build.rs','crates/riley-runtime/src/llama/graph_decode_full.rs']")
s=s.replace('Separate full K16 decode value tiles from masked partial tails','Fuse decode gate rows and projection residual normalization')
a=s.index("'sampling_backend':");b=s.index('};(out/',a)
s=s[:a]+"'sampling_backend':'gpu-greedy; V7 token-major V, fused gate/up row warps and projection merge residual normalization; full-logit fallback supported'"+s[b:]
(r/'freeze_ffn_v56.py').write_text(s)
s=(r/'export_integration_v55.py').read_text().replace('v55','v56').replace('attention-v56','ffn-v56').replace('qualify_attention_v56','qualify_ffn_v56').replace('freeze_attention_v56','freeze_ffn_v56').replace('round61','round62')
(r/'export_integration_v56.py').write_text(s)
