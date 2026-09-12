from pathlib import Path
r=Path('/tmp/riley-opt-260912')
s=(r/'qualify_gqa_v50.py').read_text();a=s.index("p=s/'crates/riley-runtime");b=s.index('env=os.environ',a);s=s[:a]+s[b:];a=s.index("for kind in ['memcheck'");b=s.index("run('gqa-v50-build.log'",a);s=s[:a]+s[b:];s=s.replace('gqa-v50','prefill-v51');(r/'qualify_prefill_v51.py').write_text(s)
for kind in ['http','fallback']:(r/f'run_v7_{kind}_v51.py').write_text((r/f'run_v7_{kind}_v50.py').read_text().replace('v50','v51'))
s=(r/'qualify_serving_v50.py').read_text().replace('gqa-v50','prefill-v51').replace('v50','v51');(r/'qualify_serving_v51.py').write_text(s)
with (r/'qualify_prefill_v51.py').open('a') as f:f.write("\nrun('prefill-v51-serving-qualification-controller.log',['python3',str(r/'qualify_serving_v51.py')])\n")
