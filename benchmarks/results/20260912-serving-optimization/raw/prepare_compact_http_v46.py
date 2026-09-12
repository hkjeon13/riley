from pathlib import Path
r=Path('/tmp/riley-opt-260912')
s=(r/'run_v4_http_v44_c32.py').read_text().replace("out=r/'v4-http-v44-c32-final'","out=r/'v4-http-v46-c32-final'").replace("'--sampling-backend','cpu'","'--sampling-backend','gpu-greedy'")
(r/'run_v4_http_v46_c32.py').write_text(s)
s=(r/'run_compact_owned_v46.py').read_text();s=s[:s.index("args=['cargo'")]+'''jobs=[('memcheck',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'loaded_compact','--ignored','--test-threads=1','--nocapture']),('http',['python3',str(r/'run_v4_http_v46_c32.py')])]
for name,args in jobs:
 with (r/f'compact-v46-{name}-integration.log').open('w') as log:subprocess.run(args,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
'''
(r/'run_compact_integration_v46.py').write_text(s)
