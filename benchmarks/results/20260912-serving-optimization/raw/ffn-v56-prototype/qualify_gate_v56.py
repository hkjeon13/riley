from pathlib import Path
import subprocess,os,json
r=Path('/tmp/riley-opt-260912');d=r/'decode-gate-v56'
assert len(json.loads((r/'variable-serving-screen-round61/completion.json').read_text())['records'])==40
profile=json.loads((r/'native-trace-v55/analysis.json').read_text())
assert any('shared32_gate_up_swiglu' in k['kernel'] for k in profile['cases']['32']['stages']['decode']['kernels'])
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
def run(name,args):
 with (d/name).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
run('build.log',['/data/riley-g04-cuda13/bin/nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(d/'probe.cu'),'-o',str(d/'probe')])
run('resources.txt',['/data/cuda-12.8.1/bin/cuobjdump','--dump-resource-usage',str(d/'probe')])
run('correctness.log',[str(d/'probe')])
for name in ['memcheck','racecheck']:run(name+'.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',str(d/'probe')])
with (d/'timing.jsonl').open('w') as out,(d/'timing-check.log').open('w') as err:subprocess.run([str(d/'probe'),'--timing'],env=env,stdout=out,stderr=err,check=True)
print('PASS timing',flush=True)
