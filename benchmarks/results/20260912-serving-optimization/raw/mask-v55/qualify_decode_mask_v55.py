from pathlib import Path
import os,subprocess,json
r=Path('/tmp/riley-opt-260912');d=r/'decode-mask-v55'
assert len(list((r/'variable-serving-screen-round60').glob('*-summary.json')))==32
assert json.loads((r/'serving-round60-analysis.json').read_text())['requests']==12288
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
def run(name,args):
 with (d/name).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
run('build.log',['/data/riley-g04-cuda13/bin/nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(d/'probe.cu'),'-o',str(d/'probe')])
run('correctness.log',[str(d/'probe')])
for name in ['memcheck','racecheck']:
 run(name+'.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',str(d/'probe')])
with (d/'timing.jsonl').open('w') as out,(d/'timing-check.log').open('w') as err:subprocess.run([str(d/'probe'),'--timing'],env=env,stdout=out,stderr=err,check=True)
print('PASS timing',flush=True)
