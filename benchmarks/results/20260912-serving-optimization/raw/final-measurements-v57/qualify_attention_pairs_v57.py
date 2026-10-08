from pathlib import Path
import os,subprocess,json
r=Path('/tmp/riley-opt-260912');d=r/'attention-pairs-v57'
assert len(json.loads((r/'variable-serving-screen-round62/completion.json').read_text())['records'])==32
assert (r/'native-trace-v56/analysis.json').exists()
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
def run(name,args):
 with (d/name).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
for name in ['mixed','boundary','decode']:
 run(name+'-build.log',['/data/riley-g04-cuda13/bin/nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(d/(name+'.cu')),'-o',str(d/name)])
 run(name+'-correctness.log',[str(d/name)])
 for check in (['memcheck'] if name=='boundary' else ['memcheck','racecheck']):run(name+'-'+check+'.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',check,'--error-exitcode','99',str(d/name)])
 if name!='boundary':
  run(name+'-resources.txt',['/data/cuda-12.8.1/bin/cuobjdump','--dump-resource-usage',str(d/name)])
  with (d/(name+'-timing.jsonl')).open('w') as out,(d/(name+'-timing-check.log')).open('w') as err:subprocess.run([str(d/name),'--timing'],env=env,stdout=out,stderr=err,check=True)
  print('PASS',name,'timing',flush=True)
