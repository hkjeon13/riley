"""Serial build and correctness of separate prefill candidate after Qwen terminal."""
import hashlib,json,subprocess,time,os
from pathlib import Path
R=Path('/data/riley-serving-261007');O=R/'kernel-batch02-validation-attempt01';O.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
pins={str(R/n):sha(R/n) for n in ['kernel-batch02-source.tar','kernel-batch02-source-receipt.json','controller/kernel_batch02_http_screen.py']}
steps=[];failure=None
write('preparation.json',{'pins':pins,'wait_pid':3630166,'scope':'source-bound native and HTTP correctness only; no serving timing claim'})
def run(label,argv,env,cwd):
 before={'time_ns':time.time_ns(),'compute_pids':gpu(),'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']}}
 assert not gpu(),'GPU overlap before '+label
 with (O/(label+'.log')).open('x') as log:p=subprocess.run(argv,env=env,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
 steps.append({'label':label,'argv':argv,'exit':p.returncode,'before':before,'ended_ns':time.time_ns(),'log_sha256':sha(O/(label+'.log'))})
 write('progress.json',steps);print(label+' exit'+str(p.returncode),flush=True)
 assert p.returncode==0,label+' failed'
 deadline=time.monotonic()+40
 while gpu() and time.monotonic()<deadline:time.sleep(1)
 assert not gpu(),'GPU not reclaimed after '+label
try:
 deadline=time.monotonic()+10800
 while True:
  p=subprocess.run(['ps','-p','3630166','-o','args='],capture_output=True,text=True)
  if not(p.returncode==0 and 'run_qwen_after_kernel_batch01.py' in p.stdout):break
  if time.monotonic()>deadline:raise RuntimeError('Qwen/serving still live; wait deadline, left untouched')
  time.sleep(5)
 terminal=R/'qwen-m1-regular-continuation01/completion.json';assert terminal.exists(),'Qwen terminal receipt missing'
 write('qwen-terminal-snapshot.json',json.loads(terminal.read_text()))
 assert all(sha(p)==h for p,h in pins.items()),'queued source changed'
 receipt=json.loads((R/'kernel-batch02-source-receipt.json').read_text());assert sha(R/'kernel-batch02-source.tar')==receipt['archive_sha256']
 source=R/'kernel-batch02-source';source.mkdir()
 subprocess.run(['tar','-xf',str(R/'kernel-batch02-source.tar'),'-C',str(source)],check=True)
 for f,h in receipt['files'].items():assert sha(source/f)==h,'source file differs '+f
 tool=Path('/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13')
 env={'HOME':'/home/psyche','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PATH':'/home/psyche/.cargo/bin:'+str(tool/'bin')+':/usr/bin:/bin','CUDA_HOME':str(tool),'CUDAToolkit_ROOT':str(tool),'CMAKE':'/data/cmake-3.31.12/bin/cmake','CMAKE_BUILD_PARALLEL_LEVEL':'4','CARGO_BUILD_JOBS':'4','CARGO_TARGET_DIR':str(R/'kernel-batch02-target'),'LD_LIBRARY_PATH':str(tool/'lib'),'CUDA_VISIBLE_DEVICES':'0'}
 write('toolchain.json',{n:subprocess.check_output(argv,env=env,cwd=source,text=True,stderr=subprocess.STDOUT) for n,argv in {'rustc':['rustc','--version'],'cargo':['cargo','--version'],'nvcc':[str(tool/'bin/nvcc'),'--version'],'cmake':[env['CMAKE'],'--version']}.items()})
 primitive=O/'primitive-check'
 run('primitive-build',[str(tool/'bin/nvcc'),'-O3','-DNDEBUG','-std=c++17','-arch=sm_89','--use_fast_math','-I'+str(source/'kernels/src'),str(source/'kernels/tests/kernel_batch02_correctness.cu'),'-o',str(primitive)],env,source)
 run('primitive-correctness',[str(primitive)],env,source)
 result=json.loads((O/'primitive-correctness.log').read_text().splitlines()[-1]);assert result['passed'] is True and result['cases']==198,'incorrect probe coverage'
 run('server-build',['cargo','build','--locked','--release','-p','riley-server','--features','cuda,server'],env,source)
 binary=R/'kernel-batch02-target/release/riley';write('binary-receipt.json',{'source_commit':receipt['source_commit'],'sha256':sha(binary),'primitive_sha256':sha(primitive)})
 run('http-correctness',[str(R/'vllm0271-venv/bin/python'),str(R/'controller/kernel_batch02_http_screen.py')],env,source)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'steps':steps,'serving_performance':'미실행','adopted':False,'goal_achieved':False})
