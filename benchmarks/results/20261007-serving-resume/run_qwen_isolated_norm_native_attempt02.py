import hashlib,json,os,subprocess,time
from pathlib import Path
from materialize_qwen_m2_source import materialize
R=Path('/data/riley-serving-261007');O=R/'qwen-isolated-norm-native-validation-attempt02';O.mkdir();failure=None
HF=R/'qwen-layer3-isolated-norm-hf-validation-attempt01';I=R/'qwen-layer3-isolated-norm-inputs-attempt01';S=R/'qwen-isolated-norm-source-attempt02'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
try:
 assert json.loads((HF/'controller-completion.json').read_text())['failure'] is None,'HF prerequisite incomplete'
 assert not gpu(),'GPU occupied'
 inputs=[R/n for n in ['qwen-isolated-norm-source-attempt02.tar','qwen-isolated-norm-commit-attempt02.pack','qwen-isolated-norm-source-receipt-attempt02.json']]
 pins={str(p):sha(p) for p in [*inputs,*sorted(I.iterdir()),HF/'norm-weight.bf16',HF/'result.json']}
 write('preparation.json',{'pins':pins,'gpu_before':subprocess.check_output(['nvidia-smi','-q'],text=True),'serving_performance':'미실행'})
 receipt=materialize(*inputs,S);write('source-git-receipt.json',receipt)
 env={'HOME':'/home/psyche','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PATH':'/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin','CUDA_HOME':'/data/cuda-12.8.1','CUDAToolkit_ROOT':'/data/cuda-12.8.1','CUDACXX':'/data/cuda-12.8.1/bin/nvcc','CMAKE':'/data/cmake-3.31.12/bin/cmake','CARGO_BUILD_JOBS':'1','CMAKE_BUILD_PARALLEL_LEVEL':'1','CARGO_TARGET_DIR':str(R/'qwen-isolated-norm-target-attempt02'),'RUSTUP_TOOLCHAIN':'1.85.0','LD_LIBRARY_PATH':'/data/cuda-12.8.1/lib64','CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','RILEY_QWEN_ISOLATED_NORM_INPUTS':str(I),'RILEY_QWEN_ISOLATED_NORM_HF':str(HF),'RILEY_QWEN_ISOLATED_NORM_OUTPUT':str(O)}
 build=['/home/psyche/.cargo/bin/cargo','test','--locked','-p','riley-cuda','--features','cuda','--test','primitives_gpu','--no-run','--message-format=json']
 write('build-launch.json',{'argv':build,'env':env,'source_revision':receipt['source_commit']})
 with (O/'build.log').open('x') as f:p=subprocess.run(build,cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT)
 write('build-process.json',{'exit':p.returncode,'log_sha256':sha(O/'build.log')});assert p.returncode==0,'build failed'
 executables=[]
 for line in (O/'build.log').read_text().splitlines():
  try:d=json.loads(line)
  except json.JSONDecodeError:continue
  if d.get('reason')=='compiler-artifact' and d.get('target',{}).get('name')=='primitives_gpu' and d.get('executable'):executables.append(Path(d['executable']))
 assert len(executables)==1;binary=executables[0];argv=[str(binary),'--ignored','--exact','qwen_layer3_isolated_rms_norm_captured_input_replay','--nocapture'];assert not gpu(),'GPU occupied before native run'
 write('native-launch.json',{'argv':argv,'binary_sha256':sha(binary),'source_revision':receipt['source_commit']})
 with (O/'native.log').open('x') as f:p=subprocess.run(argv,cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT)
 write('native-process.json',{'exit':p.returncode,'binary_sha256_after':sha(binary),'log_sha256':sha(O/'native.log')});assert p.returncode==0,'native isolated replay failed'
 after={path:sha(path) for path in pins};write('pins-after.json',after);assert after==pins
 files=json.loads(inputs[2].read_text())['files'];source_after={n:sha(S/n) for n in files};clean=not subprocess.check_output(['git','-C',str(S),'status','--porcelain','--untracked-files=no'],text=True);write('source-integrity-after.json',{'files':source_after,'code_worktree_clean':clean});assert source_after==files and clean
 deadline=time.monotonic()+40
 while gpu() and time.monotonic()<deadline:time.sleep(1)
 assert not gpu(),'GPU resource reclamation failed'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'serving_performance':'미실행','goal_achieved':False})
