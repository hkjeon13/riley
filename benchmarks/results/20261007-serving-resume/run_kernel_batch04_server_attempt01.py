import json,hashlib,subprocess,time
from pathlib import Path
R=Path('/data/riley-serving-261007');O=R/'kernel-batch04-server-validation-attempt01';O.mkdir();failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 assert json.loads((R/'kernel-batch04-primitive-validation-attempt01/completion.json').read_text())['failure'] is None
 source=R/'kernel-batch04-source-attempt01';receipt=json.loads((R/'kernel-batch04-source-receipt-attempt01.json').read_text());assert all(sha(source/n)==h for n,h in receipt['files'].items())
 tool=Path('/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13');env={'HOME':'/home/psyche','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PATH':'/home/psyche/.cargo/bin:'+str(tool/'bin')+':/usr/bin:/bin','CUDA_HOME':str(tool),'CUDAToolkit_ROOT':str(tool),'CMAKE':'/data/cmake-3.31.12/bin/cmake','CMAKE_BUILD_PARALLEL_LEVEL':'4','CARGO_BUILD_JOBS':'4','CARGO_TARGET_DIR':str(R/'kernel-batch04-target-attempt01'),'LD_LIBRARY_PATH':str(tool/'lib'),'CUDA_VISIBLE_DEVICES':'0'}
 argv=['cargo','build','--locked','--release','-p','riley-server','--features','cuda,server'];(O/'preparation.json').write_text(json.dumps({'source_commit':receipt['source_commit'],'argv':argv,'env':env,'scope':'native server build plus HTTP60 exact-token correctness; no serving timing claim'},indent=2)+'\n')
 with (O/'build.log').open('x') as f:p=subprocess.run(argv,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'server build failed; original source/log preserved'
 binary=R/'kernel-batch04-target-attempt01/release/riley';h=sha(binary);(O/'binary-receipt.json').write_text(json.dumps({'source_commit':receipt['source_commit'],'sha256':h},indent=2)+'\n')
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; HTTP correctness not started'
 with (O/'http-screen.log').open('x') as f:p=subprocess.run([str(R/'vllm0271-venv/bin/python'),str(R/'controller/kernel_batch04_http_screen_attempt01.py')],env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'HTTP correctness failed; no serving promotion'
 assert sha(binary)==h and all(sha(source/n)==h for n,h in receipt['files'].items()),'binary/source changed during HTTP screen'
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'])
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','adopted':False,'goal_achieved':False},indent=2)+'\n')
