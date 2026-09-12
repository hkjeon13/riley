"""Build a new immutable scratch snapshot after the previous measurement exits."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912');src=ROOT/'batch2-source';target=ROOT/'batch2-target'
def run(argv,**kwargs):return subprocess.run(argv,check=True,**kwargs)
# Do not overlap CPU build/IO with retained GPU/serving measurements.
while not (ROOT/'candidate-engine/summary.json').exists():
 if (ROOT/'candidate-engine/failure.json').exists():raise RuntimeError('prior engine sweep failed; inspect before proceeding')
 time.sleep(5)
assert json.loads((ROOT/'candidate-engine/summary.json').read_text())['completed']
run(['python3',str(ROOT/'remote_session.py'),'extend'])
run(['git','clone','--no-hardlinks','--quiet',str(ROOT/'candidate-source'),str(src)])
run(['tar','-xzf',str(ROOT/'batch2-source-overlay.tar.gz'),'-C',str(src)])
files=json.loads((ROOT/'batch2-source-overlay.json').read_text())
assert all(hashlib.sha256((src/name).read_bytes()).hexdigest()==sha for name,sha in files.items())
run(['git','add',*files],cwd=src)
run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','--quiet','-m','snapshot: bounded P128 prefill graph batch'],cwd=src)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src,text=True)
(target/'release').mkdir(parents=True)
run(['rsync','-a','--exclude=riley-cuda-*',str(ROOT/'candidate-target/release')+'/',str(target/'release')+'/'])
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(target),LD_LIBRARY_PATH='/data/riley-g04-cuda13/lib')
with (ROOT/'batch2-build.log').open('x') as log:
 for cmd in [['cargo','test','--release','-p','riley-runtime','--features','cuda','--lib','--no-run'],['cargo','build','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley','--bin','riley-profile']]:
  run(cmd,cwd=src,env=env,stdout=log,stderr=log)
receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'source_files':files,'binaries':{str(target/'release'/name):hashlib.sha256((target/'release'/name).read_bytes()).hexdigest() for name in ['riley','riley-profile']}}
(ROOT/'batch2-build.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
