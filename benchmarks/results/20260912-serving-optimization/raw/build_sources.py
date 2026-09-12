import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912')
BASE=Path('/tmp/riley-g04-vllm-profile-source-260911')
def command(argv,cwd=None):subprocess.run(argv,cwd=cwd,check=True)
def setup():
 src=ROOT/'baseline-test-source'
 command(['git','clone','--no-hardlinks','--quiet',str(BASE),str(src)])
 import shutil
 module=Path('crates/riley-runtime/src/llama/graph_decode_stage_parity_gpu.rs')
 shutil.copyfile(ROOT/'candidate-source'/module,src/module)
 full=src/'crates/riley-runtime/src/llama/graph_decode_full.rs'
 full.write_text('#[cfg(all(test, feature = "cuda"))]\n#[path = "graph_decode_stage_parity_gpu.rs"]\nmod stage_parity_gpu;\n'+full.read_text().replace('//! Full M=1 model capture on actual executor weights, plans, scratch and KV parents.','// Full M=1 model capture on actual executor weights, plans, scratch and KV parents.',1))
 (ROOT/'baseline-test-target/release').mkdir(parents=True)
 command(['rsync','-a','--exclude=riley-cuda-*','/tmp/riley-g04-vllm-profile-target/release/',str(ROOT/'baseline-test-target/release')+'/'])
 for name in ['baseline-test','candidate']:
  src=ROOT/(name+'-source')
  command(['git','add','crates/riley-runtime/src/llama/graph_decode_full.rs',str(module),'kernels/src/graph_resources.cu'],cwd=src)
  command(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','--quiet','-m','snapshot: '+name+' stage execution verification'],cwd=src)
def build(name):
 src=ROOT/(name+'-source');env=os.environ.copy()
 env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(ROOT/(name+'-target')),LD_LIBRARY_PATH='/data/riley-g04-cuda13/lib')
 cmds=[['cargo','test','--release','-p','riley-runtime','--features','cuda','--lib','--no-run']]
 if name=='candidate':cmds.append(['cargo','build','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley','--bin','riley-profile'])
 with (ROOT/(name+'-build.log')).open('w') as log:
  for argv in cmds:subprocess.run(argv,cwd=src,env=env,stdout=log,stderr=log,check=True)
 print(json.dumps({'built':name}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','baseline-test','candidate']);args=p.parse_args()
 if args.action=='setup':setup()
 else:build(args.action)
