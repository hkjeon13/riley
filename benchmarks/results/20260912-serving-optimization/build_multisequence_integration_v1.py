"""Compile the isolated strided native API and Rust owner; no GPU tests."""
from pathlib import Path
import os,json,subprocess,hashlib
ROOT=Path('/tmp/riley-opt-260912');source=ROOT/'multisequence-integration-source-v1';target=ROOT/'multisequence-integration-target-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
def main():
 log=ROOT/'multisequence-integration-build-v1.log';receipt=ROOT/'multisequence-integration-build-v1.json'
 assert not log.exists() and not receipt.exists() and not target.exists()
 assert not git('status','--porcelain','--untracked-files=all');commit=git('rev-parse','HEAD')
 files={n:sha(source/n) for n in git('ls-files').splitlines() if (source/n).is_file()}
 (target/'release').mkdir(parents=True)
 subprocess.run(['rsync','-a','--exclude=riley-cuda-*','--exclude=libriley_cuda-*',str(ROOT/'fusion-history-target-v1/release')+'/',str(target/'release')+'/'],check=True)
 fixed={'CUDA_HOME':'/data/riley-g04-cuda13','CUDAToolkit_ROOT':'/data/riley-g04-cuda13','CMAKE':'/data/cmake-3.31.12/bin/cmake','CMAKE_BUILD_PARALLEL_LEVEL':'4','CARGO_BUILD_JOBS':'4','CARGO_TARGET_DIR':str(target),'LD_LIBRARY_PATH':'/data/riley-g04-cuda13/lib'}
 env={**os.environ,**fixed};env['PATH']='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH']
 argv=['cargo','test','--release','-p','riley-cuda','--features','cuda','--lib','--no-run']
 with log.open('x') as out:subprocess.run(argv,cwd=source,env=env,stdout=out,stderr=out,check=True)
 assert not git('status','--porcelain','--untracked-files=all') and git('rev-parse','HEAD')==commit
 assert all(sha(source/n)==h for n,h in files.items())
 result={'source_root':str(source),'source_commit':commit,'source_files':files,'argv':argv,'environment':fixed,'log_sha256':sha(log),'builder_sha256':sha(__file__),'gpu_tests_executed':False,'performance_measured':False}
 with receipt.open('x') as out:json.dump(result,out,indent=2)
 print(json.dumps({'built':True,'source_commit':commit,'receipt_sha256':sha(receipt),'gpu_tests_executed':False}))
if __name__=='__main__':main()
