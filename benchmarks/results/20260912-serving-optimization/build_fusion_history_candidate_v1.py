"""Build a separate F108 memory-access candidate; accepted Batch7 is untouched."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912')
SOURCE=ROOT/'fusion-history-source-v1'
TARGET=ROOT/'fusion-history-target-v1'
GRAPH='kernels/src/graph_numerics.cu'
CANDIDATE_SHA='1bbce29025fc91758f932a4bdbbf36e6e9097805df9c58157a90b18d7000ea9f'
BASE_SHA='629418e19f129436d9ee2d2a753a61317c5b8527e3f4fe438b9f9aa59a8392c6'
BASE_COMMIT='8329c1aeec6e013f581128888c536e15f8bf7300'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def run(argv,**kwargs):return subprocess.run(argv,check=True,**kwargs)

def gate():
 p=ROOT/'paired-rope-attention-round15/completion.json';d=json.loads(p.read_text());assert d['completed'] and len(d['processes'])==8
 f=Path(d['finalization']['path']);assert sha(f)==d['finalization']['sha256'];final=json.loads(f.read_text());assert final['failure'] is None
 ref=d['restoration'];assert ref==final['restoration'] and sha(ref['path'])==ref['sha256']
 restored=json.loads(Path(ref['path']).read_text());assert restored['alive_and_listening'] and restored['commands_and_gui_environment_match'] and restored['all_relaunched_processes_have_pinned_vendor_maps'] and len(restored['processes'])==3
 for x in d['processes']:
  cp=Path(x['completion']['path']);assert sha(cp)==x['completion']['sha256'];c=json.loads(cp.read_text())['cleanup'];assert c['cleanup_verified'] and not c['remaining_owned_pids']
 return {'completion_sha256':sha(p),'finalization_sha256':sha(f),'restoration':ref}

def main():
 closed=gate();base=ROOT/'batch8-source';build_path=ROOT/'batch8-build.json';assert sha(build_path)==BASE_SHA
 build=json.loads(build_path.read_text());assert build['source_commit']==BASE_COMMIT and git(base,'rev-parse','HEAD')==BASE_COMMIT
 assert not git(base,'status','--porcelain','--untracked-files=all')
 for name,h in build['source_files'].items():assert sha(base/name)==h,name
 for name,h in build['binaries'].items():assert sha(name)==h,name
 candidate=ROOT/'fusion-history-candidate-v1/graph_numerics.cu';assert sha(candidate)==CANDIDATE_SHA
 logpath=ROOT/'fusion-history-build-v1.log';receipt_path=ROOT/'fusion-history-build-v1.json'
 assert not any(p.exists() for p in (SOURCE,TARGET,logpath,receipt_path))
 run(['git','clone','--no-hardlinks','--quiet',str(base),str(SOURCE)])
 shutil.copyfile(candidate,SOURCE/GRAPH)
 assert git(SOURCE,'diff','--name-only')==GRAPH
 run(['git','add','--',GRAPH],cwd=SOURCE)
 run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','--quiet','-m','snapshot: specialize fused historical KV loads and packed reads'],cwd=SOURCE)
 assert not git(SOURCE,'status','--porcelain','--untracked-files=all')
 (TARGET/'release').mkdir(parents=True)
 run(['rsync','-a','--exclude=riley-cuda-*','--exclude=libriley_cuda-*',str(ROOT/'batch8-target/release')+'/',str(TARGET/'release')+'/'])
 env=dict(os.environ);fixed=dict(build['build_environment']);fixed['CARGO_TARGET_DIR']=str(TARGET);env.update(fixed)
 env['PATH']='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH']
 commands=[['cargo','test','--release','-p','riley-runtime','--features','cuda','--lib','--no-run'],['cargo','build','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley','--bin','riley-profile']]
 with logpath.open('x') as log:
  for cmd in commands:run(cmd,cwd=SOURCE,env=env,stdout=log,stderr=log)
 source_files=dict(build['source_files']);source_files[GRAPH]=CANDIDATE_SHA
 for name,h in source_files.items():assert sha(SOURCE/name)==h,name
 for name,h in build['source_files'].items():assert sha(base/name)==h,name
 for name,h in build['binaries'].items():assert sha(name)==h,name
 assert not git(SOURCE,'status','--porcelain','--untracked-files=all') and gate()==closed
 result={'source_commit':git(SOURCE,'rev-parse','HEAD'),'source_root':str(SOURCE),'parent_source_commit':BASE_COMMIT,'parent_build_sha256':BASE_SHA,'source_files':source_files,'changed_files':[GRAPH],'binaries':{str(TARGET/'release'/n):sha(TARGET/'release'/n) for n in ('riley','riley-profile')},'build_argv':commands,'build_environment':fixed,'build_log_sha256':sha(logpath),'builder_sha256':sha(__file__),'prior_diagnostic_finalization':closed,'candidate_id':'fusion-history-v1','capability_contract':'unchanged F108; new implementation is identified by source and binary hashes','gpu_tests_executed':False,'performance_measured':False}
 with receipt_path.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
