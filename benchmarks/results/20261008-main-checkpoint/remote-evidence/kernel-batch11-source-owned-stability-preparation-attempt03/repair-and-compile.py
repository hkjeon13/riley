"""Source-bound test-only build after the complete benchmark and evidence collectors."""
from pathlib import Path
import json,hashlib,subprocess,time,os
R=Path('/data/riley-serving-261007');O=R/'kernel-batch11-source-owned-stability-preparation-attempt03'
S=R/'kernel-batch11-source-attempt02';D=R/'kernel-batch11-source-owned-stability-harness-draft-attempt01'
W=R/'kernel-batch11-source-owned-stability-source-attempt01'
BASE='2fd37841e302fc2b34dc79567713dea6241b49ac'
steps=[];failure=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):
 with (O/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def live(pid,expected):
 p=Path('/proc')/str(pid)
 if not p.exists():return False
 argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return False
 assert expected in argv,'PID reused; do not infer completion'
 return True
def run(label,argv,cwd=None,env=None):
 with (O/(label+'.log')).open('x') as f:
  p=subprocess.run(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 steps.append({'label':label,'argv':argv,'cwd':str(cwd),'exit':p.returncode})
 assert p.returncode==0,label+' failed; evidence preserved'
try:
 handles=[(2368834,'kernel-batch11-controller-attempt02/run.py'),(2384242,'kernel-batch11-independent-analysis-attempt02/collect-and-verify.py'),(2384243,'kernel-batch11-independent-analysis-attempt02/host-after-replay.py')]
 while True:
  states=[{'pid':pid,'live':live(pid,argv)} for pid,argv in handles]
  (O/'verified-wait.json').write_text(json.dumps({'time_ns':time.time_ns(),'handles':states}))
  if not any(x['live'] for x in states):break
  time.sleep(30)
 receipts=[]
 for rel in ['kernel-batch11-quiet-attempt02/completion.json','kernel-batch11-independent-analysis-attempt02/completion.json','kernel-batch11-host-independent-analysis-attempt02/completion.json']:
  p=R/rel;v=json.loads(p.read_text());assert v['failure'] is None,rel
  receipts.append({'path':str(p),'sha256':sha(p)})
 replay=json.loads((R/'kernel-batch11-independent-analysis-attempt02/raw-replay.json').read_text())
 assert replay['matrix_complete'] and replay['raw_replay_passed'] and len(replay['lanes'])==96
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'foreign GPU occupied; postpone build'
 assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()==BASE
 assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain=v1'],text=True).strip(),'source dirty; preserve and stop'
 draft=D/'matched_all8_source_owned_http_gpu.rs'
 assert sha(draft)=='8532aa5bdeb737f977b0cb0619dfa67ae472ac73619c7cadca1afa39195dd7d2'
 write('prerequisite-receipts.json',receipts)
 assert W.exists() and subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()==BASE,'preserved harness state differs'
 tracked=subprocess.check_output(['git','-C',str(S),'ls-tree','-r','--name-only',BASE],text=True).splitlines()
 before={f:sha(S/f) for f in tracked}
 assert all(sha(W/f)==digest for f,digest in before.items() if f!='crates/riley-server/Cargo.toml'),'production worktree differs; staged Cargo addition verified separately'
 test=W/'crates/riley-server/tests/matched_all8_source_owned_http_gpu.rs'
 assert sha(test)==sha(draft),'preserved test changed'
 cargo=W/'crates/riley-server/Cargo.toml'
 expected=(S/'crates/riley-server/Cargo.toml').read_text()+'\n[[test]]\nname = "matched_all8_source_owned_http_gpu"\npath = "tests/matched_all8_source_owned_http_gpu.rs"\nrequired-features = ["server", "cuda"]\n'
 assert cargo.read_text()==expected,'preserved Cargo registration differs'
 assert all(sha(W/f)==digest for f,digest in before.items() if f!='crates/riley-server/Cargo.toml'),'production code changed'
 run('git-add',['git','add','crates/riley-server/Cargo.toml','crates/riley-server/tests/matched_all8_source_owned_http_gpu.rs'],W)
 run('git-commit',['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','test: capture matched serving ownership and shutdown evidence'],W)
 commit=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()
 write('source-receipt.json',{'base_commit':BASE,'commit':commit,'parent_file_sha256':before,'production_unchanged':True,'test_sha256':sha(test),'cargo_sha256':sha(cargo),'GPU_execution':'not executed'})
 env=json.loads((R/'kernel-batch11-server-validation-attempt02/preparation.json').read_text())['env']
 env['CARGO_TARGET_DIR']=str(R/'kernel-batch11-source-owned-stability-target-attempt03')
 env['CUDA_VISIBLE_DEVICES']=''
 argv=['cargo','test','--locked','--release','-p','riley-server','--features','cuda,server','--test','matched_all8_source_owned_http_gpu','--no-run','--message-format=json']
 write('build-preparation.json',{'argv':argv,'env':env,'source_commit':commit,'scope':'compile only; noGPU/test execution, no serving performance result'})
 run('compile-only',argv,W,env)
 artifacts=[]
 for line in (O/'compile-only.log').read_text().splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if x.get('reason')=='compiler-artifact' and x.get('executable') and x.get('target',{}).get('name')=='matched_all8_source_owned_http_gpu':
   artifacts.append({'path':x['executable'],'sha256':sha(x['executable'])})
 assert len(artifacts)==1,'test executable missing or ambiguous'
 write('test-executable.json',{'source_commit':commit,'artifacts':artifacts,'GPU_execution':'not executed'})
 assert all(sha(S/f)==digest for f,digest in before.items()),'original source changed'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)};raise
finally:
 write('completion.json',{'failure':failure,'steps':steps,'GPU_execution':'not executed','serving_performance':'미실행','stability_passed':False,'goal_achieved':False})
