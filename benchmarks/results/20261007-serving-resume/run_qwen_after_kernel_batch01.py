"""Wait for the current matched benchmark wrapper, then run Qwen correctness alone."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path('/data/riley-serving-261007');O=R/'qwen-m1-regular-continuation01';O.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
pins={str(R/n):sha(R/n) for n in ['qwen-m1-regular-candidate-source.tar','qwen-m1-regular-source-provenance.json','run_qwen_m1_regular.py','qwen-m1-regular-candidate-source-receipt.json']}
write('preparation.json',{'pins':pins,'wait_pid':3605799,'required_native_commit':'9eeeeae90cb92eda6dbb3a291dfceda8100a5889','serving_performance':'미실행'})
failure=None
try:
 deadline=time.monotonic()+7200
 while True:
  p=subprocess.run(['ps','-p','3605799','-o','args='],capture_output=True,text=True)
  if not(p.returncode==0 and 'kernel-batch01-controller/run.py' in p.stdout):break
  if time.monotonic()>deadline:raise RuntimeError('live serving job wait deadline; left untouched')
  time.sleep(5)
 quiet=R/'kernel-batch01-quiet-attempt01/completion.json';assert quiet.exists(),'no quiet terminal receipt'
 receipts={'quiet':json.loads(quiet.read_text())}
 if receipts['quiet']['failure']:
  diagnostic=R/'kernel-batch01-diagnostic-attempt01/completion.json';assert diagnostic.exists(),'no diagnostic terminal receipt'
  receipts['diagnostic']=json.loads(diagnostic.read_text())
 write('serving-terminal-snapshot.json',receipts)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 assert all(sha(p)==h for p,h in pins.items()),'queued inputs changed'
 source=R/'qwen-m1-regular-source';source.mkdir()
 subprocess.run(['tar','-xf',str(R/'qwen-m1-regular-candidate-source.tar'),'-C',str(source)],check=True)
 receipt=json.loads((R/'qwen-m1-regular-candidate-source-receipt.json').read_text())
 for f,h in receipt['files'].items():assert sha(source/f)==h,'native source differs '+f
 assert receipt['commit']=='9eeeeae90cb92eda6dbb3a291dfceda8100a5889'
 provenance=json.loads((R/'qwen-m1-regular-source-provenance.json').read_text())
 for item in provenance['sources'].values():assert sha(source/item['path'])==item['sha256'],'reference source differs'
 with (O/'qwen-controller.log').open('x') as log:
  p=subprocess.run([str(R/'vllm0271-venv/bin/python'),str(R/'run_qwen_m1_regular.py')],stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,'Qwen native/controller failed; preserved logs'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'serving_performance':'미실행','goal_achieved':False})
