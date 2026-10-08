"""Defer even source upload until the current serving/correctness archive chain ends."""
import hashlib,json,shlex,subprocess,time
from pathlib import Path

R=Path(__file__).resolve().parent
O=R/'kernel-batch03-local-dispatch-attempt01';O.mkdir()
REMOTE='/data/riley-serving-261007'
INPUTS=['kernel-batch03-source.tar','kernel-batch03-source-receipt.json',
        'run_kernel_batch03_after_archive.py','kernel_batch03_http_screen.py']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,value):(O/n).write_text(json.dumps(value,indent=2)+'\n')
pins={name:sha(R/name) for name in INPUTS}
write('preparation.json',{'pins':pins,'wait_pid':3860060,'wait_process':'archive_after_full52.py',
 'scope':'native816/build/HTTP60 only; serving benchmark unrun','source_upload':'after prior chain ends, no new upload or GPU work during the serving comparison'})
probe="""import subprocess,pathlib,json,time
p=subprocess.run(['ps','-p','3860060','-o','args='],capture_output=True,text=True)
t=pathlib.Path('/data/riley-serving-261007/terminal-evidence-archive-attempt01/completion.json')
print(json.dumps({'time_ns':time.time_ns(),'live':p.returncode==0 and 'archive_after_full52.py' in p.stdout,'terminal':t.exists()}))
"""
failure=None;steps=[]
try:
 while True:
  command=REMOTE+'/vllm0271-venv/bin/python -c '+shlex.quote(probe)
  state=json.loads(subprocess.check_output(['ssh','ai-assistant',command],text=True))
  with (O/'wait-observations.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
  if not state['live']:
   assert state['terminal'],'prior process missing without terminal receipt; inspect before any restart'
   break
  time.sleep(30)
 assert all(sha(R/n)==h for n,h in pins.items()),'local queued source changed'
 for name,dest in [('kernel-batch03-source.tar',REMOTE+'/kernel-batch03-source.tar'),
                   ('kernel-batch03-source-receipt.json',REMOTE+'/kernel-batch03-source-receipt.json'),
                   ('kernel_batch03_http_screen.py',REMOTE+'/controller/kernel_batch03_http_screen.py'),
                   ('run_kernel_batch03_after_archive.py',REMOTE+'/run_kernel_batch03_after_archive.py')]:
  subprocess.run(['scp',str(R/name),'ai-assistant:'+dest],check=True)
  steps.append({'uploaded':name,'sha256':pins[name]});write('progress.json',steps)
 print('prior chain terminal; pinned batch03 sources uploaded',flush=True)
 command=REMOTE+'/vllm0271-venv/bin/python '+REMOTE+'/run_kernel_batch03_after_archive.py'
 with (O/'remote-controller.log').open('x') as f:
  result=subprocess.run(['ssh','ai-assistant',command],stdout=f,stderr=subprocess.STDOUT)
 steps.append({'native_controller_exit':result.returncode});write('progress.json',steps)
 assert result.returncode==0,'batch03 native/build/HTTP failed; preserve remote evidence'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'steps':steps,'serving_performance':'미실행','adopted':False,'goal_achieved':False})
