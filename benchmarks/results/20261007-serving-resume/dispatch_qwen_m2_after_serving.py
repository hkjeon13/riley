"""Wait for actual serving-controller terminal before any source upload."""
import hashlib,json,shlex,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'qwen-m2-local-dispatch-attempt01';O.mkdir()
REMOTE='/data/riley-serving-261007'
FILES=['qwen-m2-bounded-source-attempt01.tar','qwen-m2-bounded-commit-attempt01.pack',
       'qwen-m2-bounded-source-receipt-attempt01.json','materialize_qwen_m2_source.py','run_qwen_m2_bounded_attempt01.py']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
pins={name:sha(R/name) for name in FILES}
write('preparation.json',{'pins':pins,'wait_pid':270594,'wait_process':'run_batch02_matched_after_batch03.py',
    'scope':'M1/M2 exact104 correctness, no serving or full128 generation claim','source_upload':'only after both actual benchmark terminals and GPU empty'})
probe="""import json,subprocess,time
from pathlib import Path
p=subprocess.run(['ps','-p','270594','-o','args='],text=True,capture_output=True)
g=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
root=Path('/data/riley-serving-261007')
print(json.dumps({'time_ns':time.time_ns(),'live':p.returncode==0 and 'run_batch02_matched_after_batch03.py' in p.stdout,
 'terminal_batch03':(root/'kernel-batch03-quiet-attempt01/completion.json').exists(),
 'terminal_batch02_queue':(root/'kernel-batch02-matched-queue-attempt01/completion.json').exists(),'compute_pids':g}))
"""
failure=None;steps=[]
try:
    deadline=time.monotonic()+14400
    while True:
        state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True))
        with (O/'wait-observations.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        write('wait-state.json',state)
        if not state['live']:
            assert state['terminal_batch03'] and state['terminal_batch02_queue'],'prior process missing without both terminals; no restart'
            assert not state['compute_pids'],'GPU occupied after terminal; leave all processes untouched'
            break
        if time.monotonic()>deadline:raise RuntimeError('actual serving process remains live at dispatch deadline; no source uploaded')
        time.sleep(30)
    assert all(sha(R/name)==digest for name,digest in pins.items()),'queued local inputs changed'
    for name in FILES:
        subprocess.run(['scp',str(R/name),'ai-assistant:'+REMOTE+'/'+name],check=True)
        steps.append({'uploaded':name,'sha256':pins[name]});write('progress.json',steps)
    with (O/'remote-controller.log').open('x') as log:
        result=subprocess.run(['ssh','ai-assistant',REMOTE+'/vllm0271-venv/bin/python '+REMOTE+'/run_qwen_m2_bounded_attempt01.py'],stdout=log,stderr=subprocess.STDOUT)
    steps.append({'native_controller_exit':result.returncode});write('progress.json',steps)
    assert result.returncode==0,'remote exact104 job failed; all source/log/raw preserved'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'steps':steps,'serving_performance':'미실행','goal_achieved':False})
