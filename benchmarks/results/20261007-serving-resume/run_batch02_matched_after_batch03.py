"""Serial candidate HTTP comparison; no overlap, restart, or pressure exemption."""
import hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/data/riley-serving-261007')
O=R/'kernel-batch02-matched-queue-attempt01';O.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def live():
    p=subprocess.run(['ps','-p','183749','-o','args='],capture_output=True,text=True)
    return p.returncode==0 and 'python run.py --attempt 1 --mode quiet' in p.stdout
pins={str(f):sha(f) for f in (R/'kernel-batch02-controller').iterdir() if f.is_file()}
write('preparation.json',{'time_ns':time.time_ns(),'pins':pins,'wait_pid':183749,'mode':'quiet','scope':'separate full96 candidate performance attempt after prior controller terminal; never pool candidates'})
failure=None
try:
    assert live(),'prior controller no longer live at queue start; inspect authoritative terminal before fresh dispatch'
    deadline=time.monotonic()+14400
    while live():
        write('wait-state.json',{'time_ns':time.time_ns(),'prior_controller_live':True,'compute_pids':gpu()})
        if time.monotonic()>deadline:raise RuntimeError('prior controller still live at queue deadline; no new benchmark started')
        time.sleep(5)
    terminal=R/'kernel-batch03-quiet-attempt01/completion.json'
    assert terminal.exists(),'prior controller missing without terminal receipt; do not restart or launch'
    write('previous-terminal.json',json.loads(terminal.read_text()))
    assert not gpu(),'GPU allocation remains; no new benchmark started'
    assert all(sha(f)==h for f,h in pins.items()),'queued controller inputs changed'
    write('launch.json',{'time_ns':time.time_ns(),'argv':[str(R/'vllm0271-venv/bin/python'),'run.py','--attempt','1','--mode','quiet'],'compute_pids':gpu()})
    with (O/'controller.log').open('x') as log:
        process=subprocess.Popen([str(R/'vllm0271-venv/bin/python'),'run.py','--attempt','1','--mode','quiet'],cwd=R/'kernel-batch02-controller',stdout=log,stderr=subprocess.STDOUT)
        write('process.json',{'pid':process.pid,'started_ns':time.time_ns()})
        result=process.wait()
    write('process-completion.json',{'exit':result,'log_sha256':sha(O/'controller.log'),'compute_pids':gpu()})
    assert result==0,'whole candidate attempt failed; all raw rows retained'
except BaseException as e:
    failure={'type':type(e).__name__,'message':str(e)};raise
finally:
    write('completion.json',{'failure':failure,'goal_achieved':False,'adopted':False,'performance_qualified':False})
