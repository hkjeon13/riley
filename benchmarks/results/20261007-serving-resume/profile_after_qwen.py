"""Serialize the CUDA13 profiler after the specific ongoing Qwen diagnostic."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/riley-serving-261007')
HERE=Path(__file__).parent
OUT=ROOT/'qwen-to-profile-continuation01'
QWEN=ROOT/'qwen-m1-layer14-qkav-attempt01'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir();failure=None
    pins={str(p):digest(p) for p in [HERE/'run_serving_profile.py',HERE/'profiling-plan.json']}
    (OUT/'preparation.json').write_text(json.dumps({'qwen_pid':3400885,'pins':pins,
        'controller_sha256':digest(__file__),'scope':'independent correctness task followed by diagnostic profiling; no promotion'},indent=2)+'\n')
    try:
        deadline=time.monotonic()+7200
        while True:
            proc=subprocess.run(['ps','-p','3400885','-o','args='],capture_output=True,text=True)
            if not (proc.returncode==0 and 'run_qwen_m1_layer14_qkav.py' in proc.stdout):break
            if time.monotonic()>deadline:raise RuntimeError('Qwen live wait deadline; existing job left untouched')
            time.sleep(5)
        assert (QWEN/'completion.json').exists(),'Qwen exited without terminal receipt'
        # A numerical diagnostic failure is preserved and does not change SmolLM2's independent work.
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU still owned; profile not launched'
        assert all(digest(p)==h for p,h in pins.items()),'queued profile source changed'
        (OUT/'qwen-completion-snapshot.json').write_bytes((QWEN/'completion.json').read_bytes())
        print('Qwen terminal and GPU empty; starting supported-stack profiling attempt04',flush=True)
        with (OUT/'profile-controller.log').open('x') as log:
            code=subprocess.run([sys.executable,str(HERE/'run_serving_profile.py'),'--attempt','4',
                '--plan',str(HERE/'profiling-plan.json'),'--verified-baseline',
                str(ROOT/'baseline-to-profile-continuation01/baseline-independent-replay.json')],stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError('profile attempt04 failed; raw artifacts retained')
    except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
    finally:(OUT/'completion.json').write_text(json.dumps({'failure':failure,'goal_achieved':False},indent=2)+'\n')

if __name__=='__main__':main()
