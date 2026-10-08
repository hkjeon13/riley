"""Wait for the specific live baseline, replay all raw lanes, then profile serially."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from verify_baseline import archive_files, verify

ROOT=Path('/data/riley-serving-261007')
BASE=ROOT/'diagnostic-baseline-attempt01'
HERE=Path(__file__).parent
OUT=ROOT/'baseline-to-profile-continuation01'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def main():
    OUT.mkdir()
    write('preparation.json',{'baseline_pid':3083990,'baseline':str(BASE),
        'controller_sha256':sha(__file__),'verifier_sha256':sha(HERE/'verify_baseline.py'),
        'profile_controller_sha256':sha(HERE/'run_serving_profile.py'),
        'profile_plan_sha256':sha(HERE/'profiling-plan.json'),'start_time_ns':time.time_ns(),
        'goal_achieved':False})
    deadline=time.monotonic()+7200
    failure=None
    try:
        while True:
            process=subprocess.run(['ps','-p','3083990','-o','args='],capture_output=True,text=True)
            alive=process.returncode==0 and 'run_diagnostic_baseline.py' in process.stdout
            if not alive:
                if not (BASE/'completion.json').exists():raise RuntimeError('baseline process ended without terminal receipt')
                break
            if time.monotonic()>deadline:raise RuntimeError('baseline wait deadline; live baseline left untouched')
            time.sleep(5)
        completion=json.loads((BASE/'completion.json').read_text())
        assert completion['failure'] is None and completion['all_lanes_complete'] and len(completion['records'])==64
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        archive=ROOT/'diagnostic-baseline-attempt01.tar.gz'
        with tarfile.open(archive,'x:gz') as tar:tar.add(BASE,arcname=BASE.name)
        result=verify(archive_files(archive),(ROOT/'controller/fixtures.json').read_bytes())
        assert result['matrix_complete'] and len(result['lanes'])==64
        result.update(raw_archive=str(archive),raw_archive_sha256=sha(archive),
                      preparation_sha256=sha(BASE/'preparation.json'),completion_sha256=sha(BASE/'completion.json'))
        write('baseline-independent-replay.json',result)
        print('full64lane independent raw replay passed; start profile',flush=True)
        with (OUT/'profile-controller.log').open('x') as log:
            code=subprocess.run([sys.executable,str(HERE/'run_serving_profile.py'),'--attempt','1',
                '--plan',str(HERE/'profiling-plan.json'),'--verified-baseline',str(OUT/'baseline-independent-replay.json')],
                stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError('profile controller exit '+str(code)+'; retained logs and completion')
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
        raise
    finally:write('completion.json',{'failure':failure,'end_time_ns':time.time_ns(),'goal_achieved':False})

if __name__=='__main__':main()
