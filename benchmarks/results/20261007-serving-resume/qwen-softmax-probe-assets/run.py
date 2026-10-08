"""Compile/run the preserved native arithmetic only after current profiling exits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/riley-serving-261007');HERE=Path(__file__).parent
OUT=ROOT/'qwen-softmax-probe-attempt01'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def host():return {'time_ns':time.time_ns(),'compute_pids':gpu(),
    'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']}}

def main():
    OUT.mkdir();plan=json.loads((HERE/'plan.json').read_text());provenance=json.loads((HERE/'provenance.json').read_text())
    pins={str(p):sha(p) for p in HERE.iterdir() if p.is_file()}
    write('preparation.json',{'plan':plan,'provenance':provenance,'pins':pins,'profile_pid':3443585,
          'host':host(),'serving_performance':'미실행'})
    steps=[];failure=None
    try:
        deadline=time.monotonic()+7200
        while True:
            p=subprocess.run(['ps','-p','3443585','-o','args='],capture_output=True,text=True)
            if not(p.returncode==0 and 'run_serving_profile.py' in p.stdout):break
            if time.monotonic()>deadline:raise RuntimeError('profiling wait deadline; live job left untouched')
            time.sleep(5)
        require=ROOT/'profiling-attempt04/completion.json';assert require.exists(),'profile exited without terminal receipt'
        (OUT/'profile-completion-snapshot.json').write_bytes(require.read_bytes())
        assert not gpu(),'GPU still occupied; probe not launched'
        assert all(sha(p)==h for p,h in pins.items()),'probe inputs changed while queued'
        assert all(sha(HERE/p)==h for p,h in provenance['files'].items()),'source/input provenance differs'
        for variant in plan['variants']:
            assert not gpu(),'GPU overlap'
            tool=Path(variant['toolchain']);library=tool/'lib64' if (tool/'lib64').is_dir() else tool/'lib'
            env={'PATH':str(tool/'bin')+':/usr/bin:/bin','HOME':'/home/psyche',
                 'LANG':'C.UTF-8','CUDA_VISIBLE_DEVICES':'0','LD_LIBRARY_PATH':str(library)}
            label=variant['id'];binary=OUT/label;output=OUT/(label+'.bf16')
            argv=[str(tool/'bin/nvcc'),*variant['flags'],str(HERE/'probe.cu'),'-o',str(binary)]
            with (OUT/(label+'-build.log')).open('x') as log:
                build=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert build.returncode==0,'compile failed: '+label
            before=host();run=subprocess.run([str(binary),str(HERE/'attention_scores.bf16'),
                str(HERE/'attention_probabilities.bf16'),str(output)],env=env,capture_output=True,text=True)
            (OUT/(label+'-stdout.log')).write_text(run.stdout);(OUT/(label+'-stderr.log')).write_text(run.stderr)
            assert run.returncode==0,'native execution failed: '+label
            result=json.loads(run.stdout);assert result['elements']==plan['required_elements']
            if label=='native-cu128':assert result['unequal']==plan['required_baseline_unequal'],'native baseline did not reproduce182 mismatches'
            assert output.stat().st_size==65568
            assert (sha(output)==sha(HERE/'attention_probabilities.bf16')) is (result['unequal']==0),'native equality inconsistent with raw bytes'
            steps.append({'variant':label,'compile_argv':argv,'nvcc_sha256':sha(tool/'bin/nvcc'),
                          'binary_sha256':sha(binary),'output_sha256':sha(output),'result':result,'before':before,'after':host()})
            write('progress.json',steps);print(label+' unequal '+str(result['unequal']),flush=True)
            deadline=time.monotonic()+40
            while gpu() and time.monotonic()<deadline:time.sleep(1)
            assert not gpu(),'GPU resource reclamation failed'
    except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
    finally:write('completion.json',{'steps':steps,'failure':failure,'serving_performance':'미실행',
        'goal_achieved':False,'performance_claim_eligible':False,'full_M1_eligibility':False})

if __name__=='__main__':main()
