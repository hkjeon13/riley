from pathlib import Path
import json,subprocess,shutil,sys
r=Path('/data/riley-serving-260913-recovery');out=r/'vllm029-install-v1';out.mkdir();envdir=r/'vllm029-venv'
assert not envdir.exists(),'preserve existing environment'
assert shutil.disk_usage(r).free>25*1024**3,'need 25 GiB free for isolated dependencies'
python='/data/riley-vllm-interim.CfrT9T/venv/bin/python'
commands=[('venv',[python,'-m','venv',str(envdir)]),('install',[str(envdir/'bin/python'),'-m','pip','install','--only-binary=:all:','vllm==0.29.0'])]
results=[]
for name,argv in commands:
 with (out/(name+'.log')).open('w') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 results.append({'name':name,'exit_code':p.returncode});(out/'execution.json').write_text(json.dumps(results));print(results[-1],flush=True)
 if p.returncode:sys.exit(p.returncode)
with (out/'packages.txt').open('w') as f:subprocess.run([str(envdir/'bin/python'),'-m','pip','freeze'],stdout=f,check=True)
print('isolated environment installed; GPU readiness is not yet verified',flush=True)
