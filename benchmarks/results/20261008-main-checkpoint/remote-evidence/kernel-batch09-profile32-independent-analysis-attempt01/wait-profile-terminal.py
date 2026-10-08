import pathlib,json,time,hashlib,subprocess
r=pathlib.Path('/data/riley-serving-261007')
o=r/'kernel-batch09-profile32-independent-analysis-attempt01'
p=pathlib.Path('/proc/483918')
failure=None;executed=False
try:
 while p.exists():
  if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
  assert str(r/'kernel-batch09-profile32-preparation-attempt01/run.py').encode() in (p/'cmdline').read_bytes().split(b'\0'),'profile PID identity changed'
  time.sleep(30)
 terminal=json.loads((r/'kernel-batch09-cpu-capture-all8-profile-attempt01/completion.json').read_text())
 assert terminal['failure'] is None and len(terminal['completed'])==32,'profile capture failed; original evidence retained'
 pins=json.loads((o/'pipeline-source-preparation.json').read_text())['new_helper_hashes']
 for n,digest in pins.items():assert hashlib.sha256((o/n).read_bytes()).hexdigest()==digest
 argv=['sudo','-n','/data/riley-serving-261007/vllm0271-venv/bin/python',str(o/'collect-terminal32.py')]
 (o/'analysis-launch.json').write_text(json.dumps({'argv':argv,'time_ns':time.time_ns()},indent=2))
 executed=True
 with (o/'analysis-dispatch.log').open('x') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,'terminal32 analysis failed; original evidence retained'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(o/'queue-completion.json').write_text(json.dumps({'failure':failure,'executed':executed,'adopted':False,'goal_achieved':False},indent=2))
