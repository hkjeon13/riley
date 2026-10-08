import pathlib,json,time,hashlib,subprocess,traceback
r=pathlib.Path('/data/riley-serving-261007')
a=r/'qwen-http-logit-trace-independent-analysis-attempt01'
proc=pathlib.Path('/proc/774205');failure=None;executed=False
try:
 while proc.exists():
  if (proc/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
  assert str(r/'qwen-http-logit-trace-preparation-attempt01/run-after-serving.py').encode() in (proc/'cmdline').read_bytes().split(b'\0'),'trace PID identity changed'
  time.sleep(30)
 assert json.loads((r/'qwen-http-logit-trace-preparation-attempt01/completion.json').read_text())['failure'] is None,'trace build/execution failed; preserve evidence'
 expected=json.loads((a/'preparation.json').read_text())['verifier_sha256']
 assert hashlib.sha256((a/'verify.py').read_bytes()).hexdigest()==expected
 executed=True
 argv=['/usr/bin/python3',str(a/'verify.py')]
 (a/'verification-launch.json').write_text(json.dumps({'argv':argv,'time_ns':time.time_ns()},indent=2))
 with (a/'verify.log').open('x') as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,'independent logit verification failed'
except BaseException:failure=traceback.format_exc()
finally:(a/'queue-completion.json').write_text(json.dumps({'failure':failure,'executed':executed,'goal_achieved':False},indent=2))
