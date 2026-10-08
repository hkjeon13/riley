import pathlib,json,subprocess,time,hashlib
r=pathlib.Path('/data/riley-serving-261007')
o=r/'kernel-batch08-profile32-independent-analysis-attempt01'
dispatch=r/'kernel-batch08-profile-dispatch-attempt01'
failure=None;executed=False
try:
 p=pathlib.Path('/proc/3894788')
 while p.exists():
  if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
  assert str(dispatch/'after-screen.py').encode() in (p/'cmdline').read_bytes().split(b'\0'),'profile dispatcher PID identity changed'
  time.sleep(30)
 terminal=json.loads((dispatch/'completion.json').read_text())
 assert terminal['failure'] is None,'profile gate/capture failed; original evidence retained'
 if not terminal['launched']:
  (o/'skipped.json').write_text(json.dumps({'reason':'profile not launched by full8 screen gate','serving_performance':'미실행 (profile analysis)','goal_achieved':False},indent=2))
 else:
  script=o/'collect-terminal32.py'
  expected=json.loads((o/'pipeline-source-preparation.json').read_text())['new_helper_hashes'][script.name]
  assert hashlib.sha256(script.read_bytes()).hexdigest()==expected,'analysis script changed'
  argv=['sudo','-n',str(r/'vllm0271-venv/bin/python'),str(script)]
  (o/'analysis-launch.json').write_text(json.dumps({'argv':argv,'time_ns':time.time_ns()},indent=2))
  executed=True
  with (o/'analysis-dispatch.log').open('x') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
  assert result.returncode==0,'terminal32 independent analysis failed; preserve logs'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(o/'queue-completion.json').write_text(json.dumps({'failure':failure,'executed':executed,'adopted':False,'goal_achieved':False},indent=2))
