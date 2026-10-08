import pathlib,json,hashlib,subprocess,time,re
b=pathlib.Path('/data/riley-serving-261007');o=b/'kernel-batch11-independent-analysis-attempt02';prep=b/'kernel-batch11-preparation-attempt02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
failure=None;executed=False
try:
 dispatch=prep/'matched96-dispatch.json'
 while not dispatch.exists():
  p=pathlib.Path('/proc/2368653')
  assert p.exists(),'serving producer gone before dispatch; inspect terminal failure, do not restart'
  assert str(prep/'matched96-after-HTTP-verifier-repair01.py').encode() in (p/'cmdline').read_bytes().split(b'\0'),'serving producer PID changed'
  time.sleep(30)
 d=json.loads(dispatch.read_text());planpath=b/'kernel-batch11-controller-attempt02/batch-plan.json';plan=json.loads(planpath.read_text())
 assert sha(planpath)==d['plan_sha256']
 assert plan['candidate_source_commit']=='2fd37841e302fc2b34dc79567713dea6241b49ac'
 digest=plan['candidate_binary_sha256'];assert re.fullmatch('[0-9a-f]{64}',digest)
 assert sha(b/'kernel-batch11-target-attempt02/release/riley')==digest
 assert sha(b/'kernel-batch11-controller-attempt02/kernel-batch11-native-http-independent-verification-attempt02.json')==plan['independent_correctness_sha256']==d['independent_correctness_sha256']
 templatepins=json.loads((o/'template-source-preparation.json').read_text())['templates']
 hashes={}
 for name,pin in templatepins.items():
  p=o/'templates'/name;assert sha(p)==pin
  s=p.read_text().replace('BATCH11_EXPECTED_BINARY_SHA256',digest);compile(s,str(o/name),'exec');assert not (o/name).exists();(o/name).write_text(s);hashes[name]=sha(o/name)
 (o/'pipeline-source-preparation.json').write_text(json.dumps({'new_helper_hashes':hashes,'candidate_binary_sha256':digest,'serving_dispatch':d,'candidate_source_commit':plan['candidate_source_commit'],'change_scope':'Batch11 paths/commit, actual frozen binary SHA and dispatch-bound controller/collector PID only; verification and statistical method unchanged'},indent=2))
 argv=[str(b/'vllm0271-venv/bin/python'),str(o/'collect-and-verify.py')]
 with (o/'collector-dispatch.log').open('x') as log:p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (o/'collector-dispatch.json').write_text(json.dumps({'pid':p.pid,'argv':argv,'time_ns':time.time_ns()},indent=2))
 argv=[str(b/'vllm0271-venv/bin/python'),str(o/'host-after-replay.py')]
 with (o/'host-dispatch.log').open('x') as log:h=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (o/'host-dispatch.json').write_text(json.dumps({'pid':h.pid,'argv':argv,'time_ns':time.time_ns()},indent=2))
 executed=True
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(o/'dispatch-completion.json').write_text(json.dumps({'failure':failure,'executed':executed,'adopted':False,'goal_achieved':False},indent=2))
