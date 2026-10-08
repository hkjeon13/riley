import pathlib,json,hashlib,subprocess,time
r=pathlib.Path('/data/riley-serving-261007')
code=r/'kernel-batch12-independent-analysis-attempt02'
out=r/'kernel-batch12-host-independent-analysis-attempt02'
out.mkdir()
wait_pid=json.loads((code/'collector-dispatch.json').read_text())['pid']
failure=None
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
scripts=['analyze_batch04_phase_host_samples.py','analyze_serving_host_samples.py','analyze_serving_host_samples_v2.py','summarize_baseline.py']
pins={n:sha(code/n) for n in scripts}
def live(pid,expected):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return False
 if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return False
 assert str(expected).encode() in (p/'cmdline').read_bytes().split(b'\0'),'PID identity changed'
 return True
(out/'preparation.json').write_text(json.dumps({'wait_pid':wait_pid,'wait_script':str(code/'collect-and-verify.py'),'source_pins':pins,'scope':'all source observations and conservative phase-bound samples; no sample exclusions or causal claims'},indent=2))
try:
 while live(wait_pid,code/'collect-and-verify.py'):time.sleep(30)
 terminal=json.loads((code/'completion.json').read_text())
 assert terminal['failure'] is None and all(x['exit']==0 for x in terminal['steps'])
 replay=json.loads((code/'raw-replay.json').read_text())
 assert replay['matrix_complete'] and replay['raw_replay_passed'] and replay['source_commits']['candidate']=='da5ec52aefbcb4216d087ef2053974a720b54601'
 receipt=json.loads((code/'collection-receipt.json').read_text())
 archive=code/'kernel-batch12-quiet-attempt02.tar.gz'
 assert sha(archive)==receipt['archive_sha256']
 assert all(sha(code/n)==h for n,h in pins.items())
 for script,name in [('analyze_serving_host_samples_v2.py','all-source-host-observations.json'),('analyze_batch04_phase_host_samples.py','phase-bounded-host-observations.json')]:
  argv=[str(r/'vllm0271-venv/bin/python'),str(code/script),str(archive),'--output',str(out/name)]
  with (out/(script+'.log')).open('x') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True)
 result=json.loads((out/'phase-bounded-host-observations.json').read_text())
 assert result['lanes_with_actual_host_samples']==96 and len(result['reports'])==192
 (out/'archive-binding.json').write_text(json.dumps({'archive_sha256':receipt['archive_sha256'],'source_pins':pins,'lanes':96,'phases':192,'exclusions':[],'goal_achieved':False},indent=2))
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(out/'completion.json').write_text(json.dumps({'failure':failure,'adopted':False,'goal_achieved':False},indent=2))
