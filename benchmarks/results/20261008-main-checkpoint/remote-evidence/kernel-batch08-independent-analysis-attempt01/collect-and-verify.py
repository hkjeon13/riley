import pathlib,json,hashlib,subprocess,tarfile,time,os,sys
r=pathlib.Path('/data/riley-serving-261007')
o=r/'kernel-batch08-independent-analysis-attempt01'
raw=r/'kernel-batch08-quiet-attempt01'
pid=3850243
failure=None;steps=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def live():
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return False
 state=(p/'stat').read_text().split(') ',1)[1].split()[0]
 if state=='Z':return False
 argv=(p/'cmdline').read_bytes().split(b'\0')
 assert str(r/'kernel-batch08-controller/run.py').encode() in argv,'PID identity changed; do not infer terminal'
 return True
def run(label,argv):
 with (o/(label+'.log')).open('x') as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 steps.append({'label':label,'argv':argv,'exit':p.returncode})
 assert p.returncode==0,label+' failed; preserve original evidence'
try:
 while live():
  (o/'wait-state.json').write_text(json.dumps({'pid':pid,'confirmed_live':True,'time_ns':time.time_ns(),'records_observed':len(json.loads((raw/'completion.json').read_text()).get('records',[])) if (raw/'completion.json').exists() else None}))
  time.sleep(30)
 assert (raw/'completion.json').exists(),'controller handle gone but terminal receipt missing; no restart'
 terminal=json.loads((raw/'completion.json').read_text())
 (o/'controller-terminal-observation.json').write_text(json.dumps({'pid':pid,'confirmed_live':False,'time_ns':time.time_ns(),'terminal':terminal},indent=2))
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; postpone archive to avoid impacting serving'
 before={str(p.relative_to(raw)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in raw.rglob('*') if p.is_file()}
 archive=o/(raw.name+'.tar.gz')
 with tarfile.open(archive,'w:gz') as t:t.add(raw,arcname=raw.name)
 after={str(p.relative_to(raw)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in raw.rglob('*') if p.is_file()}
 assert before==after,'raw source changed during archive'
 (o/'collection-receipt.json').write_text(json.dumps({'source':str(raw),'source_before':before,'source_after':after,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'terminal':terminal},indent=2))
 assert terminal['failure'] is None and terminal['all_lanes_complete'] is True and len(terminal['records'])==96,'full96 failed; failed archive preserved without exclusions'
 python=str(r/'vllm0271-venv/bin/python')
 run('raw-replay',[python,str(o/'verify_kernel_batch08.py'),str(archive),'--fixtures',str(r/'kernel-batch08-controller/fixtures.json'),'--output',str(o/'raw-replay.json')])
 run('launch-audit',[python,str(o/'audit_kernel_batch08_serving_contract.py'),str(archive),'--source-commit','f3389f3512c03f6185bf94a4daf140bf55a70a25','--output',str(o/'launch-audit.json')])
 run('summary',[python,str(o/'summarize_kernel_batch01.py'),str(o/'raw-replay.json'),'--output',str(o/'summary.json')])
 source=r/'kernel-batch08-source-attempt01'
 receipt=json.loads((r/'kernel-batch08-source-receipt-attempt01.json').read_text())
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==receipt['source_commit']
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'])
 assert all(sha(source/n)==h for n,h in receipt['files'].items()),'source changed'
 assert sha(r/'kernel-batch08-target-attempt01/release/riley')=='3797b7f45198a9c61383270943bc9c0c115126e701972d11e5b84238fa41b296'
 summary=json.loads((o/'summary.json').read_text())
 report=['All four repeats preserved; absolute throughput, latency tails, errors, memory and sample SD.','Archive SHA256 '+sha(archive),'All-core screen '+str(summary['all_core_minimum_mean_screen'])]
 for cell in summary['cells']:
  report.append('\n'+cell['cell'])
  for engine in ['v52','candidate','vllm']:
   report.append(engine)
   for metric,v in sorted(cell['all_available_repeat_statistics'][engine].items()):report.append(metric+' '+json.dumps(v))
  report.append('minimum_mean_screen '+str(cell['candidate_minimum_mean_screen']))
 (o/'all8-report.txt').write_text('\n'.join(report)+'\n')
 (o/'decision.json').write_text(json.dumps({'all_core_minimum_mean_screen':summary['all_core_minimum_mean_screen'],'all_core_challenge_mean_screen':summary['all_core_challenge_mean_screen'],'adopted':False,'stability':'미실행','serving_screen':'independently verified full96; stability and all-cell pass remain required','source_files_after_verified':len(receipt['files']),'goal_achieved':False},indent=2))
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:
 (o/'completion.json').write_text(json.dumps({'failure':failure,'steps':steps,'adopted':False,'goal_achieved':False},indent=2))
