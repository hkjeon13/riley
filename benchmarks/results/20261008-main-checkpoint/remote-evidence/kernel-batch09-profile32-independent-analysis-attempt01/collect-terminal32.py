import os,json,hashlib,pathlib,subprocess,shutil,tarfile
R=pathlib.Path(__file__).resolve().parent;N='kernel-batch09-cpu-capture-all8-profile-attempt01';S=pathlib.Path('/data/riley-serving-261007')/N
C=R/'kernel-batch09-cpu-capture-all8-profile-independent-collection-attempt01';C.mkdir();T=R/'tmp';T.mkdir()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(n,d):(R/n).write_text(json.dumps(d,indent=2)+'\n')
def inventory():return {str(p.relative_to(S)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(S.rglob('*')) if p.is_file() and 'profiler-private' not in p.relative_to(S).parts}
failure=None;steps=[]
try:
 terminal=json.loads((S/'completion.json').read_text());assert terminal['failure'] is None and len(terminal['completed'])==32
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 profile_pid=json.loads((pathlib.Path('/data/riley-serving-261007/kernel-batch09-profile32-preparation-attempt01')/'dispatch.json').read_text())['pid']
 assert not (pathlib.Path('/proc')/str(profile_pid)).exists(),'profile launcher still exists; no archive while capture live'
 write('preparation.json',{'reason':'independent terminal32 archive, exact raw replay and CPU/CUDA analysis; no profile restart or deletion','actual_remote_terminal':terminal,'analysis_code_sha256':{p.name:sha(p) for p in R.glob('*.py')},'serving_performance':'미실행; diagnostic only'})
 before=inventory();write('source-before.json',before);archive=C/(N+'.tar.gz')
 with archive.open('xb') as out:
  tar=subprocess.Popen(['tar','--exclude=./*/profiler-private','-C',str(S),'-cf','-','.'],stdout=subprocess.PIPE)
  z=subprocess.run(['gzip','-1'],stdin=tar.stdout,stdout=out);tar.stdout.close();assert z.returncode==0 and tar.wait()==0
 D=C/N;D.mkdir()
 with tarfile.open(archive) as t:
  for m in t:
   if m.isdir():continue
   n=m.name.removeprefix('./');assert m.isfile() and n in before and not pathlib.Path(n).is_absolute() and '..' not in pathlib.Path(n).parts
   p=D/n;p.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as i,p.open('xb') as o:shutil.copyfileobj(i,o,1048576)
   assert p.stat().st_size==before[n]['bytes'] and sha(p)==before[n]['sha256']
 after=inventory();assert before==after
 receipt={'completion':terminal,'source_before':before,'source_after':after,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files_verified':len(before),'source_preserved':True,'serving_performance':'미실행; diagnostic only'}
 (C/(N+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');write('source-after.json',after)
 py='/data/riley-serving-261007/vllm0271-venv/bin/python';env={**os.environ,'TMPDIR':str(T)}
 cmds=[('raw32',[py,str(R/'verify_kernel_batch09_cpu_capture_all8_profile_attempt01.py'),str(archive),'--fixtures','/data/riley-serving-261007/kernel-batch09-controller/fixtures.json','--output',str(C/'independent32-profile-verification.json')]),('phase-intervals',[py,str(R/'analyze_kernel_batch09_cpu_capture_all8_profile_phases_attempt01.py')]),('CPU-stacks',[py,str(R/'analyze_kernel_batch09_cpu_stacks_attempt01.py')])]
 for label,cmd in cmds:
  with (R/(label+'.log')).open('x') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  steps.append({'step':label,'exit':p.returncode});write('progress.json',steps);assert p.returncode==0,label+' failed; original evidence preserved'
 proof=json.loads((C/'independent32-profile-verification.json').read_text());assert proof['complete_profiles']==32 and proof['HTTP_requests_replayed']==6144
 write('verified-scope.json',{'profiles':32,'HTTP_requests_replayed':6144,'raw_files_SHA_verified':len(before),'source_archive_sha256':receipt['archive_sha256'],'serving_performance':'미실행; diagnostic only','goal_achieved':False})
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'steps':steps,'adopted':False,'goal_achieved':False})
