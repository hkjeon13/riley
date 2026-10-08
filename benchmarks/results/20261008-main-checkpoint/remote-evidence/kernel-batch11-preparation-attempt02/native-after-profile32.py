import pathlib,json,hashlib,subprocess,time,os
b=pathlib.Path('/data/riley-serving-261007')
prep=b/'kernel-batch11-preparation-attempt02'
source=b/'kernel-batch11-source-attempt02'
out=b/'kernel-batch11-primitive-validation-attempt02'
manifest_path=b/'kernel-batch11-source-receipt-attempt02.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu(): return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def pinned_source():
 m=json.loads(manifest_path.read_text())
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==m['source_commit']
 for rel,digest in m['files'].items(): assert sha(source/rel)==digest
 return m
failure=None;executed=False
try:
 for pid,token in [(2241670,'kernel-batch10-profile32-preparation-attempt01/run.py'),(2243831,'kernel-batch10-profile32-independent-analysis-attempt01/wait-profile-terminal.py')]:
  p=pathlib.Path('/proc')/str(pid)
  while p.exists():
   assert any(token in x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x),'predecessor PID identity changed'
   time.sleep(30)
 a=b/'kernel-batch10-profile32-independent-analysis-attempt01'
 for p in [a/'completion.json',a/'queue-completion.json']:
  assert json.loads(p.read_text())['failure'] is None
 scope=json.loads((a/'verified-scope.json').read_text())
 assert scope['profiles']==32 and scope['HTTP_requests_replayed']==6144
 phase_path=a/'kernel-batch10-cpu-capture-all8-profile-phase-analysis-attempt01/summary.json'
 phase=json.loads(phase_path.read_text());assert phase['profiles_analyzed']==32
 av=[]
 for lane in phase['reports']:
  p=next(x for x in lane['phases'] if x['phase']=='retained')
  k=next(x for x in p['kernels_by_work'] if 'independent_values' in x['name'])
  av.append({'case':lane['case'],'events':k['events'],'kernel_work_sum_ns':k['work_sum_ns'],'sqlite_sha256':lane['sqlite_sha256']})
 assert len(av)==32
 (prep/'full32-before-native-review.json').write_text(json.dumps({'all32_AV_observations':av,'phase_summary_sha256':sha(phase_path),'verified_scope':scope,'source_archive_sha256':scope['source_archive_sha256'],'scope':'all8 measured, recorded spans diagnostic only; restored V52 AV geometry and arithmetic','adopted':False,'goal_achieved':False},indent=2))
 assert not gpu()
 m=pinned_source()
 out.mkdir()
 old=json.loads((b/'kernel-batch10-primitive-validation-attempt01/preparation.json').read_text())
 argv=[x.replace('batch10','batch11').replace('kernel-batch11-source-attempt01','kernel-batch11-source-attempt02').replace('kernel-batch11-primitive-validation-attempt01','kernel-batch11-primitive-validation-attempt02') for x in old['argv']]
 env=os.environ.copy();env['CUDA_HOME']='/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13';env['LD_LIBRARY_PATH']=env['CUDA_HOME']+'/lib'
 (out/'preparation.json').write_text(json.dumps({'source_commit':m['source_commit'],'source_manifest_sha256':sha(manifest_path),'argv':argv,'environment':{k:env[k] for k in ['CUDA_HOME','LD_LIBRARY_PATH']},'serving_performance':'미실행'},indent=2))
 executed=True
 with (out/'build.log').open('x') as log:result=subprocess.run(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,'native compilation failed'
 binary=out/'native3161';before=sha(binary)
 assert not gpu()
 (out/'native-launch.json').write_text(json.dumps({'argv':[str(binary)],'sha256':before,'time_ns':time.time_ns()}))
 with (out/'native.log').open('x') as log:result=subprocess.run([str(binary)],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
 rows=[json.loads(x) for x in (out/'native.log').read_text().splitlines() if x.startswith('{')]
 proof=next(x for x in rows if x.get('cases')==3161)
 assert result.returncode==0 and proof['passed'] and proof['BF16_and_FP32_exact']
 assert proof['decode_value_cases']==360 and proof['mapped_value_cases']==1593
 assert sha(binary)==before and not gpu()
 pinned_source()
 (out/'native-process.json').write_text(json.dumps({'exit':result.returncode,'binary_before':before,'binary_after':sha(binary),'log_sha256':sha(out/'native.log'),'GPU_compute_after':gpu(),'source_manifest_sha256':sha(manifest_path),'native_result':proof,'time_ns':time.time_ns()},indent=2))
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:
 d={'failure':failure,'executed':executed,'serving_performance':'미실행','adopted':False,'goal_achieved':False,'time_ns':time.time_ns()}
 (prep/'native-queue-completion.json').write_text(json.dumps(d,indent=2))
 if out.exists(): (out/'completion.json').write_text(json.dumps(d,indent=2))
