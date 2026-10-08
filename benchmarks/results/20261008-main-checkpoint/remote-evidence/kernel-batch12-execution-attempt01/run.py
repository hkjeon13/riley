import pathlib,json,hashlib,subprocess,time,os
r=pathlib.Path('/data/riley-serving-261007');o=r/'kernel-batch12-execution-attempt01'
s=r/'kernel-batch12-source-attempt01';mp=r/'kernel-batch12-source-receipt-attempt01.json'
n=r/'kernel-batch12-primitive-validation-attempt01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
failure=None;steps=[]
try:
 while not mp.exists():time.sleep(10)
 a=r/'kernel-batch11-profile32-independent-analysis-attempt01'
 while not (a/'queue-completion.json').exists():time.sleep(10)
 for p in [a/'completion.json',a/'queue-completion.json']:assert json.loads(p.read_text())['failure'] is None
 scope=json.loads((a/'verified-scope.json').read_text());assert scope['profiles']==32 and scope['HTTP_requests_replayed']==6144
 pins=json.loads((o/'pipeline-pins.json').read_text())
 for p,h in pins['scripts'].items():assert sha(pathlib.Path(p))==h,p
 manifest=json.loads(mp.read_text())
 assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==manifest['source_commit']
 assert manifest['changed_files']==['kernels/src/mixed_attention_v49.cuh']
 for p,h in manifest['files'].items():assert sha(s/p)==h,p
 assert not gpu(),'GPU occupied; do not overlap'
 n.mkdir()
 old=json.loads((r/'kernel-batch11-primitive-validation-attempt02/preparation.json').read_text())
 argv=[x.replace('kernel-batch11-source-attempt02','kernel-batch12-source-attempt01').replace('kernel-batch11-primitive-validation-attempt02','kernel-batch12-primitive-validation-attempt01') for x in old['argv']]
 env=os.environ.copy();env['CUDA_HOME']='/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13';env['LD_LIBRARY_PATH']=env['CUDA_HOME']+'/lib'
 (n/'preparation.json').write_text(json.dumps({'argv':argv,'source_commit':manifest['source_commit'],'source_manifest_sha256':sha(mp),'full32_verified_scope':scope,'serving_performance':'미실행'},indent=2))
 with (n/'build.log').open('x') as log:p=subprocess.run(argv,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,'native compile failed'
 binary=n/'native3161';digest=sha(binary);assert not gpu()
 (n/'native-launch.json').write_text(json.dumps({'argv':[str(binary)],'sha256':digest,'time_ns':time.time_ns()}))
 with (n/'native.log').open('x') as log:p=subprocess.run([str(binary)],cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT)
 rows=[json.loads(x) for x in (n/'native.log').read_text().splitlines() if x.startswith('{')]
 proof=next(x for x in rows if x.get('cases')==3161)
 assert p.returncode==0 and proof['passed'] and proof['BF16_and_FP32_exact'] and proof['mapped_value_cases']==1593 and proof['decode_value_cases']==360
 assert sha(binary)==digest and not gpu()
 for name,h in manifest['files'].items():assert sha(s/name)==h,name
 (n/'native-process.json').write_text(json.dumps({'exit':p.returncode,'binary_before':digest,'binary_after':sha(binary),'log_sha256':sha(n/'native.log'),'native_result':proof,'GPU_compute_after':gpu(),'source_manifest_sha256':sha(mp)},indent=2))
 (n/'completion.json').write_text(json.dumps({'failure':None,'serving_performance':'미실행','adopted':False,'goal_achieved':False}))
 steps.append({'step':'native3161','passed':True})
 (o/'progress.json').write_text(json.dumps(steps))
 with (o/'server-controller.log').open('x') as log:p=subprocess.run(['/usr/bin/python3',str(r/'run_kernel_batch12_server_attempt01.py')],stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,'server build or HTTP60 correctness failed'
 steps.append({'step':'server_HTTP60','passed':True})
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 if n.exists() and not (n/'completion.json').exists():(n/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','adopted':False,'goal_achieved':False}))
 raise
finally:(o/'completion.json').write_text(json.dumps({'failure':failure,'steps':steps,'serving_performance':'미실행','adopted':False,'goal_achieved':False},indent=2))
