import pathlib,json,subprocess,hashlib,time,tarfile
b=pathlib.Path('/data/riley-serving-261007');o=b/'kernel-batch11-preparation-attempt01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
failure=None;executed=False
try:
 p=pathlib.Path('/proc/2296022')
 while p.exists():
  assert str(o/'native-after-profile32.py').encode() in (p/'cmdline').read_bytes().split(b'\0'),'native PID identity changed'
  time.sleep(30)
 n=b/'kernel-batch11-primitive-validation-attempt01'
 assert json.loads((n/'completion.json').read_text())['failure'] is None
 manifest=json.loads((b/'kernel-batch11-source-receipt-attempt01.json').read_text())
 src=b/'kernel-batch11-source-attempt01'
 assert subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()==manifest['source_commit']
 for rel,digest in manifest['files'].items():
  assert sha(src/rel)==digest
  assert hashlib.sha256(subprocess.check_output(['git','-C',str(src),'show','HEAD:'+rel])).hexdigest()==digest
 launch=json.loads((n/'native-launch.json').read_text());process=json.loads((n/'native-process.json').read_text())
 assert process['exit']==0 and sha(pathlib.Path(launch['argv'][0]))==launch['sha256']==process['binary_before']==process['binary_after']
 assert process['log_sha256']==sha(n/'native.log')
 results=[json.loads(x) for x in (n/'native.log').read_text().splitlines() if x.startswith('{')]
 result=next(x for x in results if x.get('cases')==3161)
 assert result['passed'] and result['BF16_and_FP32_exact'] and result['decode_value_cases']==360 and result['mapped_value_cases']==1593
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 proof={'source_commit':manifest['source_commit'],'native_cases':3161,'native_binary_sha256':launch['sha256'],'source_files_git_verified':len(manifest['files']),'native_log_sha256':sha(n/'native.log'),'BF16_and_FP32_exact':True,'serving_performance':'미실행','goal_achieved':False}
 (o/'independent-native-proof.json').write_text(json.dumps(proof,indent=2))
 before={str(p.relative_to(n)):sha(p) for p in n.rglob('*') if p.is_file()}
 archive=o/'native-evidence.tar.gz'
 with archive.open('xb') as f:
  with tarfile.open(fileobj=f,mode='w:gz') as t:t.add(n,arcname=n.name)
 assert before=={str(p.relative_to(n)):sha(p) for p in n.rglob('*') if p.is_file()}
 (o/'native-evidence-receipt.json').write_text(json.dumps({'archive_sha256':sha(archive),'source_before':before,'source_after':before,'originals_preserved':True},indent=2))
 pins=json.loads((o/'server-pipeline-preparation.json').read_text())['new_script_hashes']
 for path,digest in pins.items():assert sha(pathlib.Path(path))==digest
 executed=True
 argv=['/usr/bin/python3',str(b/'run_kernel_batch11_server_attempt01.py')]
 (o/'server-execution-launch.json').write_text(json.dumps({'argv':argv,'time_ns':time.time_ns()},indent=2))
 with (o/'server-controller.log').open('x') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,'server build/HTTP screen failed; originals preserved'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(o/'server-queue-completion.json').write_text(json.dumps({'failure':failure,'executed':executed,'adopted':False,'goal_achieved':False},indent=2))
