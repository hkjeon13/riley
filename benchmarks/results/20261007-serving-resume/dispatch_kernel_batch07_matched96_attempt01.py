import json,hashlib,subprocess,pathlib,time,shlex
R=pathlib.Path(__file__).resolve().parent;C=R/'kernel-batch07-controller';O=R/'kernel-batch07-matched96-local-dispatch-attempt01';O.mkdir();failure=None
try:
 proof=json.loads((R/'kernel-batch07-native-http-independent-verification.json').read_bytes());plan=json.loads((C/'batch-plan.json').read_bytes());assert proof['source_commit']==plan['candidate_source_commit'] and proof['binary_receipt']['sha256']==plan['candidate_binary_sha256']
 assert proof['native_exact_cases']==3161 and proof['HTTP_raw_requests_replayed']==60 and plan['orders']==[['v52','candidate','vllm'],['vllm','candidate','v52']]*2
 probe="import json,subprocess;from pathlib import Path;print(json.dumps({'GPU':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'destination_exists':Path('/data/riley-serving-261007/kernel-batch07-controller').exists()}))"
 state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True));assert not state['GPU'] and not state['destination_exists']
 (O/'preparation.json').write_text(json.dumps({'state':state,'source':proof['source_commit'],'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.iterdir() if p.is_file()},'created_ns':time.time_ns(),'expected_lanes':96,'repeats':4,'orders':plan['orders'],'no_exclusions':True},indent=2)+'\n')
 subprocess.run(['scp','-r',str(C),'ai-assistant:/data/riley-serving-261007/'],check=True)
 with (O/'remote-controller.log').open('x') as log:proc=subprocess.run(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant','/data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/kernel-batch07-controller/run.py --attempt 1 --mode quiet'],stdout=log,stderr=subprocess.STDOUT)
 (O/'SSH-exit.json').write_text(json.dumps({'exit':proc.returncode,'actual_terminal_inferred':False})+'\n');assert proc.returncode==0,'inspect actual controller; no restart on observation failure'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'adopted':False,'goal_achieved':False},indent=2)+'\n')
