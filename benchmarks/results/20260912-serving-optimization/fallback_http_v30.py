import pathlib,json,os,subprocess,socket,time,urllib.request
r=pathlib.Path('/tmp/riley-opt-260912');out=r/'fallback-http-v30';out.mkdir()
launch=json.loads((r/'v3-http-v30-shared-final/process.json').read_text())['command'];corpus=json.loads((r/'variable-corpus-v11/requests.json').read_text());results={}
for lane,binary in [('previous',r/'variable-candidate-v29/riley'),('new',r/'prefill-shapes-target-v11/release/riley')]:
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 argv=launch.copy();argv[0]=str(binary);argv[argv.index('--bind')+1]=f'127.0.0.1:{port}'
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib';env['RILEY_SHUTDOWN_METRICS_PATH']=str(out/(lane+'-shutdown.json'))
 with (out/(lane+'.log')).open('w') as log:
  p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,text=True)
  try:
   deadline=time.monotonic()+120
   while True:
    assert p.poll() is None
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=1) as res:assert res.status==200
     break
    except OSError:
     assert time.monotonic()<deadline;time.sleep(.2)
   rows=[]
   for c in [x for x in corpus if x['prompt_tokens'] in (16,128,398)][:3]:
    for params in [{'temperature':0,'top_p':0.1},{'temperature':0.7,'top_p':0.9,'seed':7},{'temperature':1.2,'top_p':1.0,'seed':42}]:
     body={'model':'v3','prompt':c['prompt'],'max_tokens':32,'return_token_ids':True,**params}
     req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
     with urllib.request.urlopen(req,timeout=120) as resp:data=json.load(resp)
     rows.append({'prompt_tokens':c['prompt_tokens'],'params':params,'choices':data['choices'],'usage':data['usage']})
   results[lane]=rows
  finally:
   if p.poll() is None:p.stdin.write('\n');p.stdin.flush();p.wait(timeout=60)
   assert p.returncode==0
assert len(results['new'])==9 and results['new']==results['previous']
(out/'comparison.json').write_text(json.dumps({'exact_match':True,'responses_per_lane':9,'results':results},indent=2)+'\n');print('PASS 9 sampling requests exactly equal V29/V30')
