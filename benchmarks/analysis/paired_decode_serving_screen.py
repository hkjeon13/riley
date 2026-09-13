"""Offline closed-loop SSE screen. No Python code runs inside Riley serving.
Run against one retained recovery root; every lane is launched and drained serially.
"""
import argparse,concurrent.futures,hashlib,json,os,pathlib,signal,socket,subprocess,time,urllib.request

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_hash(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def request(port,fixture,extra=None):
    body={'model':'g04-smol','prompt':fixture['prompt'],'max_tokens':fixture['max_tokens'],'temperature':0,'seed':0,'stream':True,'return_token_ids':True,'stream_options':{'include_usage':True}}
    body.update(extra or {})
    started=time.monotonic_ns();tokens=[];arrivals=[];texts=[];prompt=None;usage=None;finish=None;done=False;frames=[]
    try:
        req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response:
            for line in response:
                now=time.monotonic_ns()
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if data==b'[DONE]':done=True;break
                event=json.loads(data);frames.append(event)
                if event.get('usage') is not None:usage=event['usage']
                for choice in event.get('choices',[]):
                    assert choice['index']==0
                    ids=choice.get('token_ids') or []
                    assert all(isinstance(t,int) and 0<=t<49152 for t in ids)
                    tokens.extend(ids);arrivals.extend([now]*len(ids));texts.append(choice.get('text') or '')
                    if choice.get('prompt_token_ids') is not None:
                        assert prompt is None;prompt=choice['prompt_token_ids']
                    if choice.get('finish_reason') is not None:
                        assert finish is None;finish=choice['finish_reason']
        ended=time.monotonic_ns()
        assert done and finish is not None and tokens and usage is not None
        assert usage['completion_tokens']==len(tokens) and usage['prompt_tokens']==len(prompt)
        checks={'prompt':prompt==fixture['prompt_token_ids'],'tokens':tokens==fixture['token_ids'],'text':canonical(''.join(texts))==fixture['text_sha256'],'finish':finish==fixture['finish_reason']}
        return {'id':fixture['id'],'started_ns':started,'ended_ns':ended,'token_ids':tokens,'prompt_token_ids':prompt,'arrivals_ns':arrivals,'text':''.join(texts),'finish_reason':finish,'usage':usage,'checks':checks,'valid':True,'frames':frames}
    except Exception as error:
        return {'id':fixture['id'],'started_ns':started,'ended_ns':time.monotonic_ns(),'valid':False,'error':repr(error),'frames':frames}

def cancel_request(port,fixture):
    body={'model':'g04-smol','prompt':fixture['prompt'],'max_tokens':fixture['max_tokens'],'temperature':0,'stream':True,'return_token_ids':True}
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    received=0
    with urllib.request.urlopen(req,timeout=120) as response:
        for line in response:
            if not line.startswith(b'data:'):continue
            data=line[5:].strip()
            if data==b'[DONE]':break
            for choice in json.loads(data).get('choices',[]):received+=len(choice.get('token_ids') or [])
            if received>=4:break
    assert 4<=received<fixture['max_tokens']
    return {'received_tokens':received,'connection_closed_before_output_budget':True}

def phase(port,fixtures,count,concurrency):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda i:request(port,fixtures[i%len(fixtures)]),range(count)))
def percentile(values,p):
    values=sorted(values);at=(len(values)-1)*p;lo=int(at);return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(at-lo)
def summary(rows):
    valid=[r for r in rows if r['valid']];out={'requests':len(rows),'errors':len(rows)-len(valid),'reference_matches':sum(all(r['checks'].values()) for r in valid)}
    if not valid:return out
    out['throughput_tokens_s']=sum(len(r['token_ids']) for r in valid)*1e9/(max(r['ended_ns'] for r in rows)-min(r['started_ns'] for r in rows))
    for name,values in [('ttft',[(r['arrivals_ns'][0]-r['started_ns'])/1e6 for r in valid]),('tpot',[(r['arrivals_ns'][-1]-r['arrivals_ns'][0])/max(1,len(r['token_ids'])-1)/1e6 for r in valid]),('e2e',[(r['ended_ns']-r['started_ns'])/1e6 for r in valid]),('itl',[(b-a)/1e6 for r in valid for a,b in zip(r['arrivals_ns'],r['arrivals_ns'][1:])])]:
        out[name+'_ms']={str(p):percentile(values,p) for p in [.5,.95,.99]}
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=pathlib.Path);parser.add_argument('fixture',type=pathlib.Path);parser.add_argument('out',type=pathlib.Path);parser.add_argument('--warmup',type=int,default=192);parser.add_argument('--retained',type=int,default=768);parser.add_argument('--concurrency',type=int,default=32);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    spec=json.loads(args.fixture.read_text());args.out.mkdir();binary=args.root/'target/release/riley'
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'0','VLLM_BATCH_INVARIANT':'0','LD_LIBRARY_PATH':str(args.root/'toolchain130/nvidia/cu13/lib')+':/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib'})
    write(args.out/'preparation.json',{'binary_sha256':file_hash(binary),'fixture_sha256':file_hash(args.fixture),'controller_sha256':file_hash(__file__),'concurrency':args.concurrency,'warmup':args.warmup,'retained':args.retained,'smoke':args.smoke,'timing_scope':'client-observed SSE frames; no token timestamp interpolation','qualified':False,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'model_files':{str(path):file_hash(path) for path in pathlib.Path(spec['riley_argv'][spec['riley_argv'].index('--model')+1]).glob('*') if path.is_file()}})
    records=[];stop_references=None;orders=[['single','paired']] if args.smoke else [['single','paired','vllm'],['vllm','paired','single']]
    for pair,order in enumerate(orders):
        for lane in order:
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not compute,compute
            deadline=time.monotonic()+120
            while int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True))>48:
                assert time.monotonic()<deadline;time.sleep(1)
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            argv=list(spec['vllm_argv' if lane=='vllm' else 'riley_argv'])
            def option(key,value):argv[argv.index(key)+1]=str(value)
            if lane=='vllm':option('--port',port);option('--max-num-seqs',args.concurrency)
            else:
                argv[0]=str(binary);option('--bind',f'127.0.0.1:{port}');option('--max-active-sequences',args.concurrency)
                if lane=='paired':argv+=['--decode-window','paired-experimental-v1']
            lane_env=env.copy()
            if lane=='vllm':
                lane_env['LD_LIBRARY_PATH']='/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib'
                lane_env['PATH']='/data/riley-vllm-interim.CfrT9T/venv/bin:'+env.get('PATH','/usr/bin:/bin')
            name=f'c{args.concurrency}-p{pair}-{lane}';write(args.out/(name+'-launch.json'),{'argv':argv,'gpu_before':subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used','--format=csv,noheader'],text=True).strip(),'loadavg':pathlib.Path('/proc/loadavg').read_text().strip(),'env':{k:lane_env[k] for k in ['PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','VLLM_BATCH_INVARIANT']}})
            with (args.out/(name+'.log')).open('w') as log:
                process=subprocess.Popen(argv,env=lane_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    deadline=time.monotonic()+300
                    while True:
                        assert process.poll() is None,'server exited before readiness'
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as response:
                                if response.status==200:break
                        except Exception:pass
                        assert time.monotonic()<deadline,'readiness timeout';time.sleep(.5)
                    warmup=phase(port,spec['natural'],args.warmup,args.concurrency);write(args.out/(name+'-warmup.json'),warmup)
                    assert all(r['valid'] for r in warmup),'warmup protocol failure'
                    if lane!='vllm':assert all(all(r['checks'].values()) for r in warmup),'warmup reference mismatch'
                    if args.smoke:
                        stops=[]
                        for fixture in spec['natural']:
                            full=next(row for row in warmup if row['id']==fixture['id'])
                            stop=full['text'][40:60];assert len(stop)==20
                            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                                stopped=list(pool.map(lambda _:request(port,fixture,{'stop':stop}),range(args.concurrency)))
                            assert all(row['valid'] and row['finish_reason']=='stop' for row in stopped)
                            stops.extend(stopped)
                        write(args.out/(name+'-stop.json'),stops)
                        identities=[(row['token_ids'],row['text'],row['finish_reason'],row['usage']) for row in stops]
                        if stop_references is None:stop_references=identities
                        else:assert identities==stop_references,'paired stop publication differs from serial'
                        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                            cancelled=list(pool.map(lambda _:cancel_request(port,spec['natural'][-1]),range(args.concurrency)))
                        write(args.out/(name+'-cancel.json'),cancelled)
                    rows=phase(port,spec['natural'],args.retained,args.concurrency);write(args.out/(name+'-retained.json'),rows)
                    stats=summary(rows);write(args.out/(name+'-summary.json'),stats);records.append({'name':name,**stats});write(args.out/'progress.json',records)
                    assert stats['errors']==0,'retained protocol failure'
                    if lane!='vllm':assert stats['reference_matches']==args.retained,'retained reference mismatch'
                finally:
                    if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=40)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    write(args.out/(name+'-exit.json'),{'exit_code':process.returncode})
    write(args.out/'completion.json',records)
if __name__=='__main__':main()
