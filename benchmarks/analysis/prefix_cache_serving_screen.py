"""Offline closed-loop cache screen; no Python runs in the Riley serving path."""
import argparse,contextlib,hashlib,json,os,pathlib,signal,socket,subprocess,time,urllib.request
from paired_decode_serving_screen import phase,summary,file_hash,write,canonical

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=pathlib.Path);parser.add_argument('out',type=pathlib.Path)
    parser.add_argument('--concurrency',type=int,default=32);parser.add_argument('--warmup',type=int,default=64);parser.add_argument('--retained',type=int,default=256)
    args=parser.parse_args();assert args.concurrency in (8,16,32) and args.warmup>=32 and args.retained>=128
    args.out.mkdir();spec=json.loads((args.root/'workload.json').read_text());binary=args.root/'target/release/riley';prior=args.root/'riley-before-prefix-cache-v1'
    hashes={'current':file_hash(binary),'prior':file_hash(prior)}
    base=max(spec['natural'],key=lambda x:len(x['prompt_token_ids']))['prompt']
    def fixture(kind,index):
        nonce=hashlib.sha256(f'{kind}-{index}'.encode()).hexdigest()
        tail='\nAdditional request details: '+(' '.join(f'item{index}_{j}' for j in range(20)))+'. Continue the explanation.'
        prompt=base+tail if kind=='shared' else nonce+' '+base+tail
        return {'id':f'{kind}-{index}','prompt':prompt,'max_tokens':32,'prompt_token_ids':[],'token_ids':[],'text_sha256':canonical(''),'finish_reason':None}
    corpora={'shared':[fixture('shared',i) for i in range(32)],'unique':[fixture('unique',i) for i in range(args.warmup+args.retained)]}
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',VLLM_BATCH_INVARIANT='0')
    env.pop('RILEY_PREFIX_CACHE_PAGES',None);env.pop('RILEY_SERVING_PHASE_TIMING',None)
    env['LD_LIBRARY_PATH']=str(args.root/'toolchain130/nvidia/cu13/lib')+':/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib'
    def gpu(fields,kind='gpu'):return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    @contextlib.contextmanager
    def server(lane,name):
        assert not gpu('pid','compute-apps'),'foreign GPU compute process'
        deadline=time.monotonic()+120
        while int(gpu('temperature.gpu'))>48:
            assert time.monotonic()<deadline,'cooldown timeout';time.sleep(1)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        argv=list(spec['vllm_argv' if lane=='vllm' else 'riley_argv']);child=env.copy()
        def option(key,value):argv[argv.index(key)+1]=str(value)
        if lane=='vllm':
            option('--port',port);option('--max-num-seqs',args.concurrency)
            argv.remove('--no-enable-prefix-caching');argv+=['--enable-prefix-caching','--block-size','16','--kv-cache-memory-bytes',str(30*3*64*2*2*16*2048)]
            child['LD_LIBRARY_PATH']='/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib'
            child['PATH']='/data/riley-vllm-interim.CfrT9T/venv/bin:'+child.get('PATH','/usr/bin:/bin')
        else:
            argv[0]=str(prior if lane=='prior' else binary);assert file_hash(argv[0])==hashes['prior' if lane=='prior' else 'current']
            option('--bind',f'127.0.0.1:{port}');option('--max-active-sequences',args.concurrency);option('--kv-blocks',2048)
            child['RILEY_PREFIX_CACHE_PAGES']='512' if lane=='cache' else '0'
        write(args.out/(name+'-launch.json'),{'argv':argv,'environment':{k:child[k] for k in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','VLLM_BATCH_INVARIANT','RILEY_PREFIX_CACHE_PAGES'] if k in child},'gpu_before':gpu('uuid,temperature.gpu,power.draw,clocks.sm,memory.used')})
        with (args.out/(name+'.log')).open('w') as log:
            process=subprocess.Popen(argv,env=child,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+300
                while True:
                    assert process.poll() is None,'server exited before readiness: '+name
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as response:
                            if response.status==200:break
                    except Exception:pass
                    assert time.monotonic()<deadline,'readiness timeout';time.sleep(.5)
                yield port
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=40)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                write(args.out/(name+'-exit.json'),{'exit_code':process.returncode,'gpu_after':gpu('temperature.gpu,memory.used')})
    write(args.out/'preparation.json',{'hashes':hashes,'controller_sha256':file_hash(__file__),'client_sha256':file_hash(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py')),'source_fixture_sha256':file_hash(args.root/'workload.json'),'concurrency':args.concurrency,'warmup':args.warmup,'retained':args.retained,'kv_payload_bytes_per_engine':754974720,'riley_cache_page_limit':512,'vllm_prefix_caching':True,'qualification':False,'reference_scope':'same-model prior Riley greedy baseline; vLLM agreement reported separately','gpu':gpu('uuid,name,driver_version,memory.total')})
    # Capture immutable references before any timed comparison. Placeholders
    # intentionally fail comparison checks; protocol completeness must still pass.
    with server('prior','references') as port:
        for kind,fixtures in corpora.items():
            rows=phase(port,fixtures,len(fixtures),args.concurrency);assert all(r['valid'] for r in rows)
            write(args.out/(kind+'-reference-rows.json'),rows)
            for f,r in zip(fixtures,rows):
                f.update(prompt_token_ids=r['prompt_token_ids'],token_ids=r['token_ids'],text_sha256=canonical(r['text']),finish_reason=r['finish_reason'])
    assert len({tuple(f['prompt_token_ids'][:16]) for f in corpora['unique']})==len(corpora['unique']),'unique workload shares a first cache page'
    assert len({tuple(f['prompt_token_ids']) for f in corpora['shared']})==32
    write(args.out/'fixtures.json',corpora);print('references complete',flush=True)
    records=[]
    for kind,fixtures in corpora.items():
        for pair,order in enumerate([['prior','off','cache','vllm'],['vllm','cache','off','prior']]):
            for lane in order:
                name=f'{kind}-p{pair}-{lane}'
                with server(lane,name) as port:
                    warm=fixtures[:args.warmup] if kind=='unique' else fixtures
                    retained=fixtures[args.warmup:] if kind=='unique' else fixtures
                    for label,cases,count in [('warmup',warm,args.warmup),('retained',retained,args.retained)]:
                        rows=phase(port,cases,count,args.concurrency);write(args.out/(name+'-'+label+'.json'),rows)
                        stats=summary(rows);assert stats['errors']==0,name+' protocol failure'
                        if lane!='vllm':assert stats['reference_matches']==count,name+' reference mismatch'
                        if label=='retained':records.append({'name':name,**stats});write(args.out/'progress.json',records)
                print(name+' complete',flush=True)
    write(args.out/'complete.json',{'lanes':len(records),'all_complete':True})

if __name__=='__main__':main()
