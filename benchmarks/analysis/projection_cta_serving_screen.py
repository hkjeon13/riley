"""Projection CTA experiment. Offline closed-loop dense ownership validation serving screen with rolling decode in every Riley lane; no Python runs in the Riley serving path."""
import gc,resource,argparse,concurrent.futures,contextlib,hashlib,json,os,pathlib,signal,socket,subprocess,time,urllib.request
from paired_decode_serving_screen import phase,summary,file_hash,write,canonical,request,cancel_request


def host_snapshot():
    result={'monotonic_ns':time.monotonic_ns(),'pressure':{},'loadavg':pathlib.Path('/proc/loadavg').read_text().split()[:3]}
    for resource in ('cpu','io','memory'):
        result['pressure'][resource]={}
        for line in pathlib.Path('/proc/pressure',resource).read_text().splitlines():
            category,*fields=line.split()
            result['pressure'][resource][category]={key:float(value) if key!='total' else int(value) for key,value in (field.split('=') for field in fields)}
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=pathlib.Path);parser.add_argument('out',type=pathlib.Path)
    parser.add_argument('--concurrency',type=int,default=32);parser.add_argument('--warmup',type=int,default=64);parser.add_argument('--retained',type=int,default=256)
    parser.add_argument('--cooldown-timeout-seconds',type=int,default=120)
    args=parser.parse_args();assert args.concurrency in (8,16,32,64) and args.warmup>=32 and args.retained>=128
    assert 120<=args.cooldown_timeout_seconds<=900
    assert args.out.parent.resolve()==pathlib.Path('/dev/shm'),'controlled artifact spool must be tmpfs'
    fs=os.statvfs(args.out.parent);assert fs.f_bavail*fs.f_frsize>=12*1024**3,'tmpfs capacity guard'
    mem={line.split(':')[0]:int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    assert mem['MemAvailable']>=16*1024**2,'host available memory guard'
    args.out.mkdir()
    (args.out/'controller-snapshot.py').write_bytes(pathlib.Path(__file__).read_bytes())
    (args.out/'client-snapshot.py').write_bytes(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py').read_bytes())
    spec=json.loads((args.root/'workload.json').read_text());prior=args.root/'riley-dense-wire-serving-v1'
    binary=args.root/'riley-projection-cta-serving-v1'
    hashes={'prior':file_hash(prior),'candidate':file_hash(binary)}
    model_gate=json.loads((args.root/'projection-cta-model-lifecycle-v1/execution.json').read_text())
    assert len(model_gate)==1 and model_gate[0]['name']=='model' and model_gate[0]['exit_code']==0,'model gate missing or failed'
    assert hashes['candidate']==json.loads((args.root/'projection-cta-serving-build-v1/binary.json').read_text())['sha256']
    assert json.loads((args.root/'projection-cta-serving-build-v1/runtime-ticket-tests-exit.json').read_text())['exit_code']==0
    assert hashes['prior']=='a525729d037b519e9c796b7574f960820fb6cbeb1e0d60e4a8a504c4cd616403','frozen binary mismatch'
    venvs={'vllm029':args.root/'vllm029-venv'}
    versions={}
    for lane,venv in venvs.items():
        code='import importlib.metadata as m,json,sys;print(json.dumps({"python":sys.version,"packages":{k:m.version(k) for k in ["vllm","torch","transformers","flashinfer-python"]}}))'
        versions[lane]=json.loads(subprocess.check_output([str(venv/'bin/python'),'-c',code],text=True))
        assert versions[lane]['packages']['vllm']=={'vllm027':'0.27.1','vllm029':'0.29.0'}[lane]
    write(args.out/'versions.json',versions)
    base=max(spec['natural'],key=lambda x:len(x['prompt_token_ids']))['prompt']
    def fixture(kind,index):
        nonce=hashlib.sha256(f'{kind}-{index}'.encode()).hexdigest()
        tail='\nAdditional request details: '+(' '.join(f'item{index}_{j}' for j in range(20)))+'. Continue the explanation.'
        prompt=base+tail if kind=='shared' else nonce+' '+base+tail
        return {'id':f'{kind}-{index}','prompt':prompt,'max_tokens':32,'prompt_token_ids':[],'token_ids':[],'text_sha256':canonical(''),'finish_reason':None}
    corpora={'shared':[fixture('shared',i) for i in range(32)],'unique':[fixture('unique',i) for i in range(args.warmup+args.retained)]}
    env=os.environ.copy();env.pop('RILEY_EXPERIMENT_PROJECTION_CTAS',None);env.pop('RILEY_SPECULATIVE_DECODE',None);env.pop('RILEY_PREFILL_FFN_SPLIT',None);env.pop('RILEY_PREFILL_FFN_ADAPTIVE',None);env.pop('RILEY_MIXED_GQA_STAGING',None);env.pop('RILEY_ROLLING_DECODE',None);env.pop('RILEY_PREFILL_PROJECTION_PIPELINE',None);env.update(CUDA_VISIBLE_DEVICES='0',VLLM_BATCH_INVARIANT='0')
    # Select each version's runner/backend defaults, independent of shell overrides.
    for key in list(env):
        if key.startswith('VLLM_'):env.pop(key)
    env['VLLM_BATCH_INVARIANT']='0'
    env.pop('RILEY_MIXED_QUERY_REUSE',None);env.pop('RILEY_PREFIX_CACHE_PAGES',None);env.pop('RILEY_SERVING_PHASE_TIMING',None)
    env['LD_LIBRARY_PATH']=str(args.root/'toolchain130/nvidia/cu13/lib')+':/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib'
    def gpu(fields,kind='gpu'):return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    @contextlib.contextmanager
    def server(lane,name):
        assert not gpu('pid','compute-apps'),'foreign GPU compute process'
        started=time.monotonic();deadline=started+args.cooldown_timeout_seconds;samples=[]
        try:
            while True:
                temperature=int(gpu('temperature.gpu'))
                samples.append({'elapsed_s':time.monotonic()-started,'temperature_c':temperature})
                if temperature<=48:break
                assert time.monotonic()<deadline,'cooldown timeout'
                time.sleep(1)
        finally:
            write(args.out/(name+'-cooldown.json'),{'threshold_c':48,'timeout_s':args.cooldown_timeout_seconds,'samples':samples})
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        argv=list(spec['vllm_argv' if lane.startswith('vllm') else 'riley_argv']);child=env.copy()
        def option(key,value):argv[argv.index(key)+1]=str(value)
        if lane.startswith('vllm'):
            argv[0]=str(venvs[lane]/'bin/vllm')
            option('--port',port);option('--max-num-seqs',min(args.concurrency,32))
            argv.remove('--no-enable-prefix-caching');argv+=['--enable-prefix-caching','--block-size','16','--kv-cache-memory-bytes',str(30*3*64*2*2*16*2048)]
            child['LD_LIBRARY_PATH']=str(venvs[lane]/'lib/python3.13/site-packages/nvidia/cu13/lib')
            child['PATH']=str(venvs[lane]/'bin')+':'+child.get('PATH','/usr/bin:/bin')
        else:
            argv[0]=str(binary if lane=='candidate' else prior);assert file_hash(argv[0])==hashes[lane]
            option('--bind',f'127.0.0.1:{port}');option('--max-active-sequences',min(args.concurrency,32));option('--kv-blocks',2048)
            child['RILEY_EXPERIMENT_PROJECTION_CTAS']='1' if lane=='candidate' else '0'
            child['RILEY_PREFIX_CACHE_PAGES']='512'
            child['RILEY_ROLLING_DECODE']='1'
            child['RILEY_PREFILL_PROJECTION_PIPELINE']='1'
            child['RILEY_PREFILL_FFN_ADAPTIVE']='0'
            child['RILEY_PREFILL_FFN_SPLIT']='1'
            child['RILEY_MIXED_QUERY_REUSE']='0'
            argv+=['--decode-window','paired-experimental-v1','--ffn-backend','prefill-pipeline-experimental-v1','--decode-projection','adaptive-rows-experimental-v1']
        write(args.out/(name+'-launch.json'),{'argv':argv,'environment':{k:child[k] for k in ['RILEY_EXPERIMENT_PROJECTION_CTAS','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','VLLM_BATCH_INVARIANT','RILEY_PREFIX_CACHE_PAGES','RILEY_MIXED_QUERY_REUSE','RILEY_ROLLING_DECODE','RILEY_PREFILL_PROJECTION_PIPELINE','RILEY_PREFILL_FFN_ADAPTIVE','RILEY_PREFILL_FFN_SPLIT'] if k in child},'gpu_before':gpu('uuid,temperature.gpu,power.draw,clocks.sm,memory.used')})
        with (args.out/(name+'.log')).open('w') as log:
            startup_begin=time.monotonic_ns()
            process=subprocess.Popen(argv,env=child,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+600
                while True:
                    assert process.poll() is None,'server exited before readiness: '+name
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as response:
                            if response.status==200:break
                    except Exception:pass
                    assert time.monotonic()<deadline,'readiness timeout';time.sleep(.5)
                ready_ns=time.monotonic_ns()
                status=pathlib.Path(f'/proc/{process.pid}/status').read_text().splitlines()
                rss_kib=int(next(line.split()[1] for line in status if line.startswith('VmRSS:')))
                write(args.out/(name+'-startup.json'),{'ready_wall_ms':(ready_ns-startup_begin)/1e6,'server_rss_kib':rss_kib,'gpu_at_ready':gpu('memory.used,temperature.gpu'),'scope':'process launch to HTTP readiness; global GPU memory includes display; not isolated graph allocation'})
                yield port
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=40)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                write(args.out/(name+'-exit.json'),{'exit_code':process.returncode,'gpu_after':gpu('temperature.gpu,memory.used')})
    write(args.out/'preparation.json',{'hashes':hashes,'controller_sha256':file_hash(__file__),'client_sha256':file_hash(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py')),'source_fixture_sha256':file_hash(args.root/'workload.json'),'concurrency':args.concurrency,'active_capacity':min(args.concurrency,32),'warmup':args.warmup,'retained':args.retained,'kv_payload_bytes_per_engine':754974720,'riley_cache_page_limit':512,'vllm_prefix_caching':True,'client_treatment':'gc-phase-disabled-v1','artifact_storage':'tmpfs:/dev/shm','phase_cleanup':'release previous rows and full collect before every phase','readiness_timeout_seconds':600,'qualification':False,'reference_scope':'same-model prior Riley greedy baseline; vLLM agreement reported separately','gpu':gpu('uuid,name,driver_version,memory.total')})
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
    # Establish latest-runner compatibility before any timed comparisons.
    with server('vllm029','preflight-vllm029') as port:
        rows=phase(port,corpora['shared'],32,args.concurrency)
        write(args.out/'preflight-vllm029-rows.json',rows)
        assert len(rows)==32 and all(r['valid'] and len(r['token_ids'])==32 for r in rows)
    stop_reference=None
    stop_text=json.loads((args.out/"shared-reference-rows.json").read_text())[0]["text"][40:60];assert len(stop_text)==20
    records=[]
    for kind,fixtures in corpora.items():
        for pair,order in enumerate([['prior','candidate','vllm029'],['vllm029','candidate','prior']]):
            for lane in order:
                name=f'{kind}-p{pair}-{lane}'
                with server(lane,name) as port:
                    warm=fixtures[:args.warmup] if kind=='unique' else fixtures
                    retained=fixtures[args.warmup:] if kind=='unique' else fixtures
                    for label,cases,count in [('warmup',warm,args.warmup),('retained',retained,args.retained)]:
                        # Identical cleanup outside both treatments; never subtract it from latency.
                        rows=None
                        assert gc.isenabled()
                        pre_phase_collected=gc.collect()
                        before=host_snapshot()
                        events=[];starts={}
                        def observe_gc(action,info):
                            now=time.monotonic_ns();generation=info['generation']
                            if action=='start':starts[generation]=now
                            elif generation in starts:events.append((generation,starts.pop(generation),now,info['collected'],info['uncollectable']))
                        gc.disable()
                        gc_before={'enabled':gc.isenabled(),'threshold':gc.get_threshold(),'counts':gc.get_count(),'stats':gc.get_stats()}
                        cpu_before=time.process_time_ns();rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        gc.callbacks.append(observe_gc)
                        try:rows=phase(port,cases,count,args.concurrency)
                        finally:
                            gc.callbacks.remove(observe_gc)
                            gc_after={'counts':gc.get_count(),'stats':gc.get_stats()}
                            gc.enable()
                        gc_receipt={'policy':'disabled','pre_phase_collected':pre_phase_collected,'restored_enabled':gc.isenabled(),'before':gc_before,'after':gc_after,'events':events,'unclosed_generations':starts,'process_cpu_ns':time.process_time_ns()-cpu_before,'ru_maxrss_before_kib':rss_before,'ru_maxrss_after_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                        assert gc.isenabled() and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024*1024,'client state or RSS guard'
                        after=host_snapshot()
                        write(args.out/(name+'-'+label+'-gc.json'),gc_receipt)
                        write(args.out/(name+'-'+label+'-host.json'),{'before':before,'after':after})
                        write(args.out/(name+'-'+label+'.json'),rows)
                        stats=summary(rows);assert stats['errors']==0,name+' protocol failure'
                        if not lane.startswith('vllm'):assert stats['reference_matches']==count,name+' reference mismatch'
                        if label=='retained':records.append({'name':name,**stats});write(args.out/'progress.json',records)
                    if kind=='shared' and pair==0 and not lane.startswith('vllm'):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                            stopped=list(pool.map(lambda _:request(port,fixtures[0],{'stop':stop_text}),range(args.concurrency)))
                        write(args.out/(name+'-stop.json'),stopped)
                        assert all(r['valid'] and r['finish_reason']=='stop' for r in stopped)
                        identities=[(r['token_ids'],r['text'],r['finish_reason'],r['usage']) for r in stopped]
                        if stop_reference is None:stop_reference=identities
                        else:assert identities==stop_reference,'Split FFN stop publication mismatch'
                        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                            cancelled=list(pool.map(lambda _:cancel_request(port,fixtures[0]),range(args.concurrency)))
                        write(args.out/(name+'-cancel.json'),cancelled)
                        recovery=phase(port,fixtures,args.concurrency,args.concurrency)
                        write(args.out/(name+'-recovery.json'),recovery)
                        assert all(r['valid'] and all(r['checks'].values()) for r in recovery)
                print(name+' complete',flush=True)
    write(args.out/'complete.json',{'lanes':len(records),'all_complete':True})

if __name__=='__main__':main()
