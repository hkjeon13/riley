"""Diagnostic KV-sharing census using frozen serving references. Timing is not performance qualification; the Riley runtime remains native."""
import ast,re,gc,resource,argparse,concurrent.futures,contextlib,hashlib,json,os,pathlib,signal,socket,subprocess,time,urllib.request
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
    spec=json.loads((args.root/'workload.json').read_text());prior=args.root/'riley-ffn-split-serving-v1'
    binary=args.root/'riley-kv-census-v2'
    hashes={'prior':file_hash(prior),'candidate':file_hash(binary)}
    assert b'RILEY_KV_TILE_GROUPS' in binary.read_bytes(),'diagnostic missing from executable'
    assert hashes['candidate']==json.loads((args.root/'kv-census-build-v2.json').read_text())['sha256']
    assert json.loads((args.root/'dense-wire-serving-build-v2/runtime-ticket-tests-exit.json').read_text())['exit_code']==0
    assert hashes['prior']=='658c00b7b99b57d54a12dd345dd44a3ba705fdffa6749687eed9a8e1a04306cc','frozen binary mismatch'
    corpora=json.loads((args.root/'kv-census-fixtures-v1.json').read_text())
    env=os.environ.copy();env.pop('RILEY_SPECULATIVE_DECODE',None);env.pop('RILEY_PREFILL_FFN_SPLIT',None);env.pop('RILEY_PREFILL_FFN_ADAPTIVE',None);env.pop('RILEY_MIXED_GQA_STAGING',None);env.pop('RILEY_ROLLING_DECODE',None);env.pop('RILEY_PREFILL_PROJECTION_PIPELINE',None);env.update(CUDA_VISIBLE_DEVICES='0',VLLM_BATCH_INVARIANT='0')
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
        argv=list(spec['riley_argv']);child=env.copy()
        def option(key,value):argv[argv.index(key)+1]=str(value)
        assert lane=='candidate'
        argv[0]=str(binary if lane=='candidate' else prior);assert file_hash(argv[0])==hashes[lane]
        option('--bind',f'127.0.0.1:{port}');option('--max-active-sequences',min(args.concurrency,32));option('--kv-blocks',2048)
        child['RILEY_SERVING_PHASE_TIMING']='1'
        child['RILEY_PREFIX_CACHE_PAGES']='512'
        child['RILEY_ROLLING_DECODE']='1'
        child['RILEY_PREFILL_PROJECTION_PIPELINE']='1'
        child['RILEY_PREFILL_FFN_ADAPTIVE']='0'
        child['RILEY_PREFILL_FFN_SPLIT']='1'
        child['RILEY_MIXED_QUERY_REUSE']='0'
        argv+=['--decode-window','paired-experimental-v1','--ffn-backend','prefill-pipeline-experimental-v1','--decode-projection','adaptive-rows-experimental-v1']
        write(args.out/(name+'-launch.json'),{'argv':argv,'environment':{k:child[k] for k in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','VLLM_BATCH_INVARIANT','RILEY_SERVING_PHASE_TIMING','RILEY_PREFIX_CACHE_PAGES','RILEY_MIXED_QUERY_REUSE','RILEY_ROLLING_DECODE','RILEY_PREFILL_PROJECTION_PIPELINE','RILEY_PREFILL_FFN_ADAPTIVE','RILEY_PREFILL_FFN_SPLIT'] if k in child},'gpu_before':gpu('uuid,temperature.gpu,power.draw,clocks.sm,memory.used')})
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
    write(args.out/'preparation.json',{'hashes':hashes,'controller_sha256':file_hash(__file__),'client_sha256':file_hash(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py')),'source_fixture_sha256':file_hash(args.root/'workload.json'),'concurrency':args.concurrency,'active_capacity':min(args.concurrency,32),'warmup':args.warmup,'retained':args.retained,'kv_payload_bytes_per_engine':754974720,'riley_cache_page_limit':512,'client_treatment':'gc-phase-disabled-v1','artifact_storage':'tmpfs:/dev/shm','phase_cleanup':'release previous rows and full collect before every phase','readiness_timeout_seconds':600,'qualification':False,'reference_scope':'frozen C32 prior Riley greedy baseline; no performance qualification','diagnostic_only':True,'gpu':gpu('uuid,name,driver_version,memory.total')})
    write(args.out/'fixtures.json',corpora)
    records=[]
    for kind,fixtures in corpora.items():
        for pair,order in enumerate([['candidate']]):
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
                lines=re.findall(r'RILEY_KV_TILE_GROUPS decode_batches=(\d+) groups_by_owner_count=(\[[^\n]+?\])',(args.out/(name+'.log')).read_text())
                assert lines,'missing runtime census: '+name
                batches,histogram=lines[-1];histogram=ast.literal_eval(histogram)
                assert int(batches)>0 and len(histogram)==33 and all(isinstance(n,int) and n>=0 for n in histogram),'invalid census'
                write(args.out/(name+'-census.json'),{'decode_batches':int(batches),'groups_by_owner_count':histogram,'scope':'pure decode observations including warmup; not performance qualification','reports':len(lines)})
                print(name+' complete',flush=True)
    write(args.out/'complete.json',{'lanes':len(records),'all_complete':True})

if __name__=='__main__':main()
