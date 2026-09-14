"""Bounded best-serving rolling host timing Nsight comparison; diagnostic, not serving timing.

Caller owns exclusive GPU access and Blender recovery. Public viewers stay up.
Raw Nsight/SQLite outputs remain private on the measurement host.
"""
import gc,argparse,json,os,pathlib,signal,socket,subprocess,threading,time,urllib.request
from paired_decode_serving_screen import phase as client_phase,file_hash,write


def owned_processes(root_pid,proc_root=pathlib.Path('/proc')):
    """Include the exact detached Nsight session agent and its descendants."""
    processes={};owned=set()
    for path in proc_root.glob('[0-9]*'):
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            pid=int(path.name);parent=int(fields[1]);group=int(fields[2]);start=fields[19]
            rss=int((path/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            processes[pid]=(parent,start,rss)
            if pid==root_pid or group==root_pid:owned.add(pid)
            elif parent==1:
                argv=(path/'cmdline').read_bytes().decode().split('\0')
                if '--start-agent' in argv and '--session-name' in argv and argv[argv.index('--session-name')+1]==f'profile-{root_pid}':owned.add(pid)
        except (OSError,ValueError,IndexError,UnicodeError):pass
    while True:
        children={pid for pid,(parent,_,_) in processes.items() if parent in owned}
        if children<=owned:break
        owned.update(children)
    return {pid:processes[pid] for pid in owned if pid in processes}


def terminate_owned(root_pid,sig):
    for pid,(_,start,_) in owned_processes(root_pid).items():
        try:
            if pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]==start:
                os.kill(pid,sig)
        except (OSError,IndexError):pass


def watch(process,done,receipt):
    start=time.monotonic();peak=0
    while not done.wait(.5) and process.poll() is None:
        rss=sum(row[2] for row in owned_processes(process.pid).values())
        peak=max(peak,rss);receipt['peak_owned_tree_rss_bytes']=peak
        if rss>8*1024**3 or time.monotonic()-start>360:
            receipt['limit_exceeded']='rss' if rss>8*1024**3 else 'wall_time'
            terminate_owned(process.pid,signal.SIGTERM)
            if not done.wait(5):terminate_owned(process.pid,signal.SIGKILL)
            return


def phase(port,fixtures,count,concurrency):
    assert gc.isenabled()
    gc.collect();before=gc.get_stats();gc.disable()
    try:rows=client_phase(port,fixtures,count,concurrency)
    finally:
        after=gc.get_stats();gc.enable()
    assert before==after,'client GC occurred during profiling requests'
    return rows,{'policy':'disabled','before':before,'after':after,'restored_enabled':gc.isenabled()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=pathlib.Path);ap.add_argument('out',type=pathlib.Path);ap.add_argument('--fixtures',type=pathlib.Path,required=True);ap.add_argument('--requests',type=int,default=96);ap.add_argument('--concurrency',type=int,default=32);ap.add_argument('--active-capacity',type=int,choices=[1,2,4,8,16,32]);ap.add_argument('--prefill-ffn-pipeline',action='store_true');ap.add_argument('--adaptive-decode',action='store_true');args=ap.parse_args()
    assert 1<=args.concurrency<=128 and args.concurrency<=args.requests<=1024
    root=args.root;out=args.out;out.mkdir();spec=json.loads((root/'workload.json').read_text())
    fixtures=json.loads(args.fixtures.read_text())
    for lane in ['control-shared','instrumented-shared','instrumented-unique','control-unique']:
        backend,workload=lane.split('-')
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        argv=list(spec['riley_argv']);argv[0]=str(root/('riley-ffn-split-serving-v1' if backend=='control' else 'riley-rolling-host-serving-v1'));argv[argv.index('--bind')+1]=f'127.0.0.1:{port}';argv+=['--shutdown-on-stdin','--decode-window','paired-experimental-v1']
        argv[argv.index('--kv-blocks')+1]='2048'
        if args.active_capacity is not None:argv[argv.index('--max-active-sequences')+1]=str(args.active_capacity)
        if True:
            if args.prefill_ffn_pipeline:argv+=['--ffn-backend','prefill-pipeline-experimental-v1']
            if args.adaptive_decode:argv+=['--decode-projection','adaptive-rows-experimental-v1']
        command=['/data/cuda-12.8.1/bin/nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--output',str(out/lane),'--force-overwrite=false']+argv
        env={'PATH':'/usr/bin:/bin','HOME':os.environ['HOME'],'RILEY_PREFIX_CACHE_PAGES':'512','RILEY_SERVING_PHASE_TIMING':'1'};env.update(RILEY_ROLLING_DECODE='1',RILEY_SPECULATIVE_DECODE='0',RILEY_PREFILL_FFN_SPLIT='1',RILEY_MIXED_GQA_STAGING='0',RILEY_PREFILL_PROJECTION_PIPELINE='1',RILEY_MIXED_QUERY_REUSE='0');env.update(CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(root/'toolchain130/nvidia/cu13/lib')+':/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib')
        write(out/(lane+'-launch.json'),{'argv':command,'rolling_decode':True,'projection_pipeline':True,'speculative_decode':False,'rolling_timing':backend=='instrumented','client_gc_policy':'disabled','kv_payload_bytes':754974720,'requests':args.requests,'client_concurrency':args.concurrency,'active_capacity':int(argv[argv.index('--max-active-sequences')+1]),'fixture_sha256':file_hash(args.fixtures),'binary_sha256':file_hash(argv[0]),'controller_sha256':file_hash(__file__),'client_sha256':file_hash(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py')),'rss_limit_bytes':8*1024**3,'wall_limit_s':360})
        receipt={};done=threading.Event()
        with (out/(lane+'.log')).open('w') as log:
            p=subprocess.Popen(command,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            watcher=threading.Thread(target=watch,args=(p,done,receipt),daemon=True);watcher.start()
            try:
                deadline=time.monotonic()+240
                while True:
                    assert p.poll() is None,'profiler/server exited before readiness'
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as r:
                            if r.status==200:break
                    except OSError:pass
                    assert time.monotonic()<deadline,'readiness timeout';time.sleep(.5)
                warm,warm_gc=phase(port,fixtures[workload][:64],64,args.concurrency)
                assert all(r['valid'] and all(r['checks'].values()) for r in warm)
                cases=fixtures[workload][64:] if workload=='unique' else fixtures[workload]
                rows,rows_gc=phase(port,cases,args.requests,args.concurrency);write(out/(lane+'-rows.json'),rows)
                assert all(r['valid'] and all(r['checks'].values()) for r in rows)
                write(out/(lane+'-warmup.json'),warm)
                p.stdin.write(b'\n');p.stdin.flush();assert p.wait(timeout=60)==0
                receipt.update(requests=args.requests,reference_exact=True,exit_code=p.returncode,client_gc={'warmup':warm_gc,'rows':rows_gc})
            finally:
                try:
                    if p.poll() is None:
                        terminate_owned(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=10)
                        except subprocess.TimeoutExpired:terminate_owned(p.pid,signal.SIGKILL);p.wait(timeout=10)
                finally:
                    terminate_owned(p.pid,signal.SIGTERM)
                    done.set();watcher.join(timeout=2)
                    deadline=time.monotonic()+2
                    while owned_processes(p.pid) and time.monotonic()<deadline:time.sleep(.1)
                    remaining=owned_processes(p.pid)
                    receipt.update(exit_code=p.returncode,remaining_owned_pids=sorted(remaining))
                    write(out/(lane+'-receipt.json'),receipt)
            assert 'limit_exceeded' not in receipt
            assert not receipt['remaining_owned_pids'],'owned processes remain; inspect before another launch'
        subprocess.run(['/data/cuda-12.8.1/bin/nsys','export','--type=sqlite','--output',str(out/(lane+'.sqlite')),str(out/(lane+'.nsys-rep'))],check=True,timeout=120)
if __name__=='__main__':main()
