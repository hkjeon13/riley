"""Bounded offline Nsight serving trace; timings are diagnostic, not a benchmark."""
import argparse,json,os,pathlib,signal,socket,subprocess,threading,time,urllib.request
from paired_decode_serving_screen import phase,file_hash,write


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


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=pathlib.Path);ap.add_argument('out',type=pathlib.Path);args=ap.parse_args()
    root=args.root;out=args.out;out.mkdir();spec=json.loads((root/'workload.json').read_text())
    for lane in ['single','paired']:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        argv=list(spec['riley_argv']);argv[0]=str(root/'target/release/riley');argv[argv.index('--bind')+1]=f'127.0.0.1:{port}';argv+=['--shutdown-on-stdin']
        if lane=='paired':argv+=['--decode-window','paired-experimental-v1']
        command=['/data/cuda-12.8.1/bin/nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--output',str(out/lane),'--force-overwrite=false']+argv
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(root/'toolchain130/nvidia/cu13/lib')+':/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib')
        write(out/(lane+'-launch.json'),{'argv':command,'binary_sha256':file_hash(argv[0]),'controller_sha256':file_hash(__file__),'client_sha256':file_hash(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py')),'rss_limit_bytes':8*1024**3,'wall_limit_s':360})
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
                rows=phase(port,spec['natural'],96,32);write(out/(lane+'-rows.json'),rows)
                assert all(r['valid'] and all(r['checks'].values()) for r in rows)
                p.stdin.write(b'\n');p.stdin.flush();assert p.wait(timeout=60)==0
                receipt.update(requests=96,reference_exact=True,exit_code=p.returncode)
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
