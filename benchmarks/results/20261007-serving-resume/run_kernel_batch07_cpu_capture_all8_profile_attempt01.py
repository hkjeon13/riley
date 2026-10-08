"""Create-only HTTP profiling after a completed diagnostic baseline; no promotion."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/data/riley-serving-261007')
CODE = ROOT/'kernel-batch07-controller'
sys.path.insert(0, str(CODE))
from serving_token_client_v2 import TokenHttpClient, TokenReference
from mixed_phase_v34 import mixed_phase

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def clocks():
    before=time.perf_counter_ns()
    observed={'realtime_ns':time.time_ns(), 'monotonic_ns':time.monotonic_ns(),
              'monotonic_raw_ns':time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}
    observed.update(perf_counter_before_ns=before,perf_counter_after_ns=time.perf_counter_ns(),
                    perf_counter_implementation=time.get_clock_info('perf_counter').implementation)
    return observed

def gpu():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                   '--format=csv,noheader'],text=True).strip()

def host():
    return {'clocks':clocks(), 'pressure':{k:Path('/proc/pressure',k).read_text()
            for k in ['cpu','io','memory']}, 'compute_apps':gpu(),
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,temperature.gpu,power.draw,clocks.sm,memory.used',
                                         '--format=csv,noheader'],text=True).strip()}

def write(out, name, value):
    with (out/name).open('x') as f:
        json.dump(value, f, indent=2);f.write('\n')

def ready(port, process, timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline and process.poll() is None:
        connection=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
        try:
            connection.request('GET','/v1/models');response=connection.getresponse()
            body=json.loads(response.read())
            if response.status==200 and any(x.get('id')=='g04-smol' for x in body.get('data',[])):
                return
        except (OSError,http.client.HTTPException,json.JSONDecodeError):
            pass
        finally:
            connection.close()
        time.sleep(.1)
    raise RuntimeError('profile target HTTP readiness failed; exit='+str(process.poll()))

def cli(out, label, argv, env):
    started=clocks()
    with (out/(label+'.log')).open('x') as log:
        result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    write(out,label+'-receipt.json',{'argv':argv,'exit':result.returncode,'start':started,
                                   'end':clocks(),'log_sha256':sha(out/(label+'.log'))})
    if result.returncode:
        raise RuntimeError(label+' exited '+str(result.returncode))

def stop_session(out, session, process, nsys, env, target_argv):
    # The launched app may use a different process group from the profiler.
    # Require exact argv, same UID, expected binary SHA and active CUDA PID.
    compute=gpu()
    cuda_pids={int(line.split(',')[0].strip()) for line in compute.splitlines() if line.strip()}
    matches=[];observed=[]
    for pid in sorted(cuda_pids):
        folder=Path('/proc')/str(pid)
        try:
            argv=[x.decode(errors='replace') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]
            uid=folder.stat().st_uid
            item={'pid':pid,'argv':argv,'uid':uid,'pgid':os.getpgid(pid),'process_start_ticks':(folder/'stat').read_text().split(') ',1)[1].split()[19]}
            observed.append(item)
            if argv==target_argv and uid==os.getuid() and sha(target_argv[0])==json.loads((out/'launch.json').read_text())['target_binary_sha256']:matches.append(pid)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    write(out,'owned-target-shutdown.json',{'owned_pids':matches,'GPU_processes_observed':observed,'process_group':process.pid,'target_argv':target_argv,'clock':clocks(),'ownership_basis':'exact argv including unique bind port, same UID, pinned binary SHA, active CUDA PID; profiler PGID not assumed','policy':'SIGTERM exact target only; profiler auto-stop on exit'})
    error=None
    if len(matches)==1:
        try:os.kill(matches[0],signal.SIGTERM)
        except ProcessLookupError:pass
    elif process.poll() is None:
        error='owned target selector is not unique'
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:process.wait(timeout=180)
    except subprocess.TimeoutExpired:
        error=(error or '')+'; full capture profiler wait expired'
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait(timeout=10)
    observations=[];deadline=time.monotonic()+40
    while True:
        observations.append(host())
        if not observations[-1]['compute_apps'] or time.monotonic()>=deadline:break
        time.sleep(1)
    write(out,'target-exit.json',{'launcher_exit':process.returncode,'shutdown_error':error,'host':observations[-1],'reclamation_observations':observations,'reclamation_timeout_seconds':40})
    if error or process.returncode!=0 or observations[-1]['compute_apps']:
        raise RuntimeError('full capture shutdown/reclamation failed: '+str(error))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',required=True,type=int)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--verified-baseline',required=True,type=Path)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());baseline=Path(plan['baseline'])
    assert os.getuid()==plan['expected_target_uid']==0,'root-only read-only CPU profiling environment required'
    completion=json.loads((baseline/'completion.json').read_text())
    assert completion['failure'] is None and completion['all_lanes_complete'] and len(completion['records'])==96
    verified=json.loads(args.verified_baseline.read_text())
    assert verified['raw_replay_passed'] and verified['matrix_complete'] and len(verified['lanes'])==96
    for name,digest in plan['baseline_source_file_shas'].items():assert sha(baseline/name)==digest
    assert sha(args.verified_baseline)==plan['baseline_verification_file_sha256']
    assert verified['source_commits']['candidate']==plan['candidate_source_commit']
    assert plan['orders']==[['v52','candidate'],['candidate','v52']]
    assert plan['concurrency']==[1,8,16,32] and plan['workloads']==['fixed','natural']
    assert plan['warmup']==plan['retained']==96
    prep=json.loads((baseline/'preparation.json').read_text())
    assert prep['plan']['concurrency']==[1,8,16,32] and prep['plan']['orders']==[['v52','candidate','vllm'],['vllm','candidate','v52']]*2
    assert verified['input_pins']==prep['pins']
    frozen_code=Path(prep['controller_path']).parent
    for name in ['fixtures.json','mixed_phase_v34.py','serving_token_client_v2.py']:
        assert sha(CODE/name)==prep['pins'][str(frozen_code/name)],'profiling client differs from verified serving client'
    for engine,key in [('v52','expected_baseline_binary_sha256'),('candidate','expected_candidate_binary_sha256')]:
        launch=json.loads((baseline/('c1-fixed-r0-'+engine+'-launch.json')).read_text())
        assert launch['binary_sha256']==plan[key] and sha(launch['argv'][0])==plan[key]
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==plan['GPU_UUID']
    fixtures=json.loads((CODE/'fixtures.json').read_text())
    nsys=plan['tool'];version=subprocess.check_output([nsys,'--version'],text=True).strip()
    assert plan['expected_tool_version'] in version
    out=ROOT/f'kernel-batch07-cpu-capture-all8-profile-attempt{args.attempt:02d}';out.mkdir()
    (out/'controller-snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'plan-snapshot.json').write_bytes(args.plan.read_bytes())
    (out/'baseline-verification-snapshot.json').write_bytes(args.verified_baseline.read_bytes())
    write(out,'preparation.json',{'plan':plan,'baseline_preparation_sha256':sha(baseline/'preparation.json'),
          'baseline_verification_sha256':sha(args.verified_baseline),'raw_archive_sha256':plan['verified_serving_archive_sha256'],
          'baseline_completion_sha256':sha(baseline/'completion.json'),'pins':prep['pins'],
          'controller_sha256':sha(__file__),'plan_sha256':sha(args.plan),'nsys_version':version,'host':host(),
          'serving_performance_comparison':'미실행; instrumented HTTP observations are diagnostic only',
          'performance_claim_eligible':False,'goal_achieved':False})
    completed=[];failure=None
    try:
        for concurrency in plan['concurrency']:
            for workload in plan['workloads']:
                if {'concurrency':concurrency,'workload':workload} not in plan['cells']:continue
                data=fixtures[workload];corpus=data['corpus']
                refs=[TokenReference('g04-smol',tuple(r['choices'][0]['prompt_token_ids']) if 'prompt_token_ids' in r['choices'][0] else tuple(c['prompt_token_ids']),
                      tuple(r['choices'][0]['token_ids']),r['choices'][0]['text'],r['choices'][0]['finish_reason'])
                      for c,r in zip(corpus,data['responses'])]
                for repeat,order in enumerate(plan['orders']):
                    for engine in order:
                        assert not gpu(),'foreign GPU compute process'
                        for path,digest in prep['pins'].items():
                            assert sha(path)==digest,'immutable baseline input changed: '+path
                        label=f'c{concurrency}-{workload}-r{repeat}-{engine}'
                        lane=out/label;lane.mkdir()
                        launch=json.loads((baseline/f'c{concurrency}-{workload}-r0-{engine}-launch.json').read_text())
                        argv=list(launch['argv']);env=launch['env']
                        (lane/'frozen-serving-launch.json').write_bytes((baseline/f'c{concurrency}-{workload}-r0-{engine}-launch.json').read_bytes())
                        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
                        if engine in ('v52','candidate'):argv[argv.index('--bind')+1]=f'127.0.0.1:{port}'
                        else:argv[argv.index('--port')+1]=str(port)
                        session=f'Riley261007-b7cpu{args.attempt}-{label}'
                        private=lane/'profiler-private';private.mkdir()
                        for directory in ['home','config','cache','tmp']:(private/directory).mkdir()
                        profiler_env={**env,'HOME':str(private/'home'),'XDG_CONFIG_HOME':str(private/'config'),'XDG_CACHE_HOME':str(private/'cache'),'TMPDIR':str(private/'tmp')}
                        assert all(',' not in k+v for k,v in env.items())
                        target_env_override=','.join(k+'='+v for k,v in sorted(env.items()))
                        command=[nsys,'profile','--session-new='+session,'--trace='+plan['launch_trace'],
                                 '--cuda-graph-trace='+plan['cuda_graph_trace'],
                                 '--capture-range=none','--stop-on-exit=true',
                                 '--sample='+plan['sample'],'--cpuctxsw='+plan['cpuctxsw'],'--backtrace='+plan['backtrace'],
                                 '--output='+str(lane/'trace'),'--force-overwrite=false',
                                 '--env-var='+target_env_override,
                                 '--wait=all','--show-output=true',*argv]
                        write(lane,'launch.json',{'argv':command,'target_argv':argv,'env':env,'host':host(),
                                                'target_binary_sha256':sha(argv[0]),'session':session,'profiler_env':profiler_env,'target_uid':os.getuid()})
                        with (lane/'target.log').open('x') as log:
                            process=subprocess.Popen(command,env=profiler_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            primary_error=None
                            try:
                                ready(port,process,plan['startup_timeout_seconds'])
                                with TokenHttpClient() as client:
                                    def request_one(index):
                                        i=index%len(corpus);c=corpus[i]
                                        row=client.request(port,{'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0},
                                                           refs[i],streaming=True,mode='observe')
                                        row['corpus_id']=c['id'];return row
                                    for phase in ['warmup','retained']:
                                        write(lane,phase+'-host-before.json',host())
                                        rows,account,summary=mixed_phase(client,request_one,concurrency=concurrency,count=plan[phase],
                                                                         phase=phase,corpus_ids=[c['id'] for c in corpus])
                                        write(lane,phase+'-rows.json',rows);write(lane,phase+'-accounting.json',account)
                                        write(lane,phase+'-summary.json',summary);write(lane,phase+'-host-after.json',host())
                                        assert account['completed'],'profile HTTP transport incomplete'
                                        if engine in ('v52','candidate'):assert account['strict_reference_pass'],'profile target frozen reference mismatch'
                            except BaseException as error:
                                primary_error={'type':type(error).__name__,'message':str(error)}
                                write(lane,'primary-failure.json',primary_error)
                                raise
                            finally:
                                try:stop_session(lane,session,process,nsys,env,argv)
                                except BaseException as cleanup_error:
                                    if primary_error is None:raise
                                    write(lane,'cleanup-failure.json',{'type':type(cleanup_error).__name__,'message':str(cleanup_error)})
                        assert (lane/'trace.nsys-rep').is_file(),'missing raw trace'
                        cli(lane,'sqlite-export',[nsys,'export','--type=sqlite','--force-overwrite=false',
                            '--output='+str(lane/'trace.sqlite'),str(lane/'trace.nsys-rep')],env)
                        import sqlite3
                        with sqlite3.connect(lane/'trace.sqlite') as database:
                            tables={r[0] for r in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                            diagnostics=[r[0] for r in database.execute('SELECT text FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
                            write(lane,'trace-diagnostics.json',{'events':diagnostics,'scope':'profiler diagnostics; absence of warnings alone is not capture completeness'})
                            assert not any('not supported by this build' in text for text in diagnostics),'profiler CUDA driver unsupported; retain trace without attribution qualification'
                            count=database.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
                        assert count>0,'no CUDA kernel evidence; tool compatibility unproven'
                        write(lane,'artifact-manifest.json',{'files':{p.name:sha(p) for p in lane.iterdir() if p.is_file()},
                              'kernel_events':count,'serving_performance_comparison':'미실행','performance_claim_eligible':False})
                        completed.append(label)
                        (out/'progress.json').write_text(json.dumps({'completed':completed,'expected_lanes':32,
                            'performance_claim_eligible':False},indent=2)+'\n')
                        print(label+' profile complete',flush=True)
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)};raise
    finally:
        write(out,'completion.json',{'completed':completed,'failure':failure,'expected_lanes':32,
                                   'performance_claim_eligible':False,'goal_achieved':False})

if __name__=='__main__':main()
