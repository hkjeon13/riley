import json,hashlib,subprocess,time,os,signal
from pathlib import Path
R=Path('/data/riley-serving-261007');P=R/'kernel-batch04-gate-counter-plan-attempt02.json';plan=json.loads(P.read_text());O=R/'kernel-batch04-gate-counters-attempt02';O.mkdir();failure=None;completed=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def host():return {'time_ns':time.time_ns(),'compute_pids':gpu(),'gpu':subprocess.check_output(['nvidia-smi','-q'],text=True),'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']}}
try:
 assert not gpu(),'foreign GPU process; leave unchanged';assert not set(subprocess.check_output(['ps','-eo','comm'],text=True).splitlines()) & {'ncu','ncu-ui','ncu-cli'},'foreign counter profiler';Path(plan['temporary_directory']).mkdir(mode=0o700);assert sha(plan['binary'])==plan['binary_sha256'];assert plan['clock_control']==plan['cache_control']=='none'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==plan['GPU_UUID'];version=subprocess.check_output([plan['tool'],'--version'],text=True);assert plan['expected_tool_version'] in version
 for folder in ['qwen-full128-validation-attempt05','qwen-m2-bounded-validation-attempt04']:assert json.loads((R/folder/'completion.json').read_text())['failure'] is None
 receipt=json.loads((R/'kernel-batch04-source-receipt-attempt01.json').read_text());assert receipt['source_commit']==plan['source_commit'];S=R/'kernel-batch04-source-attempt01';source={n:sha(S/n) for n in receipt['files']};assert source==receipt['files']
 test=(S/'kernels/tests/kernel_batch03_correctness.cu').read_text();assert 'for(int rows=0;rows<=33;++rows)' in test and 'for(int seed:{1,17,99})' in test and 'cases+=gate<true>(seed);cases+=gate<false>(seed);' in test
 write(O/'preparation.json',{'plan':plan,'tool_version':version,'tool_sha256':sha(plan['tool']),'source_files':source,'host':host(),'plan_sha256':sha(P),'controller_sha256':sha(__file__)});(O/'controller-snapshot.py').write_bytes(Path(__file__).read_bytes())
 for rows in plan['active_rows']:
  for repeat,order in enumerate(plan['orders']):
   for engine in order:
    assert not gpu(),'GPU occupied before diagnostic';assert sha(plan['binary'])==plan['binary_sha256']
    label=f'rows{rows}-r{repeat}-{engine}';L=O/label;L.mkdir();trace=L/'trace'
    argv=['sudo','-n','/usr/bin/env','HOME=/home/psyche','TMPDIR='+plan['temporary_directory'],'PATH=/data/cuda-12.8.1/bin:/usr/bin:/bin',plan['tool'],'--config-file','off','--clock-control','none','--cache-control','none','--graph-profiling','node','--replay-mode','kernel','--kernel-name-base','function','--kernel-name',plan['kernel_filter_by_engine'][engine],'--launch-skip',str(plan['launch_skip_by_rows'][str(rows)]),'--launch-count','1','--export',str(trace)]
    for section in plan['sections']:argv+=['--section',section]
    argv.append(plan['binary']);write(L/'launch.json',{'argv':argv,'source_commit':plan['source_commit'],'binary_sha256':sha(plan['binary']),'host':host(),'active_rows_from_frozen_source_invocation':rows})
    with (L/'target.log').open('x') as log:
     process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);timed_out=False
     try:process.wait(timeout=plan['timeout_seconds'])
     except subprocess.TimeoutExpired:
      timed_out=True;subprocess.run(['sudo','-n','kill','-TERM','--','-'+str(process.pid)],check=False)
      try:process.wait(timeout=40)
      except subprocess.TimeoutExpired:subprocess.run(['sudo','-n','kill','-KILL','--','-'+str(process.pid)],check=False);process.wait(timeout=10)
    write(L/'process.json',{'exit':process.returncode,'timeout':timed_out,'host_after':host(),'log_sha256':sha(L/'target.log'),'binary_sha256_after':sha(plan['binary'])});assert not timed_out and process.returncode==0,'counter diagnostic failed; preserve all failures'
    assert (L/'trace.ncu-rep').is_file(),'missing profile result; filter/counter unavailable'
    log=(L/'target.log').read_text();assert '\"cases\":1088' in log and '\"fusion_cases\":272' in log and '\"passed\":true' in log,'instrumented exact qualifier incomplete'
    command=[plan['tool'],'--import',str(L/'trace.ncu-rep'),'--csv','--page','details']
    with (L/'metrics.csv').open('x') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode==0,'counter export failed';write(L/'export.json',{'argv':command,'exit':p.returncode,'raw_profile_sha256':sha(L/'trace.ncu-rep'),'csv_sha256':sha(L/'metrics.csv')})
    assert not gpu(),'GPU cleanup incomplete';completed.append(label);write(O/(label+'-terminal.json'),{'completed':True,'serving_performance':'미실행'})
 after={n:sha(S/n) for n in receipt['files']};write(O/'source-integrity-after.json',{'files':after,'binary_sha256':sha(plan['binary'])});assert after==source and sha(plan['binary'])==plan['binary_sha256']
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write(O/'completion.json',{'failure':failure,'completed':completed,'expected_lanes':8,'serving_performance':'미실행','performance_claim_eligible':False,'goal_achieved':False})
