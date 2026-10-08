"""Use captured clock bridges to bound host observations, without causal claims."""
import argparse,json,re,tarfile,hashlib
from pathlib import Path
from analyze_serving_host_samples import pressure
from summarize_baseline import stats
PATTERN=r'(c\d+-(?:fixed|natural)-r\d+-(?:v52|candidate|vllm))'
def analyze(archive):
 hosts={};phases={};controller=None;prep=None
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   if not member.isfile():continue
   name=Path(member.name).name
   if name=='controller-snapshot.py':controller=tar.extractfile(member).read()
   elif name=='preparation.json':prep=json.load(tar.extractfile(member))
   elif m:=re.fullmatch(PATTERN+'-host.json',name):hosts[m[1]]=json.load(tar.extractfile(member))
   elif m:=re.fullmatch(PATTERN+'-(warmup|retained)-accounting.json',name):phases[(m[1],m[2])]=json.load(tar.extractfile(member))
 assert prep is not None and controller is not None and hashlib.sha256(controller).hexdigest()==prep['pins'][prep['controller_path']]
 reports=[]
 for case,samples in sorted(hosts.items()):
  for sample in samples:
   c=sample['clocks'];assert c['perf_counter_before_ns']<=c['perf_counter_after_ns'] and c['epoch_ns']==sample['time_ns']
  for phase in ['warmup','retained']:
   account=phases.get((case,phase));selected=[];uncertain=[]
   if account:
    start,end=account['phase_started_ns'],account['phase_finished_ns']
    for i,sample in enumerate(samples):
     c=sample['clocks'];lo=c['perf_counter_before_ns']
     # Host collection runs serially in one monitoring thread. The next
     # observation's start is a conservative upper bound on this call's end.
     hi=samples[i+1]['clocks']['perf_counter_before_ns'] if i+1<len(samples) else None
     if hi is not None:assert c['perf_counter_after_ns']<=hi
     if hi is not None and start<=lo<=hi<=end:selected.append(sample)
     else:uncertain.append(i)
   metrics={k:[] for k in ['CPU_some_avg10','IO_full_avg10','memory_full_avg10','GPU_memory_MiB','RSS_sum_MiB']}
   for sample in selected:
    metrics['CPU_some_avg10'].append(pressure(sample,'cpu','some')['avg10']);metrics['IO_full_avg10'].append(pressure(sample,'io','full')['avg10']);metrics['memory_full_avg10'].append(pressure(sample,'memory','full')['avg10'])
    metrics['GPU_memory_MiB'].append(float(sample['gpu'].split(',')[4]))
    if 'server_process_memory' in sample:metrics['RSS_sum_MiB'].append(sample['server_process_memory']['rss_sum_KiB']/1024)
   reports.append({'case':case,'phase':phase,'phase_completed':account.get('completed') if account else None,'source_host_samples':len(samples),'wholly_phase_bounded_samples':len(selected),'unclassified_source_indices_retained':uncertain,'statistics':{k:stats(v) for k,v in metrics.items() if v}})
 return {'lanes_with_actual_host_samples':len(hosts),'reports':reports,'scope':'conservative phase-bounded monitoring calls; source observations not dropped; PSI avg10 reflects trailing global host pressure, not instantaneous request-specific waiting','sample_end_bound':'next monitoring call starts after preceding call and process-group memory sampling; last call has no recorded end bound','continuous_memory_highwater':'unverified','host_wait_HTTP_scheduler_causal_cost':'unmeasured; no attribution or performance sample exclusion','goal_achieved':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();j=analyze(a.archive);a.output.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'lanes':j['lanes_with_actual_host_samples'],'scope':j['scope']}))
