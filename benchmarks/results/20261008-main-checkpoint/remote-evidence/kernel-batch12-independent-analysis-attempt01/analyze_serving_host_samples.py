"""Describe all source host samples; never map unbridged clocks or exclude lanes."""
import argparse,hashlib,json,re,statistics,tarfile
from pathlib import Path
from summarize_baseline import stats
PATTERN=r'(c\d+-(?:fixed|natural)-r\d+-(?:v52|candidate|vllm))'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def pressure(sample,resource,kind):
 row=next(x for x in sample['pressure'][resource].splitlines() if x.startswith(kind+' '))
 return {k:float(v) for k,v in (part.split('=') for part in row.split()[1:])}
def analyze(archive):
 hosts={};gates={};preparation=None;controller=None
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   if not member.isfile():continue
   name=Path(member.name).name
   if name=='preparation.json':
    assert preparation is None;preparation=json.load(tar.extractfile(member))
   elif name=='controller-snapshot.py':controller=tar.extractfile(member).read()
   elif match:=re.fullmatch(PATTERN+'-host.json',name):
    case=match[1];assert case not in hosts;hosts[case]=json.load(tar.extractfile(member))
   elif match:=re.fullmatch(PATTERN+'-(start|warmup|retained)-host-gate.json',name):
    key=(match[1],match[2]);assert key not in gates;gates[key]=json.load(tar.extractfile(member))
 assert preparation is not None and controller is not None
 assert hashlib.sha256(controller).hexdigest()==preparation['pins'][preparation['controller_path']]
 policy=preparation['plan']['host_start_gate'];records=[]
 for case,samples in sorted(hosts.items()):
  assert samples,'empty actual lane host observations'
  values={k:[] for k in ['temperature_C','power_W','SM_clock_MHz','memory_MiB','GPU_utilization_percent','CPU_some_avg10','IO_full_avg10','memory_full_avg10','process_group_RSS_MiB']};uuids=set();exceeded={k:0 for k in ['CPU_some_avg10','IO_full_avg10','memory_full_avg10']}
  for sample in samples:
   gpu=[x.strip() for x in sample['gpu'].split(',')];assert len(gpu)==6;uuids.add(gpu[0])
   for metric,value in zip(['temperature_C','power_W','SM_clock_MHz','memory_MiB','GPU_utilization_percent'],gpu[1:]):values[metric].append(float(value))
   for metric,resource,kind,limit in [('CPU_some_avg10','cpu','some',policy['cpu_some_max']),('IO_full_avg10','io','full',policy['io_full_max']),('memory_full_avg10','memory','full',policy['memory_full_max'])]:
    value=pressure(sample,resource,kind)['avg10'];values[metric].append(value);exceeded[metric]+=value>limit
   if 'server_process_memory' in sample:values['process_group_RSS_MiB'].append(sample['server_process_memory']['rss_sum_KiB']/1024)
  actual_gates=[]
  for phase in ['start','warmup','retained']:
   gate=gates.get((case,phase))
   if gate is None:actual_gates.append({'phase':phase,'state':'absent; no pass inferred'});continue
   obs=gate['samples'];assert gate['policy']==policy
   actual_gates.append({'phase':phase,'passed':gate['passed'],'observations':len(obs),'epoch_sample_span_seconds':(obs[-1]['time_ns']-obs[0]['time_ns'])/1e9 if obs else None,'scope':'observed sample span; not exact gate wall duration; all samples retained'})
  times=[x['time_ns'] for x in samples]
  records.append({'case':case,'GPU_UUIDs_observed':sorted(uuids),'sample_count':len(samples),'statistics':{k:stats(v) for k,v in values.items() if v},'samples_above_admission_threshold':exceeded,'threshold_fraction_scope':'fraction of source samples over whole startup/warmup/gates/retained; not retained-duration fraction and not an exclusion rule','epoch_timestamp_decreases':sum(b<a for a,b in zip(times,times[1:])),'gates':actual_gates})
 orphan=[{'case':case,'phase':phase,'passed':gate['passed'],'sample_count':len(gate['samples'])} for (case,phase),gate in sorted(gates.items()) if case not in hosts]
 return {'archive':str(archive),'archive_sha256':sha(archive),'controller_snapshot_sha256':hashlib.sha256(controller).hexdigest(),'lanes_with_actual_host_samples':len(records),'lane_records':records,'gates_without_host_lane_file':orphan,'all_source_samples_retained':True,'clock_scope':'host time_ns is epoch; request/phase times are perf_counter; no captured clock bridge; retained-only sample alignment unverified','memory_scope':'sampled whole lane, no continuous high-water proof','causal_attribution':'unverified; these observations do not prove host wait, HTTP cost or a cause of throughput variance','performance_qualification':False,'adopted':False,'goal_achieved':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=analyze(a.archive);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'host_lanes':r['lanes_with_actual_host_samples'],'archive_sha256':r['archive_sha256'],'causal_attribution':r['causal_attribution']}))
