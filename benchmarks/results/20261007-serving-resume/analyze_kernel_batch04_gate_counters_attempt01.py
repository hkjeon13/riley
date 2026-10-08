import csv,json,statistics,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;C=R/'kernel-batch04-gate-counter-independent-collection-attempt01';D=C/'kernel-batch04-gate-counters-attempt02';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();prep=json.loads((D/'preparation.json').read_text());done=json.loads((D/'completion.json').read_text());assert done['failure'] is None and len(done['completed'])==8
receipt=json.loads((C/(D.name+'-receipt.json')).read_text());assert receipt['source_before']==receipt['source_after'] and sha(C/(D.name+'.tar.gz'))==receipt['archive_sha256'];assert json.loads((D/'source-integrity-after.json').read_text())['files']==prep['source_files'];lanes=[]
for label in done['completed']:
 L=D/label;launch=json.loads((L/'launch.json').read_text());process=json.loads((L/'process.json').read_text());export=json.loads((L/'export.json').read_text());assert process['exit']==0 and not process['timeout'];assert process['binary_sha256_after']==launch['binary_sha256']==prep['plan']['binary_sha256'];assert sha(L/'target.log')==process['log_sha256'] and sha(L/'trace.ncu-rep')==export['raw_profile_sha256'] and sha(L/'metrics.csv')==export['csv_sha256']
 data=list(csv.DictReader((L/'metrics.csv').open()));assert {x['ID'] for x in data}=={'0'};assert {x['Block Size'] for x in data}=={'(64, 1, 1)'} and {x['Grid Size'] for x in data}=={'(192, 1, 1)'};v52=label.endswith('v52');assert all(('shared32_gate_up_swiglu_v52<1>' if v52 else 'shared32_gate_up_swiglu<1>') in x['Kernel Name'] for x in data)
 metrics={}
 for x in data:
  if not x['Metric Name']:continue
  try:numeric=float(x['Metric Value'].replace(',',''))
  except ValueError:numeric=None
  metrics[x['Section Name']+'::'+x['Metric Name']]={'unit':x['Metric Unit'],'raw_value':x['Metric Value'],'value':numeric}
 rules=[{k:x[k] for k in ['Rule Name','Rule Type','Rule Description']} for x in data if x['Rule Name']];lanes.append({'label':label,'active_rows_from_source_order':launch['active_rows_from_frozen_source_invocation'],'metrics':metrics,'rules':rules})
keys=['Launch Statistics::Registers Per Thread','Occupancy::Achieved Occupancy','Scheduler Statistics::No Eligible','Scheduler Statistics::Eligible Warps Per Scheduler','Warp State Statistics::Warp Cycles Per Issued Instruction','Memory Workload Analysis::L2 Hit Rate','Memory Workload Analysis::Max Bandwidth'];cells=[]
for rows in [1,32]:
 for engine in ['v52','candidate']:
  subset=[x for x in lanes if x['label'].startswith(f'rows{rows}-') and x['label'].endswith(engine)];assert len(subset)==2;stats={}
  for key in keys:
   v=[x['metrics'][key]['value'] for x in subset];assert all(x is not None for x in v);mean=statistics.mean(v);sd=statistics.stdev(v);stats[key]={'values':v,'mean':mean,'sample_sd':sd,'cv_percent':100*sd/mean if mean else None,'unit':subset[0]['metrics'][key]['unit']}
  cells.append({'active_rows':rows,'engine':engine,'statistics':stats})
result={'independent_counter_raw_replay':'PASS8','source_commit':prep['plan']['source_commit'],'native_binary_sha256':prep['plan']['binary_sha256'],'lanes':lanes,'cells':cells,'archive_sha256':receipt['archive_sha256'],'archive_files_verified':receipt['files_verified'],'cache_clock_scope':prep['plan']['cache_state_limit'],'unmeasured':['actual HTTP model-weight counter profile','instruction/source-level stall samples; optional PCSAMP missing'],'serving_performance':'미실행','performance_claim_eligible':False,'goal_achieved':False};(C/'independent-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(cells))
