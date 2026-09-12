import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
result={'schema_version':'riley.optimization-batch-comparison.v1','scope':'SmolLM2-135M BF16 c1 P128 O32 GUI retained','goal_complete':False,'modes':{}}
for mode in ['http','engine']:
 paths=[root/(name+'-'+mode)/'summary.json' for name in ['baseline','candidate']]
 if not all(x.exists() for x in paths):continue
 summaries=[json.loads(x.read_text()) for x in paths];assert all(s['completed'] and len(s['pair_results'])==5 for s in summaries)
 metrics=['e2e_ms','first_text_event_ms'] if mode=='http' else ['e2e_ms','ttft_ms','tpot_ms']
 out={'paired_processes_per_campaign':10,'measured_requests_per_lane':150,'metrics':{}}
 for metric in metrics:
  vals=[[statistics.median(pair[role][metric]['median'] for pair in s['pair_results']) for role in ['riley','vllm']] for s in summaries]
  out['metrics'][metric]={'baseline_riley':vals[0][0],'baseline_vllm':vals[0][1],'candidate_riley':vals[1][0],'candidate_vllm':vals[1][1],'candidate_change_pct':100*(vals[1][0]/vals[0][0]-1),'candidate_vs_vllm_ratio':vals[1][0]/vals[1][1]}
 throughput='output_tokens_per_wall_second' if mode=='http' else 'output_tokens_per_request_service_second'
 vals=[[statistics.median(pair[role][throughput] for pair in s['pair_results']) for role in ['riley','vllm']] for s in summaries]
 out['output_tokens_per_second']={'boundary':throughput,'baseline_riley':vals[0][0],'baseline_vllm':vals[0][1],'candidate_riley':vals[1][0],'candidate_vllm':vals[1][1],'candidate_change_pct':100*(vals[1][0]/vals[0][0]-1)}
 for metric in metrics:
  out['metrics'][metric]['tails']={q:{name:statistics.median(pair['riley'][metric][q] for pair in s['pair_results']) for name,s in zip(['baseline','candidate'],summaries)} for q in ['p95','p99']}
 result['modes'][mode]=out
print(json.dumps(result,indent=2))
