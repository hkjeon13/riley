"""Summarize independently replayed lanes without pooling or excluding repeats."""
import argparse
import json
import re
import statistics
from pathlib import Path

def stats(values):
    mean=statistics.mean(values)
    sd=statistics.stdev(values) if len(values)>1 else None
    return {'n':len(values),'mean':mean,'median':statistics.median(values),
            'min':min(values),'max':max(values),'sample_sd':sd,
            'cv_percent':100*sd/mean if sd is not None and mean else None}

def flatten(m):
    out={k:m[k] for k in ['throughput_tokens_s','error_rate','gpu_peak_observed_MiB']}
    if 'cpu_process_group_peak_observed_RSS_MiB' in m:
        out['cpu_process_group_peak_observed_RSS_MiB']=m['cpu_process_group_peak_observed_RSS_MiB']
    for metric in ['ttft_ms','tpot_ms','e2e_ms']:
        for stat in ['median','p95','p99']:out[metric+'_'+stat]=m[metric][stat]
    return out

def summarize(replay):
    assert replay['raw_replay_passed']
    cells={}
    for lane in replay['lanes']:
        c,w,r,e=re.fullmatch(r'c(\d+)-(fixed|natural)-r(\d+)-(v52|vllm)',lane['case']).groups()
        cells.setdefault('c'+c+'-'+w,{}).setdefault(e,{})[int(r)]=lane['metrics']
    output=[]
    for key,engines in cells.items():
        pairs=[]
        for repeat in sorted(set(engines.get('v52',{}))&set(engines.get('vllm',{}))):
            a=flatten(engines['v52'][repeat]);b=flatten(engines['vllm'][repeat])
            pairs.append({'repeat':repeat,'v52':a,'vllm':b,
                'v52_change_percent':{k:100*(a[k]/b[k]-1) for k in a.keys()&b.keys() if b[k]},
                'actual_input_tokens_equal':engines['v52'][repeat]['input_tokens']==engines['vllm'][repeat]['input_tokens'],
                'actual_output_tokens_equal':engines['v52'][repeat]['output_tokens']==engines['vllm'][repeat]['output_tokens']})
        summaries={}
        for e,rows in engines.items():
            flat=[flatten(m) for _,m in sorted(rows.items())]
            summaries[e]={k:stats([x[k] for x in flat]) for k in set.intersection(*(set(x) for x in flat))}
        output.append({'cell':key,'paired_repeats':len(pairs),'pairs':pairs,'all_available_repeat_statistics':summaries,
                       'complete_four_pairs':len(pairs)==4})
    return {'raw_replay_passed':True,'matrix_complete':replay['matrix_complete'],
            'completed_lanes':len(replay['lanes']),'expected_lanes':replay['expected_lanes'],
            'failure':replay['failure'],'cells':output,'scope':replay['qualification_scope'],
            'aggregation':'per-repeat comparisons and descriptive spread; no request pooling, exclusions or confidence claim',
            'performance_promoted':False,'goal_achieved':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('replay',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=summarize(json.loads(a.replay.read_text()))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed_lanes':result['completed_lanes'],'complete_cells':sum(c['complete_four_pairs'] for c in result['cells'])}))
