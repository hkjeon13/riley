"""Report every verified repeat and all eight cells; never promote a partial screen."""
import argparse,json,re
from pathlib import Path
from summarize_baseline import flatten,stats

ENGINES=['v52','candidate','vllm']
METRICS=['throughput_tokens_s']+[m+'_'+s for m in ['ttft_ms','tpot_ms','e2e_ms'] for s in ['median','p95','p99']]
def direction_pass(metric,change):return change>=0 if metric=='throughput_tokens_s' else change<=0

def summarize(replay):
    assert replay['raw_replay_passed']
    cells={f'c{c}-{w}':{e:{} for e in ENGINES} for c in [1,8,16,32] for w in ['fixed','natural']}
    for lane in replay['lanes']:
        c,w,r,e=re.fullmatch(r'c(\d+)-(fixed|natural)-r(\d+)-(v52|candidate|vllm)',lane['case']).groups()
        cell=f'c{c}-{w}';repeat=int(r);assert repeat not in cells[cell][e]
        cells[cell][e][repeat]=lane['metrics']
    results=[]
    for cell,engines in cells.items():
        pairs=[];spread={};comparisons={}
        for repeat in sorted(set.intersection(*(set(engines[e]) for e in ENGINES))):
            rows={e:engines[e][repeat] for e in ENGINES};flat={e:flatten(rows[e]) for e in ENGINES}
            assert len({rows[e]['input_tokens'] for e in ENGINES})==1,'actual input count differs'
            assert len({rows[e]['output_tokens'] for e in ENGINES})==1,'actual output count differs'
            pair={'repeat':repeat,'absolute_metrics':flat,'input_tokens_each':rows['v52']['input_tokens'],'output_tokens_each':rows['v52']['output_tokens'],'comparisons':{}}
            for target,reference in [('candidate','v52'),('candidate','vllm'),('v52','vllm')]:
                label=target+'_vs_'+reference
                pair['comparisons'][label]={k:100*(flat[target][k]/flat[reference][k]-1) for k in flat[target].keys()&flat[reference].keys() if flat[reference][k]}
            pairs.append(pair)
        for e,rows in engines.items():
            data=[flatten(m) for _,m in sorted(rows.items())]
            spread[e]={k:stats([m[k] for m in data]) for k in set.intersection(*(set(m) for m in data))} if data else {}
        complete=all(set(engines[e])==set(range(4)) for e in ENGINES)
        if complete:
            for target,reference in [('candidate','v52'),('candidate','vllm'),('v52','vllm')]:
                label=target+'_vs_'+reference;comparisons[label]={}
                for metric in METRICS:
                    changes=[pair['comparisons'][label][metric] for pair in pairs]
                    change_of_means=100*(spread[target][metric]['mean']/spread[reference][metric]['mean']-1)
                    comparisons[label][metric]={'per_repeat_change_percent':changes,'descriptive_change_spread':stats(changes),'change_of_four_repeat_means_percent':change_of_means,'mean_direction_nonregression':direction_pass(metric,change_of_means),'every_repeat_direction_nonregression':all(direction_pass(metric,x) for x in changes)}
        minimum=None;challenge=None
        if complete:
            cv=comparisons['candidate_vs_vllm']
            minimum=all(cv[k]['mean_direction_nonregression'] for k in METRICS)
            challenge=cv['throughput_tokens_s']['change_of_four_repeat_means_percent']>=15 and cv['ttft_ms_median']['change_of_four_repeat_means_percent']<=-10 and cv['tpot_ms_median']['change_of_four_repeat_means_percent']<=-10 and minimum
        results.append({'cell':cell,'complete_four_three_engine_repeats':complete,'per_repeat_records':pairs,'all_available_repeat_statistics':spread,'complete_cell_comparisons':comparisons,'candidate_minimum_mean_screen':minimum,'candidate_challenge_mean_screen':challenge})
    complete=bool(replay['matrix_complete'] and all(r['complete_four_three_engine_repeats'] for r in results))
    return {'input_pins':replay['input_pins'],'source_commits':replay['source_commits'],'verified_lane_count':len(replay['lanes']),'expected_lanes':96,'matrix_complete':complete,'failure':replay['failure'],'pressure_policy':replay['pressure_policy'],'cells':results,'all_core_minimum_mean_screen':all(r['candidate_minimum_mean_screen'] is True for r in results) if complete else None,'all_core_challenge_mean_screen':all(r['candidate_challenge_mean_screen'] is True for r in results) if complete else None,'method':'all four repeats retained individually and described without pooling/excluding requests; mean-direction screen and every-repeat direction shown separately; no statistical confidence or adoption claim','remaining_gates':['quiet admission if diagnostic','independent correctness/model contracts','sustained load','cancellation','re-request','resource reclamation'],'adopted':False,'goal_achieved':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('replay',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=summarize(json.loads(a.replay.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['verified_lane_count','matrix_complete','all_core_minimum_mean_screen']}))
