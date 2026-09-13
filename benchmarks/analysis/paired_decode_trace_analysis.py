"""Read-only accounting that separates within-pair and after-pair graph gaps."""
import argparse,collections,json,pathlib,sqlite3
from overlap_headroom import analyze,gap_accounting,quantiles


def main():
    ap=argparse.ArgumentParser();ap.add_argument('sqlite',type=pathlib.Path);ap.add_argument('output',type=pathlib.Path);args=ap.parse_args()
    report,launches=analyze(args.sqlite);assert len(report['groups'])==1
    db=sqlite3.connect(args.sqlite.resolve().as_uri()+'?mode=ro',uri=True)
    kernels=collections.defaultdict(list)
    for start,end,correlation,name in db.execute('SELECT k.start,k.end,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id'):
        kernels[correlation].append((start,end,name))
    launches.sort(key=lambda r:r['gpu_start']);chosen=launches[len(launches)//10:len(launches)*9//10];assert len(chosen)>2
    for row in chosen:
        names=[x[2] for x in kernels[row['correlation']]]
        row['future']=any('riley_future_token::resolve' in name for name in names)
        row['stage']='future_decode' if row['future'] else ('ordinary_decode' if any('riley_shared32_model::embedding' in name for name in names) else 'prefill_or_mixed')
    gaps=collections.defaultdict(list);unsubmitted=collections.defaultdict(list);kernel_cost=collections.Counter();stage_spans=collections.defaultdict(list)
    for row in chosen:
        stage_spans[row['stage']].append(row['gpu_end']-row['gpu_start'])
        for start,end,name in kernels[row['correlation']]:kernel_cost[name]+=end-start
    for prev,cur in zip(chosen,chosen[1:]):
        if cur['future']:
            assert prev['stage']=='ordinary_decode','future launch lacks immediate ordinary decode predecessor'
            label='within_pair'
        elif prev['future']:label='after_pair'
        else:label='ordinary'
        gaps[label].append(cur['gpu_start']-prev['gpu_end'])
        unsubmitted[label].append(max(0,min(cur['gpu_start'],cur['cpu_start'])-prev['gpu_end']))
    api=collections.defaultdict(list);lo=chosen[0]['cpu_start'];hi=chosen[-1]['gpu_end'];tid=chosen[0]['global_tid'];assert all(r['global_tid']==tid for r in chosen)
    for start,end,name in db.execute('SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE r.globalTid=? AND r.start>=? AND r.end<=?',(tid,lo,hi)):
        api[name].append(end-start)
    out={'source':report,'scope':'middle 80 percent of graph launch count; includes profiling overhead and client pacing; no speedup claim','selected':gap_accounting(chosen),'stage_graph_spans':{k:quantiles(v) for k,v in stage_spans.items()},'gaps':{k:{'gpu_gap':quantiles(v),'post_gpu_until_cpu_launch':quantiles(unsubmitted[k])} for k,v in gaps.items()},'runtime_apis':{k:quantiles(v) for k,v in sorted(api.items(),key=lambda p:sum(p[1]),reverse=True)},'kernel_ms':{k:v/1e6 for k,v in kernel_cost.most_common()},'launches':chosen}
    db.close();args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['selected','stage_graph_spans','gaps']},indent=2))
if __name__=='__main__':main()
