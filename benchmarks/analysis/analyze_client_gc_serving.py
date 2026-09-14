"""Correlate observed GC intervals and captured SSE timing, without correcting latency."""
import argparse,bisect,hashlib,json,pathlib
from paired_decode_serving_screen import summary,percentile
from serving_evidence_validation import validate_row

def analyze(root):
    read=lambda name:json.loads((root/name).read_text())
    assert read('complete.json')=={'lanes':4,'all_complete':True}
    prep=read('preparation.json')
    assert prep['retained']==8192 and prep['concurrency']==32 and prep['warmup']==256
    fixtures={f['id']:f for f in read('fixtures.json')['shared']}
    results=[]
    for name in ['shared-p0-control','shared-p0-vllm','shared-p1-vllm','shared-p1-control']:
        assert read(name+'-exit.json')['exit_code']==0
        rows=read(name+'-retained.json');gc=read(name+'-retained-gc.json');host=read(name+'-retained-host.json')
        assert len(rows)==8192 and gc['before']['enabled'] and not gc['unclosed_generations']
        for row in rows:
            checks=validate_row(row,fixtures[row['id']])
            assert name.endswith('vllm') or all(checks.values())
            assert host['before']['monotonic_ns']<=row['started_ns']<=row['ended_ns']<=host['after']['monotonic_ns']
        events=sorted(gc['events'],key=lambda e:e[1])
        for i,(gen,start,end,collected,uncollectable) in enumerate(events):
            assert 0<=gen<=2 and start<=end and host['before']['monotonic_ns']<=start<=end<=host['after']['monotonic_ns']
            assert i==0 or events[i-1][2]<=start
        for gen in range(3):
            assert gc['after']['stats'][gen]['collections']-gc['before']['stats'][gen]['collections']==sum(e[0]==gen for e in events)
        long=[e for e in events if e[2]-e[1]>=10_000_000]
        ends=[e[2] for e in long]
        def overlap(a,b):
            i=bisect.bisect_right(ends,a)
            return i<len(long) and long[i][1]<b
        e2e=[(r['ended_ns']-r['started_ns'])/1e6 for r in rows];p99=percentile(e2e,.99)
        slow=[r for r,t in zip(rows,e2e) if t>=p99]
        gaps=[(a,b) for r in rows for a,b in zip(r['arrivals_ns'],r['arrivals_ns'][1:]) if b-a>=10_000_000]
        results.append({'name':name,'summary':summary(rows),'gc_generations':[{ 'generation':g,'count':sum(e[0]==g for e in events),'total_ms':sum((e[2]-e[1])/1e6 for e in events if e[0]==g),'max_ms':max([(e[2]-e[1])/1e6 for e in events if e[0]==g],default=0)} for g in range(3)],'gc_at_least_10ms':len(long),'requests_overlapping_long_gc':sum(overlap(r['started_ns'],r['ended_ns']) for r in rows),'e2e_at_or_above_p99':len(slow),'p99_requests_overlapping_long_gc':sum(overlap(r['started_ns'],r['ended_ns']) for r in slow),'itl_at_least_10ms':len(gaps),'long_itl_overlapping_long_gc':sum(overlap(a,b) for a,b in gaps),'max_gc_ms':max([(e[2]-e[1])/1e6 for e in events],default=0),'process_cpu_s':gc['process_cpu_ns']/1e9,'peak_rss_mib':gc['ru_maxrss_after_kib']/1024})
    return {'qualification':False,'scope':'Observational shared-prefix C32; interval overlap is not causal attribution. No pause subtraction or GC policy change. Callback overhead is included.','preparation':prep,'lanes':results}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=pathlib.Path);parser.add_argument('output',type=pathlib.Path);args=parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.root),indent=2)+'\n')
