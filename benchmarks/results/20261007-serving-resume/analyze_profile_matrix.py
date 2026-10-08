"""Analyze every archived trace locally; original archive remains intact."""
import collections,hashlib,json,tarfile,tempfile
from pathlib import Path
from analyze_profile_intervals import analyze
from summarize_baseline import stats
ROOT=Path(__file__).parent
archive=ROOT/'profiling-attempt04.tar.gz'
OUT=ROOT/'profile-matrix-analysis';OUT.mkdir(exist_ok=False)
results=[]
with tempfile.TemporaryDirectory(prefix='riley-matrix-profile-') as tmp:
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('/trace.sqlite'):continue
            label=member.name.split('/')[-2];path=Path(tmp)/'trace.sqlite';h=hashlib.sha256()
            with path.open('wb') as f:
                source=tar.extractfile(member)
                for b in iter(lambda:source.read(1024*1024),b''):f.write(b);h.update(b)
            result=analyze(path);result['sqlite_sha256']=h.hexdigest();result['archive_member']=member.name
            (OUT/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            work=sum(k['summed_event_duration_ns'] for k in result['kernels_by_summed_work'])
            kernels=[{'name':k['name'],'work_share_percent':100*k['summed_event_duration_ns']/work,**{x:k[x] for x in ['count','median_duration_ns','summed_event_duration_ns']}} for k in result['kernels_by_summed_work']]
            results.append({'case':label,'interval_scopes':result['interval_scopes'],'GPU_union_ns':result['gpu_recorded_intervals_union_ns'],'overlaps':result['pairwise_overlap_ns'],'kernels':kernels})
            print(label+' analyzed',flush=True)
assert len(results)==32
cells={}
for row in results:
    parts=row['case'].split('-');cell='-'.join(parts[:2]);engine=parts[-1];cells.setdefault(cell,{}).setdefault(engine,[]).append(row)
summary=[]
for cell,engines in sorted(cells.items()):
    report={'cell':cell,'engines':{}}
    for engine,rows in engines.items():
        assert len(rows)==2
        names={k['name'] for r in rows for k in r['kernels']};by_name=[]
        for name in names:
            shares=[next((k['work_share_percent'] for k in r['kernels'] if k['name']==name),0) for r in rows]
            by_name.append({'name':name,'per_order_work_share_percent':shares,'work_share_descriptive_statistics':stats(shares)})
        by_name.sort(key=lambda r:r['work_share_descriptive_statistics']['mean'],reverse=True)
        report['engines'][engine]={'order_cases':[r['case'] for r in rows],'kernel_interval_union_ns':stats([r['interval_scopes']['kernel']['interval_union_ns'] for r in rows]),'GPU_interval_union_ns':stats([r['GPU_union_ns'] for r in rows]),'kernels_by_work_share':by_name}
    summary.append(report)
(OUT/'summary.json').write_text(json.dumps({'profiles_analyzed':32,'cells':summary,'scope':'entire instrumented capture; work shares are kernel work, not serving improvement estimates','no_additive_API_GPU_walltime':True,'CPU_stacks_context_switches':'미측정','HTTP_scheduler_attribution':'미측정','serving_performance':'미실행; diagnostic profiles only','goal_achieved':False},indent=2)+'\n')
