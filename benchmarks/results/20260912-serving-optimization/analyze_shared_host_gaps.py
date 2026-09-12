import pathlib,sqlite3,statistics,json
r=pathlib.Path('/tmp/riley-opt-260912');report={}
for version in (23,24):
 p=r/f'mixed-p128-nsys-v{version}-c4-client/c4.sqlite';c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 copies=list(c.execute('select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where graphNodeId>0 order by start'))
 starts=[a for a,b,k,n in copies if k==1 and n==17536];ends=[b for a,b,k,n in copies if k==2 and n==787456];assert len(starts)==len(ends)>1
 gaps=[(starts[i+1]-ends[i])/1000 for i in range(len(ends)-1)];assert min(gaps)>=0
 report[str(version)]={'steps':len(starts),'between_graph_gap_us_median':statistics.median(gaps),'between_graph_gap_us_p95':sorted(gaps)[int(len(gaps)*.95)],'between_graph_gap_ms_total':sum(gaps)/1000}
x={'scope':'Nonexclusive diagnostic; D2H completion to next H2D start includes host scheduling, sampling, transport and CUDA dispatch, not isolated sampling cost','versions':report}
(r/'shared-v24-host-gap.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
