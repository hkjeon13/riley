"""Read exported trace intervals without treating overlapping times as additive."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

def merged(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<a:raise ValueError('negative trace interval')
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out

def length(intervals):return sum(b-a for a,b in intervals)

def intersection(a,b):
    i=j=total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total

def analyze(path):
    with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True) as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        schema={t:[row[1] for row in db.execute('PRAGMA table_info("'+t.replace('"','""')+'")')] for t in tables}
        names=dict(db.execute('SELECT id,value FROM StringIds')) if 'StringIds' in tables else {}
        scopes={};unions={};kernel_names=collections.defaultdict(list);calls_by_name={}
        for label,table in [('kernel','CUPTI_ACTIVITY_KIND_KERNEL'),('memcpy','CUPTI_ACTIVITY_KIND_MEMCPY'),
                            ('memset','CUPTI_ACTIVITY_KIND_MEMSET'),('cuda_api','CUPTI_ACTIVITY_KIND_RUNTIME'),
                            ('driver_api','CUPTI_ACTIVITY_KIND_DRIVER'),('osrt','OSRT_API')]:
            if table not in tables or not {'start','end'}<=set(schema[table]):
                scopes[label]={'state':'unavailable','table':table};continue
            columns=schema[table];name_column=next((x for x in ['demangledName','shortName','nameId'] if x in columns),None)
            selected='start,end'+(','+name_column if name_column else '')
            intervals=[];named=collections.defaultdict(list)
            for row in db.execute('SELECT '+selected+' FROM '+table):
                start,end=row[:2];intervals.append((start,end))
                if name_column:named[names.get(row[2],str(row[2]))].append((start,end))
                if label=='kernel':kernel_names[names.get(row[2],str(row[2])) if name_column else 'unresolved'].append((start,end))
            union=merged(intervals);unions[label]=union
            scopes[label]={'state':'measured' if intervals else 'no recorded events','events':len(intervals),
                          'summed_event_duration_ns':length(intervals),'interval_union_ns':length(union),
                          'overlap_duration_ns':length(intervals)-length(union),
                          'first_start_ns':union[0][0] if union else None,'last_end_ns':union[-1][1] if union else None}
            if label in ['cuda_api','driver_api','osrt']:
                calls_by_name[label]=sorted([{'name':name,'count':len(spans),
                    'summed_event_duration_ns':length(spans),'interval_union_ns':length(merged(spans))}
                    for name,spans in named.items()],key=lambda x:x['summed_event_duration_ns'],reverse=True)
        kernels=[]
        for name,intervals in kernel_names.items():
            durations=[b-a for a,b in intervals]
            kernels.append({'name':name,'count':len(intervals),'summed_event_duration_ns':sum(durations),
                'median_duration_ns':statistics.median(durations),'max_duration_ns':max(durations),
                'interval_union_ns':length(merged(intervals))})
        kernels.sort(key=lambda x:x['summed_event_duration_ns'],reverse=True)
        metadata={}
        for t in sorted(tables):
            if t.startswith(('TARGET_INFO','EXPORT_','META_DATA')):
                rows=db.execute('SELECT * FROM "'+t.replace('"','""')+'" LIMIT 201').fetchall()
                metadata[t]={'columns':schema[t],'rows':[[v.hex() if isinstance(v,bytes) else v for v in row] for row in rows[:200]],'truncated':len(rows)>200}
        diagnostics=[dict(zip(schema['DIAGNOSTIC_EVENT'],row)) for row in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else None
    return {'sqlite_path':str(path),'schema':schema,'metadata':metadata,'interval_scopes':scopes,
        'diagnostics':diagnostics,'calls_by_name':calls_by_name,
        'pairwise_overlap_ns':{a+'__'+b:intersection(unions[a],unions[b]) for a in unions for b in unions if a<b},
        'kernels_by_summed_work':kernels,
        'gpu_recorded_intervals_union_ns':length(merged([i for k in ['kernel','memcpy','memset'] for i in unions.get(k,[])])) if any(k in unions for k in ['kernel','memcpy','memset']) else None,
        'gpu_recorded_categories':[k for k in ['kernel','memcpy','memset'] if k in unions],
        'scope':'entire recorded trace; HTTP retained-envelope alignment is unverified until clock metadata is checked',
        'attribution_limits':['event sums are work totals, not additive wall time','CUDA API/OSRT durations may overlap GPU execution and other threads',
            'CPU stacks/context switches unavailable','unattributed gaps are not proof of host wait or HTTP cost'],
        'serving_performance':'미실행; profiler measurements are diagnostic only','goal_achieved':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=analyze(a.sqlite);h=hashlib.sha256()
    with a.sqlite.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    result['sqlite_sha256']=h.hexdigest();a.output.write_text(json.dumps(result,indent=2)+'\n')
