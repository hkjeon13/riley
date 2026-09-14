"""Reconstruct SSE checks and run-level metrics for the speculative screen."""
import hashlib,json,statistics,tarfile
from pathlib import Path
from paired_decode_serving_screen import summary
from serving_evidence_validation import validate_row
root=Path(__file__).resolve().parents[2]
d=root/'benchmarks/results/20260914-speculative-serving-c32'
m=json.loads((d/'manifest.json').read_text())
assert hashlib.sha256((d/'evidence.tar.gz').read_bytes()).hexdigest()==m['archive_sha256']
with tarfile.open(d/'evidence.tar.gz') as t:files={p.name:t.extractfile(p).read() for p in t.getmembers() if p.isfile()}
assert set(files)==set(m['files'])
for n,h in m['files'].items():assert hashlib.sha256(files[n]).hexdigest()==h,n
for n,h in m['sources'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
for n,h in m['local_evidence'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,n
assert '72 passed; 0 failed' in (d/'scheduler-tests.log').read_text()
assert '7 passed; 0 failed' in (d/'runtime-tests.log').read_text()
assert '69 passed; 0 failed; 1 ignored' in (d/'server-tests.log').read_text()
load=lambda n:json.loads(files[n])
b='serving/'
assert load(b+'complete.json')=={'lanes':16,'all_complete':True}
fixtures=load(b+'fixtures.json');reported={x['name']:x for x in load(b+'progress.json')}
results={};total=0
for kind in ['shared','unique']:
    byid={x['id']:x for x in fixtures[kind]}
    for lane in ['prior','control','speculative','vllm']:
        stats=[]
        for pair in [0,1]:
            name=f'{kind}-p{pair}-{lane}'
            for phase,count in [('warmup',64),('retained',512)]:
                rows=load(b+name+'-'+phase+'.json');assert len(rows)==count
                for row in rows:
                    checks=validate_row(row,byid[row['id']]);assert len(row['token_ids'])==32
                    if lane!='vllm':assert all(checks.values())
                total+=count
                if phase=='retained':
                    value=summary(rows);assert value=={k:v for k,v in reported[name].items() if k!='name'};stats.append(value)
            assert load(b+name+'-exit.json')['exit_code']==0
        results[kind+'-'+lane]={
            'tokens_s':statistics.median(x['throughput_tokens_s'] for x in stats),
            **{key:statistics.median(x[group+'_ms'][p] for x in stats) for key,group,p in [('ttft50_ms','ttft','0.5'),('tpot50_ms','tpot','0.5'),('e2e95_ms','e2e','0.95'),('e2e99_ms','e2e','0.99'),('itl99_ms','itl','0.99')]}}
    for lane in ['prior','control','speculative']:
        if kind!='shared':continue
        name=f'{kind}-p0-{lane}'
        rows=load(b+name+'-stop.json');assert len(rows)==32
        for row in rows:validate_row(row,byid[row['id']]);assert row['finish_reason']=='stop'
        ids=[(r['token_ids'],r['text'],r['finish_reason'],r['usage']) for r in rows]
        if lane=='prior':stop_reference=ids
        else:assert ids==stop_reference
        cancels=load(b+name+'-cancel.json');assert len(cancels)==32
        assert all(x['connection_closed_before_output_budget'] and 4<=x['received_tokens']<32 for x in cancels)
        rows=load(b+name+'-recovery.json');assert len(rows)==32
        assert all(all(validate_row(r,byid[r['id']]).values()) for r in rows)
assert load('lifecycle/blender-restored.json')['restored']
assert load('lifecycle/execution.json')==[{'name':'serving','exit_code':0}]
assert hashlib.sha256((d/'profile-numeric.tar.gz').read_bytes()).hexdigest()==m['profile_archive_sha256']
with tarfile.open(d/'profile-numeric.tar.gz') as t:profile={p.name:json.loads(t.extractfile(p).read()) for p in t.getmembers() if p.isfile()}
assert profile['lifecycle/blender-restored.json']['restored']
assert profile['lifecycle/execution.json']==[{'name':'profile','exit_code':0}]
for n,h in profile['provenance.json']['scripts'].items():assert hashlib.sha256((root/'benchmarks/analysis'/n).read_bytes()).hexdigest()==h,n
for lane in ['control-shared','speculative-shared','speculative-unique','control-unique']:
    v=profile[lane+'-receipt.json'];assert v['exit_code']==0 and v['reference_exact'] and not v['remaining_owned_pids']
    assert v['peak_owned_tree_rss_bytes']<8*1024**3
    kind=lane.split('-')[1];byid={x['id']:x for x in fixtures[kind]}
    for phase in ['warmup','rows']:
        rows=profile[lane+'-'+phase+'.json'];assert len(rows)==64
        assert all(all(validate_row(r,byid[r['id']]).values()) for r in rows)
    analysis=profile[lane+'-areas.json']
    if lane.startswith('speculative'):
        v=analysis['stages']['verification'];assert v['launches']>0
        assert 0.55<v['areas']['attention']['kernel_fraction']<0.61
        assert v['areas']['output_head']['kernel_fraction']<0.03
receipt={'requests_including_warmup':total,'aggregation':'median of two run-level estimates; not pooled percentiles','results':results}
(d/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
