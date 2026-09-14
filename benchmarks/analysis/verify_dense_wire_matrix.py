"""Reconstruct dense-wire concurrency evidence; reject incomplete runs."""
import argparse,hashlib,json,statistics,tarfile
from pathlib import Path
from paired_decode_serving_screen import summary
from serving_evidence_validation import validate_row
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
parser.add_argument('--concurrency',type=int,required=True,choices=[8,16,32,64])
parser.add_argument('--retained',type=int,default=2048)
parser.add_argument('--require-cooldown',action='store_true')
args=parser.parse_args()
assert args.retained>=128
d=args.directory
m=json.loads((d/'manifest.json').read_text())
assert hashlib.sha256((d/'vllm029-packages.txt').read_bytes()).hexdigest()==m['vllm029_packages_sha256']
assert 'vllm==0.29.0' in (d/'vllm029-packages.txt').read_text().splitlines()
assert hashlib.sha256((d/'evidence.tar.gz').read_bytes()).hexdigest()==m['archive_sha256']
with tarfile.open(d/'evidence.tar.gz') as t:files={p.name:t.extractfile(p).read() for p in t.getmembers() if p.isfile()}
assert set(files)==set(m['files'])
for n,h in m['files'].items():assert hashlib.sha256(files[n]).hexdigest()==h,n
assert hashlib.sha256((d/'source-snapshot.tar.gz').read_bytes()).hexdigest()==m['source_snapshot_sha256']
with tarfile.open(d/'source-snapshot.tar.gz') as t:sources={p.name:t.extractfile(p).read() for p in t.getmembers() if p.isfile()}
assert set(sources)==set(m['sources'])
for n,h in m['sources'].items():assert hashlib.sha256(sources[n]).hexdigest()==h,n
load=lambda n:json.loads(files[n])
b='serving/'
assert load(b+'complete.json')=={'lanes':12,'all_complete':True}
fixtures=load(b+'fixtures.json');reported={x['name']:x for x in load(b+'progress.json')}
results={};total=0
for kind in ['shared','unique']:
    byid={x['id']:x for x in fixtures[kind]}
    for lane in ['prior','candidate','vllm029']:
        stats=[]
        for pair in [0,1]:
            name=f'{kind}-p{pair}-{lane}'
            for phase,count in [('warmup',64),('retained',args.retained)]:
                rows=load(b+name+'-'+phase+'.json');assert len(rows)==count
                for row in rows:
                    checks=validate_row(row,byid[row['id']]);assert len(row['token_ids'])==32
                    if not lane.startswith('vllm'):assert all(checks.values())
                total+=count
                if phase=='retained':
                    value=summary(rows);assert value=={k:v for k,v in reported[name].items() if k!='name'};stats.append(value)
            assert load(b+name+'-exit.json')['exit_code']==0
        results[kind+'-'+lane]={
            'tokens_s':statistics.median(x['throughput_tokens_s'] for x in stats),
            **{key:statistics.median(x[group+'_ms'][p] for x in stats) for key,group,p in [('ttft50_ms','ttft','0.5'),('tpot50_ms','tpot','0.5'),('e2e95_ms','e2e','0.95'),('e2e99_ms','e2e','0.99'),('itl99_ms','itl','0.99')]}}
    for lane in ['prior','candidate']:
        if kind!='shared':continue
        name=f'{kind}-p0-{lane}'
        rows=load(b+name+'-stop.json');assert len(rows)==args.concurrency
        for row in rows:validate_row(row,byid[row['id']]);assert row['finish_reason']=='stop'
        ids=[(r['token_ids'],r['text'],r['finish_reason'],r['usage']) for r in rows]
        if lane=='prior':stop_reference=ids
        else:assert ids==stop_reference
        cancels=load(b+name+'-cancel.json');assert len(cancels)==args.concurrency
        assert all(x['connection_closed_before_output_budget'] and 4<=x['received_tokens']<32 for x in cancels)
        rows=load(b+name+'-recovery.json');assert len(rows)==args.concurrency
        assert all(all(validate_row(r,byid[r['id']]).values()) for r in rows)
assert load('lifecycle/blender-restored.json')['restored']
assert load('lifecycle/execution.json')==[{'name':'serving','exit_code':0}]
preflight=load('serving/preflight-vllm029-rows.json')
assert len(preflight)==32
byid={x['id']:x for x in fixtures['shared']}
for row in preflight:validate_row(row,byid[row['id']]);assert len(row['token_ids'])==32
assert load('serving/preflight-vllm029-exit.json')['exit_code']==0
versions=load('serving/versions.json')
assert versions['vllm029']['packages']['vllm']=='0.29.0'
for kind in ['shared','unique']:
    for pair in [0,1]:
        for lane in ['prior','candidate','vllm029']:
            launch=load(f'serving/{kind}-p{pair}-{lane}-launch.json')
            argv=launch['argv'];env=launch['environment']
            if args.require_cooldown:
                cooldown=load(f'serving/{kind}-p{pair}-{lane}-cooldown.json')
                assert cooldown['threshold_c']==48 and cooldown['timeout_s']==600
                assert cooldown['samples'] and cooldown['samples'][-1]['temperature_c']<=48
                assert all(x['temperature_c']>48 for x in cooldown['samples'][:-1])
                assert all(a['elapsed_s']<=b['elapsed_s'] for a,b in zip(cooldown['samples'],cooldown['samples'][1:]))
            active='--max-num-seqs' if lane=='vllm029' else '--max-active-sequences'
            assert int(argv[argv.index(active)+1])==min(args.concurrency,32)
            if lane in ['prior','candidate']:
                assert env['RILEY_ROLLING_DECODE']=='1' and env['RILEY_PREFILL_FFN_SPLIT']=='1'
                assert argv[argv.index('--kv-blocks')+1]=='2048'
            else:
                assert '--enable-prefix-caching' in argv
                assert argv[argv.index('--kv-cache-memory-bytes')+1]=='754974720'
                assert argv[argv.index('--dtype')+1]=='bfloat16'
                expected='vllm029-venv' if lane=='vllm029' else 'riley-vllm-interim.CfrT9T/venv'
                assert expected in argv[0]
assert load('build/exit.json')['exit_code']==0
assert load('build/runtime-ticket-tests-exit.json')['exit_code']==0
assert b'9 passed; 0 failed' in files['build/runtime-ticket-tests.log']
assert load('serving/preparation.json')['hashes']['candidate']==load('build/binary.json')['sha256']
receipt={'concurrency':args.concurrency,'retained_per_lane':args.retained,'requests_including_warmup':total,'aggregation':'median of two run-level estimates; not pooled percentiles','results':results}
(d/'matrix-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
