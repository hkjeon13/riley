"""Verify archived per-request evidence and derive the C32 cache comparison."""
import hashlib,json,pathlib,re,statistics,sys,tarfile
from paired_decode_serving_screen import summary
root=pathlib.Path(sys.argv[1]);archive=root/'evidence.tar.gz';prefix=('projection-serving-c64-v1/' if root.name.endswith('c64') else 'projection-serving-c8-v1/' if root.name.endswith('c8') else 'projection-serving-c32-v1/')
with tarfile.open(archive) as tar:
    def read(name):return tar.extractfile(prefix+name).read()
    def load(name):return json.loads(read(name))
    assert load('complete.json')=={'lanes':16,'all_complete':True}
    preparation=load('preparation.json');concurrency=preparation['concurrency'];assert concurrency in (8,32,64);active=preparation.get('active_capacity',concurrency)
    assert preparation['kv_payload_bytes_per_engine']==754974720
    assert preparation['controller_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('projection_pipeline_serving_screen.py').read_bytes()).hexdigest()
    measured_client=read('client-snapshot.py')
    assert preparation['client_sha256']==hashlib.sha256(measured_client).hexdigest()
    canonical_client=pathlib.Path(__file__).with_name('paired_decode_serving_screen.py').read_bytes()
    # Only the helper functions are imported; the older standalone CLI is unused.
    # Require the imports and every helper byte to match the canonical client.
    assert measured_client.split(b'def main():',1)[0]==canonical_client.split(b'def main():',1)[0]

    fixtures=load('fixtures.json');assert len(fixtures['shared'])==32
    assert len({tuple(f['prompt_token_ids'][:16]) for f in fixtures['unique']})==len(fixtures['unique'])
    records=load('progress.json');assert len(records)==16;compared=[];agreements={};telemetry={}
    for record in records:
        name=record['name'];lane=name.rsplit('-',1)[1];launch=load(name+'-launch.json');argv=launch['argv']
        option=lambda key:argv[argv.index(key)+1]
        if lane=='vllm':
            assert option('--kv-cache-memory-bytes')=='754974720' and option('--block-size')=='16'
            assert '--enable-prefix-caching' in argv and option('--max-num-seqs')==str(active)
            assert 'GPU KV cache size: 32,768 tokens' in read(name+'.log').decode()
        else:
            assert option('--kv-blocks')=='2048' and option('--max-active-sequences')==str(active)
            assert launch['environment']['RILEY_PREFIX_CACHE_PAGES']=='512'
            assert launch['environment']['RILEY_ROLLING_DECODE']=='1'
            assert launch['environment']['RILEY_PREFILL_PROJECTION_PIPELINE']==('1' if lane=='projection' else '0')
            assert ('RILEY_PROJECTION_PIPELINE prepared=true' in read(name+'.log').decode())==(lane=='projection')
            if lane!='vllm':
                progress=re.search(r'RILEY_ROLLING_DECODE completed_steps=(\d+) drains=(\d+)',read(name+'.log').decode())
                assert progress and int(progress[1])>0 and int(progress[2])>0
            assert launch['environment']['RILEY_MIXED_QUERY_REUSE']=='0'
            assert option('--decode-window')=='paired-experimental-v1' and option('--ffn-backend')=='prefill-pipeline-experimental-v1'
            assert option('--decode-projection')=='adaptive-rows-experimental-v1'
            log=read(name+'.log').decode()
            assert 'RILEY_QUERY_REUSE prepared=true' not in log
            windows=re.search(r'RILEY_DECODE_WINDOW completed=(\d+) widest=(\d+)',log)
            assert windows and int(windows[1])>0
        assert load(name+'-exit.json')['exit_code']==0
        for phase,count in [('warmup',64),('retained',256)]:
            rows=load(name+'-'+phase+'.json');assert len(rows)==count
            assert all(r['valid'] and len(r['token_ids'])==32 and r['checks']['prompt'] and r['checks']['finish'] for r in rows)
            if lane!='vllm':assert all(all(r['checks'].values()) for r in rows)
        stats=summary(rows);assert all(record[k]==v for k,v in stats.items())
        agreements[name]={k:sum(r['checks'][k] for r in rows) for k in ('prompt','tokens','text','finish')}
        if lane in ('projection','prior'):
            match=re.search(r'RILEY_PREFIX_CACHE entries=(\d+) pages=(\d+) hits=(\d+) reused_tokens=(\d+) host_capacity_bytes=(\d+)',read(name+'.log').decode())
            assert match;values=list(map(int,match.groups()));assert values[1]<=512
            assert (values[2]==0) if name.startswith('unique') else (values[2]>0)
            telemetry[name]=dict(zip(('entries','pages','hits','reused_tokens','host_capacity_bytes'),values))
    stop_reference=None
    for lane in ('prior','control','projection'):
        name='shared-p0-'+lane
        stopped=load(name+'-stop.json');assert len(stopped)==concurrency
        assert all(r['valid'] and r['finish_reason']=='stop' for r in stopped)
        identities=[(r['token_ids'],r['text'],r['finish_reason'],r['usage']) for r in stopped]
        if stop_reference is None:stop_reference=identities
        else:assert identities==stop_reference
        cancelled=load(name+'-cancel.json');assert len(cancelled)==concurrency
        assert all(r['connection_closed_before_output_budget'] and 4<=r['received_tokens']<32 for r in cancelled)
        recovery=load(name+'-recovery.json');assert len(recovery)==concurrency
        assert all(r['valid'] and all(r['checks'].values()) for r in recovery)
    for kind in ('shared','unique'):
        for lane in ('prior','control','projection','vllm'):
            group=[r for r in records if r['name'].startswith(kind+'-') and r['name'].endswith('-'+lane)];assert len(group)==2
            row={'workload':kind,'lane':lane,'throughput_tokens_s':statistics.median(r['throughput_tokens_s'] for r in group)}
            for metric,q in [('ttft','0.5'),('tpot','0.5'),('e2e','0.95'),('e2e','0.99'),('itl','0.5'),('itl','0.95'),('itl','0.99')]:
                row[metric+'_'+q+'_ms']=statistics.median(r[metric+'_ms'][q] for r in group)
            compared.append(row)
    result={'scope':f'closed-loop C{concurrency} screen; two reversed-order runs; not overall qualification','aggregation':'median of two lane throughput / percentile estimates, not pooled percentiles','retained_requests':4096,'retained_tokens':131072,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'preparation':preparation,'prompt_token_ranges':{k:[min(len(f['prompt_token_ids']) for f in v),max(len(f['prompt_token_ids']) for f in v)] for k,v in fixtures.items()},'comparison':compared,'reference_agreement':agreements,'cache_telemetry_including_warmup':telemetry}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in compared:print(r)
