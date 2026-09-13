"""Verify archived per-request evidence and derive the C32 cache comparison."""
import hashlib,json,pathlib,re,statistics,sys,tarfile
from paired_decode_serving_screen import summary
root=pathlib.Path(sys.argv[1]);archive=root/'evidence.tar.gz';prefix='prefix-cache-serving-c32-v1/'
with tarfile.open(archive) as tar:
    def read(name):return tar.extractfile(prefix+name).read()
    def load(name):return json.loads(read(name))
    assert load('complete.json')=={'lanes':16,'all_complete':True}
    preparation=load('preparation.json');assert preparation['concurrency']==32
    assert preparation['kv_payload_bytes_per_engine']==754974720
    assert preparation['controller_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('prefix_cache_serving_screen.py').read_bytes()).hexdigest()
    assert preparation['client_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py').read_bytes()).hexdigest()
    fixtures=load('fixtures.json');assert len(fixtures['shared'])==32
    assert len({tuple(f['prompt_token_ids'][:16]) for f in fixtures['unique']})==len(fixtures['unique'])
    records=load('progress.json');assert len(records)==16;compared=[];agreements={};telemetry={}
    for record in records:
        name=record['name'];lane=name.rsplit('-',1)[1];launch=load(name+'-launch.json');argv=launch['argv']
        option=lambda key:argv[argv.index(key)+1]
        if lane=='vllm':
            assert option('--kv-cache-memory-bytes')=='754974720' and option('--block-size')=='16'
            assert '--enable-prefix-caching' in argv and option('--max-num-seqs')=='32'
            assert 'GPU KV cache size: 32,768 tokens' in read(name+'.log').decode()
        else:
            assert option('--kv-blocks')=='2048' and option('--max-active-sequences')=='32'
            assert launch['environment']['RILEY_PREFIX_CACHE_PAGES']==('512' if lane=='cache' else '0')
        for phase,count in [('warmup',64),('retained',256)]:
            rows=load(name+'-'+phase+'.json');assert len(rows)==count
            assert all(r['valid'] and len(r['token_ids'])==32 and r['checks']['prompt'] and r['checks']['finish'] for r in rows)
            if lane!='vllm':assert all(all(r['checks'].values()) for r in rows)
        stats=summary(rows);assert all(record[k]==v for k,v in stats.items())
        agreements[name]={k:sum(r['checks'][k] for r in rows) for k in ('prompt','tokens','text','finish')}
        if lane=='cache':
            match=re.search(r'RILEY_PREFIX_CACHE entries=(\d+) pages=(\d+) hits=(\d+) reused_tokens=(\d+) host_capacity_bytes=(\d+)',read(name+'.log').decode())
            assert match;values=list(map(int,match.groups()));assert values[1]<=512
            assert (values[2]==0) if name.startswith('unique') else (values[2]>0)
            telemetry[name]=dict(zip(('entries','pages','hits','reused_tokens','host_capacity_bytes'),values))
    for kind in ('shared','unique'):
        for lane in ('prior','off','cache','vllm'):
            group=[r for r in records if r['name'].startswith(kind+'-') and r['name'].endswith('-'+lane)];assert len(group)==2
            row={'workload':kind,'lane':lane,'throughput_tokens_s':statistics.median(r['throughput_tokens_s'] for r in group)}
            for metric,q in [('ttft','0.5'),('tpot','0.5'),('e2e','0.95'),('e2e','0.99')]:
                row[metric+'_'+q+'_ms']=statistics.median(r[metric+'_ms'][q] for r in group)
            compared.append(row)
    result={'scope':'closed-loop C32 screen; two reversed-order runs; not overall qualification','aggregation':'median of two lane throughput / percentile estimates, not pooled percentiles','retained_requests':4096,'retained_tokens':131072,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'preparation':preparation,'prompt_token_ranges':{k:[min(len(f['prompt_token_ids']) for f in v),max(len(f['prompt_token_ids']) for f in v)] for k,v in fixtures.items()},'comparison':compared,'reference_agreement':agreements,'cache_telemetry_including_warmup':telemetry}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in compared:print(r)
