"""Verify archived per-request evidence and derive the C32 cache comparison."""
import contextlib,hashlib,json,pathlib,re,statistics,sys,tarfile
from paired_decode_serving_screen import summary
from serving_evidence_validation import validate_row
from serving_comparison_assessment import assess
root=pathlib.Path(sys.argv[1]);prefix=('ffn-split-serving-c64-v2/' if root.name.endswith('c64') else 'ffn-split-serving-c8-v2/' if root.name.endswith('c8') else 'ffn-split-serving-c32-v2/')
archives=sorted((root/'evidence').glob('*.tar.gz'));assert len(archives)==17
manifest=json.loads((root/'evidence/manifest.json').read_text())
assert set(manifest)=={p.name for p in archives}
for archive in archives:
    assert archive.stat().st_size==manifest[archive.name]['bytes'] and archive.stat().st_size<64*1024**2
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest[archive.name]['sha256']
with contextlib.ExitStack() as stack:
    members={}
    for archive in archives:
        tar=stack.enter_context(tarfile.open(archive))
        for member in tar.getmembers():
            if member.isfile():
                assert member.name not in members, 'duplicate archive member'
                members[member.name]=(tar,member)
    def read(name):
        tar,member=members[prefix+name];return tar.extractfile(member).read()
    def load(name):return json.loads(read(name))
    assert load('complete.json')=={'lanes':16,'all_complete':True}
    assert load('blender-restored.json')['restored']
    assert all(x['exit_code']==0 for x in load('execution.json'))
    preparation=load('preparation.json');assert preparation['warmup']==256 and preparation['retained']==8192;concurrency=preparation['concurrency'];assert concurrency in (8,32,64);active=preparation.get('active_capacity',concurrency)
    assert preparation['hashes']=={'current':'658c00b7b99b57d54a12dd345dd44a3ba705fdffa6749687eed9a8e1a04306cc','prior':'3ae029380abc732a96f65e361548daef92a794ea9a74905dd6e245ccaf4786fc'}
    assert preparation['kv_payload_bytes_per_engine']==754974720
    assert preparation['readiness_timeout_seconds']==600
    assert preparation['client_treatment']=='gc-phase-disabled-v1'
    assert preparation['artifact_storage']=='tmpfs:/dev/shm'
    assert preparation['controller_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('ffn_split_serving_screen.py').read_bytes()).hexdigest()
    assert preparation['controller_sha256']==hashlib.sha256(read('controller-snapshot.py')).hexdigest()
    measured_client=read('client-snapshot.py')
    assert preparation['client_sha256']==hashlib.sha256(measured_client).hexdigest()
    canonical_client=pathlib.Path(__file__).with_name('paired_decode_serving_screen.py').read_bytes()
    # Only the helper functions are imported; the older standalone CLI is unused.
    # Require the imports and every helper byte to match the canonical client.
    assert measured_client.split(b'def main():',1)[0]==canonical_client.split(b'def main():',1)[0]

    fixtures=load('fixtures.json');assert len(fixtures['shared'])==32 and len(fixtures['unique'])==8448
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
            assert launch['environment']['RILEY_PREFILL_PROJECTION_PIPELINE']=='1'
            assert launch['environment']['RILEY_PREFILL_FFN_ADAPTIVE']==('1' if lane=='prior' else '0')
            assert launch['environment']['RILEY_PREFILL_FFN_SPLIT']==('1' if lane=='split' else '0')
            assert 'RILEY_PROJECTION_PIPELINE prepared=true' in read(name+'.log').decode()
            assert ('RILEY_PREFILL_FFN_ADAPTIVE enabled threshold=192' in read(name+'.log').decode())==(lane=='prior')
            assert ('RILEY_PREFILL_FFN_SPLIT enabled threshold=192' in read(name+'.log').decode())==(lane=='split')
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
        for phase,count in [('warmup',256),('retained',8192)]:
            receipt=load(name+'-'+phase+'-gc.json')
            assert receipt['policy']=='disabled' and not receipt['before']['enabled'] and receipt['restored_enabled']
            assert not receipt['events'] and not receipt['unclosed_generations']
            assert receipt['before']['stats']==receipt['after']['stats']
            assert receipt['ru_maxrss_after_kib']<6*1024*1024
            rows=load(name+'-'+phase+'.json');assert len(rows)==count
            kind=name.split('-')[0];cases=fixtures[kind] if kind=='shared' else fixtures[kind][:256] if phase=='warmup' else fixtures[kind][256:]
            bounds=load(name+'-'+phase+'-host.json')
            for index,row in enumerate(rows):
                validate_row(row,cases[index%len(cases)])
                assert bounds['before']['monotonic_ns']<=row['started_ns']<=row['ended_ns']<=bounds['after']['monotonic_ns']
            assert all(r['valid'] and len(r['token_ids'])==32 and r['checks']['prompt'] and r['checks']['finish'] for r in rows)
            if lane!='vllm':assert all(all(r['checks'].values()) for r in rows)
        stats=summary(rows);assert all(record[k]==v for k,v in stats.items())
        agreements[name]={k:sum(r['checks'][k] for r in rows) for k in ('prompt','tokens','text','finish')}
        if lane in ('split','prior'):
            match=re.search(r'RILEY_PREFIX_CACHE entries=(\d+) pages=(\d+) hits=(\d+) reused_tokens=(\d+) host_capacity_bytes=(\d+)',read(name+'.log').decode())
            assert match;values=list(map(int,match.groups()));assert values[1]<=512
            assert (values[2]==0) if name.startswith('unique') else (values[2]>0)
            telemetry[name]=dict(zip(('entries','pages','hits','reused_tokens','host_capacity_bytes'),values))
    stop_reference=None
    for lane in ('prior','control','split'):
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
        for lane in ('prior','control','split','vllm'):
            group=[r for r in records if r['name'].startswith(kind+'-') and r['name'].endswith('-'+lane)];assert len(group)==2
            row={'workload':kind,'lane':lane,'throughput_tokens_s':statistics.median(r['throughput_tokens_s'] for r in group)}
            for metric,q in [('ttft','0.5'),('tpot','0.5'),('e2e','0.95'),('e2e','0.99'),('itl','0.5'),('itl','0.95'),('itl','0.99')]:
                row[metric+'_'+q+'_ms']=statistics.median(r[metric+'_ms'][q] for r in group)
            compared.append(row)
    result={'scope':f'closed-loop C{concurrency} extended fixed-count comparison with common timed-phase client GC disabled; two reversed-order runs; not overall qualification','aggregation':'median of two lane throughput / percentile estimates, not pooled percentiles','retained_requests':131072,'retained_tokens':4194304,'archive_sha256':{archive.name:hashlib.sha256(archive.read_bytes()).hexdigest() for archive in archives},'preparation':preparation,'prompt_token_ranges':{k:[min(len(f['prompt_token_ids']) for f in v),max(len(f['prompt_token_ids']) for f in v)] for k,v in fixtures.items()},'comparison':compared,'reference_agreement':agreements,'cache_telemetry_including_warmup':telemetry}
    pressure={}
    for record in records:
        name=record['name'];bounds=load(name+'-retained-host.json');before=bounds['before'];after=bounds['after']
        elapsed=(after['monotonic_ns']-before['monotonic_ns'])/1e3;assert elapsed>0
        pressure[name]={'elapsed_ms':elapsed/1000,'fraction_percent':{}}
        for resource in ('cpu','io','memory'):
            for category in ('some','full'):
                delta=after['pressure'][resource][category]['total']-before['pressure'][resource][category]['total'];assert delta>=0
                assert delta<=elapsed*1.02, 'pressure interval inconsistent'
                pressure[name]['fraction_percent'][resource+'_'+category]=100*delta/elapsed
    result['host_pressure_by_retained_lane']=pressure
    result['descriptive_assessment']=assess(compared,candidate='split')
    result['per_run_measurements']=records
    result['startup_by_lane']={}
    for record in records:
        startup=load(record['name']+'-startup.json')
        assert 0<startup['ready_wall_ms']<610000 and startup['server_rss_kib']>0
        result['startup_by_lane'][record['name']]=startup
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in compared:print(r)
