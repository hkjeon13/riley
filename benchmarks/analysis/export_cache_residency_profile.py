"""Recompute diagnostic accounting from the curated archive; verify actual requests."""
import hashlib,json,pathlib,subprocess,sys,tarfile,tempfile,sqlite3
root=pathlib.Path(sys.argv[1]); archive=root/'evidence.tar.gz';prefix='cache-residency-trace-export-v1/'
results={}
with tarfile.open(archive) as tar, tempfile.TemporaryDirectory() as tmp:
    def read(name):return tar.extractfile(prefix+name).read()
    def load(name):return json.loads(read(name))
    assert not read('compute-after.csv').strip()
    for lane in ('shared','unique'):
        receipt=load(lane+'-receipt.json');assert receipt['reference_exact'] and receipt['requests']==64
        assert receipt['exit_code']==0 and receipt['remaining_owned_pids']==[] and receipt['peak_owned_tree_rss_bytes']<8*1024**3
        launch=load(lane+'-launch.json')
        assert launch['binary_sha256']=='4eeea8cc3b899c41352c0a28b098c42e45a987f472470a2e11503279ad669ece'
        argv=launch['argv']
        assert argv[argv.index('--kv-blocks')+1]=='2048'
        assert argv[argv.index('--max-active-sequences')+1]=='32'
        assert argv[argv.index('--decode-window')+1]=='paired-experimental-v1'
        assert argv[argv.index('--ffn-backend')+1]=='prefill-pipeline-experimental-v1'
        assert argv[argv.index('--decode-projection')+1]=='adaptive-rows-experimental-v1'
        assert launch['controller_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('prefix_cache_composition_trace.py').read_bytes()).hexdigest()
        assert launch['client_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('paired_decode_serving_screen.py').read_bytes()).hexdigest()
        for phase in ('warmup','rows'):
            rows=load(lane+'-'+phase+'.json');assert len(rows)==64
            assert all(r['valid'] and all(r['checks'].values()) and len(r['token_ids'])==32 for r in rows)
        dbpath=pathlib.Path(tmp)/(lane+'.sqlite');dbpath.write_bytes(read(lane+'.sqlite'))
        with sqlite3.connect(dbpath) as db:
            tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
            assert tables=={'CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET','StringIds','RILEY_TRACE_REDACTION'}
            raw_hash=db.execute('select source_sha256 from RILEY_TRACE_REDACTION').fetchone()[0]
        target=pathlib.Path(tmp)/(lane+'.json')
        subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('paired_decode_trace_analysis.py')),str(dbpath),str(target)],check=True,stdout=subprocess.DEVNULL)
        derived=json.loads(target.read_text());stored=load(lane+'-analysis.json')
        derived['source']['source']=stored['source']['source'];assert derived==stored
        stage_kernels={}
        by_correlation={r['correlation']:r['stage'] for r in derived['launches']}
        with sqlite3.connect(dbpath) as db:
            for start,end,correlation,name in db.execute('SELECT k.start,k.end,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id'):
                if correlation in by_correlation:
                    costs=stage_kernels.setdefault(by_correlation[correlation],{})
                    costs[name]=costs.get(name,0)+(end-start)/1e6
        stages=derived['stage_graph_spans'];kernels=derived['kernel_ms'];total=sum(kernels.values())
        mapped=sum(v for k,v in kernels.items() if k.startswith('riley_mixed_attention::mapped_attention'))
        results[lane]={'receipt':receipt,'original_sqlite_sha256':raw_hash,'curated_sqlite_sha256':hashlib.sha256(dbpath.read_bytes()).hexdigest(),'selected':derived['selected'],'stage_graph_spans':stages,'gaps':derived['gaps'],'kernel_ms':kernels,'stage_kernel_ms':stage_kernels,'mixed_fraction_of_graph_spans':stages['prefill_or_mixed']['sum_ms']/derived['selected']['graph_span_ms'],'mapped_attention_fraction_of_kernel_time':mapped/total}
(root/'receipt.json').write_text(json.dumps({'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'scope':'diagnostic profile, middle 80% of graph count including warmup, phase transitions and profiler overhead; not serving performance','verified_requests':256,'results':results},indent=2)+'\n')
print('verified both curated traces, numeric accounting, 256 reference-exact requests and clean process receipts')
