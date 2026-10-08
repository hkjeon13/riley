"""Verify archived profile bytes and HTTP workload fidelity without performance promotion."""
import argparse,collections,hashlib,json,math,sqlite3,tarfile,tempfile
from pathlib import Path
from verify_kernel_batch01 import replay,require,decode

def verify(archive,fixture_bytes):
    payload={};hashes={};kernel_counts={}
    with tempfile.TemporaryDirectory(prefix='riley-profile-replay-') as tmp:
        with tarfile.open(archive,mode='r|gz') as t:
            for m in t:
                if not m.isfile():continue
                name=m.name.removeprefix('profiling-attempt04/');src=t.extractfile(m);h=hashlib.sha256()
                parts=[];db=None
                if name.endswith('/trace.sqlite'):db=open(Path(tmp)/'trace.sqlite','wb')
                collect=name.endswith('.json') or name.endswith('/target.log') or name=='controller-snapshot.py'
                for chunk in iter(lambda:src.read(1024*1024),b''):
                    h.update(chunk)
                    if db:db.write(chunk)
                    if collect:parts.append(chunk)
                hashes[name]=h.hexdigest()
                if collect:payload[name]=b''.join(parts)
                if db:
                    db.close()
                    with sqlite3.connect(Path(tmp)/'trace.sqlite') as conn:
                        count=conn.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
                        require(count>0,'no CUDA kernels '+name);kernel_counts[name.removesuffix('/trace.sqlite')]=count
    prep=decode(payload['preparation.json']);plan=prep['plan'];completion=decode(payload['completion.json']);fixtures=decode(fixture_bytes)
    require(hashes['controller-snapshot.py']==prep['controller_sha256'],'controller hash differs')
    require(hashes['plan-snapshot.json']==prep['plan_sha256'] and decode(payload['plan-snapshot.json'])==plan,'plan hash differs')
    require(hashlib.sha256(fixture_bytes).hexdigest()==prep['pins']['/data/riley-serving-261007/controller/fixtures.json'],'fixture differs')
    declared=[f'c{c}-{w}-r{r}-{e}' for c in plan['concurrency'] for w in plan['workloads'] for r,order in enumerate(plan['orders']) for e in order]
    require(len(declared)==32 and completion['failure'] is None and completion['completed']==declared,'incomplete or reordered profile matrix')
    require(plan['cuda_graph_trace']=='node' and plan['sample']==plan['cpuctxsw']=='none','instrumentation differs')
    require(plan['expected_tool_version'] in prep['nsys_version'],'profiler version differs')
    lanes=[]
    for label in declared:
        engine=label.rsplit('-',1)[-1];c=int(label.split('-')[0][1:]);w=label.split('-')[1]
        def read(name):return decode(payload[label+'/'+name])
        manifest=read('artifact-manifest.json')
        for name,h in manifest['files'].items():require(hashes[label+'/'+name]==h,'artifact hash differs '+label+'/'+name)
        require(manifest['kernel_events']==kernel_counts[label],'CUDA event count differs')
        require(not any('not supported by this build' in text for text in read('trace-diagnostics.json')['events']),'unsupported stack')
        launch=read('launch.json');argv=launch['target_argv'];binary=argv[0]
        require(launch['target_binary_sha256']==prep['pins'][binary],'runtime binary differs')
        opt='--max-active-sequences' if engine=='v52' else '--max-num-seqs'
        require(int(argv[argv.index(opt)+1])==c,'wrong serving concurrency')
        phases=[]
        for phase in ['warmup','retained']:
            rows=read(phase+'-rows.json');account=read(phase+'-accounting.json');require(account['completed'] and len(rows)==plan[phase],'incomplete HTTP phase')
            require([r['index'] for r in rows]==list(range(len(rows))),'omitted HTTP rows')
            corpus=fixtures[w]['corpus'];responses=fixtures[w]['responses'];events=[];matches=0
            for i,row in enumerate(rows):
                k=i%len(corpus);require(row['phase']==phase and row['corpus_id']==corpus[k]['id'],'corpus differs')
                _,_,_,exact=replay(row,corpus[k],responses[k],engine);matches+=exact
                require(account['phase_started_ns']<=row['call_started_ns']<=row['started_ns']<=row['finished_ns']<=row['call_finished_ns']<=account['phase_finished_ns'],'HTTP clock envelope differs')
                events.extend([(row['started_ns'],1),(row['finished_ns'],-1)])
            active=peak=0
            for _,delta in sorted(events):active+=delta;peak=max(peak,active)
            require(active==0 and peak==c==account['observed_max_request_in_flight'],'HTTP overlap differs')
            require(dict(collections.Counter(r['corpus_id'] for r in rows))==account['actual_corpus_counts']==account['expected_corpus_counts'],'HTTP corpus balance differs')
            require(matches==account['reference_matches'],'reference accounting differs')
            phases.append({'phase':phase,'requests':len(rows),'reference_matches':matches,'errors':0})
        exit=read('target-exit.json');require(not exit['host']['compute_apps'] and exit['shutdown_error'] is None,'profile GPU not reclaimed')
        lanes.append({'case':label,'kernel_events':kernel_counts[label],'phases':phases,'trace_diagnostics':read('trace-diagnostics.json')['events']})
    return {'complete_profiles':32,'archived_files_sha_verified':len(hashes),'HTTP_requests_replayed':sum(p['requests'] for l in lanes for p in l['phases']), 'lanes':lanes,'serving_performance':'미실행; instrumented timings are not serving evidence','CPU_stacks_context_switches':'미측정','HTTP_GPU_clock_correlation':'not established by this verifier','goal_achieved':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=verify(a.archive,a.fixtures.read_bytes());a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['complete_profiles','archived_files_sha_verified','HTTP_requests_replayed']}))
