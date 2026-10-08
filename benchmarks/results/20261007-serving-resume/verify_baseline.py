"""Reconstruct HTTP metrics from preserved SSE frames, independent of client summaries."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

def require(ok, message):
    if not ok: raise ValueError(message)

def unique(items):
    result={}
    for k,v in items:
        require(k not in result, 'duplicate JSON field '+k);result[k]=v
    return result

def decode(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',',':'), allow_nan=False).encode()).hexdigest()

def distribution(values):
    ordered=sorted(values)
    return {'samples':len(values),'median':statistics.median(values),
            'p95':ordered[math.ceil(.95*len(values))-1],
            'p99':ordered[math.ceil(.99*len(values))-1],'max':ordered[-1]}

def replay(row, corpus, response, engine):
    choice=response['choices'][0]
    prompt=choice.get('prompt_token_ids',corpus.get('prompt_token_ids'))
    reference={'model':'g04-smol','prompt_token_ids':prompt,'output_token_ids':choice['token_ids'],
               'text':choice['text'],'finish_reason':choice['finish_reason']}
    require(row['reference_sha256']==digest(reference),'reference hash mismatch')
    require(row['status']=='success' and row['transport_complete'] and row['protocol_valid'], 'failed transport')
    require(row['http_status']==200 and row['owned_connection_closed'] and not row['cleanup_errors'],'HTTP/cleanup failed')
    require(row['request']['prompt']==corpus['prompt'] and row['request']['max_tokens']==corpus['max_tokens'], 'request corpus mismatch')
    tokens=[];arrivals=[];texts=[];actual_prompt=None;identity=None;finish=None;usage=None;done=None
    previous=row['started_ns']
    for frame in row['frames']:
        t=frame['arrived_ns'];require(previous<=t<=row['call_finished_ns'],'frame timestamp order');previous=t
        require(done is None,'frame after DONE')
        if frame['data']=='[DONE]':
            require(finish is not None and usage is not None,'premature DONE');done=t;continue
        value=decode(frame['data']);require('error' not in value,'error-shaped HTTP 200')
        observed={k:value[k] for k in ['id','model','object','created']}
        require(observed['model']=='g04-smol' and observed['object']=='text_completion','response identity')
        if identity is None:identity=observed
        require(identity==observed,'SSE identity changed')
        choices=value['choices']
        if choices:
            require(len(choices)==1 and choices[0]['index']==0,'choice shape')
            c=choices[0];ids=c.get('token_ids') or []
            require(all(type(x) is int and 0<=x<2**32 for x in ids),'invalid token IDs')
            if ids:
                require(finish is None and usage is None,'token after finish')
                if not tokens:actual_prompt=c['prompt_token_ids']
                else:require(c.get('prompt_token_ids') is None,'repeated prompt IDs')
                tokens.extend(ids);arrivals.extend([t]*len(ids));texts.append(c['text'])
            else:require(c['text']=='' and c.get('prompt_token_ids') is None,'tokenless content')
            if c.get('finish_reason') is not None:
                require(finish is None and c['finish_reason'] in ['length','stop'],'invalid/repeated finish')
                finish=c['finish_reason']
        else:
            require(finish is not None and usage is None,'invalid usage ordering');usage=value['usage']
    require(done is not None and len(tokens)>1,'incomplete SSE')
    require(actual_prompt==prompt,'actual prompt tokens differ')
    require(len(tokens)==len(reference['output_token_ids']),'actual output count differs')
    require(usage=={'prompt_tokens':len(prompt),'completion_tokens':len(tokens),'total_tokens':len(prompt)+len(tokens)},'actual usage differs')
    require(tokens==row['token_ids'] and arrivals==row['token_arrival_ns'] and actual_prompt==row['prompt_token_ids'], 'row/frames token mismatch')
    require(''.join(texts)==row['text'] and finish==row['finish_reason'] and identity==row['response_identity'],'row/frames content mismatch')
    require(done==row['finished_ns']==row['done_ns'],'terminal timing mismatch')
    exact=(tokens==reference['output_token_ids'] and ''.join(texts)==reference['text'] and finish==reference['finish_reason'])
    require(exact==row['reference_match'],'reference flag differs')
    if engine=='v52':require(exact,'Riley frozen reference mismatch')
    metrics={'e2e_ns':done-row['started_ns'],'token_ttft_ns':arrivals[0]-row['started_ns'],
             'token_tpot_ns':(arrivals[-1]-arrivals[0])/(len(tokens)-1)}
    for k,v in metrics.items():require(math.isclose(v,row['metrics'][k],rel_tol=1e-12,abs_tol=1e-6),'forged metric '+k)
    return metrics,len(tokens),len(prompt),exact

def verify(files, fixtures_bytes):
    prep=decode(files['preparation.json']);plan=prep['plan'];fixtures=decode(fixtures_bytes)
    require(hashlib.sha256(fixtures_bytes).hexdigest()==prep['pins']['/data/riley-serving-261007/controller/fixtures.json'],'fixture snapshot pin mismatch')
    require(hashlib.sha256(files['controller-snapshot.py']).hexdigest()==prep['pins'][prep.get('controller_path','/data/riley-serving-261007/controller/run_baseline.py')],'controller snapshot pin mismatch')
    if 'plan_path' in prep:
        require(hashlib.sha256(files['plan-snapshot.json']).hexdigest()==prep['pins'][prep['plan_path']],'plan snapshot pin mismatch')
        require(decode(files['plan-snapshot.json'])==plan,'plan metadata differs from snapshot')
    completion=decode(files['completion.json']) if 'completion.json' in files else None
    lanes=[]
    for path in sorted(files):
        match=re.fullmatch(r'(c(\d+)-(fixed|natural)-r(\d+)-(v52|vllm))-retained-rows.json',path)
        if not match:continue
        prefix,c,w,r,engine=match.groups();c=int(c);r=int(r)
        require(c in plan['concurrency'] and r<len(plan['orders']), 'undeclared lane')
        rows=decode(files[path]);account=decode(files[prefix+'-retained-accounting.json'])
        require(account['completed'] and len(rows)==plan['retained'], 'incomplete retained phase')
        require([x['index'] for x in rows]==list(range(len(rows))),'omitted/duplicated request')
        corpus=fixtures[w]['corpus'];responses=fixtures[w]['responses'];metrics=[];outputs=inputs=matches=0;events=[]
        identities=set()
        for index,row in enumerate(rows):
            k=index%len(corpus);require(row['corpus_id']==corpus[k]['id'] and row['phase']=='retained','wrong corpus/phase')
            m,o,i,exact=replay(row,corpus[k],responses[k],engine);metrics.append(m);outputs+=o;inputs+=i;matches+=exact
            require(row['response_identity']['id'] not in identities,'duplicate response identity');identities.add(row['response_identity']['id'])
            require(account['phase_started_ns']<=row['call_started_ns']<=row['started_ns']<=row['finished_ns']<=row['call_finished_ns']<=account['phase_finished_ns'],'phase timing envelope')
            events.extend([(row['started_ns'],1),(row['finished_ns'],-1)])
        active=peak=0
        for _,delta in sorted(events):active+=delta;peak=max(peak,active)
        require(active==0 and peak==c==account['observed_max_request_in_flight'],'concurrency overlap differs')
        counts=dict(collections.Counter(x['corpus_id'] for x in rows));require(counts==account['actual_corpus_counts']==account['expected_corpus_counts'],'unbalanced corpus')
        launch=decode(files[prefix+'-launch.json']);argv=launch['argv']
        opt='--max-active-sequences' if engine=='v52' else '--max-num-seqs'
        require(int(argv[argv.index(opt)+1])==c,'server admission differs')
        require(launch['binary_sha256']==prep['pins'][argv[0]],'binary identity differs')
        for phase in ['start','warmup','retained']:
            gate=decode(files[prefix+'-'+phase+'-host-gate.json'])
            observed_only=plan.get('host_pressure_policy')=='observe_only'
            if observed_only:
                require(gate.get('pressure_policy')=='observe_only' and gate.get('performance_claim_eligible') is False,'mislabelled diagnostic host observation')
            else: require(gate['passed'],'failed host gate')
            policy=gate['policy'];require(policy==plan['host_start_gate'],'host policy changed')
            require(len(gate['samples'])>=policy['consecutive'],'insufficient host observations')
            quiet=True
            for h in gate['samples'][-policy['consecutive']:]:
                for resource,kind,limit in [('cpu','some',policy['cpu_some_max']),('io','full',policy['io_full_max']),('memory','full',policy['memory_full_max'])]:
                    line=next(x for x in h['pressure'][resource].splitlines() if x.startswith(kind+' '));value=float(line.split()[1].split('=')[1]);quiet=quiet and value<=limit
                    if not observed_only:require(value<=limit,'host gate falsely passed')
            require(gate['passed'] is quiet,'host observation quiet flag differs')
        first=min(x['started_ns'] for x in rows);last=max(x['finished_ns'] for x in rows)
        calculated={'throughput_tokens_s':outputs*1e9/(last-first),'input_tokens':inputs,'output_tokens':outputs,
                    'request_count':len(rows),'errors':0,'error_rate':0,'reference_matches':matches}
        for k,v in [('e2e_ms','e2e_ns'),('ttft_ms','token_ttft_ns'),('tpot_ms','token_tpot_ns')]:calculated[k]=distribution([m[v]/1e6 for m in metrics])
        host=decode(files[prefix+'-host.json']);require(bool(host),'missing memory observation')
        calculated['gpu_peak_observed_MiB']=max(float(h['gpu'].split(',')[4]) for h in host)
        require(calculated['gpu_peak_observed_MiB']*1024**2<=plan['gpu_operating_cap_bytes'],'GPU cap exceeded')
        calculated['memory_scope']='sampled device-wide startup and measurement, not continuous high-water'
        if host and 'server_process_memory' in host[0]:
            calculated['cpu_process_group_peak_observed_RSS_MiB']=max(h['server_process_memory']['rss_sum_KiB']/1024 for h in host)
            calculated['cpu_memory_scope']='sampled instantaneous RSS sum; shared pages may be counted more than once'
        summary=decode(files[prefix+'-summary.json'])
        require(math.isclose(calculated['throughput_tokens_s'],summary['successful_output_tokens_per_wall_second'],rel_tol=1e-12),'summary throughput differs from raw')
        for a,b in [('e2e_ms','e2e_ms'),('ttft_ms','token_ttft_ms'),('tpot_ms','token_tpot_ms')]:
            for stat in ['median','p95','p99']:require(math.isclose(calculated[a][stat],summary[b][stat],rel_tol=1e-12),'summary latency differs from raw')
        lanes.append({'case':prefix,'metrics':calculated})
    expected=len(plan['concurrency'])*len(plan['workloads'])*plan['repeats']*2
    complete=bool(completion and completion['failure'] is None and len(lanes)==expected)
    if complete:
        declared=[f'c{c}-{w}-r{r}-{engine}' for c in plan['concurrency']
                  for w in plan['workloads'] for r,order in enumerate(plan['orders']) for engine in order]
        require(completion['all_lanes_complete'] is True,'completion falsely incomplete')
        require([r['case'] for r in completion['records']]==declared,'AB/BA sequence differs')
        require(set(lane['case'] for lane in lanes)==set(declared),'full workload matrix differs')
    return {'raw_replay_passed':True,'lanes':lanes,'expected_lanes':expected,'matrix_complete':complete,
            'failure':completion['failure'] if completion else None,'performance_promoted':False,'goal_achieved':False,
            'pressure_policy':plan.get('host_pressure_policy','admission'),
            'qualification_scope':'diagnosis only; full admission-controlled validation remains required' if plan.get('host_pressure_policy')=='observe_only' else 'admission-controlled observations',
            'scope':'all preserved retained lanes; aborted attempt observations remain non-qualifying'}

def archive_files(archive):
    with tarfile.open(archive) as tar:
        members=[m for m in tar if m.isfile() and '/baseline-attempt' not in m.name]
        prep=next(m for m in members if m.name.endswith('/preparation.json'));prefix=prep.name.removesuffix('preparation.json')
        return {m.name.removeprefix(prefix):tar.extractfile(m).read() for m in members if m.name.startswith(prefix)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--fixtures',type=Path,default=Path(__file__).with_name('fixtures.json'));p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=verify(archive_files(args.archive),args.fixtures.read_bytes());args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'replay_passed':result['raw_replay_passed'],'complete_lanes':len(result['lanes']),'matrix_complete':result['matrix_complete'],'failure':result['failure']}))
