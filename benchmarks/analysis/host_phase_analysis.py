"""Aggregate successful serving host wall-time phases without calling waits CPU work."""
import argparse,json,pathlib

def parse(path):
    engine={};runtime={};fallback=0
    for line in path.read_text().splitlines():
        if not line.startswith(('RILEY_HOST_PHASE ','RILEY_RUNTIME_PHASE ','RILEY_HOST_PHASE_FALLBACK ')):continue
        tag,*fields=line.split();values=dict(field.split('=',1) for field in fields)
        if tag=='RILEY_HOST_PHASE_FALLBACK':fallback=int(values['wall_ns']);continue
        kind=values.pop('kind');target=engine if tag=='RILEY_HOST_PHASE' else runtime
        assert kind not in target
        target[kind]={k:int(v) for k,v in values.items()}
    assert set(engine)=={'decode','prefill_or_mixed','paired_decode'}
    assert len(runtime)==6
    total_steps=sum(r['steps'] for r in engine.values())
    assert runtime['retain_encode']['calls']==runtime['read_validate']['calls']==total_steps
    assert runtime['sync_transfer']['calls']+runtime['buffered_submit']['calls']==total_steps
    assert runtime['buffered_wait']['calls']==runtime['buffered_submit']['calls']
    assert runtime['future_prepare']['calls'] in (0,engine['paired_decode']['steps'])
    execute=sum(r['execute_wall_ns'] for r in engine.values())
    native=sum(runtime[k]['wall_ns'] for k in ['sync_transfer','buffered_submit','buffered_wait'])
    explicit=sum(runtime[k]['wall_ns'] for k in ['retain_encode','read_validate','future_prepare'])
    adapter=execute-native-explicit;assert adapter>=0
    outer=sum(r[k] for r in engine.values() for k in ['plan_ns','sample_ns','commit_publish_ns'])+fallback
    return {'future_preparation_scope':'prepared handoff: construction remains in adapter' if engine['paired_decode']['steps'] and not runtime['future_prepare']['calls'] else 'runtime construction counted separately','engine':engine,'runtime':runtime,'fallback_ns':fallback,'total_observed_ns':execute+outer,'native_transfer_submit_wait_ns':native,'runtime_prepare_validate_ns':explicit,'adapter_and_other_execute_ns':adapter,'outer_plan_sample_commit_ns':outer,'scheduled_tokens':sum(r['scheduled_tokens'] for r in engine.values()),'decode_mean_rows':{kind:(r['scheduled_tokens']/r['steps']/(2 if kind=='paired_decode' else 1) if r['steps'] else None) for kind,r in engine.items() if kind!='prefill_or_mixed'}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
    records={path.stem:parse(path) for path in sorted(a.directory.glob('c32-p*-*.log'))}
    assert len(records)==4
    assert len({r['scheduled_tokens'] for r in records.values()})==1
    a.output.write_text(json.dumps({'scope':'server lifetime including warmup; wall times, not CPU cycles or pure GPU time; successful steps only','records':records},indent=2)+'\n')
    for name,r in records.items():print(name,{k:round(r[k]/1e6,2) for k in ['total_observed_ns','native_transfer_submit_wait_ns','runtime_prepare_validate_ns','adapter_and_other_execute_ns','outer_plan_sample_commit_ns']},r['decode_mean_rows'])
