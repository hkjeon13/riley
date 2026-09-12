import collections,concurrent.futures,threading,time
from serving_token_client_v2 import overlap_peak,summarize_phase

def mixed_phase(client, request_one, *, concurrency, count, phase, corpus_ids):
    assert count % len(corpus_ids) == 0 and 1 <= concurrency <= 8
    lock=threading.Lock();halt=threading.Event();state={'next':0,'started':0};rows=[];errors=[]
    barrier=threading.Barrier(concurrency+1,action=lambda:state.update(started=time.perf_counter_ns()))
    def worker(worker_id):
        barrier.wait(timeout=30)
        while not halt.is_set():
            with lock:
                if halt.is_set() or state['next']>=count:return
                index=state['next'];state['next']+=1
            started=time.perf_counter_ns()
            try:
                row=request_one(index)
                assert started<=row['started_ns']<=row['finished_ns']<=row['call_finished_ns']
                assert row['corpus_id']==corpus_ids[index%len(corpus_ids)]
                assert all(row['started_ns']<=t<=row['finished_ns'] for t in row['token_arrival_ns'])
                row.update(index=index,worker_id=worker_id,phase=phase,call_started_ns=started)
                with lock:rows.append(row)
                if row['status']!='success':halt.set()
            except BaseException as e:
                with lock:errors.append({'index':index,'type':type(e).__name__,'message':str(e)})
                halt.set();client.abort_pending('mixed phase worker failed');return
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(worker,i) for i in range(concurrency)]
        barrier.wait(timeout=30)
        for future in futures:future.result()
    finished=time.perf_counter_ns();rows.sort(key=lambda x:x['index'])
    good=[x for x in rows if x['status']=='success' and x['protocol_valid'] and x['transport_complete']]
    observed=overlap_peak(rows) if rows else 0
    identities=[x['response_identity']['id'] for x in good]
    expected={k:count//len(corpus_ids) for k in corpus_ids};actual=dict(collections.Counter(x['corpus_id'] for x in rows))
    complete=len(good)==count and observed==concurrency and not errors and len(set(identities))==count and actual==expected
    account={'schema_version':'riley.mixed-p128-phase.v1','phase':phase,'requested':count,'attempted':state['next'],
      'unresolved_attempts':state['next']-len(rows),'not_started':count-state['next'],'succeeded':len(good),'failed':len(rows)-len(good),
      'phase_started_ns':state['started'],'phase_finished_ns':finished,'completed':complete,'expected_corpus_counts':expected,
      'actual_corpus_counts':actual,'observed_max_request_in_flight':observed,'reference_matches':sum(x['reference_match'] for x in rows),
      'strict_reference_pass':complete and all(x['reference_match'] for x in rows),'worker_errors':errors,
      'performance_qualified':False,'correctness_qualified':False,'failure_policy':'stop refill on transport/protocol failure; retain numerical observations; no retry'}
    return rows,account,summarize_phase(rows,account) if rows else None
