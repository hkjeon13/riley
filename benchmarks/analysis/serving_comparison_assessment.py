"""Descriptive per-workload ratios; this does not qualify a serving implementation."""
LATENCIES=('ttft_0.5_ms','tpot_0.5_ms','e2e_0.95_ms','e2e_0.99_ms','itl_0.95_ms','itl_0.99_ms')

def assess(rows,candidate='projection'):
    indexed={(r['workload'],r['lane']):r for r in rows}
    assert len(indexed)==len(rows)
    result={}
    for workload in sorted({r['workload'] for r in rows}):
        current=indexed[workload,candidate];comparisons={}
        for baseline in ('prior','control','vllm'):
            reference=indexed[workload,baseline]
            ratios={key:current[key]/reference[key] for key in ('throughput_tokens_s',*LATENCIES)}
            comparisons[baseline]={'ratios':ratios,'throughput_change_percent':100*(ratios['throughput_tokens_s']-1),'reported_latency_change_percent':{key:100*(ratios[key]-1) for key in LATENCIES}}
        vllm=comparisons['vllm']['ratios']
        result[workload]={'comparisons':comparisons,'descriptive_minimum_metrics_met':vllm['throughput_tokens_s']>=1 and all(vllm[k]<=1 for k in LATENCIES),'descriptive_target_metrics_met':vllm['throughput_tokens_s']>=1.15 and vllm['ttft_0.5_ms']<=.9 and vllm['tpot_0.5_ms']<=.9 and all(vllm[k]<=1 for k in LATENCIES)}
    return {'scope':'Ratios of reported run-level median estimates only. Not statistical significance, stability, correctness, hardware or overall goal qualification.','workloads':result,'qualified':False}
