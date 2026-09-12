"""Aggregate completed paired trials; preserve raw evidence and timing boundaries."""
import json
import statistics as s
from pathlib import Path

root = Path(__file__).resolve().parent
result = {'condition': json.loads((root/'condition.json').read_text()),
          'limitations': ['GUI retained; not the canonical 256 MiB idle condition',
                         'SmolLM2-135M BF16 c1/p128/o32 only; five process pairs',
                         'Host request timing, not CUDA event kernel timing',
                         'HTTP first text event is not a proven per-token ITL',
                         'Legacy vLLM raw environment_id is stale; condition.json and per-lane live preflight identify the actual environment',
                         'HTTP uses identical <=48 C cooldown before each lane; engine used original <=50 C start gate'],
          'modes': {}}
for mode in ('engine', 'http'):
    pairs = []
    for index in range(1, 6):
        pair = {'index': index}
        for role in ('riley', 'vllm'):
            directory = root/mode/f'pair-{index:02d}-{role}'
            assert json.loads((directory/'execution-complete.json').read_text())['completed']
            if mode == 'engine':
                if role == 'riley':
                    rows = json.loads((directory/'native-profile.json').read_text())['requests']
                    values = {key: [r[field] for r in rows] for key, field in
                              [('e2e_ms','e2e_ms'),('ttft_ms','ttft_ms'),('tpot_ms','tpot_ms')]}
                else:
                    rows = [json.loads(line)['requests'][0] for line in (directory/'vllm/raw.jsonl').read_text().splitlines()]
                    values = {key: [r[field] for r in rows] for key, field in
                              [('e2e_ms','end_to_end_ms'),('ttft_ms','ttft_ms'),('tpot_ms','mean_tpot_ms')]}
            else:
                rows = [json.loads(line) for line in (directory/'http-streaming.jsonl').read_text().splitlines()]
                values = {'e2e_ms': [(r['finished_ns']-r['started_ns'])/1e6 for r in rows],
                          'first_text_event_ms': [(r['event_ns'][0]-r['started_ns'])/1e6 for r in rows]}
            assert len(rows) == 30
            pair[role] = {k: s.median(v) for k,v in values.items()}
        pair['riley_over_vllm_e2e'] = pair['riley']['e2e_ms']/pair['vllm']['e2e_ms']
        pairs.append(pair)
    result['modes'][mode] = {'requests_per_role':150, 'pairs':pairs,
        'median_of_pair_medians':{role:{k:s.median(p[role][k] for p in pairs) for k in pairs[0][role]} for role in ('riley','vllm')},
        'paired_e2e_ratio_median':s.median(p['riley_over_vllm_e2e'] for p in pairs),
        'paired_e2e_ratio_range':[min(p['riley_over_vllm_e2e'] for p in pairs),max(p['riley_over_vllm_e2e'] for p in pairs)]}
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
