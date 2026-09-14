"""Classify actual graph kernels to select an optimization area, not claim speedup."""
import collections
import json
from pathlib import Path
import sqlite3
import sys
from overlap_headroom import analyze


def area(name):
    lower = name.lower()
    if 'riley_prefill_ffn_row_reuse::' in lower or 'riley_ffn_pipeline::' in lower or 'riley_prefill_ffn_pipeline::' in lower or 'gate' in lower or ('<(int)576, (int)1536' in lower and ('projection' in lower or 'gemm_prefill_shape_vector' in lower)):
        return 'ffn_gate_activation_down'
    if 'attention' in lower or 'batchdecodewithpaged' in lower:
        return 'attention'
    if 'riley_flashinfer' in lower:
        return 'attention_metadata_scatter'
    if 'norm' in lower:
        return 'residual_norm'
    if 'projection' in lower or 'gemm_prefill_shape_vector' in lower or 'qkv' in lower or 'rope' in lower:
        return 'other_projections_qkv_rope'
    if 'cutlass' in lower or 'ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn' in lower:
        return 'output_head'
    return 'other'


def main():
    path, output = map(Path, sys.argv[1:3])
    gaps, launches = analyze(path, merge_graph_streams=True)
    launches.sort(key=lambda x: x['gpu_start'])
    assert len(gaps['groups']) == 1
    connection = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    events = collections.defaultdict(list)
    for start, end, correlation, name in connection.execute('''
     select k.start,k.end,k.correlationId,s.value
     from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName'''):
        events[correlation].append((start, end, name))
    result = {'trace': gaps, 'scope': 'Middle 80 percent of graph launches by count; startup/drain are not precisely isolated; profiler overhead remains.', 'stages': {}}
    chosen = launches[len(launches)//10:len(launches)*9//10]
    def stage_of(launch):
        names=[e[2] for e in events[launch['correlation']]]
        if any('riley_speculative_select::gather' in name for name in names):return 'verification'
        return 'decode' if any('riley_shared32_model::embedding' in name for name in names) else 'prefill_or_mixed'
    for stage in ('decode', 'prefill_or_mixed', 'verification'):
        selected = [x for x in chosen if stage_of(x)==stage]
        if not selected:continue
        totals, counts, kernels = collections.Counter(), collections.Counter(), collections.Counter()
        for launch in selected:
            entries = events[launch['correlation']]
            assert len(entries) == launch['kernels']
            for start, end, name in entries:
                assert launch['gpu_start'] <= start <= end <= launch['gpu_end']
                totals[area(name)] += end-start
                counts[area(name)] += 1
                kernels[name] += end-start
        total = sum(totals.values())
        # Retain all unclassified kernels for inspection, without assigning their cost.
        result['stages'][stage] = {'launches': len(selected), 'kernel_ms': total/1e6,
            'graph_span_ms': sum(x['gpu_end']-x['gpu_start'] for x in selected)/1e6,
            'areas': {k:{'kernel_ms':v/1e6, 'kernel_fraction':v/total, 'count':counts[k]} for k,v in totals.most_common()},
            'kernels_ms': {k:v/1e6 for k,v in kernels.most_common()}}
    connection.close()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({s:{'launches':v['launches'],'areas':v['areas']} for s,v in result['stages'].items()}, indent=2))

if __name__ == "__main__":
    main()
