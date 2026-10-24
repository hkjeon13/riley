"""Phase 0 analysis: A/A gate calibration and numerics-contract cost.

Usage: analyze-phase0.py RUNS_DIR OUT_JSON
RUNS_DIR holds <label>/result.json for aa-c16-c32, cost-fixed, cost-natural.
"""
import itertools
import json
import statistics
import sys
from pathlib import Path

LATENCY = ['ttft_ms_p95', 'ttft_ms_p99', 'tpot_ms_p95', 'tpot_ms_p99',
           'itl_ms_p95', 'itl_ms_p99', 'e2e_ms_p95', 'e2e_ms_p99']
V19B_TAIL = ['ttft_ms_p95', 'ttft_ms_p99', 'tpot_ms_p95', 'tpot_ms_p99']


def pct(after, before):
    return (after - before) / before * 100


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def load(runs, label):
    path = runs / label / 'result.json'
    return json.loads(path.read_text())['records'] if path.exists() else []


def aa(records):
    out = {}
    for conc in sorted({r['concurrency'] for r in records}):
        by = {}
        for r in records:
            if r['concurrency'] == conc:
                by.setdefault(r['repeat'], {})[r['mode']] = r
        pairs = [by[k] for k in sorted(by) if {'owned-before', 'owned-after'} <= set(by[k])]
        changes = [dict(repeat=p['owned-before']['repeat'],
                        output_tok_s=pct(p['owned-after']['output_tok_s'], p['owned-before']['output_tok_s']),
                        **{m: pct(p['owned-after'][m], p['owned-before'][m]) for m in LATENCY})
                   for p in pairs]
        subsets = list(itertools.combinations(range(len(changes)), 5)) if len(changes) >= 5 else []
        flagged, all_positive, flag_counts = [], [], []
        medians = {m: [] for m in ['output_tok_s'] + LATENCY}
        for s in subsets:
            rows = [changes[i] for i in s]
            flags = sum(r[m] > 2 for r in rows for m in V19B_TAIL)
            flag_counts.append(flags)
            flagged.append(flags > 0)
            all_positive.append(all(r['output_tok_s'] > 0 for r in rows))
            for m in medians:
                medians[m].append(statistics.median(r[m] for r in rows))
        arms = [p[m] for p in pairs for m in ('owned-before', 'owned-after')]
        out[conc] = dict(
            pairs=len(changes),
            per_pair_change_percent=changes,
            arm_cv_percent={m: statistics.stdev(a[m] for a in arms) / statistics.mean(a[m] for a in arms) * 100
                            for m in ['output_tok_s'] + LATENCY} if len(arms) > 1 else None,
            single_pair_any_v19b_tail_flag=sum(any(r[m] > 2 for m in V19B_TAIL) for r in changes),
            subsets=len(subsets),
            v19b_tail_rule_false_flag_probability=sum(flagged) / len(subsets) if subsets else None,
            mean_tail_flags_per_five_pair_campaign=statistics.mean(flag_counts) if subsets else None,
            all_five_throughput_pairs_positive_probability=sum(all_positive) / len(subsets) if subsets else None,
            median_of_five_band_percent={m: dict(p2_5=quantile(v, .025), p97_5=quantile(v, .975),
                                                 max_abs=max(abs(x) for x in v)) for m, v in medians.items()} if subsets else None,
            flagged_subset_mask=flagged)
    concs = sorted(out)
    if len(concs) == 2 and all(out[c]['subsets'] for c in concs):
        a, b = (out[c]['flagged_subset_mask'] for c in concs)
        out['either_concurrency_v19b_false_flag_probability'] = sum(x or y for x in a for y in b) / (len(a) * len(b))
    for c in concs:
        out[c].pop('flagged_subset_mask')
    return out


def cost(records):
    out = {}
    for conc in sorted({r['concurrency'] for r in records}):
        cell = {}
        for mode in ['owned-before', 'native', 'vllm']:
            rows = [r for r in records if r['concurrency'] == conc and r['mode'] == mode]
            if not rows:
                continue
            cell[mode] = dict(
                repeats=len(rows),
                output_tok_s=[r['output_tok_s'] for r in rows],
                median={m: statistics.median(r[m] for r in rows)
                        for m in ['output_tok_s', 'ttft_ms_p50', 'tpot_ms_p50', 'itl_ms_p50'] + LATENCY},
                output_tokens=[r['output_tokens'] for r in rows],
                measured_hf_token_matches=[r['measured_hf_token_matches'] for r in rows],
                measured_eager_matches=[r['measured_eager_matches'] for r in rows],
                measured_native_canonical_matches=[r['measured_native_canonical_matches'] for r in rows],
                correctness_hf_tokens=[r['hf_correctness_tokens'] for r in rows if r['correctness_requests']],
                correctness_requests=[r['correctness_requests'] for r in rows if r['correctness_requests']])
        tput = {m: v['median']['output_tok_s'] for m, v in cell.items()}
        ratios = {}
        if 'vllm' in tput:
            for m in ['owned-before', 'native']:
                if m in tput:
                    ratios[m + '/vllm'] = tput[m] / tput['vllm']
        if 'native' in tput and 'owned-before' in tput:
            ratios['native/owned-before'] = tput['native'] / tput['owned-before']
        cell['throughput_ratios'] = ratios
        out[conc] = cell
    return out


def main():
    runs = Path(sys.argv[1])
    result = dict(aa=aa(load(runs, 'aa-c16-c32')),
                  cost_fixed=cost(load(runs, 'cost-fixed')),
                  cost_natural=cost(load(runs, 'cost-natural')))
    for label in ['cost-fixed', 'cost-natural']:
        invariance = runs / label / 'native-invariance.json'
        if invariance.exists():
            result[label.replace('-', '_') + '_native_invariance'] = json.loads(invariance.read_text())
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1)[:6000])


if __name__ == '__main__':
    main()
