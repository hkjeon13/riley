#!/usr/bin/env python3
"""Read-only Nsight SQLite accounting of inter-graph gaps.

This is a trace-conditioned optimistic ceiling, not an unprofiled serving
forecast. Client pacing and profiler overhead may be included in the gaps.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def union_ns(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('negative activity interval')
        if end is None:
            total += stop-start
        elif stop > end:
            total += stop-max(start,end)
        end = max(stop, end if end is not None else stop)
    return total


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    return {'count': len(values), 'median_us': statistics.median(values)/1000,
            'p95_us': values[min(len(values)-1, int((len(values)-1)*.95))]/1000,
            'max_us': max(values)/1000, 'sum_ms': sum(values)/1e6}


def gap_accounting(rows):
    """Rows must belong to one process/context/stream, sorted by GPU start."""
    if len(rows) < 2:
        raise ValueError('at least two launches required')
    gaps, host, launch = [], [], []
    for previous, current in zip(rows, rows[1:]):
        if current['gpu_start'] < previous['gpu_end']:
            raise ValueError('overlapping graph spans: serial-stream ceiling invalid')
        gap = current['gpu_start'] - previous['gpu_end']
        unsubmitted = max(0, min(current['gpu_start'], current['cpu_start']) - previous['gpu_end'])
        gaps.append(gap); host.append(unsubmitted); launch.append(gap-unsubmitted)
    span = sum(row['gpu_end']-row['gpu_start'] for row in rows)
    window = rows[-1]['gpu_end']-rows[0]['gpu_start']
    assert window == span+sum(gaps)
    return {'launches': len(rows), 'window_ms': window/1e6, 'graph_span_ms': span/1e6,
            'inter_graph_gap': quantiles(gaps), 'post_gpu_until_next_cpu_launch': quantiles(host),
            'remaining_gap_after_cpu_launch_starts': quantiles(launch),
            'removable_fraction_if_all_inter_graph_gaps_vanish': sum(gaps)/window,
            'optimistic_trace_speedup_if_all_gaps_vanish': window/span,
            'inside_graph_unoccupied_ms': sum(row['gpu_end']-row['gpu_start']-row['gpu_union_ns'] for row in rows)/1e6}


def analyze(path):
    connection = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    api = {}
    for start, end, correlation, tid, name, result in connection.execute('''
        select r.start,r.end,r.correlationId,r.globalTid,s.value,r.returnValue
        from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id
        where s.value like 'cudaGraphLaunch%' '''):
        if correlation in api:
            raise ValueError('ambiguous launch correlation across processes/threads')
        if result != 0:
            raise ValueError('failed graph launch present')
        api[correlation] = {'cpu_start':start, 'cpu_end':end, 'global_tid':tid, 'api':name}
    activities = {}
    for table in ['CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']:
        if table not in tables:
            continue
        for start, end, correlation, pid, context, stream in connection.execute(
                f'select start,end,correlationId,globalPid,contextId,streamId from {table}'):
            if correlation in api:
                activities.setdefault((pid,context,stream,correlation), []).append((start,end,table))
    rows = []
    for (pid,context,stream,correlation), events in activities.items():
        rows.append(dict(api[correlation], global_pid=pid, context=context, stream=stream,
                         correlation=correlation, gpu_start=min(e[0] for e in events),
                         gpu_end=max(e[1] for e in events), gpu_union_ns=union_ns([(e[0],e[1]) for e in events]),
                         kernels=sum(e[2].endswith('KERNEL') for e in events), activities=len(events)))
    groups = {}
    for row in rows:
        groups.setdefault((row['global_pid'],row['context'],row['stream']), []).append(row)
    reports = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda row:row['gpu_start'])
        report = gap_accounting(group)
        report['process_context_stream'] = key
        report['kernel_count_distribution'] = {str(n):sum(row['kernels']==n for row in group) for n in sorted({row['kernels'] for row in group})}
        reports.append(report)
    matched = {row['correlation'] for row in rows}
    missing = sorted(set(api)-matched)
    if missing:
        raise ValueError(f'{len(missing)} launches lack device activities; truncated trace is not eligible')
    if any(sum(row['correlation']==correlation for row in rows)!=1 for correlation in matched):
        raise ValueError('a launch spans multiple streams; this serial-stream analysis does not apply')
    connection.close()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024), b''):
            digest.update(block)
    return {'source':str(path.resolve()),'source_sha256':digest.hexdigest(),
            'cpu_graph_launches':len(api), 'matched_device_launches':len(rows), 'groups':reports,
            'limits':['All inter-graph gaps are optimistically removable, including client pacing and launch overhead.',
                      'Profiler overhead is not removed; this is not a bound on unprofiled serving.',
                      'Inside-graph unoccupied time is kept and must not be called scheduler host bubble.',
                      'No throughput, latency, fairness or correctness gain is established.']}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report, rows = analyze(args.sqlite)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(sorted(rows,key=lambda row:row['gpu_start']))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
