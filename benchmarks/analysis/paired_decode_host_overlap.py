"""Measure the CPU interval between pair event waits against successor activity.

This is a profiled observation, not isolated token-processing cost or a speedup.
"""
import argparse
import json
import pathlib
import sqlite3
from overlap_headroom import quantiles, union_ns


def measure(launches, waits, activities):
    rows = []
    for index in range(1, len(launches)-1):
        first, second, following = launches[index-1:index+2]
        if not second['future']:
            continue
        assert first['stage'] == 'ordinary_decode'
        pair_waits = [(s, e) for s, e in waits if first['cpu_start'] <= s < following['cpu_start']]
        assert len(pair_waits) == 2, 'pair must contain exactly two event waits'
        a, b = pair_waits
        assert a[1] >= first['gpu_end'] and b[1] >= second['gpu_end']
        assert a[1] <= b[0], 'host interval cannot overlap its event waits'
        overlapped = union_ns([(max(s, a[1]), min(e, b[0]))
                              for s, e in activities[second['correlation']]
                              if max(s, a[1]) < min(e, b[0])])
        rows.append({'first_correlation': first['correlation'], 'second_correlation': second['correlation'],
                     'host_start_ns': a[1], 'host_end_ns': b[0],
                     'between_waits_ns': b[0]-a[1], 'overlapped_device_activity_ns': overlapped})
    assert rows
    return {'pairs': len(rows), 'pairs_with_device_overlap': sum(r['overlapped_device_activity_ns'] > 0 for r in rows),
            'between_waits': quantiles([r['between_waits_ns'] for r in rows]),
            'overlapped_device_activity': quantiles([r['overlapped_device_activity_ns'] for r in rows]), 'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sqlite', type=pathlib.Path)
    p.add_argument('analysis', type=pathlib.Path)
    p.add_argument('output', type=pathlib.Path)
    a = p.parse_args()
    from overlap_headroom import analyze
    source, _ = analyze(a.sqlite, merge_graph_streams=True)
    prior = json.loads(a.analysis.read_text())
    assert prior['source']['source_sha256'] == source['source_sha256']
    with sqlite3.connect(a.sqlite.resolve().as_uri()+'?mode=ro', uri=True) as db:
        tids = {r['global_tid'] for r in prior['launches']}
        assert len(tids) == 1
        waits = db.execute("SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaEventSynchronize%' AND r.globalTid=? AND r.returnValue=0 ORDER BY r.start", (next(iter(tids)),)).fetchall()
        activities = {}
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ['CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']:
            if table in tables:
                for s, e, correlation in db.execute(f'SELECT start,end,correlationId FROM {table}'):
                    activities.setdefault(correlation, []).append((s, e))
    report = measure(prior['launches'], waits, activities)
    report.update(source_sha256=source['source_sha256'], scope='Middle launch-count window; CPU interval includes result read/validation, token/stop processing and other host overhead. GPU intervals are unioned. Profiler and client overhead remain. Event association uses the verified two-wait execution order; this does not isolate callback time or establish serving gain.')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
