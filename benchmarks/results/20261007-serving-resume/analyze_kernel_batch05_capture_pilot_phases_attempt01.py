"""Epoch-anchored diagnostic phase windows; no request-level causal attribution."""
import collections
import hashlib
import json
import pathlib
import re
import sqlite3
import time
from analyze_profile_intervals import merged, length, intersection

ROOT = pathlib.Path(__file__).resolve().parent
COLLECT = ROOT / 'kernel-batch05-capture-pilot-independent-collection-attempt01'
SOURCE = COLLECT / 'kernel-batch05-capture-pilot-attempt02'
OUT = ROOT / 'kernel-batch05-capture-pilot-phase-analysis-attempt01'
OUT.mkdir()
receipt = json.loads((COLLECT / 'kernel-batch05-capture-pilot-attempt02-receipt.json').read_text())
verification = json.loads((COLLECT / 'independent8-profile-verification.json').read_text())
assert verification['complete_profiles'] == 8 and verification['HTTP_requests_replayed'] == 1536
assert receipt['source_before'] == receipt['source_after'] and receipt['files_verified'] == 174
TABLES = {'kernel': 'CUPTI_ACTIVITY_KIND_KERNEL', 'memcpy': 'CUPTI_ACTIVITY_KIND_MEMCPY',
          'memset': 'CUPTI_ACTIVITY_KIND_MEMSET', 'cuda_api': 'CUPTI_ACTIVITY_KIND_RUNTIME',
          'driver_api': 'CUPTI_ACTIVITY_KIND_DRIVER', 'osrt': 'OSRT_API'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

reports = []
failure = None
try:
    for lane in verification['lanes']:
        label = lane['case']
        folder = SOURCE / label
        db_path = folder / 'trace.sqlite'
        assert sha(db_path) == receipt['source_before'][label + '/trace.sqlite']['sha256']
        with sqlite3.connect('file:' + str(db_path.resolve()) + '?mode=ro', uri=True) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            origins = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchall()
            assert len(origins) == 1
            epoch = origins[0][0]
            names = dict(db.execute('SELECT id,value FROM StringIds'))
            diagnostics = [row[0] for row in db.execute('SELECT text FROM DIAGNOSTIC_EVENT')]
            collected = [int(re.search(r'Number of CUDA events collected:\s*(\d+)', text)[1])
                         for text in diagnostics if 'Number of CUDA events collected:' in text]
            produced = [int(re.search(r'Number of CUPTI events produced:\s*(\d+)', text)[1])
                        for text in diagnostics if 'Number of CUPTI events produced:' in text]
            full_counts = {kind: db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                           for kind, table in TABLES.items() if table in tables}
            assert sum(count for kind, count in full_counts.items() if kind != 'osrt') == collected[0]
            phase_reports = []
            for phase in ['warmup', 'retained']:
                account = json.loads((folder / (phase + '-accounting.json')).read_text())
                before = json.loads((folder / (phase + '-host-before.json')).read_text())['clocks']
                after = json.loads((folder / (phase + '-host-after.json')).read_text())['clocks']
                assert before['perf_counter_after_ns'] <= account['phase_started_ns'] <= account['phase_finished_ns'] <= after['perf_counter_before_ns']
                offsets = [(clock['realtime_ns'] - clock['perf_counter_after_ns'],
                            clock['realtime_ns'] - clock['perf_counter_before_ns']) for clock in [before, after]]
                # Observed bracketing only: retain offset spread and mark that
                # hidden clock steps and UTC origin precision remain unmeasured.
                low = min(offset[0] for offset in offsets)
                high = max(offset[1] for offset in offsets)
                sure_start = account['phase_started_ns'] + high - epoch
                sure_end = account['phase_finished_ns'] + low - epoch
                assert sure_start < sure_end
                unions = {}
                scopes = {}
                kernels = collections.defaultdict(list)
                for kind, table in TABLES.items():
                    if table not in tables:
                        scopes[kind] = {'state': 'unavailable'}
                        continue
                    columns = [row[1] for row in db.execute('PRAGMA table_info(' + table + ')')]
                    if not {'start', 'end'} <= set(columns):
                        scopes[kind] = {'state': 'unavailable'}
                        continue
                    name_column = next((column for column in ['demangledName', 'shortName', 'nameId'] if column in columns), None)
                    query = 'SELECT start,end' + (',' + name_column if name_column else '') + ' FROM ' + table + ' WHERE start>=? AND end<=?'
                    spans = []
                    for row in db.execute(query, (sure_start, sure_end)):
                        span = (row[0], row[1])
                        assert span[0] <= span[1]
                        spans.append(span)
                        if kind == 'kernel':
                            kernels[names.get(row[2], str(row[2])) if name_column else 'unresolved'].append(span)
                    unions[kind] = merged(spans)
                    crossing = db.execute('SELECT COUNT(*) FROM ' + table + ' WHERE start<? AND end>? AND NOT(start>=? AND end<=?)',
                                          (sure_end, sure_start, sure_start, sure_end)).fetchone()[0]
                    scopes[kind] = {'events_wholly_contained': len(spans), 'boundary_crossing_events_retained_separately': crossing,
                                    'work_sum_ns': length(spans), 'interval_union_ns': length(unions[kind]),
                                    'all_capture_events': full_counts[kind]}
                gpu_union = merged([span for kind in ['kernel', 'memcpy', 'memset'] for span in unions.get(kind, [])])
                work = sum(length(spans) for spans in kernels.values())
                ranked = sorted([{'name': name, 'events': len(spans), 'work_sum_ns': length(spans),
                                  'work_share_percent': 100 * length(spans) / work} for name, spans in kernels.items()],
                                key=lambda item: item['work_sum_ns'], reverse=True)
                phase_reports.append({'phase': phase, 'origin_utc_epoch_ns': epoch,
                                      'host_clock_offset_brackets_ns': offsets, 'observed_offset_spread_ns': high - low,
                                      'conservative_trace_window_ns': [sure_start, sure_end], 'scopes': scopes,
                                      'GPU_recorded_interval_union_ns': length(gpu_union),
                                      'pairwise_overlap_ns': {a + '__' + b: intersection(unions[a], unions[b])
                                                              for a in unions for b in unions if a < b},
                                      'kernels_by_work': ranked})
            reports.append({'case': label, 'sqlite_sha256': sha(db_path), 'recorded_table_counts': full_counts,
                            'diagnostic_CUDA_collected': collected, 'diagnostic_CUPTI_produced': produced,
                            'collection_warnings': [text for text in diagnostics if 'Not all CUDA' in text],
                            'all_diagnostics': diagnostics, 'phases': phase_reports})
        print(label + ' recorded counts and conservative phase windows analyzed', flush=True)
    result = {'reports': reports, 'profiles_analyzed': len(reports), 'source_archive_sha256': receipt['archive_sha256'],
              'HTTP_requests_independently_replayed': 1536,
              'time_mapping_basis': 'SQLite utcEpochNs capture origin plus sandwiched realtime/perf boundary observations',
              'clock_mapping_limits': 'no request-to-kernel causal link; hidden clock steps and profiler UTC origin precision unmeasured',
              'capture_completeness': 'recorded counts reconcile with collected diagnostics; produced category gap unclassified; total completeness unproven',
              'retention_policy': 'all original startup/warmup/retained/teardown traces and boundary events preserved; no HTTP sample exclusions',
              'work_timing_scope': 'instrumented recorded spans only; intersections/unions, not additive API+GPU elapsed time',
              'host_wait_scheduler_HTTP_cost': '미측정; no attribution from idle gaps',
              'method_reference': 'https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html',
              'serving_performance': '미실행; diagnostic only', 'goal_achieved': False}
    with (OUT / 'summary.json').open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
except BaseException as error:
    failure = {'type': type(error).__name__, 'message': str(error)}
    raise
finally:
    (OUT / 'completion.json').write_text(json.dumps({'failure': failure, 'created_ns': time.time_ns(), 'goal_achieved': False}, indent=2))
