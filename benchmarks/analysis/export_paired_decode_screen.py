"""Validate a terminal serving screen and retain reproducible compact evidence.

Offline only. Recompute every retained summary from client timestamps/tokens;
retain full raw-file hashes and replace large SSE frame arrays with hashes.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from paired_decode_serving_screen import summary


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(source, destination):
    preparation = json.loads((source / 'preparation.json').read_text())
    records = json.loads((source / 'completion.json').read_text())
    expected = 2 if preparation['smoke'] else 4 if preparation['riley_only'] else 8 if preparation['prior_paired_binary'] else 6
    assert len(records) == expected and len({r['name'] for r in records}) == expected
    groups = {}
    for record in records:
        name = record['name']
        lane = name.split('-', 2)[2]
        rows = json.loads((source / (name + '-retained.json')).read_text())
        measured = summary(rows)
        assert measured == {k: v for k, v in record.items() if k != 'name'}, name
        assert measured == json.loads((source / (name + '-summary.json')).read_text()), name
        assert measured['requests'] == preparation['retained'] and measured['errors'] == 0
        warmup = json.loads((source / (name + '-warmup.json')).read_text())
        assert len(warmup) == preparation['warmup'] and all(r['valid'] for r in warmup)
        if lane != 'vllm':
            assert measured['reference_matches'] == len(rows), name
            assert all(all(r['checks'].values()) for r in warmup), name
        assert json.loads((source / (name + '-exit.json')).read_text())['exit_code'] == 0, name
        groups.setdefault(lane, []).append(measured)
    comparison = {}
    for lane, repeats in groups.items():
        assert len(repeats) == (1 if preparation['smoke'] else 2), lane
        comparison[lane] = {
            'output_tokens_s': statistics.median(r['throughput_tokens_s'] for r in repeats),
            'errors': sum(r['errors'] for r in repeats),
            'reference_matches': sum(r['reference_matches'] for r in repeats),
            'retained_requests': sum(r['requests'] for r in repeats),
        }
        for metric in ['ttft', 'tpot', 'e2e', 'itl']:
            for percentile, label in [('0.5', 'p50'), ('0.95', 'p95'), ('0.99', 'p99')]:
                aggregate = statistics.median if percentile == '0.5' else max
                comparison[lane][metric + '_' + label + '_ms'] = aggregate(r[metric + '_ms'][percentile] for r in repeats)
    destination.mkdir()
    manifest = {}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        data = path.read_bytes()
        manifest[path.name] = {'sha256': digest(data), 'bytes': len(data)}
        if path.name.endswith(('-warmup.json', '-retained.json', '-stop.json')):
            rows = json.loads(data)
            for row in rows:
                if 'frames' in row:
                    row['frames_canonical_sha256'] = digest(json.dumps(row.pop('frames'), sort_keys=True, separators=(',', ':')).encode())
            (destination / (path.name + '.gz')).write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0))
        elif path.suffix == '.json':
            (destination / path.name).write_bytes(data)
    (destination / 'raw-manifest.json').write_text(json.dumps({'source': str(source.resolve()), 'files': manifest}, indent=2) + '\n')
    (destination / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps({'source': str(source), 'recomputed_lanes': len(records), 'requests': sum(r['requests'] for r in records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
