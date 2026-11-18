"""Compare two isolated cache-off soaks and their Linux RSS observations."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def json_file(directory, name):
    return json.loads((directory / name).read_bytes())


def same_args(args, values_to_clear):
    normalized = list(args)
    for value in values_to_clear:
        pos = normalized.index(value)
        normalized[pos + 1] = value
    return normalized


def run(directory, lease_directory):
    receipt = json_file(directory, 'receipt.json')
    require(receipt['mode'] == 'cache-off' and receipt['prefix_cache_blocks'] == 0,
            'expected prefix cache disabled')
    require(receipt['status'] == 'passed' and receipt['observed_active']
            and receipt['client_exit'] == receipt['server_exit'] == 0,
            'expected successful active run and clean exits')
    archived = directory / 'soak.json.gz'
    raw = gzip.decompress(archived.read_bytes()) if archived.exists() else (directory / 'soak.json').read_bytes()
    shutdown_raw = (directory / 'shutdown-metrics.json').read_bytes()
    require(sha256(raw).hexdigest() == receipt['soak_sha256']
            and sha256(shutdown_raw).hexdigest() == receipt['shutdown_metrics_sha256'],
            'mismatched soak or shutdown receipt digest')
    soak = json.loads(raw)
    shutdown = json.loads(shutdown_raw)
    conditions = ('active_requests', 'waiting_requests', 'kv_allocated_blocks')
    require(not any(shutdown[key] for key in conditions)
            and not any(shutdown['allocation'].values()), 'shutdown leaked resources')
    require(not any(soak['final']['metrics'][key] for key in conditions)
            and soak['initial']['metrics']['allocation'] == soak['final']['metrics']['allocation'],
            'unexpected retained allocation at drain')
    require(soak['status'] == 'passed' and not soak['problem_count'] and all(soak['checks'].values()),
            'soak workload contract failed')
    require(soak['seconds'] == receipt['duration_seconds'] == 1800
            and soak['rate'] == 8 and soak['shared_prefixes'] == 8, 'unaligned workload profile')
    holder = json_file(lease_directory, 'idle-after.json')
    require(holder['pid'] == receipt['holder_pid'] and holder['start_ticks'] == receipt['holder_start_ticks']
            and holder['gpu'] == receipt['gpu_uuid'] and holder['physical_index'] == 7 and holder['phase'] == 'idle'
            and holder['status'] == 'running' and holder['allocation_bytes'] == 20 * 1024**3
            and holder['torch_reserved_bytes'] == 20 * 1024**3 and not holder['other_processes'],
            'GPU 7 lease did not return to the same owner and allocation')
    baseline = next(s for s in soak['rss_samples'] if s['at_seconds'] >= soak['seconds'] / 2)
    end = soak['final']
    growth = end['rss_mib'] - baseline['rss_mib']
    require(growth == soak['rss_growth_second_half_mib'], 'incorrect RSS growth field')
    return receipt, soak, {
        'normal_completions': sum(counts.get('completed', 0) for counts in soak['counts'].values()),
        'intentional_disconnects': soak['counts']['cancel']['disconnected'],
        'counter_changes': soak['counter_changes'],
        'p95_change_percent': soak['p95_change_percent'],
        'rss_first_late': {k: baseline[k] for k in ('at_seconds', 'rss_mib')},
        'rss_at_drain': {k: end[k] for k in ('at_seconds', 'rss_mib')},
        'rss_growth_second_half_mib': growth,
        'rss_zero_growth': growth <= 0,
        'native_allocation_at_ready_and_drain': end['metrics']['allocation'],
        'native_allocation_after_shutdown': shutdown['allocation'],
        'soak_sha256': receipt['soak_sha256'],
        'shutdown_metrics_sha256': receipt['shutdown_metrics_sha256'],
    }


def compare(original, repeated):
    require((repeated / 'campaign.exit').read_text().strip() == '0', 'repeat campaign did not finish')
    original_receipt, original_soak, original_summary = run(original, original)
    repeat_receipt, repeat_soak, repeat_summary = run(repeated / 'cache-off', repeated)
    for key in ('gpu_uuid', 'holder_pid', 'holder_start_ticks', 'riley_source'):
        require(original_receipt[key] == repeat_receipt[key], f'changed {key}')
    digests = [{Path(path).name: digest for path, digest in rec['sha256'].items()}
               for rec in (original_receipt, repeat_receipt)]
    require(len(digests[0]) == len(digests[1]) == 6 and digests[0] == digests[1], 'changed inputs')
    require(same_args(original_receipt['server_argv'], ('--bind',))
            == same_args(repeat_receipt['server_argv'], ('--bind',)), 'changed server flags')
    clients = []
    for receipt in (original_receipt, repeat_receipt):
        argv = same_args(receipt['client_argv'], ('--pid', '--out'))
        argv[2] = Path(argv[2]).name
        argv[3] = 'loopback'
        clients.append(argv)
    require(clients[0] == clients[1], 'changed client flags')
    require(original_soak['counts'] == repeat_soak['counts'], 'changed workload outcome counts')
    before, after = (json_file(repeated, name) for name in ('idle-before.json', 'idle-after.json'))
    lease_keys = ('pid', 'start_ticks', 'phase', 'physical_index', 'gpu', 'allocation_bytes',
                  'torch_reserved_bytes', 'other_processes')
    require(all(before[key] == after[key] for key in lease_keys), 'the GPU reservation changed')
    observer = (repeated / 'rss-observer' / 'samples.jsonl').read_bytes()
    samples = [json.loads(line) for line in observer.splitlines()]
    argv = repeat_receipt['client_argv']
    pid = int(argv[argv.index('--pid') + 1])
    require(len(samples) > 1 and {s['pid'] for s in samples} == {pid}
            and len({s['start_ticks'] for s in samples}) == 1, 'observer server identity differs')
    require(all(left['unix_seconds'] < right['unix_seconds'] for left, right in zip(samples, samples[1:])),
            'unordered observer timeline')
    fields = ('RssAnon', 'RssFile', 'RssShmem')
    require(all(sum(s['kib'][field] for field in fields) == s['kib']['VmRSS'] for s in samples),
            'RSS totals and components differ')
    first, last = samples[0], samples[-1]
    return {
        'original': original_summary, 'repeat': repeat_summary,
        'observer': {
            'samples': len(samples), 'first': first, 'last': last,
            'delta_kib': {field: last['kib'][field] - first['kib'][field] for field in first['kib']},
            'distinct_kib_values': {field: sorted({s['kib'][field] for s in samples})
                                    for field in first['kib']},
            'samples_sha256': sha256(observer).hexdigest(),
        },
        'matched_input_sha256': digests[0],
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('original', type=Path)
    parser.add_argument('repeated', type=Path, nargs='?')
    args = parser.parse_args()
    output = compare(args.original, args.repeated) if args.repeated else run(args.original, args.original)[2]
    print(json.dumps(output, indent=2))
