"""Archive compact per-request evidence after a terminal serving screen.

Full HTTP frames remain in the source directory; retain their file hashes.
This is an offline exporter and never runs inside Riley.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:3])
expected_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
completion = json.loads((source / 'completion.json').read_text())
assert len(completion['records']) == expected_runs, 'screen is incomplete'
destination.mkdir()
manifest = {}
for path in sorted(source.glob('*-rows.json')):
    data = path.read_bytes()
    manifest[path.name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    rows = json.loads(data)
    for row in rows:
        for field in ('frames', 'request', 'text'):
            if field in row:
                value = row.pop(field)
                row[field + '_canonical_json_sha256'] = hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest()
    payload = json.dumps(rows, separators=(',', ':')).encode()
    (destination / (path.name + '.gz')).write_bytes(gzip.compress(payload, mtime=0))
(source / 'raw-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'source': str(source), 'files': len(manifest),
                  'rows': sum(json.loads(p.read_text()).__len__() for p in source.glob('*-rows.json'))}))
