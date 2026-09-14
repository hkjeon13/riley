"""Verify reclassification against immutable exported per-kernel timings."""
import collections
import hashlib
import json
import math
from pathlib import Path
import tarfile

from serving_stage_census import area


def main():
    root = Path(__file__).resolve().parents[2]
    directory = root / 'benchmarks/results/20260914-dense-wire-serving-c32'
    receipt = json.loads((directory / 'reclassified-areas.json').read_text())
    archive = directory / 'profile-numeric.tar.gz'
    classifier = root / 'benchmarks/analysis/serving_stage_census.py'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['source_archive_sha256']
    assert hashlib.sha256(classifier.read_bytes()).hexdigest() == receipt['classifier_sha256']
    expected_lanes = {'control-shared', 'candidate-shared', 'control-unique', 'candidate-unique'}
    assert set(receipt['lanes']) == expected_lanes
    checked = 0
    with tarfile.open(archive) as source:
        for lane in sorted(expected_lanes):
            original = json.load(source.extractfile(f'profile/{lane}-areas.json'))['stages']
            revised = receipt['lanes'][lane]
            assert set(original) == set(revised)
            for stage, values in original.items():
                actual = revised[stage]
                for key in ('launches', 'graph_span_ms', 'kernel_ms'):
                    assert actual[key] == values[key], (lane, stage, key)
                areas = collections.defaultdict(float)
                for kernel, duration in values['kernels_ms'].items():
                    assert math.isfinite(duration) and duration >= 0
                    areas[area(kernel)] += duration
                assert dict(areas) == actual['areas_ms'], (lane, stage)
                assert math.isclose(sum(areas.values()), values['kernel_ms'], abs_tol=1e-6)
                checked += 1
    print(f'Validated {checked} stage summaries across four traces; all original timings preserved.')


if __name__ == '__main__':
    main()
