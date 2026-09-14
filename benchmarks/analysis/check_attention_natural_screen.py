#!/usr/bin/env python3
"""Fail-closed adjudication of the existing fixed natural-language screen.

Consumes offline metrics; never runs in Rust serving. This checks the metric
receipt, not raw logits, resource cleanup, generation parity or general quality.
Exit 0: relative metric screen passes; 1: fails; 2: invalid evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def adjudicate(report):
    require(type(report.get('targets')) is int and report['targets'] == 256,
            'expected 256 target positions')
    rows = report.get('passages')
    require(isinstance(rows, list) and len(rows) == 8, 'expected eight passages')
    require(all(type(row.get('index')) is int for row in rows), 'invalid index')
    require([row['index'] for row in rows] == list(range(8)),
            'passages must contain each frozen index once, in order')
    aggregate = {}
    for lane in ('baseline', 'candidate'):
        for row in rows:
            for key in ('nll', 'kl_from_fp32'):
                value = row[lane][key]
                require(type(value) in (float, int) and math.isfinite(value),
                        f'{lane} {key} must be finite')
            require(row[lane]['nll'] >= 0, 'negative NLL')
            # FP32 reduction can produce tiny negative KL; do not alter metrics.
            matches = row[lane]['argmax_fp32_matches']
            require(type(matches) is int and 0 <= matches <= 32, 'invalid argmax count')
        aggregate[lane] = {
            key: sum(row[lane][key] for row in rows) /
                 (1 if key == 'argmax_fp32_matches' else 8)
            for key in ('nll', 'kl_from_fp32', 'argmax_fp32_matches')
        }
        require(report['aggregate'][lane] == aggregate[lane],
                f'{lane} stored aggregate differs from passage arithmetic')
        digest = report['logit_hashes'][lane]
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                f'{lane} missing valid logit SHA256')
    conditions = {key: aggregate['candidate'][key] <= aggregate['baseline'][key]
                  for key in ('nll', 'kl_from_fp32')}
    passed = all(conditions.values())
    require(report.get('relative_screen_passed') is passed,
            'stored pass flag disagrees with immutable NLL/KL rule')
    require(report.get('general_quality_accepted') is False,
            'this screen cannot establish general quality')
    require(report.get('serving_measured') is False,
            'this screen cannot establish serving performance')
    return {'schema': 'attention-natural-metric-check-v1',
            'aggregate': aggregate, 'conditions': conditions,
            'relative_screen_passed': passed,
            'general_quality_accepted': False, 'serving_qualified': False,
            'raw_logits_verified': False, 'resource_cleanup_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path)
    args = parser.parse_args()
    try:
        raw = args.metrics.read_bytes()
        result = adjudicate(json.loads(raw))
        result['metrics_sha256'] = hashlib.sha256(raw).hexdigest()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['relative_screen_passed'] else 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({'evidence_valid': False, 'error': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
