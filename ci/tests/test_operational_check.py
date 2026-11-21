"""Keep the cache counter acceptance checks independent of CUDA."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/live"))
from operational_check import check_prefix_cache_metrics
sys.path.pop(0)


def idle_cache(retained_blocks, hit_tokens, evicted_blocks):
    return {
        "active_requests": 0,
        "waiting_requests": 0,
        "kv_allocated_blocks": 0,
        "prefix_cache": {
            "retained_blocks": retained_blocks,
            "hit_tokens": hit_tokens,
            "evicted_blocks": evicted_blocks,
            "metrics_degraded": False,
        },
    }


class PrefixCacheMetricsTest(unittest.TestCase):
    def setUp(self):
        self.samples = {
            "before": idle_cache(4, 0, 0),
            "cold": idle_cache(6, 0, 0),
            "warm": idle_cache(6, 96, 0),
            "pressure": idle_cache(8, 112, 8),
            "after_pressure": idle_cache(8, 128, 14),
            "rewarmed": idle_cache(8, 224, 14),
            "drained": idle_cache(8, 240, 14),
            "final": idle_cache(8, 336, 14),
        }
        self.rows = {label: {"cached_tokens": value} for label, value in
                     zip(("cold", "warm", "pressure", "after_pressure", "rewarmed"), (0, 96, 16, 16, 96), strict=True)}
        self.recovery = {"cached_tokens": 96}

    def test_idle_retention_and_usage_deltas_pass_with_burst_between_samples(self):
        result = check_prefix_cache_metrics(self.samples, self.rows, self.recovery, 8)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["hit_token_deltas"], {**{label: row["cached_tokens"]
                                                         for label, row in self.rows.items()}, "recovery": 96})

    def test_invalid_occupancy_degradation_and_missing_eviction_fail(self):
        for label, field, value in (
            ("final", "retained_blocks", 9),
            ("cold", "metrics_degraded", True),
            ("pressure", "evicted_blocks", 0),
            ("warm", "hit_tokens", 80),
            ("after_pressure", "evicted_blocks", 7),
        ):
            with self.subTest(label=label, field=field):
                samples = deepcopy(self.samples)
                samples[label]["prefix_cache"][field] = value
                self.assertEqual(check_prefix_cache_metrics(samples, self.rows, self.recovery, 8)["status"], "failed")

    def test_disabled_cache_has_zero_counts_even_before_the_first_request(self):
        rows = {label: {"cached_tokens": 0} for label in self.rows}
        samples = {label: idle_cache(0, 0, 0) for label in self.samples}
        recovery = {"cached_tokens": 0}
        self.assertEqual(check_prefix_cache_metrics(samples, rows, recovery, 0)["status"], "passed")
        for sample in samples.values():
            sample["prefix_cache"]["hit_tokens"] = 16
        self.assertEqual(check_prefix_cache_metrics(samples, rows, recovery, 0)["status"], "failed")
