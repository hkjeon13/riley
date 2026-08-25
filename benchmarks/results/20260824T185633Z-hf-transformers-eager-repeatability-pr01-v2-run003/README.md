# PR-01 repeatability evidence

This directory is an append-only import of one passing, externally staged
repeatability gate for lane `hf-transformers` at Git revision
`09911ba2630845e9d4094b7c33c3ff65931a919c`.

## Exact runner invocation

```shell
/tmp/rustinfer-pr01-lock-20260824/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13 /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/scripts/run_repeatability_gate.py --lane hf-transformers --output-root /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging --uv /tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv --finalize-to benchmarks/results/20260824T185633Z-hf-transformers-eager-repeatability-pr01-v2-run003
```

## Summary

- Gate status: `passed`
- Independent runs: 5
- Predeclared cells per run: 4
- Fresh single-cell benchmark subprocesses: 20
- Combined raw observations: 455 JSONL rows
- Combined raw SHA-256: `5e02c537e1a155942e88d2086585f2b82a34660aeeba220fbaa5c111b93c1bef`

```json
{
  "passed": true,
  "status": "passed",
  "thresholds": {
    "cold_model_load_p50_cv_max": 0.1,
    "failure_count_max": 0.0,
    "peak_vram_relative_range_max": 0.01,
    "throughput_cv_max": 0.05,
    "warm_p50_cv_max": 0.05,
    "warm_p95_cv_max": 0.1
  }
}
```

## Variance and threshold evidence

The values below are copied from the passing repeatability report. Statistical
definitions and the complete report remain in `repeatability-report.json`.

```json
[
  {
    "checks": [
      {
        "name": "throughput_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.05,
        "value": 0.028357076304637914
      },
      {
        "name": "peak_vram_relative_range_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.01,
        "value": 0.0
      },
      {
        "name": "warm_p50_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.05,
        "value": 0.028012483931611893
      },
      {
        "name": "warm_p95_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.1,
        "value": 0.037965786652735575
      },
      {
        "name": "failure_count_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.0,
        "value": 0.0
      }
    ],
    "errors": [],
    "independent_run_count": 5,
    "passed": true,
    "required_independent_runs": 5,
    "required_trials_per_run": 30,
    "run_summaries": [
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 380.5293120094575,
          "r7_p95": 411.88813075132197
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 11.789003503508866,
          "r7_p95": 15.50704890905763
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 11.912175032193772,
          "r7_p95": 12.939529943574714
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-01",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 84.0934606512561
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 393.0469179977081,
          "r7_p95": 434.10128864998114
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 12.303179493756033,
          "r7_p95": 16.100708404701432
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 12.293744032156669,
          "r7_p95": 13.61585022724842
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-02",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 81.41528424011574
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 392.56663800915703,
          "r7_p95": 420.7400019564375
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 12.148554502346087,
          "r7_p95": 15.776866494707061
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 12.269532709568363,
          "r7_p95": 13.186578565810416
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-03",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 81.51499645370257
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 366.9721295009367,
          "r7_p95": 394.04870640137227
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 11.479410495667253,
          "r7_p95": 14.793005299725337
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 11.489481161275293,
          "r7_p95": 12.352107363145596
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-04",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 87.2020136669294
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 380.8355815053801,
          "r7_p95": 429.39213704667054
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 11.926034501811955,
          "r7_p95": 15.765963989542795
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 11.915511612906196,
          "r7_p95": 13.462246271134767
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-05",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 84.02602169662168
        },
        "trial_count": 30
      }
    ],
    "statistics": {
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.028357076304637914,
      "warm_end_to_end_r7_p50_sample_cv": 0.028012483931611893,
      "warm_end_to_end_r7_p95_sample_cv": 0.037965786652735575,
      "warm_pooled_itl_r7_p50_sample_cv": 0.026836399390010465,
      "warm_pooled_itl_r7_p95_sample_cv": 0.03156980360184945,
      "warm_request_mean_tpot_r7_p50_sample_cv": 0.02742669257044269,
      "warm_request_mean_tpot_r7_p95_sample_cv": 0.03793018790351967
    },
    "workload": {
      "concurrency": 1,
      "output_tokens": 32,
      "prompt_tokens": 128,
      "warm_state": "warm"
    }
  },
  {
    "checks": [
      {
        "name": "throughput_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.05,
        "value": 0.026667999961592567
      },
      {
        "name": "peak_vram_relative_range_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.01,
        "value": 0.0
      },
      {
        "name": "warm_p50_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.05,
        "value": 0.027015908189455925
      },
      {
        "name": "warm_p95_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.1,
        "value": 0.06798589215511058
      },
      {
        "name": "failure_count_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.0,
        "value": 0.0
      }
    ],
    "errors": [],
    "independent_run_count": 5,
    "passed": true,
    "required_independent_runs": 5,
    "required_trials_per_run": 30,
    "run_summaries": [
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 1788.5778245035908,
          "r7_p95": 2048.2075941479707
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 3488677888.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 12.137593002989888,
          "r7_p95": 17.99785095281547
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 12.419219480382415,
          "r7_p95": 14.457986345303823
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-01",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 71.56527891094572
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 1747.1407570046722,
          "r7_p95": 1815.0010487464897
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 3488677888.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 11.982310003077146,
          "r7_p95": 15.051055699586865
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 12.0938963740427,
          "r7_p95": 12.627139729538182
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-02",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 73.26266326711391
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 1687.9865755036008,
          "r7_p95": 1738.1990270449023
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 3488677888.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 11.493311503727455,
          "r7_p95": 14.411827448930124
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 11.627707389773668,
          "r7_p95": 12.022312088553495
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-03",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 75.83002274555938
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 1676.021221501287,
          "r7_p95": 1756.7904705509136
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 3488677888.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 11.500511005579028,
          "r7_p95": 14.43605709791881
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 11.533100003977928,
          "r7_p95": 12.169597543708202
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-04",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 76.37140174384658
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 1704.2095684955711,
          "r7_p95": 1806.8171257989889
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 3488677888.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 11.679539005854167,
          "r7_p95": 15.0097194549744
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 11.756737606268096,
          "r7_p95": 12.56151157325889
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-05",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 75.10813437166877
        },
        "trial_count": 30
      }
    ],
    "statistics": {
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.026667999961592567,
      "warm_end_to_end_r7_p50_sample_cv": 0.027015908189455925,
      "warm_end_to_end_r7_p95_sample_cv": 0.06798589215511058,
      "warm_pooled_itl_r7_p50_sample_cv": 0.02468175799557748,
      "warm_pooled_itl_r7_p95_sample_cv": 0.09712378919148693,
      "warm_request_mean_tpot_r7_p50_sample_cv": 0.030783050181047704,
      "warm_request_mean_tpot_r7_p95_sample_cv": 0.07666685359458736
    },
    "workload": {
      "concurrency": 1,
      "output_tokens": 128,
      "prompt_tokens": 4096,
      "warm_state": "warm"
    }
  },
  {
    "checks": [
      {
        "name": "throughput_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.05,
        "value": 0.024024211901044423
      },
      {
        "name": "peak_vram_relative_range_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.01,
        "value": 0.0
      },
      {
        "name": "warm_p50_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.05,
        "value": 0.024328988742864134
      },
      {
        "name": "warm_p95_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.1,
        "value": 0.04580353628011142
      },
      {
        "name": "failure_count_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.0,
        "value": 0.0
      }
    ],
    "errors": [],
    "independent_run_count": 5,
    "passed": true,
    "required_independent_runs": 5,
    "required_trials_per_run": 30,
    "run_summaries": [
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 409.28522100148257,
          "r7_p95": 460.4263799992623
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1641086976.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 12.452185503207147,
          "r7_p95": 17.496104002930224
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 12.81196243554962,
          "r7_p95": 14.478048483837366
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-01",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 625.4806843274334
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 383.91832950583193,
          "r7_p95": 412.4900470051216
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1641086976.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 11.948428495088592,
          "r7_p95": 15.379725009552203
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 12.024458032336101,
          "r7_p95": 12.970071515977203
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-02",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 666.8090236423068
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 397.4310885023442,
          "r7_p95": 438.2740129949525
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1641086976.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 12.225225000292994,
          "r7_p95": 16.73673899495043
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 12.464489516259487,
          "r7_p95": 13.795807612628016
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-03",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 644.1371869505365
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 389.54002450191183,
          "r7_p95": 428.1515490001766
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1641086976.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 12.222095996548887,
          "r7_p95": 15.978907991666347
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 12.15502282256268,
          "r7_p95": 13.435475613204613
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-04",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 657.187605965267
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 392.3309145029634,
          "r7_p95": 414.0438489994267
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 1641086976.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 12.225964506797027,
          "r7_p95": 15.568692004308105
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 12.305092370943091,
          "r7_p95": 12.979512580768056
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-05",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 652.5116086959987
        },
        "trial_count": 30
      }
    ],
    "statistics": {
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.024024211901044423,
      "warm_end_to_end_r7_p50_sample_cv": 0.024328988742864134,
      "warm_end_to_end_r7_p95_sample_cv": 0.04580353628011142,
      "warm_pooled_itl_r7_p50_sample_cv": 0.014621651959345795,
      "warm_pooled_itl_r7_p95_sample_cv": 0.05408697778865871,
      "warm_request_mean_tpot_r7_p50_sample_cv": 0.024705321795466393,
      "warm_request_mean_tpot_r7_p95_sample_cv": 0.046664336071435275
    },
    "workload": {
      "concurrency": 8,
      "output_tokens": 32,
      "prompt_tokens": 128,
      "warm_state": "warm"
    }
  },
  {
    "checks": [
      {
        "name": "peak_vram_relative_range_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.01,
        "value": 0.0
      },
      {
        "name": "cold_model_load_p50_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.1,
        "value": 0.009848457471420314
      },
      {
        "name": "failure_count_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.0,
        "value": 0.0
      }
    ],
    "errors": [],
    "independent_run_count": 5,
    "passed": true,
    "required_independent_runs": 5,
    "required_trials_per_run": 1,
    "run_summaries": [
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 4810.870804998558
        },
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 1
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-01",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 34.5434217083804
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 4766.030860002502
        },
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 1
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-02",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 36.37516295007009
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 4715.654721992905
        },
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 1
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-03",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 34.36176898049622
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 4767.365439009154
        },
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 1
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-04",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 36.81793258296836
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 4692.441357008647
        },
        "peak_vram_bytes": {
          "maximum": 1609629696.0,
          "observation_count": 1
        },
        "run_id": "hf-transformers-20260824T185646.150Z-55a397313acd-run-05",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 37.61945136884255
        },
        "trial_count": 1
      }
    ],
    "statistics": {
      "cold_model_load_r7_p50_sample_cv": 0.009848457471420314,
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.039887276250608644
    },
    "workload": {
      "concurrency": 1,
      "output_tokens": 32,
      "prompt_tokens": 128,
      "warm_state": "cold"
    }
  }
]
```

## Comparability

```json
{
  "engine_revision": "transformers-5.15.1+torch-2.13.0",
  "environment_id": "rtx4090-ubuntu22-driver580-v1",
  "git_dirty": false,
  "git_revision": "09911ba2630845e9d4094b7c33c3ff65931a919c",
  "lane_manifest_sha256": "e84ddc2ee30d5734b7490b36d95350b4c51379a57580d9665987ffa7fdabe645",
  "matrix_id": "smollm2-135m-rtx4090-bf16-v1",
  "matrix_sha256": "a979659ef9d7b3c5a7a85e423347eb6f06ccbd3ae5a370056bd056d3137c7e87",
  "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
  "prompts_sha256": "709612e45d735888b240951d51b979b7ded1e87ef6cae9296f0b1250647255d2",
  "scope": "end-to-end"
}
```

All 20 preflight stdout/stderr snapshots, the canonical baseline, lane logs,
per-cell raw files, dependency hashes, and exact argv/environment evidence are
preserved below this directory. `raw.jsonl` is the canonical deterministic
concatenation of the 20 per-cell raw files and was checked again before import.

## Cache preparation and cold scope

`cold` means a fresh benchmark subprocess and freshly loaded model state. It is
not an OS page-cache, immutable model-file cache, uv package cache, or compiled
kernel disk-cache cold start. Before measurement the runner completed offline
`uv sync --frozen --offline`; for the selected lane it also ran one unmeasured
fresh process for every distinct repeatability compile/model profile. Those prime raws are preserved under
`preparation/` and excluded from the checker. The external cache roots below
were reused unchanged by all 20 measured invocations; every invocation stored
an inventory fingerprint equal to the post-prime baseline.
The complete before/after entry lists are preserved as deterministic gzip JSON
at `preparation/cache.inventory.before.json.gz` and
`preparation/cache.inventory.after.json.gz` (level 9, `mtime=0`).

```json
{
  "cache_inventory_after": {
    "aggregate_sha256": "4f3b41ba58f1d78313fff29687270b50f8c2e8eb3a0354c112e146992da3abec",
    "captured_at_utc": "2026-08-24T18:59:05.217Z",
    "contract_version": "rustinfer.cache-inventory.v1",
    "roots": [
      {
        "entry_count": 58690,
        "environment_key": "UV_CACHE_DIR",
        "fingerprint_sha256": "5f278a6dfdcee3333863e471ef1e7ad825c03e70f8d27cd1aa3cd85cca2f5b47",
        "path": "/tmp/rustinfer-pr01-lock-20260824/cache",
        "total_regular_file_bytes": 8045975698
      },
      {
        "entry_count": 533877,
        "environment_key": "UV_PYTHON_INSTALL_DIR",
        "fingerprint_sha256": "e3cda22fccff43db40eb075cb529a5399e20609f610a22d289066270231c838e",
        "path": "/tmp/rustinfer-pr01-lock-20260824/python",
        "total_regular_file_bytes": 93584426245
      },
      {
        "entry_count": 45,
        "environment_key": "HF_HOME",
        "fingerprint_sha256": "26673b995546ee90fe082e808a7e14f1276a76c4895bd75d03a11ec9ae8bb489",
        "path": "/tmp/rustinfer-pr01-lock-20260824/hf",
        "total_regular_file_bytes": 272520960
      },
      {
        "entry_count": 828,
        "environment_key": "VLLM_CACHE_ROOT",
        "fingerprint_sha256": "c4bec8c0f82ca8c0624e9937a79eaffc40308093dad90a954c8e4a0c91eb508f",
        "path": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "total_regular_file_bytes": 65933152
      },
      {
        "entry_count": 824,
        "environment_key": "TORCHINDUCTOR_CACHE_DIR",
        "fingerprint_sha256": "967f7a41c2db7de8af61e25487548135cbae75366d1b0c87fd9928189527d72a",
        "path": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
        "total_regular_file_bytes": 65932328
      },
      {
        "entry_count": 533,
        "environment_key": "TRITON_CACHE_DIR",
        "fingerprint_sha256": "bf6eb5fc4ecf2ea5c16e943fceb7bee0163fd49b11b77f4bc828aaf94351c07e",
        "path": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
        "total_regular_file_bytes": 14164248
      },
      {
        "entry_count": 0,
        "environment_key": "CUDA_CACHE_PATH",
        "fingerprint_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "path": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
        "total_regular_file_bytes": 0
      }
    ]
  },
  "cache_inventory_artifact": {
    "content_type": "application/json",
    "contract_version": "rustinfer.cache-inventory-artifact.v1",
    "encoding": "gzip",
    "gzip_compresslevel": 9,
    "gzip_mtime": 0,
    "json_serialization": "utf8-sort-keys-compact-newline"
  },
  "cache_inventory_before": {
    "aggregate_sha256": "d2aebdddfbee437a8b6d10854265183168837dd9088bb7d62c370a0d38922877",
    "captured_at_utc": "2026-08-24T18:57:02.001Z",
    "contract_version": "rustinfer.cache-inventory.v1",
    "roots": [
      {
        "entry_count": 58675,
        "environment_key": "UV_CACHE_DIR",
        "fingerprint_sha256": "8f6d8b84dff8688e0cbb94cf7d01c6f3240475ea2bdff1ec57cfe4a27e7e8487",
        "path": "/tmp/rustinfer-pr01-lock-20260824/cache",
        "total_regular_file_bytes": 8045970677
      },
      {
        "entry_count": 511247,
        "environment_key": "UV_PYTHON_INSTALL_DIR",
        "fingerprint_sha256": "2d212824ae43191c44e79c8b960f99fc386bd301678670c378c5f0640ed92e58",
        "path": "/tmp/rustinfer-pr01-lock-20260824/python",
        "total_regular_file_bytes": 88698679515
      },
      {
        "entry_count": 45,
        "environment_key": "HF_HOME",
        "fingerprint_sha256": "26673b995546ee90fe082e808a7e14f1276a76c4895bd75d03a11ec9ae8bb489",
        "path": "/tmp/rustinfer-pr01-lock-20260824/hf",
        "total_regular_file_bytes": 272520960
      },
      {
        "entry_count": 828,
        "environment_key": "VLLM_CACHE_ROOT",
        "fingerprint_sha256": "c4bec8c0f82ca8c0624e9937a79eaffc40308093dad90a954c8e4a0c91eb508f",
        "path": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "total_regular_file_bytes": 65933152
      },
      {
        "entry_count": 824,
        "environment_key": "TORCHINDUCTOR_CACHE_DIR",
        "fingerprint_sha256": "967f7a41c2db7de8af61e25487548135cbae75366d1b0c87fd9928189527d72a",
        "path": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
        "total_regular_file_bytes": 65932328
      },
      {
        "entry_count": 533,
        "environment_key": "TRITON_CACHE_DIR",
        "fingerprint_sha256": "bf6eb5fc4ecf2ea5c16e943fceb7bee0163fd49b11b77f4bc828aaf94351c07e",
        "path": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
        "total_regular_file_bytes": 14164248
      },
      {
        "entry_count": 0,
        "environment_key": "CUDA_CACHE_PATH",
        "fingerprint_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "path": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
        "total_regular_file_bytes": 0
      }
    ]
  },
  "completed_at_utc": "2026-08-24T18:59:06.529Z",
  "contract_version": "rustinfer.repeatability-preparation.v2",
  "immutable_evidence": {
    "dependency_lock_sha256": "101d21486780e57492b3053149c0a594fcf2859d1955854250bd644b6fdaff30",
    "lane_manifest_sha256": "e84ddc2ee30d5734b7490b36d95350b4c51379a57580d9665987ffa7fdabe645",
    "model_id": "HuggingFaceTB/SmolLM2-135M",
    "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "project_environment": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/hf-transformers-101d21486780e574-55a397313acd",
    "uv_sha256": "b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46",
    "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
  },
  "measured_cache_baseline_sha256": "4f3b41ba58f1d78313fff29687270b50f8c2e8eb3a0354c112e146992da3abec",
  "policy": "the pinned managed Python is resolved and verified; uv lock synchronization is offline into one fresh external project environment; the selected lane primes every distinct repeatability compile profile in fresh unmeasured subprocesses",
  "prime_invocations": [
    {
      "argv": [
        "/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv",
        "run",
        "--frozen",
        "--offline",
        "--no-sync",
        "--project",
        "tools/python/reference",
        "rustinfer-reference",
        "benchmark",
        "--matrix",
        "/tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml",
        "--prompts",
        "/tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl",
        "--result-dir",
        "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-01-warm-c1-p128-o32/result",
        "--run-index",
        "1",
        "--run-id",
        "hf-transformers-cache-prime-55a397313acd-01",
        "--warm-state",
        "warm",
        "--concurrency",
        "1",
        "--prompt-tokens",
        "128",
        "--output-tokens",
        "32"
      ],
      "cell": {
        "concurrency": 1,
        "output_tokens": 32,
        "prompt_tokens": 128,
        "warm_state": "warm"
      },
      "environment": {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_CACHE_MAXSIZE": "4294967296",
        "CUDA_CACHE_PATH": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
        "DO_NOT_TRACK": "1",
        "HF_HOME": "/tmp/rustinfer-pr01-lock-20260824/hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/psyche",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
        "TRANSFORMERS_OFFLINE": "1",
        "TRITON_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
        "UV_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/cache",
        "UV_OFFLINE": "1",
        "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/hf-transformers-101d21486780e574-55a397313acd",
        "UV_PYTHON": "3.13.15",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
        "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1"
      },
      "finished_at_utc": "2026-08-24T18:57:24.199Z",
      "prime_index": 1,
      "raw_result": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-01-warm-c1-p128-o32/result/raw.jsonl",
      "raw_result_row_count": 30,
      "raw_result_sha256": "e1ff5daf5916617d9d60044fba3f6087bad535d47cde3f9e4339f8994e955b70",
      "returncode": 0,
      "started_at_utc": "2026-08-24T18:57:03.619Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-01-warm-c1-p128-o32/benchmark.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-01-warm-c1-p128-o32/benchmark.stdout.txt"
    },
    {
      "argv": [
        "/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv",
        "run",
        "--frozen",
        "--offline",
        "--no-sync",
        "--project",
        "tools/python/reference",
        "rustinfer-reference",
        "benchmark",
        "--matrix",
        "/tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml",
        "--prompts",
        "/tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl",
        "--result-dir",
        "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-02-warm-c1-p4096-o128/result",
        "--run-index",
        "1",
        "--run-id",
        "hf-transformers-cache-prime-55a397313acd-02",
        "--warm-state",
        "warm",
        "--concurrency",
        "1",
        "--prompt-tokens",
        "4096",
        "--output-tokens",
        "128"
      ],
      "cell": {
        "concurrency": 1,
        "output_tokens": 128,
        "prompt_tokens": 4096,
        "warm_state": "warm"
      },
      "environment": {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_CACHE_MAXSIZE": "4294967296",
        "CUDA_CACHE_PATH": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
        "DO_NOT_TRACK": "1",
        "HF_HOME": "/tmp/rustinfer-pr01-lock-20260824/hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/psyche",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
        "TRANSFORMERS_OFFLINE": "1",
        "TRITON_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
        "UV_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/cache",
        "UV_OFFLINE": "1",
        "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/hf-transformers-101d21486780e574-55a397313acd",
        "UV_PYTHON": "3.13.15",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
        "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1"
      },
      "finished_at_utc": "2026-08-24T18:58:29.855Z",
      "prime_index": 2,
      "raw_result": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-02-warm-c1-p4096-o128/result/raw.jsonl",
      "raw_result_row_count": 30,
      "raw_result_sha256": "ae43b4c9a6a3e8cb29161d6a69e9b3126c1be40aeeface23a67cf06d4b7be330",
      "returncode": 0,
      "started_at_utc": "2026-08-24T18:57:24.214Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-02-warm-c1-p4096-o128/benchmark.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-02-warm-c1-p4096-o128/benchmark.stdout.txt"
    },
    {
      "argv": [
        "/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv",
        "run",
        "--frozen",
        "--offline",
        "--no-sync",
        "--project",
        "tools/python/reference",
        "rustinfer-reference",
        "benchmark",
        "--matrix",
        "/tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml",
        "--prompts",
        "/tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl",
        "--result-dir",
        "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-03-warm-c8-p128-o32/result",
        "--run-index",
        "1",
        "--run-id",
        "hf-transformers-cache-prime-55a397313acd-03",
        "--warm-state",
        "warm",
        "--concurrency",
        "8",
        "--prompt-tokens",
        "128",
        "--output-tokens",
        "32"
      ],
      "cell": {
        "concurrency": 8,
        "output_tokens": 32,
        "prompt_tokens": 128,
        "warm_state": "warm"
      },
      "environment": {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_CACHE_MAXSIZE": "4294967296",
        "CUDA_CACHE_PATH": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
        "DO_NOT_TRACK": "1",
        "HF_HOME": "/tmp/rustinfer-pr01-lock-20260824/hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/psyche",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
        "TRANSFORMERS_OFFLINE": "1",
        "TRITON_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
        "UV_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/cache",
        "UV_OFFLINE": "1",
        "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/hf-transformers-101d21486780e574-55a397313acd",
        "UV_PYTHON": "3.13.15",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
        "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1"
      },
      "finished_at_utc": "2026-08-24T18:58:50.553Z",
      "prime_index": 3,
      "raw_result": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-03-warm-c8-p128-o32/result/raw.jsonl",
      "raw_result_row_count": 30,
      "raw_result_sha256": "d376e8374ee57d00c7aad7a79ad928ca8a4e42125b35c78f44d4a7a588bd9d66",
      "returncode": 0,
      "started_at_utc": "2026-08-24T18:58:29.877Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-03-warm-c8-p128-o32/benchmark.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/prime-03-warm-c8-p128-o32/benchmark.stdout.txt"
    }
  ],
  "prime_results_excluded_from_checker": true,
  "project_environment": {
    "covered_by_cache_inventory": "UV_PYTHON_INSTALL_DIR",
    "dependency_lock_sha256": "101d21486780e57492b3053149c0a594fcf2859d1955854250bd644b6fdaff30",
    "derivation": "UV_PYTHON_INSTALL_DIR/project-environments/<lane>-<lock-prefix16>-<nonce>",
    "lane_id": "hf-transformers",
    "path": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/hf-transformers-101d21486780e574-55a397313acd",
    "repository_external": true,
    "variable": "UV_PROJECT_ENVIRONMENT"
  },
  "python_evidence": {
    "contract_version": "rustinfer.python-runtime-evidence.v1",
    "managed_python": {
      "implementation": "cpython",
      "launcher_path": "/tmp/rustinfer-pr01-lock-20260824/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13",
      "machine": "x86_64",
      "path": "/tmp/rustinfer-pr01-lock-20260824/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13",
      "platform": "linux",
      "reported_executable": "/tmp/rustinfer-pr01-lock-20260824/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13",
      "sha256": "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866",
      "version": "3.13.15"
    },
    "project_python": {
      "implementation": "cpython",
      "launcher_path": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/hf-transformers-101d21486780e574-55a397313acd/bin/python",
      "machine": "x86_64",
      "path": "/tmp/rustinfer-pr01-lock-20260824/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13",
      "platform": "linux",
      "reported_executable": "/tmp/rustinfer-pr01-lock-20260824/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13",
      "sha256": "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866",
      "version": "3.13.15"
    },
    "same_binary_sha256": true,
    "uv_python_find": {
      "argv": [
        "/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv",
        "python",
        "find",
        "3.13.15"
      ],
      "finished_at_utc": "2026-08-24T18:57:03.266Z",
      "started_at_utc": "2026-08-24T18:57:03.257Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/uv-python-find.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/uv-python-find.stdout.txt"
    }
  },
  "reproducibility_environment": {
    "allowlisted_values": {
      "CUDA_CACHE_PATH": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
      "HF_HOME": "/tmp/rustinfer-pr01-lock-20260824/hf",
      "HF_HUB_OFFLINE": "1",
      "TORCHINDUCTOR_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
      "TRANSFORMERS_OFFLINE": "1",
      "TRITON_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
      "UV_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/cache",
      "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
      "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache"
    },
    "inheritance": "runtime subprocesses receive one exact sanitized environment; unrelated or secret ambient values are not inherited",
    "required_offline_values": {
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1"
    },
    "resolved_cache_paths": {
      "CUDA_CACHE_PATH": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
      "HF_HOME": "/tmp/rustinfer-pr01-lock-20260824/hf",
      "TORCHINDUCTOR_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
      "TRITON_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
      "UV_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/cache",
      "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
      "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache"
    }
  },
  "status": "passed",
  "uv_sync": {
    "argv": [
      "/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv",
      "sync",
      "--frozen",
      "--offline",
      "--project",
      "/tmp/rustinfer-pr01-09911ba-hf-20260825/tools/python/reference"
    ],
    "environment": {
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
      "CUDA_CACHE_MAXSIZE": "4294967296",
      "CUDA_CACHE_PATH": "/tmp/rustinfer-pr01-lock-20260824/cuda-cache",
      "DO_NOT_TRACK": "1",
      "HF_HOME": "/tmp/rustinfer-pr01-lock-20260824/hf",
      "HF_HUB_DISABLE_TELEMETRY": "1",
      "HF_HUB_OFFLINE": "1",
      "HOME": "/home/psyche",
      "LANG": "en_US.UTF-8",
      "LC_ALL": "C.UTF-8",
      "MKL_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
      "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
      "PYTHONDONTWRITEBYTECODE": "1",
      "PYTHONHASHSEED": "0",
      "TOKENIZERS_PARALLELISM": "false",
      "TORCHINDUCTOR_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache/torch_compile_cache",
      "TRANSFORMERS_OFFLINE": "1",
      "TRITON_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/triton-cache",
      "UV_CACHE_DIR": "/tmp/rustinfer-pr01-lock-20260824/cache",
      "UV_OFFLINE": "1",
      "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/hf-transformers-101d21486780e574-55a397313acd",
      "UV_PYTHON": "3.13.15",
      "UV_PYTHON_DOWNLOADS": "never",
      "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
      "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
      "VLLM_DO_NOT_TRACK": "1",
      "VLLM_NO_USAGE_STATS": "1"
    },
    "finished_at_utc": "2026-08-24T18:57:03.591Z",
    "returncode": 0,
    "started_at_utc": "2026-08-24T18:57:03.297Z",
    "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/uv-sync.stderr.txt",
    "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/preparation/uv-sync.stdout.txt"
  }
}
```

## Exact lane commands

```shell
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-01/cell-01-warm-c1-p128-o32/result --run-index 1 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-01 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-01/cell-02-warm-c1-p4096-o128/result --run-index 1 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-01 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-01/cell-03-warm-c8-p128-o32/result --run-index 1 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-01 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-01/cell-04-cold-c1-p128-o32/result --run-index 1 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-01 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-02/cell-01-warm-c1-p128-o32/result --run-index 2 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-02 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-02/cell-02-warm-c1-p4096-o128/result --run-index 2 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-02 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-02/cell-03-warm-c8-p128-o32/result --run-index 2 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-02 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-02/cell-04-cold-c1-p128-o32/result --run-index 2 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-02 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-03/cell-01-warm-c1-p128-o32/result --run-index 3 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-03 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-03/cell-02-warm-c1-p4096-o128/result --run-index 3 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-03 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-03/cell-03-warm-c8-p128-o32/result --run-index 3 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-03 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-03/cell-04-cold-c1-p128-o32/result --run-index 3 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-03 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-04/cell-01-warm-c1-p128-o32/result --run-index 4 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-04 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-04/cell-02-warm-c1-p4096-o128/result --run-index 4 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-04 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-04/cell-03-warm-c8-p128-o32/result --run-index 4 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-04 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-04/cell-04-cold-c1-p128-o32/result --run-index 4 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-04 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-05/cell-01-warm-c1-p128-o32/result --run-index 5 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-05 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-05/cell-02-warm-c1-p4096-o128/result --run-index 5 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-05 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-05/cell-03-warm-c8-p128-o32/result --run-index 5 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-05 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference benchmark --matrix /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-hf-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging/runs/run-05/cell-04-cold-c1-p128-o32/result --run-index 5 --run-id hf-transformers-20260824T185646.150Z-55a397313acd-run-05 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
```

## Known limitations

- This gate covers only the four cells predeclared by PR-01, not all 48 matrix cells.
- It establishes repeatability for one lane and one primary environment; it does
  not by itself establish cross-lane performance superiority.
- External caches are inventory-fingerprinted rather than copied into Git;
  model weights and profiler traces are not copied into this artifact.
