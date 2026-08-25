# PR-01 repeatability evidence

This directory is an append-only import of one passing, externally staged
repeatability gate for lane `vllm` at Git revision
`09911ba2630845e9d4094b7c33c3ff65931a919c`.

## Exact runner invocation

```shell
/tmp/rustinfer-pr01-lock-20260824/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13 /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/scripts/run_repeatability_gate.py --lane vllm --output-root /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging --uv /tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv --finalize-to benchmarks/results/20260824T192344Z-vllm-repeatability-pr01-v2-run003
```

## Summary

- Gate status: `passed`
- Independent runs: 5
- Predeclared cells per run: 4
- Fresh single-cell benchmark subprocesses: 20
- Combined raw observations: 455 JSONL rows
- Combined raw SHA-256: `97830a7fc574d7a30b88a4027f374d4de9ff5c47e08c51dc4139c32252ca82b8`

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
        "value": 0.005817958021218931
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
        "value": 0.005843584544117956
      },
      {
        "name": "warm_p95_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.1,
        "value": 0.03446672149009067
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
          "r7_p50": 41.40898599871434,
          "r7_p95": 43.86345590828569
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 1.1327804968459532,
          "r7_p95": 1.3190435027354397
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.1156749997978967,
          "r7_p95": 1.197297780890949
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-01",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 772.6388537864243
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 41.44429800362559,
          "r7_p95": 44.204412100225454
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 1.1303904975648038,
          "r7_p95": 1.2316371517954392
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.115257951765201,
          "r7_p95": 1.146543829306592
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-02",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 771.9986840568704
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 42.00075350672705,
          "r7_p95": 46.47609684325287
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 1.1319584955344908,
          "r7_p95": 1.309985701664118
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.115565451571808,
          "r7_p95": 1.1400084565168307
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-03",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 761.7451858984866
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 41.47828949498944,
          "r7_p95": 42.23525344859809
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 1.1301859995000996,
          "r7_p95": 1.1971231986535713
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.1142775643571852,
          "r7_p95": 1.119629074110567
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-04",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 771.3566636055916
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 41.622990007454064,
          "r7_p95": 44.64946835141745
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 930,
          "r7_p50": 1.1328090040478855,
          "r7_p95": 1.2996501493034882
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.1163101289346213,
          "r7_p95": 1.1684053193265196
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-05",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 768.6742944751069
        },
        "trial_count": 30
      }
    ],
    "statistics": {
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.005817958021218931,
      "warm_end_to_end_r7_p50_sample_cv": 0.005843584544117956,
      "warm_end_to_end_r7_p95_sample_cv": 0.03446672149009067,
      "warm_pooled_itl_r7_p50_sample_cv": 0.0011215297089984895,
      "warm_pooled_itl_r7_p95_sample_cv": 0.042453391853830784,
      "warm_request_mean_tpot_r7_p50_sample_cv": 0.0006663740265131666,
      "warm_request_mean_tpot_r7_p95_sample_cv": 0.025676557651235943
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
        "value": 0.0016112081948853592
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
        "value": 0.0016105536932594347
      },
      {
        "name": "warm_p95_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.1,
        "value": 0.0038035468526373783
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
          "r7_p50": 168.0389270040905,
          "r7_p95": 170.67597999921418
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24472780800.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 1.191559997096192,
          "r7_p95": 1.2975580000784248
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.192471350375453,
          "r7_p95": 1.2104937133670777
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-01",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 761.6560630565386
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 167.51990549528273,
          "r7_p95": 169.41896614516736
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24472780800.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 1.1912920090253465,
          "r7_p95": 1.2659466468903702
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.1896681062639765,
          "r7_p95": 1.2010133811148933
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-02",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 764.0169853408534
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 167.79437699733535,
          "r7_p95": 170.23272475707927
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24472780800.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 1.192156007164158,
          "r7_p95": 1.2969970935955644
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.1914823149623328,
          "r7_p95": 1.208047948797842
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-03",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 762.747893533746
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 167.84483649826143,
          "r7_p95": 171.07751915609697
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24472780800.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 1.1917154988623224,
          "r7_p95": 1.3129282539011908
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.1903924606149232,
          "r7_p95": 1.2165733539283012
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-04",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 762.5458210407713
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 30,
          "r7_p50": 167.36040049727308,
          "r7_p95": 170.80540895476588
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24472780800.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 3810,
          "r7_p50": 1.189886505017057,
          "r7_p95": 1.292084951273864
        },
        "request_mean_tpot_ms": {
          "observation_count": 30,
          "r7_p50": 1.1895129724779703,
          "r7_p95": 1.2126811011934995
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-05",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 764.7429735215724
        },
        "trial_count": 30
      }
    ],
    "statistics": {
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.0016112081948853592,
      "warm_end_to_end_r7_p50_sample_cv": 0.0016105536932594347,
      "warm_end_to_end_r7_p95_sample_cv": 0.0038035468526373783,
      "warm_pooled_itl_r7_p50_sample_cv": 0.0007231301676474677,
      "warm_pooled_itl_r7_p95_sample_cv": 0.013206557114471001,
      "warm_request_mean_tpot_r7_p50_sample_cv": 0.0010554908120061907,
      "warm_request_mean_tpot_r7_p95_sample_cv": 0.004800872601395884
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
        "value": 0.0011417504044486826
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
        "value": 0.001022058830437998
      },
      {
        "name": "warm_p95_cv_max",
        "operator": "<=",
        "passed": true,
        "threshold": 0.1,
        "value": 0.05009924020319628
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
          "r7_p50": 57.757304995902814,
          "r7_p95": 59.20787730065058
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24468586496.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 1.4971769996918738,
          "r7_p95": 1.572435998241417
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 1.4850117093432815,
          "r7_p95": 1.489898032178321
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-01",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 4392.793241469422
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 57.73988949658815,
          "r7_p95": 67.43306263888371
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24468586496.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 1.4943739952286705,
          "r7_p95": 1.5680580108892173
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 1.481618758031888,
          "r7_p95": 1.4883913550405732
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-02",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 4394.613846185781
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 57.731734501430765,
          "r7_p95": 60.624302706855815
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24468586496.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 1.4981259882915765,
          "r7_p95": 1.6107920091599226
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 1.486531290274504,
          "r7_p95": 1.493166935453642
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-03",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 4390.853722405927
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 57.727551997231785,
          "r7_p95": 63.31281450402457
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24468586496.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 1.4957499952288345,
          "r7_p95": 1.566724997246638
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 1.4850518707516454,
          "r7_p95": 1.4897484192436923
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-04",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 4397.697917268193
        },
        "trial_count": 30
      },
      {
        "end_to_end_ms": {
          "observation_count": 240,
          "r7_p50": 57.868652504112106,
          "r7_p95": 62.3679469943454
        },
        "failure_count": 0,
        "peak_vram_bytes": {
          "maximum": 24468586496.0,
          "observation_count": 30
        },
        "pooled_itl_ms": {
          "observation_count": 7440,
          "r7_p50": 1.4974590012570843,
          "r7_p95": 1.5881970030022785
        },
        "request_mean_tpot_ms": {
          "observation_count": 240,
          "r7_p50": 1.4857831612912817,
          "r7_p95": 1.4912681935745622
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-05",
        "successful_trial_count": 30,
        "throughput_tokens_per_second": {
          "observation_count": 30,
          "r7_p50": 4384.295638052964
        },
        "trial_count": 30
      }
    ],
    "statistics": {
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.0011417504044486826,
      "warm_end_to_end_r7_p50_sample_cv": 0.001022058830437998,
      "warm_end_to_end_r7_p95_sample_cv": 0.05009924020319628,
      "warm_pooled_itl_r7_p50_sample_cv": 0.0010064245256393076,
      "warm_pooled_itl_r7_p95_sample_cv": 0.011762235551666896,
      "warm_request_mean_tpot_r7_p50_sample_cv": 0.0012686217346748845,
      "warm_request_mean_tpot_r7_p95_sample_cv": 0.0012130519935805999
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
        "value": 0.004535517603315144
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
          "r7_p50": 15025.81374500005
        },
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 1
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-01",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 582.2660925238431
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 15027.383338005166
        },
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 1
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-02",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 597.5823723382157
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 14904.839687005733
        },
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 1
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-03",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 604.99426934029
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 14879.982108002878
        },
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 1
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-04",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 582.2597040559167
        },
        "trial_count": 1
      },
      {
        "failure_count": 0,
        "model_load_ms": {
          "observation_count": 1,
          "r7_p50": 14948.768743997789
        },
        "peak_vram_bytes": {
          "maximum": 24414060544.0,
          "observation_count": 1
        },
        "run_id": "vllm-20260824T192357.417Z-8ce3ce384d9a-run-05",
        "successful_trial_count": 1,
        "throughput_tokens_per_second": {
          "observation_count": 1,
          "r7_p50": 599.8087359054952
        },
        "trial_count": 1
      }
    ],
    "statistics": {
      "cold_model_load_r7_p50_sample_cv": 0.004535517603315144,
      "failure_count": 0,
      "peak_vram_max_relative_range": 0.0,
      "throughput_p50_sample_cv": 0.017696369048235176
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
  "engine_revision": "vllm-0.27.1",
  "environment_id": "rtx4090-ubuntu22-driver580-v1",
  "git_dirty": false,
  "git_revision": "09911ba2630845e9d4094b7c33c3ff65931a919c",
  "lane_manifest_sha256": "002bab2b7dae587c78339131e5057a7cf4e9fc6d0d83432f514a6db59e89469b",
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
    "aggregate_sha256": "baf2b060a896c77a940ae6a57e5dd082df297e755bcd066e8643be3bcf265788",
    "captured_at_utc": "2026-08-24T19:25:35.499Z",
    "contract_version": "rustinfer.cache-inventory.v1",
    "roots": [
      {
        "entry_count": 58705,
        "environment_key": "UV_CACHE_DIR",
        "fingerprint_sha256": "303b1de9002aa8c32a550c3eddb0fcbda7f6dac5258c7e8bc98b14baa2cc1e1d",
        "path": "/tmp/rustinfer-pr01-lock-20260824/cache",
        "total_regular_file_bytes": 8045981018
      },
      {
        "entry_count": 589863,
        "environment_key": "UV_PYTHON_INSTALL_DIR",
        "fingerprint_sha256": "a81599ef15c893b7e28271aaf9ea602698eed5a1e7f47023a23720d8275ceb61",
        "path": "/tmp/rustinfer-pr01-lock-20260824/python",
        "total_regular_file_bytes": 101531698219
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
    "aggregate_sha256": "4f3b41ba58f1d78313fff29687270b50f8c2e8eb3a0354c112e146992da3abec",
    "captured_at_utc": "2026-08-24T19:24:12.872Z",
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
  "completed_at_utc": "2026-08-24T19:25:36.933Z",
  "contract_version": "rustinfer.repeatability-preparation.v2",
  "immutable_evidence": {
    "dependency_lock_sha256": "90120452532be59c3d6a9064c6964995414216d076b0c360e4a97de3e3a45451",
    "lane_manifest_sha256": "002bab2b7dae587c78339131e5057a7cf4e9fc6d0d83432f514a6db59e89469b",
    "model_id": "HuggingFaceTB/SmolLM2-135M",
    "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "project_environment": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/vllm-90120452532be59c-8ce3ce384d9a",
    "uv_sha256": "b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46",
    "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
  },
  "measured_cache_baseline_sha256": "baf2b060a896c77a940ae6a57e5dd082df297e755bcd066e8643be3bcf265788",
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
        "benchmarks/lanes/vllm",
        "rustinfer-vllm-benchmark",
        "--matrix",
        "/tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml",
        "--prompts",
        "/tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl",
        "--result-dir",
        "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-01-warm-c1-p128-o32/result",
        "--run-index",
        "1",
        "--run-id",
        "vllm-cache-prime-8ce3ce384d9a-01",
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
        "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/vllm-90120452532be59c-8ce3ce384d9a",
        "UV_PYTHON": "3.13.15",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
        "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0"
      },
      "finished_at_utc": "2026-08-24T19:24:35.520Z",
      "prime_index": 1,
      "raw_result": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-01-warm-c1-p128-o32/result/raw.jsonl",
      "raw_result_row_count": 30,
      "raw_result_sha256": "0d3c3f68898aafacbb9277d814e1286959d0224ee7ce1d4d62a60ea1aa98343a",
      "returncode": 0,
      "started_at_utc": "2026-08-24T19:24:14.621Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-01-warm-c1-p128-o32/benchmark.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-01-warm-c1-p128-o32/benchmark.stdout.txt"
    },
    {
      "argv": [
        "/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv",
        "run",
        "--frozen",
        "--offline",
        "--no-sync",
        "--project",
        "benchmarks/lanes/vllm",
        "rustinfer-vllm-benchmark",
        "--matrix",
        "/tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml",
        "--prompts",
        "/tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl",
        "--result-dir",
        "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-02-warm-c1-p4096-o128/result",
        "--run-index",
        "1",
        "--run-id",
        "vllm-cache-prime-8ce3ce384d9a-02",
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
        "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/vllm-90120452532be59c-8ce3ce384d9a",
        "UV_PYTHON": "3.13.15",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
        "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0"
      },
      "finished_at_utc": "2026-08-24T19:24:59.455Z",
      "prime_index": 2,
      "raw_result": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-02-warm-c1-p4096-o128/result/raw.jsonl",
      "raw_result_row_count": 30,
      "raw_result_sha256": "1a3e10cd619ed061f6751d5ed347e920e5a556c8e342f772c2aa2657fa686309",
      "returncode": 0,
      "started_at_utc": "2026-08-24T19:24:35.530Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-02-warm-c1-p4096-o128/benchmark.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-02-warm-c1-p4096-o128/benchmark.stdout.txt"
    },
    {
      "argv": [
        "/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv",
        "run",
        "--frozen",
        "--offline",
        "--no-sync",
        "--project",
        "benchmarks/lanes/vllm",
        "rustinfer-vllm-benchmark",
        "--matrix",
        "/tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml",
        "--prompts",
        "/tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl",
        "--result-dir",
        "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-03-warm-c8-p128-o32/result",
        "--run-index",
        "1",
        "--run-id",
        "vllm-cache-prime-8ce3ce384d9a-03",
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
        "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/vllm-90120452532be59c-8ce3ce384d9a",
        "UV_PYTHON": "3.13.15",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
        "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0"
      },
      "finished_at_utc": "2026-08-24T19:25:19.226Z",
      "prime_index": 3,
      "raw_result": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-03-warm-c8-p128-o32/result/raw.jsonl",
      "raw_result_row_count": 30,
      "raw_result_sha256": "e4a8e772844eb2d0ea42a45b7b223c163f32459333e815a9ff47be9aa8213864",
      "returncode": 0,
      "started_at_utc": "2026-08-24T19:24:59.469Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-03-warm-c8-p128-o32/benchmark.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/prime-03-warm-c8-p128-o32/benchmark.stdout.txt"
    }
  ],
  "prime_results_excluded_from_checker": true,
  "project_environment": {
    "covered_by_cache_inventory": "UV_PYTHON_INSTALL_DIR",
    "dependency_lock_sha256": "90120452532be59c3d6a9064c6964995414216d076b0c360e4a97de3e3a45451",
    "derivation": "UV_PYTHON_INSTALL_DIR/project-environments/<lane>-<lock-prefix16>-<nonce>",
    "lane_id": "vllm",
    "path": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/vllm-90120452532be59c-8ce3ce384d9a",
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
      "launcher_path": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/vllm-90120452532be59c-8ce3ce384d9a/bin/python",
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
      "finished_at_utc": "2026-08-24T19:24:14.199Z",
      "started_at_utc": "2026-08-24T19:24:14.193Z",
      "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/uv-python-find.stderr.txt",
      "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/uv-python-find.stdout.txt"
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
      "/tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/lanes/vllm"
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
      "UV_PROJECT_ENVIRONMENT": "/tmp/rustinfer-pr01-lock-20260824/python/project-environments/vllm-90120452532be59c-8ce3ce384d9a",
      "UV_PYTHON": "3.13.15",
      "UV_PYTHON_DOWNLOADS": "never",
      "UV_PYTHON_INSTALL_DIR": "/tmp/rustinfer-pr01-lock-20260824/python",
      "VLLM_CACHE_ROOT": "/tmp/rustinfer-pr01-lock-20260824/vllm-cache",
      "VLLM_DO_NOT_TRACK": "1",
      "VLLM_NO_USAGE_STATS": "1"
    },
    "finished_at_utc": "2026-08-24T19:24:14.593Z",
    "returncode": 0,
    "started_at_utc": "2026-08-24T19:24:14.226Z",
    "stderr": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/uv-sync.stderr.txt",
    "stdout": "/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/preparation/uv-sync.stdout.txt"
  }
}
```

## Exact lane commands

```shell
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-01/cell-01-warm-c1-p128-o32/result --run-index 1 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-01 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-01/cell-02-warm-c1-p4096-o128/result --run-index 1 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-01 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-01/cell-03-warm-c8-p128-o32/result --run-index 1 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-01 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-01/cell-04-cold-c1-p128-o32/result --run-index 1 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-01 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-02/cell-01-warm-c1-p128-o32/result --run-index 2 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-02 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-02/cell-02-warm-c1-p4096-o128/result --run-index 2 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-02 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-02/cell-03-warm-c8-p128-o32/result --run-index 2 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-02 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-02/cell-04-cold-c1-p128-o32/result --run-index 2 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-02 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-03/cell-01-warm-c1-p128-o32/result --run-index 3 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-03 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-03/cell-02-warm-c1-p4096-o128/result --run-index 3 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-03 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-03/cell-03-warm-c8-p128-o32/result --run-index 3 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-03 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-03/cell-04-cold-c1-p128-o32/result --run-index 3 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-03 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-04/cell-01-warm-c1-p128-o32/result --run-index 4 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-04 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-04/cell-02-warm-c1-p4096-o128/result --run-index 4 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-04 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-04/cell-03-warm-c8-p128-o32/result --run-index 4 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-04 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-04/cell-04-cold-c1-p128-o32/result --run-index 4 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-04 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-05/cell-01-warm-c1-p128-o32/result --run-index 5 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-05 --warm-state warm --concurrency 1 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-05/cell-02-warm-c1-p4096-o128/result --run-index 5 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-05 --warm-state warm --concurrency 1 --prompt-tokens 4096 --output-tokens 128
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-05/cell-03-warm-c8-p128-o32/result --run-index 5 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-05 --warm-state warm --concurrency 8 --prompt-tokens 128 --output-tokens 32
/tmp/rustinfer-pr01-lock-20260824/uv-0.12.5/uv-x86_64-unknown-linux-gnu/uv run --frozen --offline --no-sync --project benchmarks/lanes/vllm rustinfer-vllm-benchmark --matrix /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/matrix.yaml --prompts /tmp/rustinfer-pr01-09911ba-vllm-20260825/benchmarks/prompts.jsonl --result-dir /home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging/runs/run-05/cell-04-cold-c1-p128-o32/result --run-index 5 --run-id vllm-20260824T192357.417Z-8ce3ce384d9a-run-05 --warm-state cold --concurrency 1 --prompt-tokens 128 --output-tokens 32
```

## Known limitations

- This gate covers only the four cells predeclared by PR-01, not all 48 matrix cells.
- It establishes repeatability for one lane and one primary environment; it does
  not by itself establish cross-lane performance superiority.
- External caches are inventory-fingerprinted rather than copied into Git;
  model weights and profiler traces are not copied into this artifact.
