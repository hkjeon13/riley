# Paired combined RoPE and attention diagnostic

`run_paired_rope_attention_profile.py` is a separate diagnostic campaign. It compares the API baseline's original RoPE/KV plus attention region, measured as one interval, with Batch8's fused interval. It does not reinterpret Round14 serving measurements or make an acceptance decision.

Preparation is the default and is available only after Round14 has finalized, every launched measurement process has an owned-session cleanup receipt, and Blender restoration is verified. Both diagnostic builds and the generated Round15 session helper must already exist. This runner never builds, instruments source, creates a session helper, or launches a remote command.

```sh
python3 /tmp/riley-opt-260912/run_paired_rope_attention_profile.py \
  --output /tmp/riley-opt-260912/paired-rope-attention-preparation
```

Use a fresh output directory with `--measure` to execute the authorized diagnostic campaign after reviewing preparation. Existing output directories are rejected.

```sh
python3 /tmp/riley-opt-260912/run_paired_rope_attention_profile.py \
  --output /tmp/riley-opt-260912/paired-rope-attention-profile --measure
```

The runner pins both builders, both profiler modules, generated native source, complete diagnostic build receipts, binaries, and build logs. The frozen Round14 plan and preparation supply unchanged source/model/numerical/raw-token proofs and the exact P128/O32 reference. Their full immutable inputs and resolved aliases are checked again. Round15's generated source is independently reconstructed through the frozen factory before import; its predecessor pins and cleanup gate are checked, followed by live checks of its three proven Blender successors. There is no hardcoded successor PID.

Eight fresh C1 server processes run serially:

| Order | Implementation | Operator timing |
|---|---|---|
| 1–2 | Baseline, candidate | Off before |
| 3–4 | Baseline, candidate | Combined interval, AB |
| 5–6 | Candidate, baseline | Combined interval, BA |
| 7–8 | Baseline, candidate | Off after |

Each process performs five nonstream warmups, five streaming warmups and six retained streaming requests. All 16 responses must match exact prompt IDs, all 32 output IDs, text, finish reason and usage using the frozen grouped-token V2 client. The diagnostic binary and working directory replace the qualified lane's original binary and source directory; all remaining HTTP argv are preserved, including GPU greedy sampling. The explicit private runtime environment gains only whole-graph profiling, projection `off`, operator `off` or `rope_attention`, and layer selection `all`.

Native records must contain exactly 512 ordered replays, each request at positions 127–158. Warmups occupy replay IDs 1–320. Retained IDs 321–512 contain six prefill and 186 decode replays. Selected mode must have one complete interval for each of all 30 layers on every decode replay; missing/failed intervals, wrong IDs or positions, projection hooks, and event inventory mismatches fail the process. The frozen profiler summarizes retained records only. Full logs and every raw HTTP response remain available, including warmups.

Before each server, the same private driver, GPU UUID, no-foreign-CUDA-compute, ≤512 MiB idle GPU memory and ≤48°C starting-temperature conditions apply. Actual private CUDA mappings are checked before and after requests; a monitor checks process ownership and the restoration watchdog while requests run. Every server is shut down through its owned session and must exit gracefully. Partial stop, request failure and ordinary termination enter the restoration path; the independent watchdog remains responsible if restoration cannot complete immediately. No unrelated process is killed.

`comparison.json` is written only after all eight processes, final immutable-input checks and verified restoration. It reports two direct combined-interval ratios and each implementation's whole-graph off-before/off-after drift and selected/off perturbation. It does not add separate family medians, subtract event overhead, report an uninstrumented serving improvement, or infer a winner. Raw client timestamps are retained as response evidence, not promoted to a serving benchmark.

Focused CPU tests cover replay boundaries, complete intervals, warmup failures, the actual frozen summarizers, environment separation, pre-stop rejection, interrupted/partial campaigns, restoration failures and fixed execution order. No CUDA build, GPU execution or performance result is established by these tests.
