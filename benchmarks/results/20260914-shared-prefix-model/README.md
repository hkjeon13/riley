# PR10 retained native capability and model parity

Status: **shared-prefix reads executed successfully in a real SmolLM2 model graph on RTX 4090, in synchronous and buffered modes. Automatic serving cache and captured-model COW are still incomplete.** No serving performance improvement is claimed.

The new opt-in model constructor enables the native V7 capability after recording and before buffered setup or first replay. Native code rejects non-V7, sealed, repeated, pending and failed-owner configuration. The capability is included in the loaded-model catalog digest and retained session identity. Scheduler expectations copy it from that identity; packet bytes cannot enable it. Ordinary and paired submissions reject capability mismatches before dispatch. Existing constructors remain exclusive.

The GPU test creates two real pool sequences, prefills identical 32-token prompts on independent pages, and records the next decode's complete BF16 logits. After device completion, it truncates the sequences, imports the first sequence's actual prefix pages into the second and decodes the same input again with independent append pages. All 49,152 vocabulary logits for both rows match byte-for-byte. This comparison uses the same numerical backend and opt-in session for exclusive and shared page layouts; it is not a vLLM comparison.

| Executed check | Result |
|---|---|
| Synchronous full-model shared/exclusive parity | 196,608 logits bytes identical |
| Buffered full-model shared/exclusive parity | 196,608 logits bytes identical |
| Host pages and CUDA allocations after each mode | Zero |
| CUDA-enabled retained session tests | 7 passed, 0 failed |
| GPU test | 1 passed, both modes executed; 0 ignored |
| CPU runtime library | 348 passed, 0 failed; 1 existing timing diagnostic ignored |
| Scheduler library, integration and doc suites | 150 passed, 0 failed |
| CUDA/server feature compile check | Passed |

The model test was explicitly run with `--ignored` on the supported GPU. Its ignore annotation prevents accidental local execution; the successful run was not skipped. CUDA compilation revealed an extra existing-wrapper argument and the test used the wrong truncate method name initially; both were corrected before the final v2 gate. Existing unused-code warnings remain.

Binary SHA256: `f5959dc0d98884c08da6087383e0c4bf5fc66ac66eb9a6448e5ce60b77d49e62`. See [receipt](receipt.json) and [model log](evidence/model-tests.log). The GPU compute-process inventory was empty after the gate. No production serving binary was rebuilt or replaced.

Reproduce on the prepared remote host with `bash benchmarks/analysis/shared_prefix_model_gate.sh /data/riley-serving-260913-recovery <new-absolute-evidence-directory>`. The server feature check was also executed independently after the model gate and added to the runner for subsequent runs.

Remaining work includes loaded-model/token binding for production cache descriptors, automatic scheduler cache lookup/publication/eviction with cache-only owners, captured-parent COW/drain, partial-tail and diverse-prompt/model qualification, and a matched cache-hit/cache-miss vLLM serving comparison. The fixture's descriptor fingerprints are controlled test values for a known common model; they do not implement production cache identity. Full-page prefix read parity does not establish partial-tail COW, multi-GPU, Hopper or Blackwell correctness.
