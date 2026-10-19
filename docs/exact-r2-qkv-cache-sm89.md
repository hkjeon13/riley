# Exact Q4 R2 batching and HF QKV/cache fusion on SM89

Status: reviewed source integrated; actual935-file snapshot rebuilt, CPU and dedicated GPU/HTTP verification passed. No deployment, commit or push. Broader stock-vLLM parity is not achieved.

Opt in with `--owned-decode-batch exact-r2-v3` alongside `--numerics-profile owned-hf-eager-bf16-v1 --execution-completion iteration-batch --owned-decode-graph segments-v1`. Performance qualification is Qwen3-4B, concurrency2, RTX4090 SM89/CUDA12.8.1/cuBLAS120804/128SM. Cold preparation gates the model/device and cuBLAS workspace/SM configuration; other batch counts have numerical checks but are outside this performance claim. Existing mode2 keeps the original107 R2 fallback.

The change reuses canonical request CSR boundaries across QKV, Gate/Up, normalization and Attention; coalesces2-row Gate/Up with the independently qualified native-order kernel; and carries the existing HF128 norm selector through the deferred KV writer. The fused operation launches the SAME existing HF128 Q/K normalization/RoPE kernel with cache destinations. No GPU scratch is added. Complete-prompt prefill, BF16 stages, original normalization geometry and canonical KV ownership/lease checks remain. HF deferred writes additionally require identical position storage and active-row counts.

The installed vLLM0.27.1 Qwen3 code uses QKVParallelLinear and merged Gate/Up projections. This change borrows metadata preparation/reuse and launch fusion while retaining Riley arithmetic. [vLLM CUDA Graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/) and [chunked-prefill design](https://docs.vllm.ai/en/v0.22.1/configuration/optimization/) informed the comparison.

## Numerical and runtime evidence

- CPU517 passed;40 ordinary GPU tests ignored. Dedicated tests actually ran Q4 C1 eager, C2 Graph and changing actual counts1..32, including reordered replicas and changing contexts. All104 complete-vocabulary HF reference rows match byte-for-byte.
- Driver-only introspection found72 capturedGraphs,36 fused HF norm/RoPE/cache nodes, zero standalone KV-copy nodes, and72 R2 Gate/Up nodes. Instrumented timings are excluded from performance.
- Four original model paths (Q4, Mistral7B, SmolLM2, Qwen2.5-3B) remain native before/after exact at C1 and C8 Graph. Fresh HF references match the first3; the existing Qwen2.5 HF discrepancy is retained separately.
- Serving audit checks498 complete native response values, owned GPU PIDs/start identities,22GiB process cap, recovery, stable allocations and zero new captures during measured windows. Equality was not relaxed.

## Same-device performance

Three fresh native pairs, with the final pair order reversed; ONE fresh stock-vLLM pilot. Natural prompts16/128/397 tokens, output caps32/64/128,48 measured requests per arm, identical3584 output-token counts. Context1024, token budget512, block16, KV64×concurrency, prefix cache disabled, neutral greedy controls and original EOS. Native CUDA12.8 versus stock Torch CUDA13; the software stacks differ.

| Metric | Before107 median | Candidate143 median | Change | Stock vLLM single pilot |
|---|---:|---:|---:|---:|
| Output tok/s | 107.052 | 131.629 | +22.96% | 186.912 |
| TTFT p95 ms | 52.915 | 52.653 | -0.50% | 46.290 |
| TTFT p99 ms | 53.042 | 52.707 | -0.63% | 51.451 |
| TPOT p99 ms | 18.400 | 14.790 | -19.62% | 10.193 |
| E2E p99 ms | 2382.677 | 1923.972 | -19.25% | 1327.079 |

Throughput improved in all3 pairs (22.77%..22.97%). All6 p95/p99 tail medians and paired medians are nonregressing. TTFT is variable: one pair had p95+4.75% and p99+4.35%. This is not an every-pair or every-workload tail guarantee,5 fresh triples, or stock-parity claim.

TPOT percentiles describe each request’s mean first-to-last-token interval, not individual token ITL. Stock generation differs: HF token/text agreement39/45 correctness requests and16/48 measured requests. Output counts are equal, but this is not a quality-ranking result. Riley full responses remain exactly equal to the pinned native eager reference.

## Rejected paths retained as evidence

Forced128/128/128/13 prefill was first numerically repaired after a last13-row RMSNorm launch selected64 rather than32 threads. Rotated/reversed/repeated397-token controls then matched14,585,856 BF16 values, but TTFTp95 worsened52.80→103.23ms. It was rejected. The metadata/R2-only candidate likewise had a small positive TTFT tail and was not adopted. Those raw results and prototype sources remain in artifacts; their token-splitting/Lt/padded-Attention/RMSNorm changes are ABSENT from this12-file selected patch.

Artifacts: `artifacts/multimodel-parity-20261017/v143-source-delta-vs107.json`, `v143-runtime-controller-result.json`, `v143-native-confirm-audit.json`, `v143-existing-model-regression-result.json`, and `v143-adoption-receipt.json`. The remote experiment root is `/data/riley-multimodel-parity-20261017` on `ssh ai-assistant`. A100 GPU2/3 owners were rechecked after every finite arm; foreign work was preserved. RTX4090 and A100 results are never combined.

Rollback of the new profile is the existing `--owned-decode-batch lt-mlp-v2` selection. The12 previous dirty source files are preserved in the adoption backup.


## Actual integrated rebuild and runtime receipt

All935 pinned files in the actual main snapshot match the private143 runtime source, with the12 previous dirty files retained in the adoption backup. Rebuild CPU results:517 passed,40 ordinary GPU tests ignored. Dedicated actual-snapshot Q4 C2 and changing1..32 Graph tests match all104 HF full-vocabulary reference rows.

The actual integrated server additionally passed 195 complete native response checks across Q4 C2 mode3 and Mistral C8 ordinary mode2, including recovery, stable allocations, owned PIDs,22GiB cap and zero captures during measurement. These are actual runtime checks, NOT replacement before/after performance medians.

Performance binary SHA256: `433b0d0da14cd2fb5b8727d1bd643c0d7e60f9dab15b5f1274576a1970a3b442`. Actual integrated binary SHA256: `49de5a917d7fdc4061be575bb444760c588b76b07667b61b90724bc7c91970ca`. Source identity is proven independently of differing build paths.

The initial actual HTTP verifier stopped BEFORE GPU execution because its before-reference SHA was incorrectly paired with the served executable. V2 retains the original eager reference/before-binary identity and separately selects the actual runtime binary/profile; no oracle was relabeled and equality was unchanged. Failed preflight logs remain. The two teacher executables’ source/binary provenance is separately reconciled after a derived helper inherited a shared JSON metadata filename; actual integrated helper metadata now uses a unique filename.
