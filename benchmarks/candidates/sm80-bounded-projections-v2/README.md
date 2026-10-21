# SM80 bounded projection batching v2

Explicit opt-in mode7 `bounded-projections-sm80-v2`, layered on Gate/Up-only v225. Defaults and modes0..6 remain unchanged. Qualified Qwen3-4B geometry only, SM80/108SM, CUDA12040, cuBLAS/Lt120405, unmodified workspace/SM configuration, at most32 sequences.

Q batches at most32 rows, O at most8, Down at most4 and LM head at most16; Gate/Up retain the existing16-row bound. Prefill extents, K/V and Attention arithmetic remain unchanged. O and Down use fixed Lt configurations with private cold, tracked workspace. Larger batches split at these boundaries; singleton leftovers retain nativeM1.

Two matched96-request C8 pairs improved158.07 to259.08 tok/s (+63.9%), TTFT p95 from121.52 to97.17ms. All nine C8 latency quantiles improved in both pairs. C1 was essentially unchanged. All full-vocabulary104teacher rows and actual Graph keys1..32 passed, as did518CPU regressions. This is an opt-in screen, not global default/release qualification.

The qualified935-file source records the mode7 qualification tree. Qualified binary SHA256: `d339fe5d39b04983f09cbb36dce32b22c4ef45d77db9b2a8604ab1a38e22f884`. The performance variant differs only by the added cold Lt version guard; final-binary serving and full numerical checks passed separately. See the [report](../../../docs/qwen-sm80-projection-batching-v2-20261019.md) for exact bounds, evidence and limitations.

`candidate.patch` applies above v225. A baseline checkout must first apply `../sm80-bounded-gateup-v225/candidate.patch`. On a tree containing this mode7 layer without later layers, `git apply --reverse --check benchmarks/candidates/sm80-bounded-projections-v2/candidate.patch` verifies this layer without mutating it. Use `--owned-decode-batch bounded-projections-sm80-v2 --sampling-backend gpu-greedy` with the other qualified owned-HF/iteration-batch serving controls. Roll back to `bounded-gateup-sm80-v1` to disable this layer.

The current tree also contains the separately qualified [mode8 layer](../sm80-native-kv-v3/README.md). Reverse-check that top layer first; this mode7 snapshot remains reproducible independently.
