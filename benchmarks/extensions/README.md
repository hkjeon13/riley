# Extension benchmark contracts

Registry v1 lands with no contracts by default. The checked-in
[`deploy/extensions/registry.json`](../../deploy/extensions/registry.json) is
authoritative: `contracts/` contains exactly the contracts named by its current
entries.

Each `contracts/<extension-id>.json` conforms to
[`benchmark-contract.schema.json`](benchmark-contract.schema.json) and freezes
the comparison before implementation begins:

- ID, status, track, allowed semantic class, experimental flag, class gate, and
  exact common-result primary, track-required, and quality/error metric paths;
- distinct Git-tracked reference and fallback files plus SHA-256 of their bytes;
- one or more Git-tracked end-to-end workload artifacts as `{path, sha256}`;
- `gpu_count: 1`, exact GPU/driver/CUDA/model ID/full 40-hex model
  revision/dtype, unique positive concurrency and prompt/output lengths,
  closed sampling configs, and both
  `cold` and `warm` states;
- at least five independent processes and five measured iterations per process;
  and
- median and p95, end-to-end scope, and fallback A/B.

The common result schema remains unchanged and single-GPU. Its current quality
fields admit `reference`/`E0` with `failure_count` and
`query-aware-kv-selection`/`A1` with
`sparse_attention.omitted_mass_bound`. Other `A1`, `E1`, and `M1` contracts
must wait for a versioned common result schema with a suitable quality field;
`sparse_attention.exact_fallback_rate` is only an operational metric. A
microbenchmark may diagnose a kernel but cannot satisfy the end-to-end gate.
Required metrics are an exact track-specific set and include the primary.
E0 tolerances match the frozen dtype exactly and stay below 1; the query-aware
omitted-mass A1 budget is a `[0, 1)` fraction. The experimental implementation
link does not constitute a passing result receipt; stable promotion needs a
versioned receipt gate.

Measured output uses the append-only [`benchmarks/results/`](../results/README.md)
layout. Validate the registry and all registered contracts with:

```sh
python3 ci/check_extension_gates.py
```
