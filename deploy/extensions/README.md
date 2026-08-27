# Extension admission registry

This directory is the fail-closed admission boundary for work described by
[PR 17](../17-extension-gates.md). Registry v1 lands empty by default; the
checked-in [`registry.json`](registry.json), not this prose or the roadmap,
is authoritative for the current approved set.

## Admission PR

One admission-only PR adds exactly one coherent four-file set:

```text
deploy/extensions/registry.json
deploy/extensions/proposals/<extension-id>.json
deploy/extensions/plans/<extension-id>.md
benchmarks/extensions/contracts/<extension-id>.json
```

The new registry entry must set `implementation_link_path` to `null`. The PR
must not add implementation code, dependencies, kernels, runtime behavior, or
performance claims. With a full base commit, CI enforces this exact diff and
the v1 append-only transition:

```sh
python3 ci/check_extension_gates.py --base-revision <full-lowercase-base-sha>
```

Bootstrap is also closed: if the base has no registry, the first registry
revision must remain empty. Once admitted, a v1 registry entry and its proposal,
plan, and benchmark contract are immutable. Rename sources count as changed
paths, so moving pre-existing files cannot masquerade as the four-file diff.

## Closed contract

Every proposal declares `approved-for-implementation`, a closed roadmap track,
an allowed semantic class, an `RILEY_EXPERIMENTAL_*` flag, both defaults
`false`, disclosure and rollback text, and structured answers to all ten PR 17
questions. `expected_resource_reduction` is a non-empty unique subset of
`flops`, `serial-depth`, and `hbm-traffic`; there is no `none` escape hatch.

`reference_path` and `fallback_path` must name two different Git-tracked regular
repository files. Git control paths and symlinks are rejected. Their
`reference_sha256` and `fallback_sha256` values are checked against the actual
bytes and copied exactly into the benchmark contract. Each workload is likewise
a closed `{path, sha256}` object naming a Git-tracked artifact, so reference,
fallback, and workload contents cannot drift silently.

The track-to-class mapping is closed:

| Track | Allowed class |
|---|---|
| quantization | `E0`, `A1` |
| low-rank-weight-compression | `A1` |
| kv-compression | `A1` |
| prefix-cache | `reference` |
| kv-offload-prefetch | `reference` |
| query-aware-kv-selection | `A1` |
| moe | `E0`, `M1` |
| mamba-ssm | `E0` |
| multimodal | `E0`, `M1` |
| speculative-decoding | `E0`, `E1` |
| jacobi-lookahead | `E0`, `A1`, `M1` |

`reference` is the exact-systems class and requires behavioral parity, token
parity, stable fallback, and lifetime/resource regression coverage. `E0`
requires reference parity, dtype tolerances, extreme cases, and token-level
regression. `E1` requires a distribution contract, statistical test,
request-local RNG isolation and restore, greedy exactness, and fixed-seed
semantics. `A1` requires a metric-bound error budget, exact fallback, opt-in,
usage disclosure, and a quality/latency curve. `M1` additionally requires the
`research` boundary and calibration/training provenance.

Metrics are exact scalar paths in the common result schema. Every track has an
exact sorted `required_metrics` set; the primary metric must be one of those
required performance/resource observations, must differ from the quality/error
metric, and the quality path is bound by track and class. The speculative set
includes every common-result acceptance, target-call, draft/verify latency,
lookahead, rejected-suffix, and rollback observation plus end-to-end latency and
throughput. An E0 tolerance set must match the frozen comparison dtype exactly, with each
`atol` and `rtol` below 1. The query-aware omitted-mass A1 budget uses the
`fraction` unit and a maximum in `[0, 1)`. Common result schema v1 can
currently admit `reference`/`E0` with `failure_count`, and
`query-aware-kv-selection`/`A1` with
`sparse_attention.omitted_mass_bound`. `sparse_attention.exact_fallback_rate`
is operational, not a quality/error budget. Other `A1`, `E1`, and `M1`
admissions fail closed until a versioned common result schema adds a suitable
quality field. Multi-GPU is not a v1 track because the common result contract is
single-GPU (`gpu_count: 1`).

The benchmark contract freezes the actual comparison environment: one GPU,
GPU/driver/CUDA/model ID/full 40-hex model revision/dtype strings, unique
positive concurrency and prompt/output-length arrays, closed greedy or
fixed-seed sampling objects, and both cold and warm states. At least five
independent processes and five measured iterations per process, median and p95,
end-to-end scope, and fallback A/B are mandatory.

## Experimental implementation link

A later implementation PR may change an admitted entry only from
`implementation_link_path: null` to
`deploy/extensions/implementations/<extension-id>.json`. That manifest binds the
approved ID, proposal, plan, contract, runtime flag, tracked implementation
paths, the source file containing the flag, and one or more validation tests as
`{path, sha256, test_id}`. Every validation test must be a top-level
auto-discovered Rust integration test in an explicitly listed Cargo workspace
member. The crate cannot disable `autotests`, and a matching explicit target
cannot disable the test harness or require a feature. The standard CPU
`--all-targets` workspace lane therefore includes it. The test ID is a Rust
identifier naming exactly one direct top-level `#[test] fn`; hidden targets,
comments/string literals, `cfg`/`cfg_attr`, and `ignore` cannot satisfy it.

After the registry bootstrap, adding a new `RILEY_EXPERIMENTAL_*` literal
under production `crates/` requires exactly one null-to-manifest implementation
link, and the new literal must equal that extension's approved flag. This closes
the mechanically identifiable namespace bypass; reviewers still classify code
that attempts to avoid the required experimental namespace entirely.

The checker proves the files, hashes, identifiers, and default-false metadata
are connected. Reviewers must still verify that executable parsing and control
flow are genuinely default-off, flag-on, and fall back to the declared stable
path; a comment or dead-code flag literal is not behavioral evidence.

Registry v1 supports admission and one immutable experimental implementation
link only. The link does not claim that benchmark results pass; a stable
promotion must introduce a versioned result-receipt gate. Stable promotion,
withdrawal, replacement, or mutation of an admitted contract requires a
reviewed schema-v2 transition first.

## Validation

```sh
python3 ci/check_extension_gates.py
python3 -m unittest discover -s ci/tests -p 'test_*.py' -v
cargo test --workspace --all-targets --no-default-features
```

The JSON schemas are portable shape contracts. CI also pins a canonical
semantic SHA-256 for each schema and tests critical `additionalProperties`,
`const`, `enum`, `pattern`, numeric-bound, and semantic conditional mutations,
so schema relaxation cannot silently replace the authoritative checker.
