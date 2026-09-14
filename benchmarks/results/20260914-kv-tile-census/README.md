# Physical KV tile sharing census

Diagnostic only: C32, active cap 32, 64 warmup + 128 retained per workload, frozen C32 prior references. Pure decode observations include warmup and rolling successor preparation; counts are not GPU traffic, successful execution counts, or serving performance estimates.

| Workload | Decode observations | Tile groups | Row-tile references | References in shared groups | Maximum owners |
|---|---:|---:|---:|---:|---:|
| shared | 171 | 9115 | 18283 | 10773 (58.9%) | 19 |
| unique | 145 | 15448 | 15448 | 0 (0.0%) | 1 |

Decision: cross-request KV reuse has an observed applicability window in shared workloads, but not unique. Groups match all eight physical page IDs at the same logical 128-token tile and exclude the append/partial tile. This is a residency/ownership observation; L2 reuse may already avoid some HBM traffic. No speedup is inferred.

Candidate runtime is SHA256 e95bd576863b7a7605c7491f1d3a18b9561e5a77d36f89466f3f8a360aaa85af, built with server,cuda and --bin riley. V1 accidentally copied the old executable because the server feature was omitted; V1 is excluded. V2 asserts the diagnostic marker in the binary and a nonempty runtime census per workload. All 384 warmup/retained responses passed the controller protocol and exact-reference checks. Lifecycle exited zero and restored all three Blender processes. Raw V2 responses, launch records and lifecycle logs are archived in evidence.tar.gz. Source snapshots and member hashes are recorded in manifest.json. Independent verification passes with verify_kv_tile_census.py and reconstructs all 384 responses plus census histograms.

Next: design a physical-tile group descriptor with fallback for unshared tiles, preserve per-request QK K16 and PV recurrence, and evaluate QK plus PV together. Do not use QK-only savings as proof of the serving objective.
