# Local benchmark artifact inventory — 2026-09-13

The progress snapshot includes source, plans, reports, textual raw measurements,
and small prototype contract fixtures. Per the [benchmark artifact contract](../README.md),
full tensor/native trace dumps, profiler databases/reports, compiled objects and
executables, and export archives remain local and are excluded from Git.

[`LOCAL_ARTIFACTS_20260913.jsonl`](LOCAL_ARTIFACTS_20260913.jsonl) records each
excluded file's original path, local URI, byte size, and SHA-256 at publication.
The original bytes have not been moved, deleted, or rewritten. Archive copies
are excluded alongside the expanded evidence; textual measurements remain
version controlled. Small descriptor fixtures under `benchmarks/prototypes`
remain version controlled.

These URIs identify the existing local copies, not a remotely accessible
artifact store. No external upload or durable remote retention was established
by this publication. Retain the local originals until an explicit archival or
deletion decision; no expiry is scheduled. Existing experiment receipts may
record their separate remote origins, but this inventory does not revalidate
those locations. References to excluded artifacts require the local copy or
the independently retained experiment source. The inventory is a point-in-time
snapshot and does not cover files produced after it was captured.
