# PR10 scheduler shared-prefix authority

Status: **ordinary and paired scheduler plans now carry shared read-only prefixes into authoritative V7 expectations. GPU/model execution and automatic cache policy remain unconnected.**

The integration test first settles three 32-token prefills, then replaces two request sequences using real pool export/import operations. Their six logical prefix pages share two physical pages. Two consumers reserve independent append tails while the third remains off-batch. Both ordinary replay and a paired decode window construct V7 authority from these actual reservations. NotDispatched rollback retains the prefix, and final cancellation reclaims every page. This is host scheduling evidence; mocked output tokens are not model correctness evidence.

Implementation:

- Scheduler-only shared plan construction accepts position-identical read pages and rejects aliased append ranges. Public transported-plan constructors retain their exclusive contract.
- `SequenceState::execution_block_table` checks every page's pool, generation and owner and rechecks write exclusivity against both shared sequence owners and external immutable leases. It performs no allocation or mutation.
- Ordinary and paired execution use one ledger builder including all live request owners and reservations. Capacity is reserved for logical owner/page pairs, not physical pool size. The old repeated scan for duplicate physical IDs is removed. Shared geometric lookup allocation occurs only when the plan actually contains aliases.

The ledger currently covers request-owned sequences. Cache-only owners must be incorporated when automatic scheduler cache storage is connected. Native graph capability remains disabled; no serving binary was rebuilt or deployed in this gate. These changes do not justify a throughput or latency claim.

## Verification

| Command / scope | Result |
|---|---|
| `cargo test -p riley-scheduler --quiet` | 150 passed across library, integration and doc suites; 0 failed |
| Scheduler library subset | 50 passed |
| `CARGO_PROFILE_TEST_OPT_LEVEL=2 cargo test -p riley-runtime --lib --quiet` | 348 passed, 0 failed, 1 existing timing diagnostic ignored |
| Paged KV subset | 32 passed |

[Scheduler log](scheduler-tests.log) and [runtime log](runtime-tests.log) contain the final terminal test output. New rejection coverage includes stale committed-head generations, external append holds, shared tail writes, shifted read positions, and the unchanged exclusive plan contract. Initial test fixture capacity/model geometry and a mutable binding compile error were corrected before these final runs.

Remaining serving batch: retained model capability and identity binding, captured-model COW/drain seam, scheduler cache lookup/publication/eviction, full-model parity, then matched vLLM cache-hit/cache-miss and cache-off regression comparison. No new serving measurement was run at this host integration boundary.
