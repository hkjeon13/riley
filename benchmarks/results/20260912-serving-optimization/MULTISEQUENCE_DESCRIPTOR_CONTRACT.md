# Proposed N2/N4 descriptor and completion contract

This is a CPU-side packet/ownership specification for the proposed genuine
multi-sequence owner. It is not an implemented ABI or a qualification result.
Scheduling policy and numerical work remain in
[MULTISEQUENCE_BATCH_DESIGN.md](MULTISEQUENCE_BATCH_DESIGN.md); this document
defines the byte contract and rejection cases needed before native recording.
No production source or remote/GPU state was changed.

## Existing contracts that constrain the new format

* `PreparedLlamaBatchMetadata::pack/validate/validate_block_table` in
  `crates/riley-runtime/src/llama/batch.rs` already checks unique sequence tags,
  one token/output per decode row, dense unique output slots, exact live block
  counts, canonical full-block/tail valid counts, pool bounds and physical-block
  uniqueness across the iteration. It produces explicit per-token positions
  from `target_logical_length - input_count`; rows need not share a position.
  Its V1 CSR format supports more shapes than this proposed owner. Passing its
  validation is necessary but does not admit arbitrary mixed/prefill shapes.
* `PreparedLlamaIteration`/`adapt_item` in
  `crates/riley-scheduler/src/execution.rs` bind each work item to a block-table
  snapshot with the same request ID and target length. Runtime `sequence_tag`
  is that scheduler request ID. It is not the HTTP engine ID or a device row.
* Existing native decode metadata has `align4(16 + 6*C) + 4` bytes for block
  capacity `C`: **32 at C2, 116 at C16, 80 at C10**. These are distinct exact
  layouts, not interchangeable versions. `OwnedLlamaDecodeExecutor` appends
  520 bytes for P128: magic + token count + 128 U32 tokens. At C10 this is
  **600 bytes**. Its replay upload is padded to the captured transfer size,
  which can be larger than metadata because the same staging owner holds output.
* The old M1 result is an 8-byte `RileyCudaBf16ArgmaxResult` and a **32-byte
  embedding error report**, optionally after BF16 logits. That 32-byte report
  is unrelated to the 32-byte C2 input layout. `graph_decode_full.rs::execute`
  clears readiness before validation and permits no publication after an
  execution failure; native `replay_transfer` also invalidates completion on
  attempted replay, even when preflight rejects it.

## Serialization and fixed request layout

Proposed name: `LlamaMultiGraphPacketV1`. Use explicit little-endian integer
encoding and checked byte readers; never serialize a Rust/C struct or padding.
Offsets below are byte offsets, intervals exclude the end, and counts are
unsigned. Bounds/`offset + width`/multiplications are checked before indexing.
Unknown version, size, enum, flag or nonzero reserved byte is rejected. Pointer
values, allocation handles and untrusted offsets are never transported.

The request is **exactly 1,280 bytes** for every stage/bucket:

| Region | Offset | Bytes |
|---|---:|---:|
| Header | 0 | 128 |
| Four row records, stride128 | 128 | 512 |
| P128-compatible C10 metadata view | 640 | 600 |
| Zero padding | 1240 | 40 |

The fixed device metadata parent and pinned input region can be shared by all
captures. This says nothing about GEMM scratch: use separate exact-M parents
for M1/M2/M4 under the common owner. Current `bind_reserved_gemm_state` rejects
M≠1 and `canonical_gemm_bf16_graph_state_is_valid` requires exact whole-parent
input/output byte lengths. A maximum-M parent is not a valid smaller-M binding.
New qualified N2/N4 bindings need their own exact geometry checks.

### Header (128 bytes)

| Offset | Type/bytes | Field and accepted value |
|---:|---|---|
| 0 | U32 | Magic `0x31444D52` (LE bytes `RMD1`) |
| 4 | U16 | Packet version1, distinct from existing CSR/block-table versions |
| 6 | U16 | Header bytes128 |
| 8 | U32 | Packet bytes1280 |
| 12 | U32 | Row stride128 |
| 16 | U32 | Stage: 0=`Prefill128`, 1=`Decode` |
| 20 | U32 | GEMM request-row bucket:1,2,4 |
| 24 | U32 | Active request rows A |
| 28 | U32 | Output count, exactly A |
| 32 | U32 | Wire row capacity4 |
| 36 | U32 | Per-request block capacity10 |
| 40 | U32 | Vocabulary49152 |
| 44 | U32 | Actual physical pool block count; equals cold owner binding |
| 48 | U64 | Nonzero owner-instance generation |
| 56 | U64 | Nonzero replay identity assigned by this owner |
| 64 | U64 | Exact scheduler iteration ID |
| 72 | 32 raw bytes | Cold catalog/signature digest, exact owner match |
| 104 | U32 | Result mode:0=greedy, 1=full BF16 logits |
| 108 | U32 | Flags0 |
| 112 | U32 | Appended P128 input tokens:128 for Prefill128, otherwise0 |
| 116 | U32 | Reserved0 |
| 120 | U64 | Reserved0 |

Owner generation is a checked, process-unique monotonically assigned instance
number, not a GPU pointer; overflow requires a new safe ownership boundary, not
wrapping/saturation. Replay identity likewise must never be reused after native
submission accepts it, including a failed/unknown submission. Preserve a
separate successful-replay counter if desired. A malformed call always clears
readiness; it cannot make an old result available under a new call.

The digest binds source/numerics, schema, physical/per-sequence capacities,
output format, exact plans/parents and captured `(stage,bucket,result_mode)`
catalog. Select a captured entry using this freshly validated header. A mode
whose D2H topology was not prepared must fail before dispatch. To support both
CPU sampling and GPU greedy in one owner, prepare both transfer modes cold;
their graphs may share the same exact-M scratch/plans. Do not change a captured
copy length at runtime or silently return full logits in greedy mode.

### Row `r`, base `128 + 128*r`

The first80 bytes are deliberately a legacy **C10 row-local** decode metadata
view. The two legacy zero slots are not the global execution/output slots.

| Relative offset | Type/bytes | Field |
|---:|---|---|
| 0 | U32 | Input token ID; for P128, the last prompt token |
| 4 | U32 | Absolute input position `p` |
| 8 | U32 | Legacy token-to-sequence slot0 |
| 12 | U32 | Live logical blocks L |
| 16 | U32[10],40 | Physical IDs in logical-block order |
| 56 | U16[10],20 | Valid counts in the same order |
| 76 | U32 | Legacy output slot0 |
| 80 | U64 | Scheduler `sequence_tag`/request ID |
| 88 | U64 | Owner-issued current reservation cookie |
| 96 | U32 | Target logical length `p+1` |
| 100 | U32 | Actual dense iteration output slot |
| 104 | U32 | This row's input count:128 or1 |
| 108 | U32 | Expected generated-output index for server routing |
| 112 | U32 | Row flags:1=active; no other bits |
| 116 | U16 | Existing `BLOCK_TABLE_V1_VERSION` (1) |
| 118 | U16 | Reserved0 |
| 120 | U64 | Reserved0 |

For every active row: token<49152; `L=ceil((p+1)/16)≤10`; each physical ID is
within the actual pool; entries0..L-2 have valid16, entryL-1 has valid
`(p%16)+1`. Remaining physical IDs **and** valid counts are zero. Physical ID0
is valid for a live entry; liveness is determined by L, not by a sentinel ID.
Sum(valid)=target length. Active output slots are a permutation of `0..A`,
preserving the scheduler mapping; they need not equal execution-row indices.
Sequence tags and reservation cookies are nonzero and unique among active rows.
These valid counts describe the reserved **post-success target**, including the
current token's destination, not already-initialized GPU bytes. Decode requires
the request's committed KV prefix `[0,p)` and exclusive ownership of its new
position p; fresh P128 may populate all128 positions. Reservation validation
must establish that lineage. Byte counts alone cannot certify initialized KV.

Cookie is proposed host bookkeeping, not an existing KV API field or proof of
ownership by itself. Allocate a new cookie when binding a planned reservation
to a submitted row; retain its immutable snapshot in the owner's bounded
in-flight row ledger until settlement. Encode only after the scheduler adapter
has verified the actual reservation, request ID, target, table and expected
generated index. A byte parser can prove pool bounds/cross-row uniqueness, but
cannot prove that an arbitrary physical block belongs to this request. The
host reservation/ledger comparison supplies that proof, including against
resident requests absent from this iteration. Native must compare the packet
with its registered submitted-row binding rather than treat a caller-supplied
cookie/tag as authority. No raw packet API may bypass that binding step. Binding,
encoding and submission are one owner-thread transaction under the exclusive
owner borrow; no other request may replace that ledger while a replay or its
settlement is outstanding. The packet retains no pointers into a reusable CSR
vector or a movable Rust request.

### Stages, compatibility view and N3 padding

`Prefill128`: A=1, bucket1, row0 active, global output slot0, input count128,
position127, target128, live8, valid[0..8]=16 and remaining entries zero. The
host input positions must be exactly0..127 and the expected generated index0.
The 600-byte region at640 is exactly:

| Offset | Bytes | Content |
|---:|---:|---|
| 640 | 80 | Byte-identical copy of row0's legacy prefix `[128,208)` |
| 720 | 4 | Existing P128 magic `0x50313238` |
| 724 | 4 | Token count128 |
| 728 | 512 | Full128 prompt U32 IDs, each<49152; last equals row0 token |

The new recorder uses the checked constant byte view `[640,1240)` at metadata parent+640 for
unchanged P128 numerical wrappers. It must not fabricate an opaque buffer
handle or pass a larger parent through the old exact-whole-parent recording
API. Repeating the80-byte prefix avoids an extra GPU conversion/upload and is
accepted only when byte-identical. Original P128 APIs and kernel bodies remain
unchanged. All other row records and trailing padding are zero.

The compatibility view is a bounded read of registered metadata bytes with
U32/U16 alignment. It is not a GEMM span and does not acquire a new ownership
lease. Its offset is a recorder constant, never a pointer/offset supplied in
the packet. Both the metadata parent and its pinned upload source remain in
the common owner ledger through completion.

`Decode`: A in1..cold active capacity; bucket is the smallest of1/2/4 that holds
A. Each active row has one token and its own p in128..159; no shared end-position
formula is allowed. The P128-compatible region and trailing padding are zero.
The per-request generated index is checked against scheduler state; under the
P128 contract it is `p-127`. Public O≤32 means the normal last decoded input is
position158 (output index31). Position159 fits the low-level numerical/160-KV
envelope but does **not** authorize an O33 serving request or an extra token
after the request's output limit.

For A3, bucket4, rows0..2 are active and row3 is **128 zero bytes**. RowsA..4 are
always zero even for smaller buckets; inactive rows have no request, cookie,
KV table or output slot despite zero-valued fields. Every inactive GEMM input
must be written as zero before consumption, including after a larger preceding
replay. Descriptor-aware embedding/norm/residual/SwiGLU paths zero their inactive
output rows; attention zeroes its inactive output and performs no KV access.
An inactive CTA return must be uniform and precede every CTA barrier. Inactive
rows may occupy qualified padded GEMM work; they cannot publish KV, request
state, an argmax success record or a token event.

## Bulk completion format and validation

Proposed result magic `0x314F4D52` (LE `RMO1`), version1. The result has a128-byte
header and four128-byte row records at128, in **execution-row order**. Its
header uses the same layout and echoes request owner/replay/iteration/digest,
stage, bucket, counts, result mode and cold geometry. Differences: packet bytes
is the selected result length; offset108 is overall status0=success/1=row error;
offset112 is logits row count0 for greedy or bucket for full logits. Reserved
bytes stay zero. No output identity is inferred from a previously selected
graph or the first returned row.

| Result row offset | Type/bytes | Content |
|---:|---|---|
| 0 | U64 | Sequence tag |
| 8 | U64 | Reservation cookie |
| 16 | U64 | Replay ID |
| 24 | U64 | Iteration ID |
| 32 | U32 | Execution row index |
| 36 | U32 | Actual output slot |
| 40 | U32 | Input position |
| 44 | U32 | Expected generated-output index |
| 48 | U32 | Greedy token ID |
| 52 | U32 | Existing argmax status (0 success;1 nonfinite) |
| 56 | 32 | Embedding report, explicitly encoded: size U32=32, code U32, token_position U64, token_id U64, reserved U64 |
| 88 | U32 | Row completion:1=complete,2=failed;0 reserved for inactive |
| 92 | U32 | Reserved0 |
| 96 | U64 | Full-logits byte offset from result start, or0 in greedy mode |
| 104 | U64 | Full-logits byte length98304, or0 in greedy mode |
| 112 | 16 | Reserved zero bytes |

Greedy result length is **640 bytes**. Full logits start at640, row stride
`49152*2=98304`, in execution-row order; lengths are **98944/197248/393856**
for bucket1/2/4. Row r's checked logits offset is `640+r*98304`, not an arbitrary
device-provided offset. The CPU adapter scatters completed rows into existing
dense slot workspaces using the already validated slot permutation. This
avoids relying on a GEMM's output row order as request routing authority.
Inactive result records and any inactive logits row in the transferred bucket
are canonical zero; a descriptor-aware final packing step may canonicalize
wire padding without treating inactive numerical scratch as meaningful output.

The result parent may be prepared for the maximum packet size with explicit
checked **copy** spans for each catalog entry. It is separate from exact-M GEMM
output parents. Zero/poison readiness storage before replay; the final reporting
operation stamps active row identities/status only after that graph's required
embedding, head and argmax operations. The bulk D2H depends on completion of
all result writers. An early CTA/header store is not a cross-grid completion
barrier. One stream completion must be proven before reading the envelope.

Prepublication sequence:

1. Clear every reusable getter/workspace readiness bit before any validation.
   Validate all prepared rows, reservations, stage, bytes and catalog binding
   before memmove/H2D/launch. No partial row upload or fallback replay.
2. After known GPU completion and full D2H, validate the entire header and every
   active/inactive record against the retained submission ledger. Require exact
   byte count, unique/dense slots, matching owner/replay/iteration/tag/cookie/
   row/position/generated index, row-complete1, argmax status0 and token<49152.
   Require embedding size32/code0/position0/token0/reserved0. For full mode,
   check exact offsets/lengths and all active BF16 logits finite; qualification
   compares full logits and argmax with the independent M1 oracle.
3. Only after every row passes, fill all output slots and expose one immutable
   downloaded-batch view. CPU sampling validates every selected token before
   forming `IterationResult`; GPU greedy uses the validated per-slot IDs.
4. `IterationResult` carries the scheduler iteration ID and each dense slot
   once. Retain `Scheduler::complete_iteration`'s result/reservation prevalidation
   and settlement before `publish_committed_item`. The engine must prevalidate
   **all** pending request mappings/delivery indices before its first external
   event, then publish only the scheduler's committed request-specific events.

Malformed/stale completion publishes no row. GPU KV writes cannot be rolled
back by rejecting a result. Distinguish NotDispatched from proven-quiesced but
mutation-unknown failure; unknown completion retains graph/parent/KV leases and
the in-flight reservations. Do not retry eagerly or free cancelled rows while
device use is possible. An in-flight cancellation remains a deferred scheduler
state change: validate its returned row normally, settle its reservation after
quiescence, suppress that request's token publication, and preserve unrelated
successful requests according to existing settlement semantics. Close graphs
before releasing shared parents; no new owner may adopt a live unsettled row.

## CPU encode/decode fixtures to implement before native integration

These are required tests for the proposed codec; no such codec was implemented
while writing this document. Begin with valid fixtures, mutate one relation per
negative case and verify output-not-ready on rejection. Input rejection means
zero submissions; result rejection means no additional submission and zero
publications, while retaining the appropriate completed/unknown ownership state.

Concrete N3 golden case: owner generation1, replay7, iteration42, physical pool40,
bucket4, active/output count3, full-logits mode, and a **codec-test catalog digest
of32 bytes `0x5a`** (registered in that synthetic fixture, not a production key).
Use the following rows with flags1, input count1, block-table version1, both
legacy slots0 and all reserved bytes0. Row3 and the entire P128 region are zero.
Cookies are bound to synthetic immutable reservations in the codec test fixture.

| Row | Tag/cookie | Token / p / target / generated index / output slot | Ten physical IDs | Ten valid counts |
|---:|---|---|---|---|
| 0 | 101 / 5001 | 28 / 128 / 129 / 1 / 2 | `[7,2,19,5,11,23,0,16,9,0]` | `[16,16,16,16,16,16,16,16,1,0]` |
| 1 | 202 / 5002 | 339 / 144 / 145 / 17 / 0 | `[34,28,31,24,36,25,39,29,33,21]` | `[16,16,16,16,16,16,16,16,16,1]` |
| 2 | 303 / 5003 | 5248 / 158 / 159 / 31 / 1 | `[3,4,6,8,10,12,13,14,15,17]` | `[16,16,16,16,16,16,16,16,16,15]` |

L is9/10/10. Row0 deliberately uses physical ID0 in its live prefix while its
unused last ID is also0. A valid completion echoes all identities, has report
size32 with other report fields0 and row-complete1, and has logits offsets
640/98944/197248 with byte length98304. The inactive fourth logits row starts at
295552 and is zero; packet length393856. The adapter orders rows1/2/0 into
output slots0/1/2. This fixture must round-trip byte-for-byte, not merely yield
equivalent host fields.

For the codec-only completion fixture, set every active logit to BF16 positive
zero and each greedy token/status to0; this tests serialization and is not model
output evidence. Direct CPU byte construction gives request SHA256
`885cfa7344b8f9ceda8a279ba1b5d7a734ba87a0e8ba56ea6d07617ad2ef96c9`
and393856-byte result SHA256
`b9b72de4923751be9f05aae5c8d73cda0fe6dd83943117bad35cf8f1b375fb86`.

| Fixture | Expected result |
|---|---|
| Fresh P128, IDs in vocabulary,8 noncontiguous blocks, exact600-byte compatibility view | Accept; bucket1, slot0, output index0 |
| Two decoders p128/p143, disjoint maps, slots `[1,0]` | Accept; L9/L9, tail1/16; returned rows scatter correctly |
| Three decoders p128/p144/p158, slots `[2,0,1]`, row3 zero | Accept bucket4; L9/10/10, tail1/1/15; no inactive publication |
| Transition4→3→1→2 with fresh replay IDs/cookies and fully zeroed padding | Accept; no previous larger-bucket output remains readable |
| Truncated/extra byte; wrong magic/version/header/stride/geometry/digest; unknown mode/flag; nonzero reserved | Reject before copy; no reinterpretation as old32/116/600-byte format |
| Integer overflow while deriving byte extents, `p+1`, replay or cookie | Reject; no wrap/saturating identity |
| A0/A>capacity, wrong smallest bucket, output count mismatch, active row after a zero hole | Reject |
| Mixed stages, partial127-token prefill, second prefill row, decode with two tokens, nonzero decode P128 region | Reject |
| P128 position/target not127/128, compatibility prefix differs, last appended token differs, any token≥49152 | Reject |
| Decode p127/p160; target not p+1; generated index differs; request O32 already exhausted at p159 | Reject at respective owner/scheduler gate |
| Duplicate tag/cookie/slot, sparse slots `[0,2]`, output slot outside A | Reject; row index must not replace slot mapping |
| Block ID outside pool, cross-row repeated ID, duplicate within one row, block belongs to off-batch live request | Reject; last case requires reservation authority, not merely byte validation |
| Incorrect L, valid15 on a non-tail block, zero tail, valid sum mismatch, nonzero unused ID/valid | Reject |
| Inactive row with a tag/token/position/block ID/slot/flag; nonzero request padding | Reject even when count says inactive |
| Complete-shaped result with old owner/replay/iteration/cookie or swapped tag/slot only | Reject whole batch; all getters unavailable |
| Missing/duplicate row or slot, wrong generated index/position, incomplete marker, token UINT32_MAX or≥vocab | Reject whole batch |
| Nonfinite argmax/report, embedding report wrong size or any success-reserved field nonzero, invalid full-logit span | Reject whole batch |
| Result byte count inconsistent with bucket/mode; nonzero inactive record/logits; previous bucket's extra rows | Reject whole batch |
| Cancellation arrives after submission; remaining rows complete | No early KV reuse; cancelled request emits no token, other committed rows retain correct routing |
| Launch/sync ambiguity; post-sync malformed result; pre-copy rejection | Three distinct failure dispositions; none may expose stale output or infer safe reuse without proof |

## Concrete source seams

| Source | Change/retained authority |
|---|---|
| `crates/riley-runtime/src/llama/batch.rs` | Retain generic CSR validation; add separate codec/profile-stage gate, fixed arrays and checked serialization |
| `crates/riley-scheduler/src/execution.rs` | Bind immutable plan/reservation rows and iteration ID; replace singular graph `greedy_token` push with whole-batch validation and dense-slot output |
| `crates/riley-runtime/src/llama/graph_decode_full.rs` | New sibling owner and bounded submitted-row ledger; preserve old M1/P128 APIs and readiness/poison discipline |
| `crates/riley-cuda/src/ffi.rs`, `graph_resources.rs`; `kernels/include/riley_cuda.h` | Additive packet/record/replay/read contract, complete array/byte count validation and shared lease lifetime |
| `kernels/src/graph_resources.cu` | Fresh header-based catalog selection, checked metadata byte views, exact-M parent bindings, one upload/launch/bulk completion and all-graph close |
| `kernels/src/gemm.cu` | Separate qualified M2/M4 exact-parent binding; do not weaken current M1 validator |
| `crates/riley-scheduler/src/scheduler.rs` | Preserve `validate_iteration_result`, `prevalidate_completion_publication`, reservation settlement and deferred cancellation |
| `crates/riley-server/src/engine.rs` | Separate request rows from `plan.total_tokens()` (P128 is128 tokens but1 request); validate complete pending publication routing before the first event |

Independent GPU projection and per-sequence full-logits/KV gates still decide
whether these descriptor shapes may be implemented/captured. A successful
CPU codec round-trip cannot qualify a numerical plan or concurrent serving.
