//! Multi-sequence wire validation; live authority is supplied by the scheduler adapter.
//!
//! This crate owns no GPU/KV resources. Expectations are supplied by an
//! authoritative scheduler/owner adapter; packet bytes never establish live
//! ownership or quiescence. GPU dispatch wiring is separate; no performance claim.

pub mod shape_progress;

use std::collections::BTreeSet;
use std::fmt;

pub const REQUEST_BYTES: usize = 1792;
pub const RESULT_PREFIX_BYTES: usize = 1152;
pub const LOGITS_BYTES: usize = 49152 * 2;
pub const MAX_RESULT_BYTES: usize = RESULT_PREFIX_BYTES + 8 * LOGITS_BYTES;
pub const CONTRACT_SHA256: &str =
    "fdeb3a8c660a5a1059685361ddf50cc950354051f663817eb6a1fb2a8bc39329";
const REQUEST_MAGIC: u32 = 0x3144_4d52;
const RESULT_MAGIC: u32 = 0x314f_4d52;
const PREFILL_MAGIC: u32 = 0x5031_3238;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub field: &'static str,
    pub reason: &'static str,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
fn check(ok: bool, field: &'static str, reason: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error { field, reason })
    }
}
fn overflow(field: &'static str) -> Error {
    Error {
        field,
        reason: "checked arithmetic overflow",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum Stage {
    Prefill128 = 0,
    Decode = 1,
}
impl Stage {
    fn parse(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Prefill128),
            1 => Ok(Self::Decode),
            _ => Err(Error {
                field: "stage",
                reason: "unknown stage",
            }),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum ResultMode {
    Greedy = 0,
    FullLogits = 1,
}
impl ResultMode {
    fn parse(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Greedy),
            1 => Ok(Self::FullLogits),
            _ => Err(Error {
                field: "result_mode",
                reason: "unknown result mode",
            }),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CatalogEntry {
    pub stage: Stage,
    pub bucket: u32,
    pub mode: ResultMode,
}

/// Explicit caller authority, not extracted from packet fields. Catalog entries
/// must correspond to actual prepared/qualified graphs in a future adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerExpectation {
    pub generation: u64,
    pub last_accepted_replay: u64,
    pub catalog_digest: [u8; 32],
    pub max_active_rows: u32,
    pub physical_block_count: u32,
    pub catalog: Vec<CatalogEntry>,
}
/// Includes resident/off-batch requests and reserved destinations. Merely being
/// within the physical pool is not ownership. The caller derives this ledger
/// from real live/reserved KV state while holding its exclusive owner borrow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockOwnership {
    pub physical_id: u32,
    pub sequence_tag: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReservationExpectation {
    pub sequence_tag: u64,
    pub cookie: u64,
    pub input_tokens: Vec<u32>,
    pub committed_length: u32,
    pub target_length: u32,
    pub output_slot: u32,
    pub generated_index: u32,
    pub max_output_tokens: u32,
    pub physical_ids: Vec<u32>,
    pub valid_tokens: Vec<u16>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionExpectation {
    pub owner: OwnerExpectation,
    pub replay_id: u64,
    pub iteration_id: u64,
    pub stage: Stage,
    pub mode: ResultMode,
    pub rows: Vec<ReservationExpectation>,
    pub block_ownership: Vec<BlockOwnership>,
}

fn bucket(count: usize) -> Result<u32> {
    match count {
        1 => Ok(1),
        2 => Ok(2),
        3 | 4 => Ok(4),
        5..=8 => Ok(8),
        _ => Err(Error {
            field: "active_rows",
            reason: "requires one through eight rows",
        }),
    }
}
pub fn result_bytes(bucket: u32, mode: ResultMode) -> Result<usize> {
    check(
        matches!(bucket, 1 | 2 | 4 | 8),
        "bucket",
        "unsupported exact bucket",
    )?;
    let rows = if mode == ResultMode::FullLogits {
        bucket as usize
    } else {
        0
    };
    RESULT_PREFIX_BYTES
        .checked_add(
            rows.checked_mul(LOGITS_BYTES)
                .ok_or_else(|| overflow("result_bytes"))?,
        )
        .ok_or_else(|| overflow("result_bytes"))
}

/// Validate expectations themselves before accepting any caller-supplied bytes.
/// This verifies structural consistency; it cannot inspect real GPU memory or
/// independently certify that the caller's scheduler snapshot is truthful.
pub fn validate_expectations(e: &SubmissionExpectation) -> Result<()> {
    check(
        e.owner.generation != 0,
        "owner_generation",
        "zero owner identity",
    )?;
    check(
        matches!(e.owner.max_active_rows, 2 | 4 | 8),
        "max_active_rows",
        "owner capacity must be two, four or eight",
    )?;
    check(
        e.owner.physical_block_count >= e.owner.max_active_rows * 10,
        "physical_block_count",
        "pool cannot cover fixed maximum reservations",
    )?;
    let next = e
        .owner
        .last_accepted_replay
        .checked_add(1)
        .ok_or_else(|| overflow("replay_id"))?;
    check(
        e.replay_id == next && e.iteration_id != 0,
        "replay_identity",
        "expected next replay and nonzero iteration required",
    )?;
    let b = bucket(e.rows.len())?;
    check(
        e.rows.len() <= e.owner.max_active_rows as usize,
        "active_rows",
        "owner capacity exceeded",
    )?;
    check(
        e.stage != Stage::Prefill128 || e.rows.len() == 1,
        "stage",
        "P128 represents exactly one request",
    )?;
    let mut entries = BTreeSet::new();
    for entry in &e.owner.catalog {
        check(
            matches!(entry.bucket, 1 | 2 | 4 | 8)
                && entry.bucket <= e.owner.max_active_rows
                && (entry.stage != Stage::Prefill128 || entry.bucket == 1),
            "catalog",
            "invalid cold catalog entry",
        )?;
        check(entries.insert(*entry), "catalog", "duplicate catalog entry")?;
    }
    check(
        entries.contains(&CatalogEntry {
            stage: e.stage,
            bucket: b,
            mode: e.mode,
        }),
        "catalog",
        "exact stage/bucket/result mode was not prepared; no fallback",
    )?;
    check(
        e.block_ownership.len() <= e.owner.physical_block_count as usize,
        "block_ownership",
        "ledger exceeds pool",
    )?;
    let mut all_ids = BTreeSet::new();
    for own in &e.block_ownership {
        check(
            own.physical_id < e.owner.physical_block_count && own.sequence_tag != 0,
            "block_ownership",
            "invalid owner ledger record",
        )?;
        check(
            all_ids.insert(own.physical_id),
            "block_ownership",
            "duplicate physical block authority",
        )?;
    }
    let (mut tags, mut cookies, mut slots, mut used) = (
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
    );
    for row in &e.rows {
        check(
            row.sequence_tag != 0 && tags.insert(row.sequence_tag),
            "sequence_tag",
            "zero or duplicate sequence tag",
        )?;
        check(
            row.cookie != 0 && cookies.insert(row.cookie),
            "reservation_cookie",
            "zero or duplicate cookie",
        )?;
        check(
            row.output_slot < e.rows.len() as u32 && slots.insert(row.output_slot),
            "output_slot",
            "slots must be dense and unique",
        )?;
        check(
            (1..=32).contains(&row.max_output_tokens)
                && row.generated_index < row.max_output_tokens,
            "generation_bound",
            "request output bound is exhausted or unsupported",
        )?;
        let input_count = if e.stage == Stage::Prefill128 { 128 } else { 1 };
        check(
            row.input_tokens.len() == input_count && row.input_tokens.iter().all(|&t| t < 49152),
            "input_tokens",
            "wrong count or vocabulary bound",
        )?;
        // The v2 wire remains deliberately bounded; the logical request-progress
        // calculation is shared with the variable-shape path being integrated.
        let progress = shape_progress::Progress {
            prompt_tokens: 128, output_limit: row.max_output_tokens, context_tokens: 160,
            committed_tokens: row.committed_length, input_tokens: input_count as u32,
            generated_index: row.generated_index,
            stage: if e.stage==Stage::Prefill128 { shape_progress::InputStage::Prefill } else { shape_progress::InputStage::Decode },
        }.validate()?;
        let target=progress.target_tokens;
        check(row.target_length==target,"target_length","target differs from committed prefix plus inputs")?;
        let pos=progress.last_position;
        let live = target.div_ceil(16) as usize;
        check(
            row.physical_ids.len() == live && row.valid_tokens.len() == live && live <= 10,
            "block_table",
            "table must exactly cover reserved target",
        )?;
        for (index, (&id, &valid)) in row.physical_ids.iter().zip(&row.valid_tokens).enumerate() {
            let expected_valid = if index + 1 == live {
                (pos % 16 + 1) as u16
            } else {
                16
            };
            check(
                valid == expected_valid,
                "valid_tokens",
                "noncanonical target valid counts",
            )?;
            check(
                id < e.owner.physical_block_count && used.insert(id),
                "physical_ids",
                "out-of-pool or aliased physical block",
            )?;
            check(
                e.block_ownership
                    .iter()
                    .any(|own| own.physical_id == id && own.sequence_tag == row.sequence_tag),
                "block_ownership",
                "block is absent or belongs to another resident request",
            )?;
        }
    }
    Ok(())
}

/// CPU ownership/readiness guard for one owner instance. `settle_success` and
/// `settle_failure` require the caller to have performed the real scheduler/KV
/// settlement; these methods do not perform or prove it. A failed completion
/// poisons this instance even after its diagnostic transaction is removed.
pub struct CodecOwner {
    expectation: OwnerExpectation,
    transaction: Option<Transaction>,
    poisoned: bool,
}
impl CodecOwner {
    pub fn new(expectation: OwnerExpectation) -> Self {
        Self {
            expectation,
            transaction: None,
            poisoned: false,
        }
    }
    pub fn expectation(&self) -> &OwnerExpectation {
        &self.expectation
    }
    pub fn admit(&mut self, packet: &[u8], e: SubmissionExpectation) -> Result<()> {
        if let Some(transaction) = self.transaction.as_mut() {
            // An attempted replacement cannot preserve a previously ready
            // result or erase the old reservation authority.
            transaction.ready = None;
            if transaction.state == State::Validated {
                transaction.state = State::FailedQuiesced;
            }
            self.poisoned = true;
            return Err(Error {
                field: "owner",
                reason: "unsettled transaction cannot be replaced",
            });
        }
        check(!self.poisoned, "owner", "owner is poisoned")?;
        check(
            e.owner == self.expectation,
            "owner",
            "caller expectation is not the current owner state",
        )?;
        let transaction = Transaction::admit(packet, e)?;
        self.expectation.last_accepted_replay = transaction.expectation.replay_id;
        self.transaction = Some(transaction);
        Ok(())
    }
    pub fn accept_result(&mut self, bytes: &[u8], completion: CompletionEvidence) -> Result<()> {
        let transaction = self.transaction.as_mut().ok_or(Error {
            field: "owner",
            reason: "no admitted transaction",
        })?;
        let result = transaction.accept_result(bytes, completion);
        if result.is_err() {
            self.poisoned = true;
        }
        if self.poisoned && transaction.state == State::Validated {
            transaction.ready = None;
            transaction.state = State::FailedQuiesced;
            return Err(Error {
                field: "owner",
                reason: "poisoned owner cannot publish a completed row",
            });
        }
        result
    }
    pub fn result(&self) -> Result<&ValidatedBulk> {
        check(
            !self.poisoned,
            "owner",
            "poisoned owner output is not ready",
        )?;
        self.transaction
            .as_ref()
            .ok_or(Error {
                field: "output",
                reason: "no admitted transaction",
            })?
            .result()
    }
    pub fn retained_expectation(&self) -> Option<&SubmissionExpectation> {
        self.transaction.as_ref().map(|t| &t.expectation)
    }
    pub fn settlement_advice(&self) -> Option<SettlementAdvice> {
        self.transaction
            .as_ref()
            .map(Transaction::settlement_advice)
    }
    pub fn establish_quiescence_after_failure(&mut self, proof: CompletionEvidence) -> Result<()> {
        self.transaction
            .as_mut()
            .ok_or(Error {
                field: "owner",
                reason: "no admitted transaction",
            })?
            .establish_quiescence_after_failure(proof)
    }
    pub fn settle_success(&mut self, scheduler_commit_confirmed: bool) -> Result<ValidatedBulk> {
        check(
            scheduler_commit_confirmed
                && !self.poisoned
                && self
                    .transaction
                    .as_ref()
                    .is_some_and(|t| t.state == State::Validated && t.ready.is_some()),
            "settlement",
            "whole result and explicit scheduler commit required",
        )?;
        Ok(self.transaction.take().unwrap().ready.take().unwrap())
    }
    pub fn settle_failure(&mut self, containment_confirmed: bool) -> Result<()> {
        check(
            containment_confirmed
                && self
                    .transaction
                    .as_ref()
                    .is_some_and(|t| t.state == State::FailedQuiesced),
            "settlement",
            "proven quiescence and explicit containment required",
        )?;
        self.transaction = None;
        self.poisoned = true;
        Ok(())
    }
}

fn span(bytes: &[u8], offset: usize, count: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(count)
        .ok_or_else(|| overflow("byte_extent"))?;
    bytes.get(offset..end).ok_or(Error {
        field: "byte_extent",
        reason: "out-of-bounds byte view",
    })
}
fn read16(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        span(bytes, offset, 2)?.try_into().unwrap(),
    ))
}
fn read32(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        span(bytes, offset, 4)?.try_into().unwrap(),
    ))
}
fn read64(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        span(bytes, offset, 8)?.try_into().unwrap(),
    ))
}
fn put(bytes: &mut [u8], offset: usize, value: &[u8]) -> Result<()> {
    let end = offset
        .checked_add(value.len())
        .ok_or_else(|| overflow("byte_extent"))?;
    bytes
        .get_mut(offset..end)
        .ok_or(Error {
            field: "byte_extent",
            reason: "out-of-bounds write",
        })?
        .copy_from_slice(value);
    Ok(())
}
fn put16(bytes: &mut [u8], offset: usize, value: u16) -> Result<()> {
    put(bytes, offset, &value.to_le_bytes())
}
fn put32(bytes: &mut [u8], offset: usize, value: u32) -> Result<()> {
    put(bytes, offset, &value.to_le_bytes())
}
fn put64(bytes: &mut [u8], offset: usize, value: u64) -> Result<()> {
    put(bytes, offset, &value.to_le_bytes())
}
fn zero(bytes: &[u8], offset: usize, count: usize, field: &'static str) -> Result<()> {
    check(
        span(bytes, offset, count)?.iter().all(|&x| x == 0),
        field,
        "nonzero reserved or inactive bytes",
    )
}

fn write_header(bytes: &mut [u8], e: &SubmissionExpectation, result: bool) -> Result<()> {
    let b = bucket(e.rows.len())?;
    put32(bytes, 0, if result { RESULT_MAGIC } else { REQUEST_MAGIC })?;
    put16(bytes, 4, 2)?;
    put16(bytes, 6, 128)?;
    let total = u32::try_from(bytes.len()).map_err(|_| overflow("packet_bytes"))?;
    let payload = if result {
        if e.mode == ResultMode::FullLogits {
            b
        } else {
            0
        }
    } else if e.stage == Stage::Prefill128 {
        128
    } else {
        0
    };
    for (offset, value) in [
        (8, total),
        (12, 128),
        (16, e.stage as u32),
        (20, b),
        (24, e.rows.len() as u32),
        (28, e.rows.len() as u32),
        (32, 8),
        (36, 10),
        (40, 49152),
        (44, e.owner.physical_block_count),
        (104, e.mode as u32),
        (108, 0),
        (112, payload),
        (116, 0),
    ] {
        put32(bytes, offset, value)?;
    }
    put64(bytes, 48, e.owner.generation)?;
    put64(bytes, 56, e.replay_id)?;
    put64(bytes, 64, e.iteration_id)?;
    put(bytes, 72, &e.owner.catalog_digest)?;
    put64(bytes, 120, 0)?;
    Ok(())
}
fn check_header(bytes: &[u8], e: &SubmissionExpectation, result: bool) -> Result<()> {
    let expected_len = if result {
        result_bytes(bucket(e.rows.len())?, e.mode)?
    } else {
        REQUEST_BYTES
    };
    check(
        bytes.len() == expected_len,
        "packet_bytes",
        "exact selected packet size required",
    )?;
    check(
        read32(bytes, 0)? == if result { RESULT_MAGIC } else { REQUEST_MAGIC },
        "magic",
        "unknown packet kind",
    )?;
    check(
        read16(bytes, 4)? == 2 && read16(bytes, 6)? == 128,
        "version",
        "unknown header/version",
    )?;
    check(
        Stage::parse(read32(bytes, 16)?)? == e.stage,
        "stage",
        "stage differs from submitted plan",
    )?;
    check(
        ResultMode::parse(read32(bytes, 104)?)? == e.mode,
        "result_mode",
        "result mode differs from catalog binding",
    )?;
    let mut expected = vec![0; expected_len];
    write_header(&mut expected, e, result)?;
    check(
        span(bytes, 0, 128)? == &expected[..128],
        "header",
        "header geometry, identity, digest, status or padding differs",
    )
}

/// New fixed request encoder. No allocation size or ownership is inferred from
/// input packet bytes; all expected source tokens/reservations are explicit.
pub fn encode_request(e: &SubmissionExpectation) -> Result<[u8; REQUEST_BYTES]> {
    validate_expectations(e)?;
    let mut bytes = [0; REQUEST_BYTES];
    write_header(&mut bytes, e, false)?;
    for (index, row) in e.rows.iter().enumerate() {
        let base = 128 + index * 128;
        put32(&mut bytes, base, *row.input_tokens.last().unwrap())?;
        put32(&mut bytes, base + 4, row.target_length - 1)?;
        put32(&mut bytes, base + 12, row.physical_ids.len() as u32)?;
        for (block, (&id, &valid)) in row.physical_ids.iter().zip(&row.valid_tokens).enumerate() {
            put32(&mut bytes, base + 16 + block * 4, id)?;
            put16(&mut bytes, base + 56 + block * 2, valid)?;
        }
        put64(&mut bytes, base + 80, row.sequence_tag)?;
        put64(&mut bytes, base + 88, row.cookie)?;
        for (offset, value) in [
            (96, row.target_length),
            (100, row.output_slot),
            (104, row.input_tokens.len() as u32),
            (108, row.generated_index),
            (112, 1),
        ] {
            put32(&mut bytes, base + offset, value)?;
        }
        put16(&mut bytes, base + 116, 1)?;
    }
    if e.stage == Stage::Prefill128 {
        bytes.copy_within(128..208, 1152);
        put32(&mut bytes, 1232, PREFILL_MAGIC)?;
        put32(&mut bytes, 1236, 128)?;
        for (index, &token) in e.rows[0].input_tokens.iter().enumerate() {
            put32(&mut bytes, 1240 + index * 4, token)?;
        }
    }
    Ok(bytes)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedRow {
    pub sequence_tag: u64,
    pub cookie: u64,
    pub token: u32,
    pub position: u32,
    pub target_length: u32,
    pub output_slot: u32,
    pub input_count: u32,
    pub generated_index: u32,
    pub physical_ids: Vec<u32>,
    pub valid_tokens: Vec<u16>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedRequest {
    pub stage: Stage,
    pub bucket: u32,
    pub mode: ResultMode,
    pub rows: Vec<DecodedRow>,
}

/// Decoded fields are data, not reservation authority. Admission still requires
/// the complete independent expectation and full byte equality to that binding.
pub fn decode_request(bytes: &[u8], e: &SubmissionExpectation) -> Result<DecodedRequest> {
    validate_expectations(e)?;
    check_header(bytes, e, false)?;
    let mut rows = Vec::with_capacity(e.rows.len());
    for (index, expected) in e.rows.iter().enumerate() {
        let b = 128 + index * 128;
        let live = usize::try_from(read32(bytes, b + 12)?).map_err(|_| overflow("live_blocks"))?;
        check(
            (1..=10).contains(&live),
            "live_blocks",
            "row count outside fixed capacity",
        )?;
        let mut ids = Vec::with_capacity(live);
        let mut valid = Vec::with_capacity(live);
        for block in 0..live {
            ids.push(read32(bytes, b + 16 + 4 * block)?);
            valid.push(read16(bytes, b + 56 + 2 * block)?);
        }
        let row = DecodedRow {
            sequence_tag: read64(bytes, b + 80)?,
            cookie: read64(bytes, b + 88)?,
            token: read32(bytes, b)?,
            position: read32(bytes, b + 4)?,
            target_length: read32(bytes, b + 96)?,
            output_slot: read32(bytes, b + 100)?,
            input_count: read32(bytes, b + 104)?,
            generated_index: read32(bytes, b + 108)?,
            physical_ids: ids,
            valid_tokens: valid,
        };
        check(
            row.sequence_tag == expected.sequence_tag && row.cookie == expected.cookie,
            "row_identity",
            "row differs from bound reservation",
        )?;
        check(
            row.token == *expected.input_tokens.last().unwrap()
                && row.position == expected.target_length - 1
                && row.target_length == expected.target_length
                && row.output_slot == expected.output_slot
                && row.input_count == expected.input_tokens.len() as u32
                && row.generated_index == expected.generated_index,
            "row_fields",
            "row differs from immutable submission",
        )?;
        check(
            row.physical_ids == expected.physical_ids && row.valid_tokens == expected.valid_tokens,
            "block_table",
            "row differs from reserved owner table",
        )?;
        check(
            read32(bytes, b + 112)? == 1 && read16(bytes, b + 116)? == 1,
            "row_flags",
            "active/schema differs",
        )?;
        zero(bytes, b + 8, 4, "legacy_slot")?;
        zero(bytes, b + 76, 4, "legacy_slot")?;
        zero(bytes, b + 16 + 4 * live, 4 * (10 - live), "unused_physical")?;
        zero(bytes, b + 56 + 2 * live, 2 * (10 - live), "unused_valid")?;
        zero(bytes, b + 118, 10, "row_reserved")?;
        rows.push(row);
    }
    zero(
        bytes,
        128 + e.rows.len() * 128,
        (8 - e.rows.len()) * 128,
        "inactive_rows",
    )?;
    if e.stage == Stage::Prefill128 {
        check(
            span(bytes, 1152, 80)? == span(bytes, 128, 80)?
                && read32(bytes, 1232)? == PREFILL_MAGIC
                && read32(bytes, 1236)? == 128,
            "prefill_view",
            "compatibility view differs",
        )?;
        for (i, &token) in e.rows[0].input_tokens.iter().enumerate() {
            check(
                read32(bytes, 1240 + i * 4)? == token,
                "prefill_tokens",
                "full prompt differs from submission",
            )?;
        }
    } else {
        zero(bytes, 1152, 600, "decode_prefill_region")?;
    }
    zero(bytes, 1752, 40, "packet_padding")?;
    Ok(DecodedRequest {
        stage: e.stage,
        bucket: bucket(e.rows.len())?,
        mode: e.mode,
        rows,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedRow {
    sequence_tag: u64,
    cookie: u64,
    execution_row: u32,
    output_slot: u32,
    position: u32,
    generated_index: u32,
    token_id: u32,
    logits: Option<Vec<u8>>,
}
impl ValidatedRow {
    pub fn sequence_tag(&self) -> u64 {
        self.sequence_tag
    }
    pub fn cookie(&self) -> u64 {
        self.cookie
    }
    pub fn execution_row(&self) -> u32 {
        self.execution_row
    }
    pub fn output_slot(&self) -> u32 {
        self.output_slot
    }
    pub fn position(&self) -> u32 {
        self.position
    }
    pub fn generated_index(&self) -> u32 {
        self.generated_index
    }
    pub fn token_id(&self) -> u32 {
        self.token_id
    }
    pub fn logits(&self) -> Option<&[u8]> {
        self.logits.as_deref()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedBulk {
    owner_generation: u64,
    replay_id: u64,
    iteration_id: u64,
    rows_by_slot: Vec<ValidatedRow>,
}
impl ValidatedBulk {
    pub fn owner_generation(&self) -> u64 {
        self.owner_generation
    }
    pub fn replay_id(&self) -> u64 {
        self.replay_id
    }
    pub fn iteration_id(&self) -> u64 {
        self.iteration_id
    }
    pub fn rows_by_slot(&self) -> &[ValidatedRow] {
        &self.rows_by_slot
    }
}
/// Supplied only after the real owner establishes stream/graph completion.
/// These values are assertions by the adapter, not hardware observations here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionEvidence {
    Quiesced,
    Unknown,
}

/// Atomic validation: no row getter or partial output escapes before every
/// active/inactive record and all transferred logits pass. Rows are returned in
/// dense scheduler-slot order, independent of execution-row order.
pub fn validate_bulk_result(
    bytes: &[u8],
    e: &SubmissionExpectation,
    completion: CompletionEvidence,
) -> Result<ValidatedBulk> {
    check(
        completion == CompletionEvidence::Quiesced,
        "completion",
        "GPU quiescence not established",
    )?;
    validate_expectations(e)?;
    check_header(bytes, e, true)?;
    let b = bucket(e.rows.len())? as usize;
    let mut rows = Vec::with_capacity(e.rows.len());
    for (index, expected) in e.rows.iter().enumerate() {
        let base = 128 + 128 * index;
        check(
            read64(bytes, base)? == expected.sequence_tag
                && read64(bytes, base + 8)? == expected.cookie
                && read64(bytes, base + 16)? == e.replay_id
                && read64(bytes, base + 24)? == e.iteration_id,
            "result_identity",
            "stale or misrouted result row",
        )?;
        check(
            read32(bytes, base + 32)? == index as u32
                && read32(bytes, base + 36)? == expected.output_slot
                && read32(bytes, base + 40)? == expected.target_length - 1
                && read32(bytes, base + 44)? == expected.generated_index,
            "result_routing",
            "execution row/slot/position/index differs",
        )?;
        let token = read32(bytes, base + 48)?;
        check(
            token < 49152 && read32(bytes, base + 52)? == 0 && read32(bytes, base + 88)? == 1,
            "result_status",
            "invalid token, nonfinite status or incomplete row",
        )?;
        check(
            read32(bytes, base + 56)? == 32,
            "embedding_report",
            "report size differs",
        )?;
        zero(bytes, base + 60, 28, "embedding_report")?;
        zero(bytes, base + 92, 4, "result_reserved")?;
        zero(bytes, base + 112, 16, "result_reserved")?;
        let logits = if e.mode == ResultMode::FullLogits {
            let offset = RESULT_PREFIX_BYTES + index * LOGITS_BYTES;
            check(
                read64(bytes, base + 96)? == offset as u64
                    && read64(bytes, base + 104)? == LOGITS_BYTES as u64,
                "logits_span",
                "arbitrary or mismatched logits offset/length",
            )?;
            let raw = span(bytes, offset, LOGITS_BYTES)?;
            check(
                raw.chunks_exact(2)
                    .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 != 0x7f80),
                "logits",
                "nonfinite BF16 output",
            )?;
            Some(raw.to_vec())
        } else {
            zero(bytes, base + 96, 16, "greedy_logits_span")?;
            None
        };
        rows.push(ValidatedRow {
            sequence_tag: expected.sequence_tag,
            cookie: expected.cookie,
            execution_row: index as u32,
            output_slot: expected.output_slot,
            position: expected.target_length - 1,
            generated_index: expected.generated_index,
            token_id: token,
            logits,
        });
    }
    zero(
        bytes,
        128 + e.rows.len() * 128,
        (8 - e.rows.len()) * 128,
        "inactive_result_rows",
    )?;
    if e.mode == ResultMode::FullLogits {
        zero(
            bytes,
            RESULT_PREFIX_BYTES + e.rows.len() * LOGITS_BYTES,
            (b - e.rows.len()) * LOGITS_BYTES,
            "inactive_logits",
        )?;
    }
    rows.sort_by_key(|row| row.output_slot);
    Ok(ValidatedBulk {
        owner_generation: e.owner.generation,
        replay_id: e.replay_id,
        iteration_id: e.iteration_id,
        rows_by_slot: rows,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettlementAdvice {
    RetainAll,
    ContainAffectedRequests,
    SchedulerCommitAllowed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Awaiting,
    Validated,
    FailedQuiesced,
    Unknown,
}

/// One admitted transaction with a retained immutable expectation snapshot.
/// No subsequent packet may replace this snapshot. This is a readiness model,
/// not a KV allocator, scheduler, CUDA completion detector or publisher.
pub struct Transaction {
    expectation: SubmissionExpectation,
    state: State,
    ready: Option<ValidatedBulk>,
}
impl Transaction {
    pub fn admit(packet: &[u8], expectation: SubmissionExpectation) -> Result<Self> {
        decode_request(packet, &expectation)?;
        Ok(Self {
            expectation,
            state: State::Awaiting,
            ready: None,
        })
    }
    pub fn accept_result(&mut self, bytes: &[u8], completion: CompletionEvidence) -> Result<()> {
        // Every attempted result read invalidates the previous getter, including
        // repeated/stale attempts after a completed validation.
        self.ready = None;
        if self.state != State::Awaiting {
            if self.state == State::Validated {
                self.state = State::FailedQuiesced;
            }
            return Err(Error {
                field: "transaction",
                reason: "result already attempted; no retry/publication",
            });
        }
        if completion == CompletionEvidence::Unknown {
            self.state = State::Unknown;
            return Err(Error {
                field: "completion",
                reason: "unknown completion retains all ownership",
            });
        }
        self.state = State::FailedQuiesced;
        let result = validate_bulk_result(bytes, &self.expectation, completion)?;
        self.ready = Some(result);
        self.state = State::Validated;
        Ok(())
    }
    pub fn result(&self) -> Result<&ValidatedBulk> {
        self.ready.as_ref().ok_or(Error {
            field: "output",
            reason: "whole-batch output is not ready",
        })
    }
    pub fn retained_expectation(&self) -> &SubmissionExpectation {
        &self.expectation
    }
    pub fn settlement_advice(&self) -> SettlementAdvice {
        match self.state {
            State::Awaiting | State::Unknown => SettlementAdvice::RetainAll,
            State::FailedQuiesced => SettlementAdvice::ContainAffectedRequests,
            State::Validated => SettlementAdvice::SchedulerCommitAllowed,
        }
    }
    /// Later external proof may permit containment, never a second result read
    /// or token publication after unknown completion.
    pub fn establish_quiescence_after_failure(&mut self, proof: CompletionEvidence) -> Result<()> {
        check(
            self.state == State::Unknown && proof == CompletionEvidence::Quiesced,
            "completion",
            "requires later explicit quiescence proof after unknown completion",
        )?;
        self.state = State::FailedQuiesced;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
