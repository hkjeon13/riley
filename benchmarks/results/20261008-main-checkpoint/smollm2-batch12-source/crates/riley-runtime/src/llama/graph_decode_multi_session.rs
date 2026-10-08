//! Retained catalog identity, submission lifetime and whole-result publication.
use super::*;
use crate::llama::multi_descriptor::{
    self as wire, CatalogEntry, CodecOwner, CompletionEvidence, OwnerExpectation, ResultMode,
    Stage, SubmissionExpectation, ValidatedBulk,
};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
const MAX_BYTES: usize = 1152 + 8 * 98304;
/// A single prepared P128/M1/N2/N4 owner with one outstanding transaction.
/// Scheduler callers must supply expectations from their live reservation authority.
pub struct OwnedLlamaMultiDecodeExecutor {
    inner: OwnedLlamaDecodeExecutor,
    codec: CodecOwner,
    next_cookie: u64,
    issued: Vec<u64>,
    input: Vec<u8>,
    output: Vec<u8>,
    started: bool,
    poisoned: bool,
}
fn wire_error(_: wire::Error) -> LlamaBatchExecutorError {
    rejected("multi descriptor or completion validation failed")
}
fn put32(p: &mut [u8], at: usize, value: u32) {
    p[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
impl PreparedLlamaBatchExecutor {
    /// Prepares all supported graph shapes before any request executes.
    pub fn into_owned_multi_decode_graph(
        self,
        context: &riley_cuda::CudaContext,
    ) -> LlamaBatchExecutorResult<OwnedLlamaMultiDecodeExecutor> {
        let shared_rows = self.config.shared_rows_graph();
        self.into_owned_multi_decode_graph_profile(context, shared_rows)
    }
    /// Experimental arithmetic-changing QKV/gate-up/head profile; not model-quality qualified.
    pub fn into_owned_shared_multi_decode_graph(self, context: &riley_cuda::CudaContext) -> LlamaBatchExecutorResult<OwnedLlamaMultiDecodeExecutor> {
        self.into_owned_multi_decode_graph_profile(context, true)
    }
    fn into_owned_multi_decode_graph_profile(self, context: &riley_cuda::CudaContext, shared_rows: bool) -> LlamaBatchExecutorResult<OwnedLlamaMultiDecodeExecutor> {
        let physical = self.owner.layout.physical_block_count() as u32;
        let inner = self.into_owned_decode_graph_with_catalog_profile(context, true, shared_rows)?;
        let generation = NEXT_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| rejected("multi owner generation exhausted"))?;
        let mut hash = Sha256::new();
        hash.update(b"riley.multi-catalog.dual-output.v2\0");
        hash.update(wire::CONTRACT_SHA256.as_bytes());
        hash.update(&inner.multi_plan_identity);
        hash.update(include_bytes!("../../../../kernels/src/graph_multisequence_packet.inc"));
        hash.update(include_bytes!("../../../../kernels/src/graph_resources.cu"));
        hash.update(inner.signature.fingerprint().as_bytes());
        hash.update(physical.to_le_bytes());
        hash.update(include_bytes!("graph_decode_multi_parents.rs"));
        hash.update(include_bytes!("../../../../kernels/src/graph_numerics.cu"));
        hash.update(include_bytes!("../../../../kernels/src/graph_numerics_precise.cu"));
        hash.update(include_bytes!("../../../../kernels/src/prefill_shape_projection.cuh"));
        hash.update(include_bytes!("../../../../kernels/src/prefill_shape_rope_kv.cuh"));
        hash.update(include_bytes!("../../../../kernels/src/prefill_shape_attention.cuh"));

        hash.update(include_bytes!("../../../../kernels/src/gemm.cu"));
        hash.update(include_bytes!(
            "../../../../kernels/src/graph_multisequence_record.inc"
        ));
        hash.update(include_bytes!(
            "../../../../kernels/src/graph_multisequence_catalog.inc"
        ));
        hash.update(include_bytes!(
            "../../../../kernels/src/graph_multisequence_io.cu"
        ));
        hash.update(include_bytes!(
            "../../../../kernels/src/batch_primitives.cu"
        ));
        let max_active_rows = if physical >= 80 { 8 } else if physical >= 40 { 4 } else { 2 };
        let mut catalog = vec![
            CatalogEntry {
                stage: Stage::Prefill128,
                bucket: 1,
                mode: ResultMode::FullLogits,
            },
            CatalogEntry {
                stage: Stage::Decode,
                bucket: 1,
                mode: ResultMode::FullLogits,
            },
            CatalogEntry {
                stage: Stage::Decode,
                bucket: 2,
                mode: ResultMode::FullLogits,
            },
            CatalogEntry {
                stage: Stage::Decode,
                bucket: 4,
                mode: ResultMode::FullLogits,
            },
            CatalogEntry { stage: Stage::Decode, bucket: 8, mode: ResultMode::FullLogits },
        ];
        let greedy: Vec<_> = catalog.iter().map(|e| CatalogEntry { mode: ResultMode::Greedy, ..*e }).collect();
        catalog.extend(greedy);
        catalog.retain(|entry| entry.bucket <= max_active_rows);
        let codec = CodecOwner::new(OwnerExpectation {
            generation,
            last_accepted_replay: 0,
            catalog_digest: hash.finalize().into(),
            max_active_rows,
            physical_block_count: physical,
            catalog,
        });
        Ok(OwnedLlamaMultiDecodeExecutor {
            inner,
            codec,
            next_cookie: 1,
            issued: Vec::with_capacity(8),
            input: vec![0; MAX_BYTES],
            output: vec![0; MAX_BYTES],
            started: false,
            poisoned: false,
        })
    }
}
impl OwnedLlamaMultiDecodeExecutor {
    /// Allocates unique cookies for the next scoped scheduler submission.
    pub fn issue_submission(
        &mut self,
        rows: usize,
    ) -> LlamaBatchExecutorResult<(OwnerExpectation, u64, Vec<u64>)> {
        if self.poisoned
            || self.codec.retained_expectation().is_some()
            || !self.issued.is_empty()
            || !(1..=8).contains(&rows)
        {
            return Err(rejected("multi owner busy or invalid row count"));
        }
        self.started = false;
        let replay = self
            .codec
            .expectation()
            .last_accepted_replay
            .checked_add(1)
            .ok_or_else(|| rejected("multi replay identity exhausted"))?;
        let end = self
            .next_cookie
            .checked_add(rows as u64)
            .ok_or_else(|| rejected("multi cookie identity exhausted"))?;
        self.issued.extend(self.next_cookie..end);
        self.next_cookie = end;
        Ok((
            self.codec.expectation().clone(),
            replay,
            self.issued.clone(),
        ))
    }
    /// Clears an identity allocation only before an actual admitted submission.
    pub fn abandon_issued_submission(&mut self) -> LlamaBatchExecutorResult<()> {
        if self.codec.retained_expectation().is_some() {
            return Err(rejected("cannot abandon admitted multi submission"));
        }
        self.issued.clear();
        Ok(())
    }
    /// True means errors require conservative GPU-completion containment.
    pub fn submission_started(&self) -> bool {
        self.started
    }
    /// Executes one authorized expectation; no row is exposed until all validate.
    pub fn execute_submission(
        &mut self,
        e: SubmissionExpectation,
    ) -> LlamaBatchExecutorResult<&ValidatedBulk> {
        if self.codec.retained_expectation().is_some() {
            self.poisoned = true;
            return Err(rejected("previous multi submission is unsettled"));
        }
        self.started = false;
        self.inner.output_ready = false;
        if self.poisoned
            || self.issued.len() != e.rows.len()
            || e.rows.iter().zip(&self.issued).any(|(r, c)| r.cookie != *c)
        {
            return Err(rejected("multi submission identity differs"));
        }
        let packet = wire::encode_request(&e).map_err(wire_error)?;
        let decoded = wire::decode_request(&packet, &e).map_err(wire_error)?;
        self.codec.admit(&packet, e.clone()).map_err(wire_error)?;
        self.issued.clear();
        let bytes =
            wire::result_bytes(decoded.bucket, e.mode).map_err(wire_error)?;
        self.output[..bytes].fill(0);
        self.started = true;
        let executed: LlamaBatchExecutorResult<()> = if decoded.bucket == 1 {
            let row = &e.rows[0];
            let table = crate::llama::LlamaBatchBlockTable::new(
                crate::paged_kv::BLOCK_TABLE_V1_VERSION,
                &row.physical_ids,
                &row.valid_tokens,
                row.target_length,
            );
            let item = crate::llama::LlamaBatchRow::new(
                row.sequence_tag,
                if e.stage == Stage::Prefill128 {
                    crate::llama::LlamaBatchRowKind::Prefill
                } else {
                    crate::llama::LlamaBatchRowKind::Decode
                },
                &row.input_tokens,
                row.target_length,
                table,
                Some(0),
            );
            self.inner.execute(&[item]).map(|_| ()).map(|()| {
                self.output[..128].copy_from_slice(&packet[..128]);
                put32(&mut self.output, 0, 0x314f4d52);
                put32(&mut self.output, 8, bytes as u32);
                put32(&mut self.output, 108, 0);
                put32(&mut self.output, 112, e.mode as u32);
                self.output[128..144].copy_from_slice(&packet[208..224]);
                self.output[144..160].copy_from_slice(&packet[56..72]);
                put32(&mut self.output, 164, row.output_slot);
                put32(&mut self.output, 168, row.target_length - 1);
                put32(&mut self.output, 172, row.generated_index);
                self.output[176..216].copy_from_slice(&self.inner.output[98304..98344]);
                put32(&mut self.output, 216, 1);
                if e.mode == ResultMode::FullLogits {
                put32(&mut self.output, 224, 1152);
                put32(&mut self.output, 232, 98304);
                self.output[1152..bytes].copy_from_slice(&self.inner.output[..98304]);
                }
            })
        } else {
            let transfer = bytes.max(wire::REQUEST_BYTES);
            self.input[..transfer].fill(0);
            self.input[..1792].copy_from_slice(&packet);
            let mut index = if decoded.bucket == 2 { 1 } else if decoded.bucket == 4 { 2 } else { 3 };
            if e.mode == ResultMode::Greedy { index += 3; }
            self.inner
                .graph
                .replay_catalog(index, &self.input[..transfer])
                .and_then(|()| {
                    self.inner
                        .graph
                        .read_catalog(index, &mut self.output[..transfer])
                })
                .map_err(|e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e))
        };
        if let Err(error) = executed {
            self.poisoned = true;
            let _ = self.codec.accept_result(&[], CompletionEvidence::Unknown);
            return Err(error);
        }
        if let Err(error) = self
            .codec
            .accept_result(&self.output[..bytes], CompletionEvidence::Quiesced)
        {
            self.poisoned = true;
            return Err(wire_error(error));
        }
        self.codec.result().map_err(wire_error)
    }
    /// Call only after the scheduler has successfully committed this iteration.
    pub fn confirm_scheduler_commit(&mut self, iteration: u64) -> LlamaBatchExecutorResult<()> {
        if self.codec.result().map_err(wire_error)?.iteration_id() != iteration {
            return Err(rejected("multi commit identity differs"));
        }
        self.codec.settle_success(true).map_err(wire_error)?;
        Ok(())
    }
    /// Native graph close remains the authority for releasing GPU parent leases.
    pub fn close(self) -> LlamaBatchExecutorResult<()> {
        self.inner.close()
    }
}
