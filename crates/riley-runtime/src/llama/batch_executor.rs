//! Owning shape-bucketed CUDA executor for mixed Llama prefill/decode batches.
//!
//! One cold-prepared owner shares uploaded weights, maximum-size graph buffers,
//! and paged KV storage across exact-`M` execution-plan/GEMM variants. The
//! rollback policy keeps `M = max_input_tokens`; the active-row policy selects
//! the smallest prepared power-of-two bucket that contains the `T` flattened
//! input rows. Indexed `RoPE`, paged KV scatter, and ragged causal attention
//! preserve each active row's absolute sequence position.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::fmt;

use riley_cuda::{
    AttentionReductionProfile, CudaContext, CudaDeviceBuffer, CudaStream,
    FIXED37_RAGGED_MAX_LOGICAL_TOKENS,
};
use riley_model::LoadedModel;

use super::batch::{LlamaBatchRow, LlamaPackedBatchMetadata};
pub use super::executor::allocation::PreparedLlamaBatchAllocationReport;
use super::executor::allocation::build_batch_allocation_report;
use super::executor::buffers::{BatchDeviceInput, U16_BYTES, U32_BYTES};
pub use super::executor::config::{
    BatchMetadataTransport, ExecutionCompletionImplementation, PreparedLlamaBatchExecutorConfig,
    RaggedAttentionImplementation, ResidualNormImplementation,
};
use super::executor::config::{
    batch_metadata_transport_id, execution_completion_implementation_id, normalize_prepared_config,
    ragged_attention_implementation_id, residual_norm_implementation_id,
    runtime_selection_policy_id,
};
use super::executor::dispatch::{
    BatchDispatchDisposition, execute_packed as dispatch_execute_packed,
};
use super::executor::error::cuda_error as batch_cuda;
pub use super::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
};
use super::executor::gemm_plan::PreparedLlamaBatchShape;
use super::executor::metadata::{
    PackedIterationLayout, pack_iteration_input, validate_for_execution,
};
pub use super::executor::metrics::{LlamaBatchShapeBucketHit, LlamaBatchShapeObservation};
use super::executor::output::{decode_greedy_tokens, greedy_result_bytes};
use super::executor::owner::{
    BatchHostWorkspace, PreparedLlamaBatchOwner, SUPPORTED_HEAD_DIMENSION,
};
use super::executor::poison::poison_for_batch_error;
use super::executor::rope::absolute_rope_position_count;
use super::executor::shape::{
    LlamaBatchShapeHistory, batch_shape_policy_id, select_prepared_dense_rows,
};
pub use super::executor::shape::{LlamaBatchShapePolicy, MAX_LLAMA_BATCH_SHAPE_BUCKETS};
use super::forward::{PreparedLlamaForward, poison_for_cuda_error};
use super::{ExecutionSite, LlamaOp, LlamaReductionProfile};
use crate::paged_kv::{KV_BLOCK_SIZE, KvLayout};

#[cfg(test)]
use super::forward::PreparedLlamaForwardConfig;

const BF16_BYTES: u64 = 2;
const BF16_BYTES_USIZE: usize = 2;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchOutputMode {
    Logits,
    GreedyTokens,
}

fn shape_history_for_config(
    config: PreparedLlamaBatchExecutorConfig,
) -> LlamaBatchExecutorResult<LlamaBatchShapeHistory> {
    LlamaBatchShapeHistory::new(
        config.shape_policy(),
        config.configured_shape_buckets(),
        config.metadata().max_input_tokens(),
    )
}

/// Shape-bucketed, shared-KV Llama continuous-batch executor.
///
/// The scheduler retains ownership of logical reservations. A successful call
/// only establishes that every synchronous native operation completed; the
/// caller may commit the matching scheduler iteration after `execute` returns.
/// A failed native operation poisons this owner and the caller must abort the
/// iteration instead of publishing any partial KV writes.
pub struct PreparedLlamaBatchExecutor {
    config: PreparedLlamaBatchExecutorConfig,
    shape_history: LlamaBatchShapeHistory,
    owner: PreparedLlamaBatchOwner,
    allocation_report: PreparedLlamaBatchAllocationReport,
    output_count: usize,
    output_mode: BatchOutputMode,
    output_ready: bool,
}

impl fmt::Debug for PreparedLlamaBatchExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLlamaBatchExecutor")
            .field("config", &self.config)
            .field("shape_history", &self.shape_history)
            .field("shape_variant_count", &self.owner.shape_variants.len())
            .field("layout", &self.owner.layout)
            .field("allocation_report", &self.allocation_report)
            .field("output_count", &self.output_count)
            .field("output_mode", &self.output_mode)
            .field("output_ready", &self.output_ready)
            .field("poisoned", &self.owner.poisoned)
            .finish_non_exhaustive()
    }
}

impl PreparedLlamaBatchExecutor {
    /// Uploads weights and allocates every host/device byte used by repeated
    /// mixed-batch execution.
    ///
    /// The current ragged kernel is deliberately D64-only. Preparation rejects
    /// other head widths before uploading weights or allocating CUDA storage.
    /// `max_input_tokens` remains the maximum dense GEMM row count and must not
    /// exceed the model's maximum sequence length. Active-row mode prepares
    /// smaller exact-M plans against the same [`PreparedLlamaForward`] owner.
    ///
    /// # Errors
    ///
    /// Returns a model/configuration, host allocation, CUDA preparation, weight
    /// upload, or checked KV-layout error. No partially prepared owner is
    /// returned.
    #[allow(clippy::too_many_lines)]
    pub fn prepare(
        model: &LoadedModel,
        context: &CudaContext,
        stream: &mut CudaStream,
        config: PreparedLlamaBatchExecutorConfig,
    ) -> LlamaBatchExecutorResult<Self> {
        let config = normalize_prepared_config(config);
        config.validate_metadata_transport()?;
        let mut shape_history = shape_history_for_config(config)?;
        let bounds = config.metadata();
        let owner = PreparedLlamaBatchOwner::prepare(model, context, stream, config)?;
        shape_history.retain_prepared_variants(
            owner.forward.plan.sequence_length(),
            |dense_rows| {
                owner
                    .shape_variants
                    .iter()
                    .any(|shape| shape.dense_rows == dense_rows)
            },
        );
        let allocation_report = build_batch_allocation_report(
            owner.forward.allocation_report(),
            bounds,
            config.metadata_transport(),
            owner.layout.total_bytes(),
            owner.absolute_rope_cos.byte_len(),
            owner
                .gathered_logits
                .as_ref()
                .map_or(0, CudaDeviceBuffer::byte_len),
            owner
                .greedy_results
                .as_ref()
                .map_or(0, CudaDeviceBuffer::byte_len),
        )?;

        Ok(Self {
            config,
            shape_history,
            owner,
            allocation_report,
            output_count: 0,
            output_mode: BatchOutputMode::Logits,
            output_ready: false,
        })
    }

    #[must_use]
    pub const fn config(&self) -> PreparedLlamaBatchExecutorConfig {
        self.config
    }

    /// Stable C02 identifier for the completion boundary frozen during cold
    /// preparation. This reads the normalized prepared configuration, not a
    /// caller's requested CLI setting.
    #[must_use]
    pub const fn execution_completion_mode_id(&self) -> &'static str {
        execution_completion_implementation_id(self.config.execution_completion_implementation())
    }

    /// Stable C02 identifier for the cold-prepared metadata transport.
    #[must_use]
    pub const fn metadata_transport_id(&self) -> &'static str {
        batch_metadata_transport_id(self.config.metadata_transport())
    }

    /// Stable C02 identifier for the prepared dense-row shape policy.
    #[must_use]
    pub const fn batch_shape_policy_id(&self) -> &'static str {
        batch_shape_policy_id(self.config.shape_policy())
    }

    /// Exact maximum dense-row budget prepared for this executor.
    ///
    /// The server verifies this equals the scheduler iteration budget before
    /// publishing C02 facts.
    #[must_use]
    pub const fn batch_token_budget(&self) -> usize {
        self.config.metadata().max_input_tokens()
    }

    /// Stable ID of the prefill backend selected during cold preparation.
    ///
    /// The returned value is the prepared forward owner's actual selection
    /// trace, after normalization and capability fallback, not an attention
    /// preference requested by the caller.
    #[must_use]
    pub fn prefill_attention_implementation_id(&self) -> &'static str {
        self.owner.forward.attention_selection().implementation_id()
    }

    /// Stable ID of the ragged paged-attention implementation bound to this
    /// prepared continuous-batch executor.
    #[must_use]
    pub const fn decode_attention_implementation_id(&self) -> &'static str {
        ragged_attention_implementation_id(
            self.config.ragged_attention_reduction_profile(),
            self.config.ragged_attention_implementation(),
        )
    }

    /// Stable aggregate ID for the role-specific prepared GEMM reduction
    /// policy vector.
    ///
    /// The forward owner resolves the role vector during preparation. Its
    /// whole-profile ID is the compact C02 aggregate: `canonical-v1` can
    /// contain the reviewed heterogeneous role vector, while fixed37 resolves
    /// its own fail-closed aggregate. Individual requested CLI values are not
    /// exposed here.
    #[must_use]
    pub const fn gemm_reduction_policy_aggregate_id(&self) -> &'static str {
        self.owner.forward.reduction_profile().id()
    }

    /// Stable C02 value for the residual-plus-RMSNorm implementation.
    #[must_use]
    pub const fn residual_rmsnorm_implementation_id(&self) -> &'static str {
        residual_norm_implementation_id(self.config.residual_norm_implementation())
    }

    /// Runtime selection contract bound to the prepared reduction profile.
    #[must_use]
    pub const fn runtime_selection_policy_id(&self) -> &'static str {
        runtime_selection_policy_id(self.owner.forward.reduction_profile())
    }

    /// Number of exact dense-row plans owned by this executor, including the
    /// maximum rollback plan held by the shared forward owner.
    #[must_use]
    pub fn prepared_shape_count(&self) -> usize {
        self.owner.shape_variants.len() + 1
    }

    /// Selects the prepared dense row count for a prospective active batch.
    ///
    /// # Errors
    ///
    /// Returns when the active row count is empty or exceeds the cold bound.
    pub fn select_dense_rows(&self, active_rows: usize) -> LlamaBatchExecutorResult<usize> {
        self.config.select_dense_rows(active_rows)?;
        Ok(select_prepared_dense_rows(
            active_rows,
            self.owner.forward.plan.sequence_length(),
            self.owner
                .shape_variants
                .iter()
                .map(|shape| shape.dense_rows),
        ))
    }

    /// Returns shape facts from the most recent successful iteration.
    ///
    /// Failed or rejected iterations do not replace the last successful
    /// observation.
    #[must_use]
    pub const fn last_shape_observation(&self) -> Option<LlamaBatchShapeObservation> {
        self.shape_history.last_success()
    }

    /// Returns cumulative hit counters in ascending cold-prepared bucket order.
    ///
    /// Fixed-maximum mode exposes exactly one entry. The returned slice borrows
    /// inline executor storage and performs no allocation.
    #[must_use]
    pub const fn shape_bucket_hits(&self) -> &[LlamaBatchShapeBucketHit] {
        self.shape_history.entries()
    }

    /// Returns the forward/decode profile selected at cold preparation.
    ///
    /// Use [`Self::reduction_profile_is_coherent`] before treating this value
    /// as the complete graph profile in logs or evidence.
    #[must_use]
    pub const fn reduction_profile(&self) -> LlamaReductionProfile {
        self.config.reduction_profile()
    }

    /// Whether every reduction family matches [`Self::reduction_profile`].
    #[must_use]
    pub fn reduction_profile_is_coherent(&self) -> bool {
        self.config.reduction_profile_is_coherent()
    }

    /// Vocabulary width of every gathered output row.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.owner.forward.plan.dimensions().vocabulary_size()
    }

    /// Number of absolute positions represented by the cold-prepared `RoPE` tables.
    ///
    /// # Errors
    ///
    /// Returns when the owned table byte shape is inconsistent or cannot be
    /// represented as a host `usize`.
    pub fn maximum_position_count(&self) -> LlamaBatchExecutorResult<usize> {
        let positions = absolute_rope_position_count(
            self.owner.absolute_rope_cos.byte_len(),
            self.owner.forward.plan.dimensions().head_dimension(),
        )?;
        usize::try_from(positions).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::RopeCos,
        })
    }

    #[must_use]
    pub const fn kv_layout(&self) -> KvLayout {
        self.owner.layout
    }

    #[must_use]
    pub const fn allocation_report(&self) -> PreparedLlamaBatchAllocationReport {
        self.allocation_report
    }

    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    #[must_use]
    pub const fn output_ready(&self) -> bool {
        self.output_ready
    }

    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.owner.poisoned
            || self.owner.forward.poisoned
            || self
                .owner
                .shape_variants
                .iter()
                .any(|shape| shape.gemms.any_poisoned())
    }

    /// Validates, packs, uploads, and executes one mixed iteration.
    ///
    /// All host/model bounds are checked before the first device mutation.
    /// Packing, encoding, upload, execution, and output routing reuse cold
    /// storage and perform no host or device allocation.
    ///
    /// # Errors
    ///
    /// Returns for malformed or over-capacity metadata, invalid token/position
    /// IDs, a poisoned owner, or any CUDA operation failure. Native execution
    /// failures poison the owner because KV mutation may be partial.
    pub fn execute(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.execute_output(rows, BatchOutputMode::Logits, stream)
    }

    /// Executes one mixed iteration and reduces gathered BF16 logits to exact
    /// deterministic greedy token IDs on the device.
    ///
    /// This path is valid only when the caller has already proven that every
    /// output row uses unconstrained temperature-zero decoding with repetition
    /// penalty one. It preserves the same post-dispatch poison contract as
    /// [`Self::execute`].
    ///
    /// # Errors
    ///
    /// Returns for the same malformed metadata, capacity, poison, and CUDA
    /// failures as [`Self::execute`], plus unavailable greedy result storage.
    pub fn execute_greedy(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.execute_output(rows, BatchOutputMode::GreedyTokens, stream)
    }

    fn execute_output(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        output_mode_requested: BatchOutputMode,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.is_poisoned() {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        self.output_ready = false;
        self.output_count = 0;
        self.owner.forward.output_ready = false;
        let Self {
            config,
            shape_history,
            owner,
            allocation_report: _,
            output_count,
            output_mode,
            output_ready,
        } = self;
        let PreparedLlamaBatchOwner {
            metadata,
            forward,
            shape_variants,
            layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            poisoned,
        } = owner;
        let packed = metadata.pack(rows)?;
        let active_rows = packed.total_input_tokens();
        config.select_dense_rows(active_rows)?;
        let selected_dense_rows = select_prepared_dense_rows(
            active_rows,
            forward.plan.sequence_length(),
            shape_variants.iter().map(|shape| shape.dense_rows),
        );
        let shape_bucket_index = shape_history.bucket_index(selected_dense_rows)?;
        validate_for_execution(
            packed,
            forward.plan.dimensions().vocabulary_size(),
            absolute_rope_position_count(
                absolute_rope_cos.byte_len(),
                forward.plan.dimensions().head_dimension(),
            )?,
            config.metadata(),
            (config.ragged_attention_reduction_profile()
                == AttentionReductionProfile::FixedContiguous37BalancedV1)
                .then_some(FIXED37_RAGGED_MAX_LOGICAL_TOKENS),
        )?;

        let mut dispatch_disposition = BatchDispatchDisposition::PreDispatch;
        let result = execute_packed(
            packed,
            *config,
            selected_dense_rows,
            forward,
            shape_variants,
            *layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            output_mode_requested,
            &mut dispatch_disposition,
            stream,
        );
        match result {
            Ok(()) => {
                shape_history.record_success(shape_bucket_index, active_rows, selected_dense_rows);
                *output_count = packed.output_count();
                *output_mode = output_mode_requested;
                *output_ready = true;
                forward.output_ready = true;
                Ok(())
            }
            Err(error) => {
                // Host packing, pinned writes, descriptor preflight, and
                // command-batch begin failures do not trigger the iteration's
                // blanket mutation-unknown poison. Once command submission can
                // have started, semantic KV state may be partial and the owner
                // must never be reused. Established error-specific and nested
                // GEMM poison handling remains active in both cases below.
                if config.execution_completion_implementation()
                    == ExecutionCompletionImplementation::IterationBatch
                    && dispatch_disposition.mutation_may_have_occurred()
                {
                    *poisoned = true;
                    forward.poisoned = true;
                }
                let forward_gemms = &forward.gemms;
                poison_for_batch_error(poisoned, &mut forward.poisoned, &error, || {
                    forward_gemms.any_poisoned()
                });
                *poisoned |= shape_variants
                    .iter()
                    .any(|shape| shape.gemms.any_poisoned());
                Err(error)
            }
        }
    }

    /// Exact BF16 byte length of the most recently gathered `[O,V]` output.
    ///
    /// # Errors
    ///
    /// Returns when the output shape cannot be represented as a host byte
    /// length.
    pub fn output_byte_len(&self) -> LlamaBatchExecutorResult<usize> {
        self.output_byte_len_for(self.output_count)
    }

    /// Exact BF16 byte length needed for a prospective `[output_count,V]` download.
    ///
    /// This pre-dispatch query lets orchestration allocate its destination
    /// before any reserved KV range can be mutated.
    ///
    /// # Errors
    ///
    /// Returns when `output_count` exceeds the cold-prepared bound or the byte
    /// length cannot be represented as a host `usize`.
    pub fn output_byte_len_for(&self, output_count: usize) -> LlamaBatchExecutorResult<usize> {
        if output_count > self.config.metadata().max_output_slots() {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "output_count",
                reason: "prospective output count exceeds the cold-prepared bound",
            });
        }
        output_count
            .checked_mul(self.vocabulary_size())
            .and_then(|elements| elements.checked_mul(BF16_BYTES_USIZE))
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::GatheredLogits,
            })
    }

    /// Exact byte length of `{token_id,status}` records for a prospective
    /// greedy output count.
    ///
    /// # Errors
    ///
    /// Returns when `output_count` exceeds the cold-prepared output bound or
    /// when the record byte length overflows `usize`.
    pub fn greedy_result_byte_len_for(
        &self,
        output_count: usize,
    ) -> LlamaBatchExecutorResult<usize> {
        if output_count > self.config.metadata().max_output_slots() {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "output_count",
                reason: "prospective output count exceeds the cold-prepared bound",
            });
        }
        greedy_result_bytes(output_count)
    }

    /// Downloads only gathered sampled rows `[O,V]`, in dense output-slot order.
    ///
    /// # Errors
    ///
    /// Returns when execution has not produced output, the owner is poisoned,
    /// the destination length differs from [`Self::output_byte_len`], or the
    /// synchronous CUDA transfer fails.
    pub fn download_logits(
        &mut self,
        destination: &mut [u8],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.is_poisoned() {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        if !self.output_ready {
            return Err(LlamaBatchExecutorError::OutputNotReady);
        }
        if self.output_mode != BatchOutputMode::Logits {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_mode",
                reason: "the completed iteration produced greedy tokens, not downloadable logits",
            });
        }
        let expected = self.output_byte_len()?;
        if destination.len() != expected {
            return Err(LlamaBatchExecutorError::InvalidDownloadLength {
                expected_bytes: expected,
                actual_bytes: destination.len(),
            });
        }
        if destination.is_empty() {
            return Ok(());
        }
        let gathered = self.owner.gathered_logits.as_mut().ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "gathered_logits",
                reason: "non-empty output has no cold-prepared device buffer",
            },
        )?;
        match gathered.download_to_slice(0, destination, &mut self.owner.forward.io_staging, stream)
        {
            Ok(()) => Ok(()),
            Err(source) => {
                poison_for_cuda_error(&mut self.owner.poisoned, &source);
                Err(batch_cuda(
                    ExecutionSite::global(LlamaOp::OutputGather),
                    source,
                ))
            }
        }
    }

    /// Downloads and validates one exact greedy token ID per output slot.
    ///
    /// Device traffic is eight bytes per row: a token ID and a status word.
    /// The caller-owned destination is filled only after every record is
    /// validated, so a non-finite or unknown status cannot publish a partial
    /// sample vector.
    ///
    /// # Errors
    ///
    /// Returns when execution did not produce greedy records, the destination
    /// has the wrong length, a device status is invalid, or transfer fails.
    pub fn download_greedy_tokens(
        &mut self,
        destination: &mut [u32],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.is_poisoned() {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        if !self.output_ready {
            return Err(LlamaBatchExecutorError::OutputNotReady);
        }
        if self.output_mode != BatchOutputMode::GreedyTokens {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_mode",
                reason: "the completed iteration produced logits, not greedy tokens",
            });
        }
        if destination.len() != self.output_count {
            return Err(LlamaBatchExecutorError::InvalidDownloadLength {
                expected_bytes: self.output_count.saturating_mul(U32_BYTES),
                actual_bytes: destination.len().saturating_mul(U32_BYTES),
            });
        }
        if destination.is_empty() {
            return Ok(());
        }
        let vocabulary_size = self.vocabulary_size();
        let result_bytes = self.greedy_result_byte_len_for(self.output_count)?;
        let device = self.owner.greedy_results.as_mut().ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "greedy_results",
                reason: "non-empty output has no cold-prepared greedy result buffer",
            },
        )?;
        let host = self
            .owner
            .host
            .greedy_results
            .get_mut(..result_bytes)
            .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                field: "greedy_results_host",
                reason: "cold-prepared host result storage is too short",
            })?;
        if let Err(source) =
            device.download_to_slice(0, host, &mut self.owner.forward.io_staging, stream)
        {
            poison_for_cuda_error(&mut self.owner.poisoned, &source);
            return Err(batch_cuda(
                ExecutionSite::global(LlamaOp::OutputGather),
                source,
            ));
        }
        if let Err(error) = decode_greedy_tokens(host, vocabulary_size, destination) {
            if matches!(&error, LlamaBatchExecutorError::InvalidGreedyResult { .. }) {
                self.owner.poisoned = true;
            }
            return Err(error);
        }
        Ok(())
    }

    /// Explicitly closes all extra batch allocations, then the reused forward.
    /// Every resource is attempted even after the first cleanup failure.
    ///
    /// # Errors
    ///
    /// Returns the first CUDA cleanup failure after attempting every owned
    /// device resource and the underlying prepared forward.
    pub fn close(self) -> LlamaBatchExecutorResult<()> {
        self.owner.close()
    }
}

// HOT_BATCH_EXECUTE_BEGIN
#[allow(clippy::too_many_arguments)]
fn execute_packed(
    packed: LlamaPackedBatchMetadata<'_>,
    config: PreparedLlamaBatchExecutorConfig,
    dense_rows: usize,
    forward: &mut PreparedLlamaForward,
    shape_variants: &mut [PreparedLlamaBatchShape],
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    device: &mut BatchDeviceInput,
    gathered_logits: &mut Option<CudaDeviceBuffer>,
    greedy_results: &mut Option<CudaDeviceBuffer>,
    host: &mut BatchHostWorkspace,
    output_mode: BatchOutputMode,
    dispatch_disposition: &mut BatchDispatchDisposition,
    stream: &mut CudaStream,
) -> LlamaBatchExecutorResult<()> {
    dispatch_execute_packed(
        packed,
        config,
        dense_rows,
        forward,
        shape_variants,
        layout,
        key_cache,
        value_cache,
        rope_cos,
        rope_sin,
        device,
        gathered_logits,
        greedy_results,
        &mut host.input,
        output_mode == BatchOutputMode::GreedyTokens,
        dispatch_disposition,
        stream,
    )
}
// HOT_BATCH_EXECUTE_END

const _: () = assert!(KV_BLOCK_SIZE == 16);
const _: () = assert!(SUPPORTED_HEAD_DIMENSION == 64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRowKind,
        PreparedLlamaBatchMetadata,
    };
    use crate::llama::executor::metadata::ByteRegion;
    use crate::paged_kv::BLOCK_TABLE_V1_VERSION;

    #[test]
    fn metadata_transport_is_synchronous_by_default_and_explicitly_reversible() {
        let metadata = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());

        assert_eq!(
            defaults.metadata_transport(),
            BatchMetadataTransport::Synchronous
        );
        assert_eq!(
            defaults
                .with_packed_async_metadata()
                .with_synchronous_metadata()
                .metadata_transport(),
            BatchMetadataTransport::Synchronous
        );
        assert!(matches!(
            defaults
                .with_packed_async_metadata()
                .validate_metadata_transport(),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "metadata_transport",
                ..
            })
        ));
        let packed = defaults
            .with_iteration_batch_completion()
            .with_packed_async_metadata();
        packed
            .validate_metadata_transport()
            .expect("iteration completion owns the pinned-source lease");
        assert_eq!(
            normalize_prepared_config(packed).metadata_transport(),
            BatchMetadataTransport::PackedAsync
        );
    }

    #[test]
    fn ragged_attention_implementation_is_legacy_by_default_reversible_and_preserved() {
        let metadata = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());

        assert_eq!(
            defaults.ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
        );
        assert_eq!(
            defaults
                .with_grouped_ragged_attention_heads()
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::GroupedHeads
        );
        assert_eq!(
            defaults
                .with_grouped_ragged_attention_heads()
                .with_legacy_ragged_attention_heads()
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
        );
        assert_eq!(
            normalize_prepared_config(defaults.with_grouped_ragged_attention_heads())
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::GroupedHeads
        );
    }

    #[test]
    fn c02_runtime_fact_ids_follow_normalized_prepared_policy() {
        let metadata =
            LlamaBatchMetadataConfig::new(2, 64, 8, 2, 8).expect("valid metadata bounds");
        let normalized = normalize_prepared_config(
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_iteration_batch_completion()
                .with_packed_async_metadata()
                .with_active_row_buckets()
                .with_grouped_ragged_attention_heads(),
        );

        assert_eq!(
            execution_completion_implementation_id(
                normalized.execution_completion_implementation()
            ),
            "iteration-batch"
        );
        assert_eq!(
            batch_metadata_transport_id(normalized.metadata_transport()),
            "packed-async"
        );
        assert_eq!(
            batch_shape_policy_id(normalized.shape_policy()),
            "power-of-two"
        );
        assert_eq!(
            residual_norm_implementation_id(normalized.residual_norm_implementation()),
            "separate"
        );
        assert_eq!(
            ragged_attention_implementation_id(
                normalized.ragged_attention_reduction_profile(),
                normalized.ragged_attention_implementation(),
            ),
            "riley.cuda.ragged-paged-attention.grouped-heads-d64-v1"
        );
        assert_eq!(
            runtime_selection_policy_id(normalized.reduction_profile()),
            "exact-fallback-allowed"
        );

        let fixed = normalized.with_fixed37_reductions();
        assert_eq!(
            ragged_attention_implementation_id(
                fixed.ragged_attention_reduction_profile(),
                fixed.ragged_attention_implementation(),
            ),
            "riley.cuda.ragged-paged-attention.fixed37-two-pass-d64-s8192-v1"
        );
        assert_eq!(
            runtime_selection_policy_id(fixed.reduction_profile()),
            "fail-closed"
        );
        assert_eq!(
            fixed.reduction_profile().id(),
            "fixed-contiguous-37-balanced-v1"
        );
        assert_eq!(
            residual_norm_implementation_id(ResidualNormImplementation::Fused),
            "fused"
        );
    }

    #[test]
    fn packed_iteration_layout_is_checked_aligned_and_capacity_bounded() {
        let layout = PackedIterationLayout::checked(4, 3, 5, 5, 3, 3, 2)
            .expect("representable packed layout");

        assert_eq!(
            layout.token_ids,
            ByteRegion {
                offset: 0,
                byte_len: 16
            }
        );
        assert_eq!(
            layout.sequence_block_offsets,
            ByteRegion {
                offset: 16,
                byte_len: 12
            }
        );
        assert_eq!(
            layout.physical_block_ids,
            ByteRegion {
                offset: 28,
                byte_len: 20
            }
        );
        assert_eq!(
            layout.valid_tokens,
            ByteRegion {
                offset: 48,
                byte_len: 10
            }
        );
        assert_eq!(
            layout.row_sequence_slots,
            ByteRegion {
                offset: 60,
                byte_len: 12
            }
        );
        assert_eq!(
            layout.row_positions,
            ByteRegion {
                offset: 72,
                byte_len: 12
            }
        );
        assert_eq!(
            layout.output_token_indices,
            ByteRegion {
                offset: 84,
                byte_len: 8
            }
        );
        assert_eq!(layout.total_bytes, 92);
        for region in [
            layout.token_ids,
            layout.sequence_block_offsets,
            layout.physical_block_ids,
            layout.row_sequence_slots,
            layout.row_positions,
            layout.output_token_indices,
        ] {
            assert_eq!(region.offset % U32_BYTES, 0);
        }
        assert_eq!(layout.valid_tokens.offset % U16_BYTES, 0);
        layout
            .validate_capacity(layout.total_bytes)
            .expect("exact capacity is accepted");
        assert!(matches!(
            layout.validate_capacity(layout.total_bytes - 1),
            Err(LlamaBatchExecutorError::InvalidBatch {
                field: "packed_iteration_input",
                ..
            })
        ));

        let bounds = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let capacity = PackedIterationLayout::capacity(bounds).expect("checked cold capacity");
        assert!(capacity.total_bytes >= layout.total_bytes);
    }

    #[test]
    fn packed_iteration_host_bytes_match_all_seven_sources_and_zero_padding() {
        let prefill_tokens = [10, 11, 12];
        let prefill_ids = [2];
        let prefill_valid = [3];
        let decode_tokens = [20];
        let decode_ids = [4, 5];
        let decode_valid = [u16::try_from(KV_BLOCK_SIZE).expect("block size"), 1];
        let rows = [
            LlamaBatchRow::new(
                41,
                LlamaBatchRowKind::Prefill,
                &prefill_tokens,
                3,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &prefill_ids, &prefill_valid, 3),
                Some(1),
            ),
            LlamaBatchRow::new(
                42,
                LlamaBatchRowKind::Decode,
                &decode_tokens,
                17,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &decode_ids, &decode_valid, 17),
                Some(0),
            ),
        ];
        let bounds = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(bounds).expect("prepare metadata");
        let packed = prepared.pack(&rows).expect("pack mixed rows");
        let layout = PackedIterationLayout::for_batch(&packed, 8).expect("dynamic layout");
        let capacity = PackedIterationLayout::capacity(bounds)
            .expect("cold layout")
            .total_bytes;
        let mut bytes = vec![0xA5; capacity];

        pack_iteration_input(&packed, 8, layout, &mut bytes).expect("pack host input");

        assert_eq!(
            &bytes[layout.token_ids.offset..layout.token_ids.offset + 4 * U32_BYTES],
            &[10_u32, 11, 12, 20]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert!(
            bytes[layout.token_ids.offset + 4 * U32_BYTES..layout.token_ids.end().expect("end")]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(
            &bytes[layout.sequence_block_offsets.offset
                ..layout.sequence_block_offsets.end().expect("end")],
            &[0_u32, 1, 3]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.physical_block_ids.offset..layout.physical_block_ids.end().expect("end")],
            &[2_u32, 4, 5]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.valid_tokens.offset..layout.valid_tokens.end().expect("end")],
            &[3_u16, 16, 1]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.row_sequence_slots.offset..layout.row_sequence_slots.end().expect("end")],
            &[0_u32, 0, 0, 1]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.row_positions.offset..layout.row_positions.end().expect("end")],
            &[0_u32, 1, 2, 16]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.output_token_indices.offset
                ..layout.output_token_indices.end().expect("end")],
            &[3_u32, 2]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert!(
            bytes[layout.valid_tokens.end().expect("end")..layout.row_sequence_slots.offset]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert!(bytes[layout.total_bytes..].iter().all(|&byte| byte == 0xA5));
    }

    #[test]
    fn packed_iteration_input_preflight_preserves_destination_bytes() {
        let tokens = [7_u32];
        let physical_block_ids = [0_u32];
        let valid_tokens = [1_u16];
        let rows = [LlamaBatchRow::new(
            41,
            LlamaBatchRowKind::Prefill,
            &tokens,
            1,
            LlamaBatchBlockTable::new(
                BLOCK_TABLE_V1_VERSION,
                &physical_block_ids,
                &valid_tokens,
                1,
            ),
            None,
        )];
        let bounds = LlamaBatchMetadataConfig::new(1, 1, 1, 0, 1).expect("valid metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(bounds).expect("prepare metadata");
        let packed = prepared.pack(&rows).expect("pack one row");
        let layout = PackedIterationLayout::for_batch(&packed, 1).expect("dynamic layout");
        let mut bytes = [0xA5_u8; 64];

        assert!(matches!(
            pack_iteration_input(&packed, 1, layout, &mut bytes[..layout.total_bytes - 1],),
            Err(LlamaBatchExecutorError::InvalidBatch {
                field: "packed_iteration_input",
                reason: "dynamic packed input exceeds the cold-prepared slab",
            })
        ));
        assert!(bytes.iter().all(|&byte| byte == 0xA5));

        let too_small = PackedIterationLayout::for_batch(&packed, 0).expect("representable layout");
        assert!(matches!(
            pack_iteration_input(&packed, 0, too_small, &mut bytes[..too_small.total_bytes],),
            Err(LlamaBatchExecutorError::InvalidBatch {
                field: "dense_rows",
                reason: "active input rows exceed the selected packed token region",
            })
        ));
        assert!(bytes.iter().all(|&byte| byte == 0xA5));
    }

    #[test]
    fn fixed_maximum_shape_is_default_and_reversible() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());

        assert_eq!(defaults.shape_policy(), LlamaBatchShapePolicy::FixedMaximum);
        assert_eq!(defaults.select_dense_rows(1).expect("select fixed M"), 512);
        assert_eq!(
            defaults
                .with_active_row_buckets()
                .with_fixed_maximum_shape()
                .shape_policy(),
            LlamaBatchShapePolicy::FixedMaximum
        );
    }

    #[test]
    fn active_row_policy_selects_smallest_prepared_bucket() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_active_row_buckets();

        for (active, expected) in [
            (1, 1),
            (2, 2),
            (3, 4),
            (8, 8),
            (9, 16),
            (127, 128),
            (128, 128),
            (129, 256),
            (256, 256),
            (257, 512),
            (512, 512),
        ] {
            assert_eq!(
                config.select_dense_rows(active).expect("select bucket"),
                expected,
                "active rows {active}"
            );
        }
    }

    #[test]
    fn active_row_policy_uses_non_power_of_two_maximum_as_final_bucket() {
        assert_eq!(
            LlamaBatchShapePolicy::ActiveRowBuckets
                .select_dense_rows(65, 100)
                .expect("select configured maximum"),
            100
        );
        assert_eq!(
            LlamaBatchShapePolicy::ActiveRowBuckets
                .select_dense_rows(200, 300)
                .expect("select power-of-two bucket"),
            256
        );
        assert_eq!(
            LlamaBatchShapePolicy::ActiveRowBuckets
                .select_dense_rows(257, 300)
                .expect("select final bucket"),
            300
        );
    }

    #[test]
    fn custom_active_row_buckets_are_stored_and_select_the_smallest_shape() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let automatic =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_active_row_buckets();
        assert_eq!(
            automatic.configured_shape_buckets(),
            &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        );

        let custom = automatic
            .with_custom_active_row_buckets(&[1, 3, 7, 64, 512])
            .expect("valid custom buckets");
        assert_eq!(
            custom.shape_policy(),
            LlamaBatchShapePolicy::ActiveRowBuckets
        );
        assert_eq!(custom.configured_shape_buckets(), &[1, 3, 7, 64, 512]);
        for (active, expected) in [(1, 1), (2, 3), (3, 3), (4, 7), (65, 512), (512, 512)] {
            assert_eq!(
                custom.select_dense_rows(active).expect("select custom"),
                expected
            );
        }
        assert_eq!(
            custom
                .with_fixed_maximum_shape()
                .select_dense_rows(1)
                .expect("fixed rollback"),
            512
        );
    }

    #[test]
    fn custom_active_row_buckets_fail_closed_for_every_list_invariant() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        let excessive = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 512];
        for invalid in [
            &[][..],
            &[0, 512][..],
            &[2, 512][..],
            &[1, 2, 2, 512][..],
            &[1, 4, 2, 512][..],
            &[1, 2, 256][..],
            &excessive[..],
        ] {
            assert!(matches!(
                defaults.with_custom_active_row_buckets(invalid),
                Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "shape_buckets",
                    ..
                })
            ));
        }
    }

    fn record_shape_success(
        history: &mut LlamaBatchShapeHistory,
        config: PreparedLlamaBatchExecutorConfig,
        active_rows: usize,
    ) {
        let dense_rows = config
            .select_dense_rows(active_rows)
            .expect("valid active rows");
        let bucket_index = history
            .bucket_index(dense_rows)
            .expect("selected bucket is tracked");
        history.record_success(bucket_index, active_rows, dense_rows);
    }

    #[test]
    fn fixed_maximum_shape_history_tracks_padding_and_one_bucket() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        let mut history = shape_history_for_config(config).expect("valid fixed history");
        assert_eq!(history.last_success(), None);

        record_shape_success(&mut history, config, 128);
        record_shape_success(&mut history, config, 1);

        let observation = history
            .last_success()
            .expect("successful shape observation");
        assert_eq!(observation.active_rows(), 1);
        assert_eq!(observation.selected_dense_rows(), 512);
        assert_eq!(observation.padding_rows(), 511);
        let hits = history.entries();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].dense_rows(), 512);
        assert_eq!(hits[0].hit_count(), 2);
    }

    #[test]
    fn active_shape_history_tracks_shape_changes_and_maximum_hits() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_active_row_buckets();
        let mut history = shape_history_for_config(config).expect("valid active history");

        for active_rows in [128, 1, 8, 256, 1, 511] {
            record_shape_success(&mut history, config, active_rows);
        }

        let observation = history
            .last_success()
            .expect("successful shape observation");
        assert_eq!(observation.active_rows(), 511);
        assert_eq!(observation.selected_dense_rows(), 512);
        assert_eq!(observation.padding_rows(), 1);
        let hits = history.entries();
        assert_eq!(hits.len(), 10);
        assert_eq!(hits[0].dense_rows(), 1);
        assert_eq!(hits[0].hit_count(), 2);
        assert_eq!(hits[3].dense_rows(), 8);
        assert_eq!(hits[3].hit_count(), 1);
        assert_eq!(hits[7].dense_rows(), 128);
        assert_eq!(hits[7].hit_count(), 1);
        assert_eq!(hits[8].dense_rows(), 256);
        assert_eq!(hits[8].hit_count(), 1);
        assert_eq!(hits[9].dense_rows(), 512);
        assert_eq!(hits[9].hit_count(), 1);
    }

    #[test]
    fn shape_selection_rejects_empty_and_over_capacity_batches() {
        for active in [0, 513] {
            assert!(matches!(
                LlamaBatchShapePolicy::ActiveRowBuckets.select_dense_rows(active, 512),
                Err(LlamaBatchExecutorError::InvalidBatch {
                    field: "active_rows",
                    ..
                })
            ));
        }
    }

    #[test]
    fn prepare_normalization_preserves_active_row_policy() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config = PreparedLlamaBatchExecutorConfig::new(
            metadata,
            PreparedLlamaForwardConfig::default().with_reference_attention(),
        )
        .with_custom_active_row_buckets(&[1, 8, 64, 512])
        .expect("valid custom buckets");

        let normalized = normalize_prepared_config(config);
        assert_eq!(
            normalized.shape_policy(),
            LlamaBatchShapePolicy::ActiveRowBuckets
        );
        assert_eq!(normalized.configured_shape_buckets(), &[1, 8, 64, 512]);
    }

    #[test]
    fn whole_reduction_profile_updates_forward_and_ragged_attention_atomically() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid metadata bounds");
        let canonical =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        assert_eq!(
            canonical.reduction_profile(),
            LlamaReductionProfile::CanonicalV1
        );
        assert_eq!(
            canonical.ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );
        assert!(canonical.reduction_profile_is_coherent());

        let fixed = canonical.with_fixed37_reductions();
        assert_eq!(
            fixed.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            fixed.forward().reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            fixed.ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
        assert!(fixed.reduction_profile_is_coherent());

        let narrow_rollback = fixed.with_canonical_ragged_attention();
        assert_eq!(
            narrow_rollback.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            narrow_rollback.ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );
        assert!(!narrow_rollback.reduction_profile_is_coherent());

        let restored = narrow_rollback.with_canonical_reductions();
        assert_eq!(
            restored.reduction_profile(),
            LlamaReductionProfile::CanonicalV1
        );
        assert_eq!(
            restored.ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );
        assert!(restored.reduction_profile_is_coherent());

        let normalized = normalize_prepared_config(fixed);
        assert_eq!(
            normalized.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            normalized.ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
    }

    #[test]
    fn new_batch_config_inherits_forward_reduction_profile() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid metadata bounds");
        let config = PreparedLlamaBatchExecutorConfig::new(
            metadata,
            PreparedLlamaForwardConfig::default().with_fixed37_reductions(),
        );

        assert_eq!(
            config.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            config.ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
    }

    #[test]
    fn fixed37_profile_rejects_t8193_in_host_preflight() {
        const LOGICAL_TOKENS: usize = 8_193;
        let block_count = LOGICAL_TOKENS.div_ceil(KV_BLOCK_SIZE);
        let physical_block_ids: Vec<u32> =
            (0..u32::try_from(block_count).expect("block count")).collect();
        let mut valid_tokens = vec![u16::try_from(KV_BLOCK_SIZE).expect("block size"); block_count];
        *valid_tokens.last_mut().expect("last block") = 1;
        let token = [1_u32];
        let rows = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Decode,
            &token,
            u32::try_from(LOGICAL_TOKENS).expect("logical token count"),
            LlamaBatchBlockTable::new(
                BLOCK_TABLE_V1_VERSION,
                &physical_block_ids,
                &valid_tokens,
                u32::try_from(LOGICAL_TOKENS).expect("logical token count"),
            ),
            Some(0),
        )];
        let metadata = LlamaBatchMetadataConfig::new(1, 1, block_count, 1, block_count)
            .expect("valid metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(metadata).expect("prepare metadata");
        let packed = prepared.pack(&rows).expect("pack T=8193 decode row");
        let fixed37 =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_fixed37_ragged_attention();
        let fixed37_position_limit = (fixed37.ragged_attention_reduction_profile()
            == AttentionReductionProfile::FixedContiguous37BalancedV1)
            .then_some(FIXED37_RAGGED_MAX_LOGICAL_TOKENS);
        assert!(matches!(
            validate_for_execution(
                packed,
                2,
                16_384,
                fixed37.metadata(),
                fixed37_position_limit,
            ),
            Err(LlamaBatchExecutorError::PositionOutOfRange {
                position: 8_192,
                maximum: 8_192,
                ..
            })
        ));

        let packed = prepared.pack(&rows).expect("repack for bound precedence");
        assert!(matches!(
            validate_for_execution(packed, 2, 4_096, fixed37.metadata(), fixed37_position_limit,),
            Err(LlamaBatchExecutorError::PositionOutOfRange {
                position: 8_192,
                maximum: 8_192,
                ..
            })
        ));

        let packed = prepared.pack(&rows).expect("repack after preflight error");
        let canonical = fixed37.with_canonical_ragged_attention();
        let canonical_position_limit = (canonical.ragged_attention_reduction_profile()
            == AttentionReductionProfile::FixedContiguous37BalancedV1)
            .then_some(FIXED37_RAGGED_MAX_LOGICAL_TOKENS);
        validate_for_execution(
            packed,
            2,
            16_384,
            canonical.metadata(),
            canonical_position_limit,
        )
        .expect("canonical ragged attention retains its existing model bound");
    }
}
