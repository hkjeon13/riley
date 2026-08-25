//! Owning, fixed-sequence CUDA execution path for the PR07 Llama reference graph.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::error;
use std::fmt;
use std::mem;

use rustinfer_cuda::{
    AttentionBackend, AttentionBackendAvailability, AttentionMask, AttentionPreference,
    AttentionSelectionTrace, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaError, CudaErrorStage, CudaExecutionStream, CudaGemmConfig,
    CudaPinnedHostBuffer, CudaPreparedGemm, CudaStream, EmbeddingError, EmbeddingParams,
    GatedMultiplyParams, GemmParams, PrefillAttentionParams, PrefillAttentionRequest,
    PreparedPrefillAttention, ResidualAddParams, RmsNormParams, RopeParams,
    RowBiasAddInPlaceParams, SiluParams, embedding, gated_multiply, residual_add, rms_norm, rope,
    row_bias_add_in_place, silu,
};
use rustinfer_model::LoadedModel;

use super::decode::PrefillKvCacheSink;
use super::{
    ExecutionSite, LlamaDimensions, LlamaExecutionPlan, LlamaOp, LlamaPlanError, PhysicalWeightId,
};
use crate::cuda_weights::{CudaUploadedWeights, CudaWeightUploadError};

const DEFAULT_UPLOAD_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_IO_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_GEMM_WORKSPACE_CAP_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_ATTENTION_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const BF16_BYTES: u64 = 2;
// Token IDs + 5 hidden + 3 key/value + 4 intermediate + 2 RoPE tables +
// logits + embedding-error scratch. Attention and GEMM workspaces are optional.
const NON_ATTENTION_GRAPH_ALLOCATION_COUNT: u64 = 17;
const TRACE_POINT_COUNT: usize = 18;

/// Result type for preparing, executing, downloading, and closing PR07 forward state.
pub type LlamaForwardResult<T> = Result<T, LlamaForwardError>;

/// Named owning resource used by cleanup and allocation diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaForwardResource {
    UploadedWeights,
    UploadStaging,
    IoStaging,
    HiddenGemm,
    KeyValueGemm,
    IntermediateGemm,
    DownGemm,
    LmHeadGemm,
    TokenIds,
    HiddenCurrent,
    HiddenNorm,
    HiddenProjection,
    HiddenRotary,
    HiddenContext,
    KeyRaw,
    ValueRaw,
    KeyRotary,
    GateRaw,
    UpRaw,
    GateActivated,
    GatedProduct,
    Attention,
    RopeCos,
    RopeSin,
    Logits,
    EmbeddingErrorScratch,
    GemmWorkspace,
    TraceCapture,
}

/// Stable diagnostic checkpoints emitted by a traced PR07 reference forward.
///
/// The order and names match the pinned Hugging Face PR07 trace artifact. A
/// production [`PreparedLlamaForward::execute`] does not capture or transfer
/// any of these tensors; callers must explicitly prepare a trace and use
/// [`PreparedLlamaForward::execute_traced`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LlamaTracePoint {
    Embedding,
    Layer0InputNorm,
    Layer0QueryProjection,
    Layer0KeyProjection,
    Layer0ValueProjection,
    Layer0AttentionProbabilities,
    Layer0AttentionContext,
    Layer0AfterAttentionResidual,
    Layer0PostAttentionNorm,
    Layer0GateProjection,
    Layer0UpProjection,
    Layer0Gated,
    Layer0DownProjection,
    Layer0Output,
    Layer14Output,
    FinalNormInput,
    FinalNormOutput,
    LastLogits,
}

impl LlamaTracePoint {
    /// Every checkpoint in canonical artifact order.
    pub const ALL: [Self; TRACE_POINT_COUNT] = [
        Self::Embedding,
        Self::Layer0InputNorm,
        Self::Layer0QueryProjection,
        Self::Layer0KeyProjection,
        Self::Layer0ValueProjection,
        Self::Layer0AttentionProbabilities,
        Self::Layer0AttentionContext,
        Self::Layer0AfterAttentionResidual,
        Self::Layer0PostAttentionNorm,
        Self::Layer0GateProjection,
        Self::Layer0UpProjection,
        Self::Layer0Gated,
        Self::Layer0DownProjection,
        Self::Layer0Output,
        Self::Layer14Output,
        Self::FinalNormInput,
        Self::FinalNormOutput,
        Self::LastLogits,
    ];

    /// Canonical tensor name in the pinned PR07 trace manifest.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Layer0InputNorm => "layer0.input_norm",
            Self::Layer0QueryProjection => "layer0.q_proj",
            Self::Layer0KeyProjection => "layer0.k_proj",
            Self::Layer0ValueProjection => "layer0.v_proj",
            Self::Layer0AttentionProbabilities => "layer0.attention_probs",
            Self::Layer0AttentionContext => "layer0.attention_context",
            Self::Layer0AfterAttentionResidual => "layer0.after_attention_residual",
            Self::Layer0PostAttentionNorm => "layer0.post_attention_norm",
            Self::Layer0GateProjection => "layer0.gate_proj",
            Self::Layer0UpProjection => "layer0.up_proj",
            Self::Layer0Gated => "layer0.gated",
            Self::Layer0DownProjection => "layer0.down_proj",
            Self::Layer0Output => "layer0.output",
            Self::Layer14Output => "layer14.output",
            Self::FinalNormInput => "final_norm.input",
            Self::FinalNormOutput => "final_norm.output",
            Self::LastLogits => "last_logits",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LlamaTraceContract {
    sequence_length: usize,
    layer_count: usize,
    dimensions: LlamaDimensions,
}

/// Caller-owned host storage for an explicit diagnostic forward trace.
///
/// Storage is allocated before execution and reused by every traced run. Each
/// successful checkpoint transfer sets one captured bit, so a failed run still
/// identifies the last confirmed tensor without publishing stale later data.
pub struct PreparedLlamaTrace {
    contract: LlamaTraceContract,
    tensors: Box<[Box<[u8]>]>,
    captured: u32,
}

impl fmt::Debug for PreparedLlamaTrace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLlamaTrace")
            .field("contract", &self.contract)
            .field("captured", &self.captured_count())
            .finish_non_exhaustive()
    }
}

impl PreparedLlamaTrace {
    fn prepare(plan: &LlamaExecutionPlan) -> LlamaForwardResult<Self> {
        if plan.layers().len() <= 14 {
            return Err(LlamaForwardError::TraceLayerUnavailable {
                required_layers: 15,
                actual_layers: plan.layers().len(),
            });
        }
        let contract = trace_contract(plan);
        let requested_bytes = trace_total_bytes(plan)?;
        let mut tensors = Vec::new();
        tensors.try_reserve_exact(TRACE_POINT_COUNT).map_err(|_| {
            LlamaForwardError::HostAllocation {
                resource: LlamaForwardResource::TraceCapture,
                requested_bytes,
            }
        })?;
        for point in LlamaTracePoint::ALL {
            tensors.push(allocate_host_bytes(
                trace_byte_len(plan, point)?,
                LlamaForwardResource::TraceCapture,
            )?);
        }
        Ok(Self {
            contract,
            tensors: tensors.into_boxed_slice(),
            captured: 0,
        })
    }

    /// Number of canonical checkpoints in this trace contract.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Number of checkpoints confirmed during the latest traced execution.
    #[must_use]
    pub const fn captured_count(&self) -> u32 {
        self.captured.count_ones()
    }

    /// Returns a captured BF16 tensor, or `None` before that checkpoint ran.
    #[must_use]
    pub fn tensor(&self, point: LlamaTracePoint) -> Option<&[u8]> {
        if self.captured & trace_bit(point) == 0 {
            return None;
        }
        self.tensors.get(point.index()).map(AsRef::as_ref)
    }

    /// Exact byte length reserved for one BF16 checkpoint tensor.
    #[must_use]
    pub fn tensor_byte_len(&self, point: LlamaTracePoint) -> usize {
        self.tensors[point.index()].len()
    }

    fn validate(&self, plan: &LlamaExecutionPlan) -> LlamaForwardResult<()> {
        if self.contract != trace_contract(plan) {
            return Err(LlamaForwardError::TracePlanMismatch);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.captured = 0;
    }

    fn destination(&mut self, point: LlamaTracePoint) -> &mut [u8] {
        self.tensors[point.index()].as_mut()
    }

    fn mark_captured(&mut self, point: LlamaTracePoint) {
        self.captured |= trace_bit(point);
    }
}

const fn trace_bit(point: LlamaTracePoint) -> u32 {
    1_u32 << point.index()
}

impl PreparedLlamaForward {
    /// Explicitly closes every prepared plan, graph allocation, pinned buffer,
    /// and uploaded weight, attempting the full set after the first failure.
    ///
    /// # Errors
    ///
    /// Returns the first close failure after all resources have been attempted.
    #[allow(clippy::too_many_lines)]
    pub fn close(self) -> LlamaForwardResult<()> {
        let Self {
            plan: _,
            weights,
            gemms,
            attention: _,
            buffers,
            io_staging,
            token_bytes: _,
            allocation_report: _,
            tokens_ready: _,
            output_ready: _,
            poisoned: _,
        } = self;
        let GemmPlans {
            hidden,
            key_value,
            intermediate,
            down,
            lm_head,
        } = gemms;
        let ForwardBuffers {
            token_ids,
            hidden_current,
            hidden_norm,
            hidden_projection,
            hidden_rotary,
            hidden_context,
            key_raw,
            value_raw,
            key_rotary,
            gate_raw,
            up_raw,
            gate_activated,
            gated_product,
            attention_workspace,
            rope_cos,
            rope_sin,
            logits,
            embedding_error_scratch,
            gemm_workspace,
        } = buffers;
        let mut first = None;
        record_close(&mut first, LlamaForwardResource::HiddenGemm, hidden.close());
        record_close(
            &mut first,
            LlamaForwardResource::KeyValueGemm,
            key_value.close(),
        );
        record_close(
            &mut first,
            LlamaForwardResource::IntermediateGemm,
            intermediate.close(),
        );
        record_close(&mut first, LlamaForwardResource::DownGemm, down.close());
        record_close(
            &mut first,
            LlamaForwardResource::LmHeadGemm,
            lm_head.close(),
        );
        for (resource, result) in [
            (LlamaForwardResource::TokenIds, token_ids.close()),
            (LlamaForwardResource::HiddenCurrent, hidden_current.close()),
            (LlamaForwardResource::HiddenNorm, hidden_norm.close()),
            (
                LlamaForwardResource::HiddenProjection,
                hidden_projection.close(),
            ),
            (LlamaForwardResource::HiddenRotary, hidden_rotary.close()),
            (LlamaForwardResource::HiddenContext, hidden_context.close()),
            (LlamaForwardResource::KeyRaw, key_raw.close()),
            (LlamaForwardResource::ValueRaw, value_raw.close()),
            (LlamaForwardResource::KeyRotary, key_rotary.close()),
            (LlamaForwardResource::GateRaw, gate_raw.close()),
            (LlamaForwardResource::UpRaw, up_raw.close()),
            (LlamaForwardResource::GateActivated, gate_activated.close()),
            (LlamaForwardResource::GatedProduct, gated_product.close()),
            (LlamaForwardResource::RopeCos, rope_cos.close()),
            (LlamaForwardResource::RopeSin, rope_sin.close()),
            (LlamaForwardResource::Logits, logits.close()),
            (
                LlamaForwardResource::EmbeddingErrorScratch,
                embedding_error_scratch.close(),
            ),
        ] {
            record_close(&mut first, resource, result);
        }
        if let Some(workspace) = attention_workspace {
            record_close(
                &mut first,
                LlamaForwardResource::Attention,
                workspace.close(),
            );
        }
        if let Some(workspace) = gemm_workspace {
            record_close(
                &mut first,
                LlamaForwardResource::GemmWorkspace,
                workspace.close(),
            );
        }
        record_close(
            &mut first,
            LlamaForwardResource::IoStaging,
            io_staging.close(),
        );
        record_close(
            &mut first,
            LlamaForwardResource::UploadedWeights,
            weights.close(),
        );
        first.map_or(Ok(()), Err)
    }
}

fn record_close(
    first: &mut Option<LlamaForwardError>,
    resource: LlamaForwardResource,
    result: Result<(), CudaError>,
) {
    if let Err(source) = result {
        if first.is_none() {
            *first = Some(LlamaForwardError::Cleanup { resource, source });
        }
    }
}

pub(super) fn execute_gemm<S: CudaExecutionStream + ?Sized>(
    plan: &mut CudaPreparedGemm,
    input: &CudaDeviceBuffer,
    weight: CudaBufferSpan<'_>,
    output: &mut CudaDeviceBuffer,
    workspace: &mut Option<CudaDeviceBuffer>,
    stream: &mut S,
    site: ExecutionSite,
) -> LlamaForwardResult<()> {
    let config = plan.config();
    let required_workspace = plan.algorithm_metadata().workspace_bytes();
    let workspace = if required_workspace == 0 {
        None
    } else {
        let buffer = workspace
            .as_mut()
            .ok_or(LlamaForwardError::InvalidConfiguration {
                field: "gemm_workspace",
                reason: "selected algorithm workspace was not allocated",
            })?;
        Some(span_mut(buffer, CudaDType::U8, required_workspace, site)?)
    };
    let mut params = GemmParams {
        input: span(input, CudaDType::BF16, config.input_bytes(), site)?,
        weight,
        output: span_mut(output, CudaDType::BF16, config.output_bytes(), site)?,
        workspace,
    };
    plan.execute(&mut params, stream)
        .map_err(|source| LlamaForwardError::cuda(site, source))
}

pub(super) fn execute_projection_bias<S: CudaExecutionStream + ?Sized>(
    weights: &CudaUploadedWeights,
    bias_id: Option<PhysicalWeightId>,
    output: &mut CudaDeviceBuffer,
    row_count: u64,
    column_count: u64,
    stream: &mut S,
    site: ExecutionSite,
) -> LlamaForwardResult<()> {
    let Some(bias_id) = bias_id else {
        return Ok(());
    };
    let bias = weight_span(weights, bias_id, site)?;
    let output_bytes = output.byte_len();
    let mut params = RowBiasAddInPlaceParams {
        matrix: span_mut(output, CudaDType::BF16, output_bytes, site)?,
        bias,
        row_count,
        column_count,
    };
    row_bias_add_in_place(&mut params, stream)
        .map_err(|source| LlamaForwardError::cuda(site, source))
}

pub(super) fn weight_span(
    weights: &CudaUploadedWeights,
    id: PhysicalWeightId,
    site: ExecutionSite,
) -> LlamaForwardResult<CudaBufferSpan<'_>> {
    weights
        .view_physical(id)
        .map(|weight| weight.span())
        .map_err(|source| LlamaForwardError::weight(site, source))
}

pub(super) fn span(
    buffer: &CudaDeviceBuffer,
    dtype: CudaDType,
    byte_len: u64,
    site: ExecutionSite,
) -> LlamaForwardResult<CudaBufferSpan<'_>> {
    CudaBufferSpan::new(buffer, dtype, 0, byte_len)
        .map_err(|source| LlamaForwardError::cuda(site, source))
}

pub(super) fn span_mut(
    buffer: &mut CudaDeviceBuffer,
    dtype: CudaDType,
    byte_len: u64,
    site: ExecutionSite,
) -> LlamaForwardResult<CudaBufferSpanMut<'_>> {
    CudaBufferSpanMut::new(buffer, dtype, 0, byte_len)
        .map_err(|source| LlamaForwardError::cuda(site, source))
}

pub(super) fn poison_for_cuda_error(poisoned: &mut bool, source: &CudaError) {
    if !matches!(source.stage(), CudaErrorStage::Validation) {
        *poisoned = true;
    }
}

pub(super) fn poison_for_forward_error(poisoned: &mut bool, error: &LlamaForwardError) {
    match error {
        LlamaForwardError::Cuda { source, .. } => poison_for_cuda_error(poisoned, source),
        LlamaForwardError::Embedding { source, .. } => {
            poison_for_cuda_error(poisoned, source.cuda_error());
        }
        LlamaForwardError::Weight { site: Some(_), .. }
        | LlamaForwardError::InvalidConfiguration { .. }
        | LlamaForwardError::ArithmeticOverflow { .. } => *poisoned = true,
        _ => {}
    }
}

fn trace_contract(plan: &LlamaExecutionPlan) -> LlamaTraceContract {
    LlamaTraceContract {
        sequence_length: plan.sequence_length(),
        layer_count: plan.layers().len(),
        dimensions: plan.dimensions(),
    }
}

fn trace_byte_len(plan: &LlamaExecutionPlan, point: LlamaTracePoint) -> LlamaForwardResult<u64> {
    let workspace = plan.workspace_spec();
    let bytes = match point {
        LlamaTracePoint::Layer0KeyProjection | LlamaTracePoint::Layer0ValueProjection => {
            workspace.key_value_buffer_bytes()
        }
        LlamaTracePoint::Layer0AttentionProbabilities => workspace.attention_buffer_bytes(),
        LlamaTracePoint::Layer0GateProjection
        | LlamaTracePoint::Layer0UpProjection
        | LlamaTracePoint::Layer0Gated => workspace.intermediate_buffer_bytes(),
        LlamaTracePoint::LastLogits => {
            let sequence = u64::try_from(plan.sequence_length()).map_err(|_| {
                LlamaForwardError::ArithmeticOverflow {
                    resource: LlamaForwardResource::Logits,
                }
            })?;
            workspace.logits_bytes().checked_div(sequence).ok_or(
                LlamaForwardError::ArithmeticOverflow {
                    resource: LlamaForwardResource::Logits,
                },
            )?
        }
        LlamaTracePoint::Embedding
        | LlamaTracePoint::Layer0InputNorm
        | LlamaTracePoint::Layer0QueryProjection
        | LlamaTracePoint::Layer0AttentionContext
        | LlamaTracePoint::Layer0AfterAttentionResidual
        | LlamaTracePoint::Layer0PostAttentionNorm
        | LlamaTracePoint::Layer0DownProjection
        | LlamaTracePoint::Layer0Output
        | LlamaTracePoint::Layer14Output
        | LlamaTracePoint::FinalNormInput
        | LlamaTracePoint::FinalNormOutput => workspace.hidden_buffer_bytes(),
    };
    Ok(bytes)
}

fn trace_total_bytes(plan: &LlamaExecutionPlan) -> LlamaForwardResult<u64> {
    LlamaTracePoint::ALL
        .into_iter()
        .try_fold(0_u64, |total, point| {
            total.checked_add(trace_byte_len(plan, point)?).ok_or(
                LlamaForwardError::ArithmeticOverflow {
                    resource: LlamaForwardResource::TraceCapture,
                },
            )
        })
}

fn capture_trace(
    trace: &mut Option<&mut PreparedLlamaTrace>,
    point: LlamaTracePoint,
    buffer: &mut CudaDeviceBuffer,
    source_offset: u64,
    io_staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
    site: ExecutionSite,
) -> LlamaForwardResult<()> {
    let Some(trace) = trace.as_deref_mut() else {
        return Ok(());
    };
    {
        let destination = trace.destination(point);
        buffer
            .download_to_slice(source_offset, destination, io_staging, stream)
            .map_err(|source| LlamaForwardError::cuda(site, source))?;
    }
    trace.mark_captured(point);
    Ok(())
}

impl LlamaForwardResource {
    const fn name(self) -> &'static str {
        match self {
            Self::UploadedWeights => "uploaded_weights",
            Self::UploadStaging => "upload_staging",
            Self::IoStaging => "io_staging",
            Self::HiddenGemm => "hidden_gemm",
            Self::KeyValueGemm => "key_value_gemm",
            Self::IntermediateGemm => "intermediate_gemm",
            Self::DownGemm => "down_gemm",
            Self::LmHeadGemm => "lm_head_gemm",
            Self::TokenIds => "token_ids",
            Self::HiddenCurrent => "hidden_current",
            Self::HiddenNorm => "hidden_norm",
            Self::HiddenProjection => "hidden_projection",
            Self::HiddenRotary => "hidden_rotary",
            Self::HiddenContext => "hidden_context",
            Self::KeyRaw => "key_raw",
            Self::ValueRaw => "value_raw",
            Self::KeyRotary => "key_rotary",
            Self::GateRaw => "gate_raw",
            Self::UpRaw => "up_raw",
            Self::GateActivated => "gate_activated",
            Self::GatedProduct => "gated_product",
            Self::Attention => "attention",
            Self::RopeCos => "rope_cos",
            Self::RopeSin => "rope_sin",
            Self::Logits => "logits",
            Self::EmbeddingErrorScratch => "embedding_error_scratch",
            Self::GemmWorkspace => "gemm_workspace",
            Self::TraceCapture => "trace_capture",
        }
    }
}

impl fmt::Display for LlamaForwardResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Structured PR07 preparation, execution, transfer, or cleanup failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum LlamaForwardError {
    Plan(LlamaPlanError),
    Weight {
        site: Option<ExecutionSite>,
        source: CudaWeightUploadError,
    },
    Cuda {
        site: ExecutionSite,
        source: CudaError,
    },
    Embedding {
        site: ExecutionSite,
        source: EmbeddingError,
    },
    InvalidConfiguration {
        field: &'static str,
        reason: &'static str,
    },
    AttentionBudgetExceeded {
        required_bytes: u64,
        maximum_bytes: u64,
    },
    HostAllocation {
        resource: LlamaForwardResource,
        requested_bytes: u64,
    },
    InvalidTokenCount {
        expected: usize,
        actual: usize,
    },
    TokenOutOfRange {
        position: usize,
        token_id: u32,
        vocabulary_size: usize,
    },
    TokensNotUploaded,
    OutputNotReady,
    Poisoned,
    InvalidDownloadLength {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    TracePlanMismatch,
    TraceRequiresReferenceAttention,
    TraceLayerUnavailable {
        required_layers: usize,
        actual_layers: usize,
    },
    Cleanup {
        resource: LlamaForwardResource,
        source: CudaError,
    },
    ArithmeticOverflow {
        resource: LlamaForwardResource,
    },
}

impl LlamaForwardError {
    pub(super) fn cuda(site: ExecutionSite, source: CudaError) -> Self {
        Self::Cuda { site, source }
    }

    fn weight(site: ExecutionSite, source: CudaWeightUploadError) -> Self {
        Self::Weight {
            site: Some(site),
            source,
        }
    }
}

impl fmt::Display for LlamaForwardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(source) => source.fmt(formatter),
            Self::Weight {
                site: Some(site),
                source,
            } => write!(formatter, "{site}: {source}"),
            Self::Weight { site: None, source } => source.fmt(formatter),
            Self::Cuda { site, source } => write!(formatter, "{site}: {source}"),
            Self::Embedding { site, source } => write!(formatter, "{site}: {source}"),
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid Llama forward configuration {field}: {reason}"
                )
            }
            Self::AttentionBudgetExceeded {
                required_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "materialized attention requires {required_bytes} bytes, exceeding budget {maximum_bytes}"
            ),
            Self::HostAllocation {
                resource,
                requested_bytes,
            } => write!(
                formatter,
                "could not reserve {requested_bytes} host bytes for {resource}"
            ),
            Self::InvalidTokenCount { expected, actual } => {
                write!(
                    formatter,
                    "forward expects {expected} token IDs, received {actual}"
                )
            }
            Self::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "token ID {token_id} at position {position} is outside vocabulary 0..{vocabulary_size}"
            ),
            Self::TokensNotUploaded => {
                formatter.write_str("token IDs must be uploaded before executing the forward")
            }
            Self::OutputNotReady => {
                formatter.write_str("logits are unavailable before a successful forward")
            }
            Self::Poisoned => formatter
                .write_str("the Llama forward was poisoned by a native CUDA execution failure"),
            Self::InvalidDownloadLength {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "download destination has {actual_bytes} bytes, expected {expected_bytes}"
            ),
            Self::TracePlanMismatch => formatter
                .write_str("prepared trace dimensions do not match this Llama execution plan"),
            Self::TraceRequiresReferenceAttention => formatter.write_str(
                "the PR07 probability trace requires the materialized reference attention backend",
            ),
            Self::TraceLayerUnavailable {
                required_layers,
                actual_layers,
            } => write!(
                formatter,
                "PR07 trace requires at least {required_layers} decoder layers, found {actual_layers}"
            ),
            Self::Cleanup { resource, source } => {
                write!(formatter, "could not close {resource}: {source}")
            }
            Self::ArithmeticOverflow { resource } => {
                write!(formatter, "byte arithmetic overflow for {resource}")
            }
        }
    }
}

impl error::Error for LlamaForwardError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Plan(source) => Some(source),
            Self::Weight { source, .. } => Some(source),
            Self::Cuda { source, .. } | Self::Cleanup { source, .. } => Some(source),
            Self::Embedding { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<LlamaPlanError> for LlamaForwardError {
    fn from(source: LlamaPlanError) -> Self {
        Self::Plan(source)
    }
}

/// Cold-path limits for one fixed-sequence forward owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct PreparedLlamaForwardConfig {
    upload_staging_bytes: u64,
    io_staging_bytes: u64,
    gemm_workspace_cap_bytes: u64,
    attention_budget_bytes: u64,
    attention_preference: AttentionPreference,
}

impl PreparedLlamaForwardConfig {
    #[must_use]
    pub const fn new(
        upload_staging_bytes: u64,
        io_staging_bytes: u64,
        gemm_workspace_cap_bytes: u64,
        attention_budget_bytes: u64,
    ) -> Self {
        Self {
            upload_staging_bytes,
            io_staging_bytes,
            gemm_workspace_cap_bytes,
            attention_budget_bytes,
            attention_preference: AttentionPreference::Optimized,
        }
    }

    /// Selects the allocation-free online prefill backend when supported,
    /// falling back to the native materialized reference during preparation.
    #[must_use]
    pub const fn with_optimized_attention(mut self) -> Self {
        self.attention_preference = AttentionPreference::Optimized;
        self
    }

    /// Requires the native materialized reference backend. This mode is used
    /// by the pinned PR07 probability trace and its exact staged-BF16 golden.
    #[must_use]
    pub const fn with_reference_attention(mut self) -> Self {
        self.attention_preference = AttentionPreference::Reference;
        self
    }

    #[must_use]
    pub const fn upload_staging_bytes(self) -> u64 {
        self.upload_staging_bytes
    }

    #[must_use]
    pub const fn io_staging_bytes(self) -> u64 {
        self.io_staging_bytes
    }

    #[must_use]
    pub const fn gemm_workspace_cap_bytes(self) -> u64 {
        self.gemm_workspace_cap_bytes
    }

    #[must_use]
    pub const fn attention_budget_bytes(self) -> u64 {
        self.attention_budget_bytes
    }

    #[must_use]
    pub const fn attention_preference(self) -> AttentionPreference {
        self.attention_preference
    }

    fn validate(self) -> LlamaForwardResult<()> {
        if self.upload_staging_bytes == 0 {
            return Err(LlamaForwardError::InvalidConfiguration {
                field: "upload_staging_bytes",
                reason: "must be non-zero",
            });
        }
        if self.io_staging_bytes == 0 {
            return Err(LlamaForwardError::InvalidConfiguration {
                field: "io_staging_bytes",
                reason: "must be non-zero",
            });
        }
        Ok(())
    }
}

impl Default for PreparedLlamaForwardConfig {
    fn default() -> Self {
        Self::new(
            DEFAULT_UPLOAD_STAGING_BYTES,
            DEFAULT_IO_STAGING_BYTES,
            DEFAULT_GEMM_WORKSPACE_CAP_BYTES,
            DEFAULT_ATTENTION_BUDGET_BYTES,
        )
    }
}

/// Exact owned CUDA allocation totals after cold preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaAllocationReport {
    weight_bytes: u64,
    graph_bytes: u64,
    gemm_workspace_bytes: u64,
    total_device_bytes: u64,
    device_allocation_count: u64,
    pinned_host_bytes: u64,
    pinned_host_allocation_count: u64,
}

impl PreparedLlamaAllocationReport {
    #[must_use]
    pub const fn weight_bytes(self) -> u64 {
        self.weight_bytes
    }
    #[must_use]
    pub const fn graph_bytes(self) -> u64 {
        self.graph_bytes
    }
    #[must_use]
    pub const fn gemm_workspace_bytes(self) -> u64 {
        self.gemm_workspace_bytes
    }
    #[must_use]
    pub const fn total_device_bytes(self) -> u64 {
        self.total_device_bytes
    }
    #[must_use]
    pub const fn device_allocation_count(self) -> u64 {
        self.device_allocation_count
    }
    #[must_use]
    pub const fn pinned_host_bytes(self) -> u64 {
        self.pinned_host_bytes
    }
    #[must_use]
    pub const fn pinned_host_allocation_count(self) -> u64 {
        self.pinned_host_allocation_count
    }
}

pub(super) struct GemmPlans {
    pub(super) hidden: CudaPreparedGemm,
    pub(super) key_value: CudaPreparedGemm,
    pub(super) intermediate: CudaPreparedGemm,
    pub(super) down: CudaPreparedGemm,
    pub(super) lm_head: CudaPreparedGemm,
}

impl GemmPlans {
    pub(super) fn any_poisoned(&self) -> bool {
        self.hidden.is_poisoned()
            || self.key_value.is_poisoned()
            || self.intermediate.is_poisoned()
            || self.down.is_poisoned()
            || self.lm_head.is_poisoned()
    }
}

pub(super) struct ForwardBuffers {
    pub(super) token_ids: CudaDeviceBuffer,
    pub(super) hidden_current: CudaDeviceBuffer,
    pub(super) hidden_norm: CudaDeviceBuffer,
    pub(super) hidden_projection: CudaDeviceBuffer,
    pub(super) hidden_rotary: CudaDeviceBuffer,
    pub(super) hidden_context: CudaDeviceBuffer,
    pub(super) key_raw: CudaDeviceBuffer,
    pub(super) value_raw: CudaDeviceBuffer,
    pub(super) key_rotary: CudaDeviceBuffer,
    pub(super) gate_raw: CudaDeviceBuffer,
    pub(super) up_raw: CudaDeviceBuffer,
    pub(super) gate_activated: CudaDeviceBuffer,
    pub(super) gated_product: CudaDeviceBuffer,
    pub(super) attention_workspace: Option<CudaDeviceBuffer>,
    pub(super) rope_cos: CudaDeviceBuffer,
    pub(super) rope_sin: CudaDeviceBuffer,
    pub(super) logits: CudaDeviceBuffer,
    pub(super) embedding_error_scratch: CudaDeviceBuffer,
    pub(super) gemm_workspace: Option<CudaDeviceBuffer>,
}

/// Owning fixed-S reference forward. All hot execution state is direct-indexed.
pub struct PreparedLlamaForward {
    pub(super) plan: LlamaExecutionPlan,
    pub(super) weights: CudaUploadedWeights,
    pub(super) gemms: GemmPlans,
    attention: PreparedPrefillAttention,
    pub(super) buffers: ForwardBuffers,
    pub(super) io_staging: CudaPinnedHostBuffer,
    pub(super) token_bytes: Box<[u8]>,
    allocation_report: PreparedLlamaAllocationReport,
    pub(super) tokens_ready: bool,
    pub(super) output_ready: bool,
    pub(super) poisoned: bool,
}

impl fmt::Debug for PreparedLlamaForward {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLlamaForward")
            .field("plan", &self.plan)
            .field("attention_selection", &self.attention.selection_trace())
            .field("allocation_report", &self.allocation_report)
            .field("tokens_ready", &self.tokens_ready)
            .field("output_ready", &self.output_ready)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl PreparedLlamaForward {
    /// Uploads the verified checkpoint and prepares one immutable fixed-S graph.
    ///
    /// The model is used only during this cold call. The returned owner contains
    /// all CUDA weights, algorithms, tables, graph buffers, and reusable I/O
    /// staging required by subsequent allocation-free execution.
    ///
    /// # Errors
    ///
    /// Returns for an invalid plan/configuration, failed weight upload, attention
    /// budget violation, host reservation, GEMM preparation, CUDA allocation, or
    /// RoPE-table upload failure.
    pub fn prepare(
        model: &LoadedModel,
        context: &CudaContext,
        stream: &mut CudaStream,
        sequence_length: usize,
        config: PreparedLlamaForwardConfig,
    ) -> LlamaForwardResult<Self> {
        config.validate()?;
        if sequence_length == 0 || sequence_length > model.spec().max_sequence_length() {
            return Err(LlamaForwardError::Plan(
                LlamaPlanError::InvalidSequenceLength {
                    requested: sequence_length,
                    maximum: model.spec().max_sequence_length(),
                },
            ));
        }

        let weights = CudaUploadedWeights::upload(
            model.weights(),
            context,
            stream,
            config.upload_staging_bytes,
        )
        .map_err(|source| LlamaForwardError::Weight { site: None, source })?;
        let plan = LlamaExecutionPlan::prepare(model.spec(), &weights, sequence_length)?;
        let workspace = plan.workspace_spec();
        let attention = prepare_attention(context, &plan, config.attention_preference)?;
        let attention_workspace_bytes = attention.workspace_bytes();
        if attention_workspace_bytes > config.attention_budget_bytes {
            return Err(LlamaForwardError::AttentionBudgetExceeded {
                required_bytes: attention_workspace_bytes,
                maximum_bytes: config.attention_budget_bytes,
            });
        }

        let gemms = prepare_gemms(context, &plan, config.gemm_workspace_cap_bytes)?;
        let gemm_workspace_bytes = [
            gemms.hidden.algorithm_metadata().workspace_bytes(),
            gemms.key_value.algorithm_metadata().workspace_bytes(),
            gemms.intermediate.algorithm_metadata().workspace_bytes(),
            gemms.down.algorithm_metadata().workspace_bytes(),
            gemms.lm_head.algorithm_metadata().workspace_bytes(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let mut buffers = allocate_buffers(
            context,
            &plan,
            attention_workspace_bytes,
            gemm_workspace_bytes,
        )?;
        let mut io_staging = context
            .allocate_pinned_host_buffer(config.io_staging_bytes)
            .map_err(|source| {
                LlamaForwardError::cuda(ExecutionSite::global(LlamaOp::Embedding), source)
            })?;

        let (rope_cos, rope_sin) = build_rope_tables(&plan)?;
        buffers
            .rope_cos
            .upload_from_slice(0, &rope_cos, &mut io_staging, stream)
            .map_err(|source| {
                LlamaForwardError::cuda(ExecutionSite::layer(0, LlamaOp::QueryRope), source)
            })?;
        buffers
            .rope_sin
            .upload_from_slice(0, &rope_sin, &mut io_staging, stream)
            .map_err(|source| {
                LlamaForwardError::cuda(ExecutionSite::layer(0, LlamaOp::QueryRope), source)
            })?;

        let token_bytes =
            allocate_host_bytes(workspace.token_ids_bytes(), LlamaForwardResource::TokenIds)?;
        let allocation_report = build_allocation_report(
            &weights,
            &plan,
            attention_workspace_bytes,
            gemm_workspace_bytes,
            config.io_staging_bytes,
        )?;

        Ok(Self {
            plan,
            weights,
            gemms,
            attention,
            buffers,
            io_staging,
            token_bytes,
            allocation_report,
            tokens_ready: false,
            output_ready: false,
            poisoned: false,
        })
    }

    /// Immutable fixed-S topology and dimensions.
    #[must_use]
    pub const fn plan(&self) -> &LlamaExecutionPlan {
        &self.plan
    }

    /// Exact owned allocations expected after successful preparation.
    #[must_use]
    pub const fn allocation_report(&self) -> PreparedLlamaAllocationReport {
        self.allocation_report
    }

    /// Cold-path attention selection and its allocation/materialization facts.
    #[must_use]
    pub fn attention_selection(&self) -> AttentionSelectionTrace {
        self.attention.selection_trace()
    }

    /// Allocates reusable caller-owned host storage for the canonical PR07
    /// diagnostic checkpoints without executing CUDA work.
    ///
    /// # Errors
    ///
    /// Returns if the plan has fewer than the layer-14 checkpoint or host byte
    /// arithmetic/reservation fails. Online attention has no probability
    /// matrix to capture, so callers must prepare a reference-attention owner.
    pub fn prepare_trace(&self) -> LlamaForwardResult<PreparedLlamaTrace> {
        if self.attention.backend() != AttentionBackend::MaterializedReference {
            return Err(LlamaForwardError::TraceRequiresReferenceAttention);
        }
        PreparedLlamaTrace::prepare(&self.plan)
    }

    /// Whether one valid exact-length token vector has been uploaded.
    #[must_use]
    pub const fn tokens_ready(&self) -> bool {
        self.tokens_ready
    }

    /// Whether logits belong to the most recent successful execution.
    #[must_use]
    pub const fn output_ready(&self) -> bool {
        self.output_ready
    }

    /// Whether a native execution failure or nested owner poison disabled reuse.
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub(super) fn validate_token_ids(&self, token_ids: &[u32]) -> LlamaForwardResult<()> {
        if token_ids.len() != self.plan.sequence_length() {
            return Err(LlamaForwardError::InvalidTokenCount {
                expected: self.plan.sequence_length(),
                actual: token_ids.len(),
            });
        }
        let vocabulary_size = self.plan.dimensions().vocabulary_size();
        for (position, &token_id) in token_ids.iter().enumerate() {
            let in_range = usize::try_from(token_id)
                .ok()
                .is_some_and(|token| token < vocabulary_size);
            if !in_range {
                return Err(LlamaForwardError::TokenOutOfRange {
                    position,
                    token_id,
                    vocabulary_size,
                });
            }
        }
        Ok(())
    }

    /// Validates and uploads exactly the plan's fixed token vector.
    ///
    /// Host conversion, pinned staging, and the device allocation are all
    /// prepared already, so this method performs no allocation.
    ///
    /// # Errors
    ///
    /// Returns for a poisoned owner, wrong token count, out-of-range token, or
    /// CUDA copy/synchronization failure.
    pub fn upload_tokens(
        &mut self,
        token_ids: &[u32],
        stream: &mut CudaStream,
    ) -> LlamaForwardResult<()> {
        if self.poisoned {
            return Err(LlamaForwardError::Poisoned);
        }
        self.validate_token_ids(token_ids)?;
        for (destination, &token_id) in self.token_bytes.chunks_exact_mut(4).zip(token_ids) {
            destination.copy_from_slice(&token_id.to_ne_bytes());
        }

        self.tokens_ready = false;
        self.output_ready = false;
        let result = self.buffers.token_ids.upload_from_slice(
            0,
            &self.token_bytes,
            &mut self.io_staging,
            stream,
        );
        match result {
            Ok(()) => {
                self.tokens_ready = true;
                Ok(())
            }
            Err(source) => {
                poison_for_cuda_error(&mut self.poisoned, &source);
                Err(LlamaForwardError::cuda(
                    ExecutionSite::global(LlamaOp::Embedding),
                    source,
                ))
            }
        }
    }

    /// Executes the prepared cache-free full-sequence graph and writes logits.
    ///
    /// The successful path performs no host or device allocation, name lookup,
    /// JSON parsing, or map lookup. Every native operation synchronizes the
    /// explicit stream before returning.
    ///
    /// # Errors
    ///
    /// Returns for missing tokens, poisoned state, an impossible bound-weight
    /// invariant, or a layer/op-qualified CUDA failure.
    pub fn execute(&mut self, stream: &mut CudaStream) -> LlamaForwardResult<()> {
        if self.poisoned {
            return Err(LlamaForwardError::Poisoned);
        }
        if !self.tokens_ready {
            return Err(LlamaForwardError::TokensNotUploaded);
        }
        self.output_ready = false;
        let result = self.execute_inner(stream, None, None);
        match result {
            Ok(()) => {
                self.output_ready = true;
                Ok(())
            }
            Err(error) => {
                poison_for_forward_error(&mut self.poisoned, &error);
                self.poisoned |= self.gemms.any_poisoned();
                Err(error)
            }
        }
    }

    /// Executes the fixed prompt graph while copying each layer's rotated K
    /// and raw V tensors into the caller-selected decode cache.
    ///
    /// This is a crate-private PR09/PR10 entry point. The public cache-free
    /// [`Self::execute`] path continues to pass no cache sink and therefore
    /// launches exactly the PR08 graph.
    pub(super) fn execute_prefill_into_cache(
        &mut self,
        cache: PrefillKvCacheSink<'_>,
        stream: &mut CudaStream,
    ) -> LlamaForwardResult<()> {
        if self.poisoned {
            return Err(LlamaForwardError::Poisoned);
        }
        if !self.tokens_ready {
            return Err(LlamaForwardError::TokensNotUploaded);
        }
        self.output_ready = false;
        let result = self.execute_inner(stream, None, Some(cache));
        match result {
            Ok(()) => {
                self.output_ready = true;
                Ok(())
            }
            Err(error) => {
                poison_for_forward_error(&mut self.poisoned, &error);
                self.poisoned |= self.gemms.any_poisoned();
                Err(error)
            }
        }
    }

    /// Executes the same graph while downloading the fixed diagnostic
    /// checkpoints into a compatible preallocated trace owner.
    ///
    /// This path is intentionally diagnostic and synchronizes for each trace
    /// transfer. The production [`Self::execute`] path performs no transfers.
    ///
    /// # Errors
    ///
    /// Returns for missing tokens, poisoned state, an incompatible trace, or
    /// any layer/op-qualified execution or trace-copy failure.
    pub fn execute_traced(
        &mut self,
        stream: &mut CudaStream,
        trace: &mut PreparedLlamaTrace,
    ) -> LlamaForwardResult<()> {
        if self.poisoned {
            return Err(LlamaForwardError::Poisoned);
        }
        if !self.tokens_ready {
            return Err(LlamaForwardError::TokensNotUploaded);
        }
        if self.attention.backend() != AttentionBackend::MaterializedReference {
            return Err(LlamaForwardError::TraceRequiresReferenceAttention);
        }
        trace.validate(&self.plan)?;
        trace.reset();
        self.output_ready = false;
        let result = self.execute_inner(stream, Some(trace), None);
        match result {
            Ok(()) => {
                self.output_ready = true;
                Ok(())
            }
            Err(error) => {
                poison_for_forward_error(&mut self.poisoned, &error);
                self.poisoned |= self.gemms.any_poisoned();
                Err(error)
            }
        }
    }

    /// Uploads one exact-length token vector and executes the graph.
    ///
    /// # Errors
    ///
    /// Returns any validation, copy, or execution failure from the two stages.
    pub fn forward(
        &mut self,
        token_ids: &[u32],
        stream: &mut CudaStream,
    ) -> LlamaForwardResult<()> {
        self.upload_tokens(token_ids, stream)?;
        self.execute(stream)
    }

    /// Downloads all BF16 logits `[S,V]` into caller-owned bytes.
    ///
    /// # Errors
    ///
    /// Returns when no successful output exists, the destination size differs,
    /// or the staged CUDA copy fails.
    pub fn download_logits(
        &mut self,
        destination: &mut [u8],
        stream: &mut CudaStream,
    ) -> LlamaForwardResult<()> {
        if self.poisoned {
            return Err(LlamaForwardError::Poisoned);
        }
        if !self.output_ready {
            return Err(LlamaForwardError::OutputNotReady);
        }
        let expected =
            usize::try_from(self.plan.workspace_spec().logits_bytes()).map_err(|_| {
                LlamaForwardError::ArithmeticOverflow {
                    resource: LlamaForwardResource::Logits,
                }
            })?;
        if destination.len() != expected {
            return Err(LlamaForwardError::InvalidDownloadLength {
                expected_bytes: expected,
                actual_bytes: destination.len(),
            });
        }
        self.download_logits_range(0, destination, stream)
    }

    /// Downloads the last-position BF16 vocabulary logits `[V]`.
    ///
    /// # Errors
    ///
    /// Returns when no successful output exists, the destination size differs,
    /// offset arithmetic overflows, or the staged CUDA copy fails.
    pub fn download_last_logits(
        &mut self,
        destination: &mut [u8],
        stream: &mut CudaStream,
    ) -> LlamaForwardResult<()> {
        if self.poisoned {
            return Err(LlamaForwardError::Poisoned);
        }
        if !self.output_ready {
            return Err(LlamaForwardError::OutputNotReady);
        }
        let row_bytes = to_u64(
            self.plan.dimensions().vocabulary_size(),
            LlamaForwardResource::Logits,
        )?
        .checked_mul(BF16_BYTES)
        .ok_or(LlamaForwardError::ArithmeticOverflow {
            resource: LlamaForwardResource::Logits,
        })?;
        let expected =
            usize::try_from(row_bytes).map_err(|_| LlamaForwardError::ArithmeticOverflow {
                resource: LlamaForwardResource::Logits,
            })?;
        if destination.len() != expected {
            return Err(LlamaForwardError::InvalidDownloadLength {
                expected_bytes: expected,
                actual_bytes: destination.len(),
            });
        }
        let row = to_u64(
            self.plan.sequence_length() - 1,
            LlamaForwardResource::Logits,
        )?;
        let offset = row
            .checked_mul(row_bytes)
            .ok_or(LlamaForwardError::ArithmeticOverflow {
                resource: LlamaForwardResource::Logits,
            })?;
        self.download_logits_range(offset, destination, stream)
    }

    pub(super) fn download_logits_range(
        &mut self,
        offset: u64,
        destination: &mut [u8],
        stream: &mut CudaStream,
    ) -> LlamaForwardResult<()> {
        let result = self.buffers.logits.download_to_slice(
            offset,
            destination,
            &mut self.io_staging,
            stream,
        );
        match result {
            Ok(()) => Ok(()),
            Err(source) => {
                poison_for_cuda_error(&mut self.poisoned, &source);
                Err(LlamaForwardError::cuda(
                    ExecutionSite::global(LlamaOp::LmHead),
                    source,
                ))
            }
        }
    }

    // HOT_EXECUTE_BEGIN
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::similar_names
    )]
    fn execute_inner(
        &mut self,
        stream: &mut CudaStream,
        mut trace: Option<&mut PreparedLlamaTrace>,
        mut cache: Option<PrefillKvCacheSink<'_>>,
    ) -> LlamaForwardResult<()> {
        let plan = &self.plan;
        let weights = &self.weights;
        let gemms = &mut self.gemms;
        let attention = &self.attention;
        let buffers = &mut self.buffers;
        let io_staging = &mut self.io_staging;
        let sequence = to_u64(plan.sequence_length(), LlamaForwardResource::HiddenCurrent)?;
        let dimensions = plan.dimensions();
        let hidden = to_u64(
            dimensions.hidden_size(),
            LlamaForwardResource::HiddenCurrent,
        )?;
        let key_value_width = to_u64(dimensions.key_value_width(), LlamaForwardResource::KeyRaw)?;
        let query_heads = to_u64(dimensions.query_heads(), LlamaForwardResource::Attention)?;
        let key_value_heads = to_u64(dimensions.key_value_heads(), LlamaForwardResource::KeyRaw)?;
        let head_size = to_u64(
            dimensions.head_dimension(),
            LlamaForwardResource::HiddenRotary,
        )?;
        let hidden_elements = plan.workspace_spec().hidden_buffer_bytes() / BF16_BYTES;
        let intermediate_elements = plan.workspace_spec().intermediate_buffer_bytes() / BF16_BYTES;

        let embedding_site = ExecutionSite::global(LlamaOp::Embedding);
        let embedding_weight = weight_span(weights, plan.embedding_weight(), embedding_site)?;
        {
            let mut params = EmbeddingParams {
                table: embedding_weight,
                token_ids: span(
                    &buffers.token_ids,
                    CudaDType::U32,
                    plan.workspace_spec().token_ids_bytes(),
                    embedding_site,
                )?,
                output: span_mut(
                    &mut buffers.hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    embedding_site,
                )?,
                error_scratch: span_mut(
                    &mut buffers.embedding_error_scratch,
                    CudaDType::U8,
                    plan.workspace_spec().embedding_error_scratch_bytes(),
                    embedding_site,
                )?,
                token_count: sequence,
                vocabulary_size: to_u64(
                    dimensions.vocabulary_size(),
                    LlamaForwardResource::Logits,
                )?,
                hidden_size: hidden,
            };
            embedding(&mut params, stream).map_err(|source| LlamaForwardError::Embedding {
                site: embedding_site,
                source,
            })?;
        }
        capture_trace(
            &mut trace,
            LlamaTracePoint::Embedding,
            &mut buffers.hidden_current,
            0,
            io_staging,
            stream,
            embedding_site,
        )?;

        for layer in plan.layers() {
            let layer_index = layer.index();
            let input_norm_site = ExecutionSite::layer(layer_index, LlamaOp::InputNorm);
            let input_norm_weight =
                weight_span(weights, layer.input_norm_weight(), input_norm_site)?;
            {
                let mut params = RmsNormParams {
                    input: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        input_norm_site,
                    )?,
                    weight: input_norm_weight,
                    output: span_mut(
                        &mut buffers.hidden_norm,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        input_norm_site,
                    )?,
                    row_count: sequence,
                    hidden_size: hidden,
                    epsilon: layer.input_norm_epsilon(),
                };
                rms_norm(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(input_norm_site, source))?;
            }
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0InputNorm,
                    &mut buffers.hidden_norm,
                    0,
                    io_staging,
                    stream,
                    input_norm_site,
                )?;
            }

            let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
            let query_weight = weight_span(weights, layer.query_weight(), query_site)?;
            execute_gemm(
                &mut gemms.hidden,
                &buffers.hidden_norm,
                query_weight,
                &mut buffers.hidden_projection,
                &mut buffers.gemm_workspace,
                stream,
                query_site,
            )?;
            execute_projection_bias(
                weights,
                layer.query_bias(),
                &mut buffers.hidden_projection,
                sequence,
                hidden,
                stream,
                query_site,
            )?;
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0QueryProjection,
                    &mut buffers.hidden_projection,
                    0,
                    io_staging,
                    stream,
                    query_site,
                )?;
            }
            let key_site = ExecutionSite::layer(layer_index, LlamaOp::KeyProjection);
            let key_weight = weight_span(weights, layer.key_weight(), key_site)?;
            execute_gemm(
                &mut gemms.key_value,
                &buffers.hidden_norm,
                key_weight,
                &mut buffers.key_raw,
                &mut buffers.gemm_workspace,
                stream,
                key_site,
            )?;
            execute_projection_bias(
                weights,
                layer.key_bias(),
                &mut buffers.key_raw,
                sequence,
                key_value_width,
                stream,
                key_site,
            )?;
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0KeyProjection,
                    &mut buffers.key_raw,
                    0,
                    io_staging,
                    stream,
                    key_site,
                )?;
            }
            let value_site = ExecutionSite::layer(layer_index, LlamaOp::ValueProjection);
            let value_weight = weight_span(weights, layer.value_weight(), value_site)?;
            execute_gemm(
                &mut gemms.key_value,
                &buffers.hidden_norm,
                value_weight,
                &mut buffers.value_raw,
                &mut buffers.gemm_workspace,
                stream,
                value_site,
            )?;
            execute_projection_bias(
                weights,
                layer.value_bias(),
                &mut buffers.value_raw,
                sequence,
                key_value_width,
                stream,
                value_site,
            )?;
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0ValueProjection,
                    &mut buffers.value_raw,
                    0,
                    io_staging,
                    stream,
                    value_site,
                )?;
            }

            let query_rope_site = ExecutionSite::layer(layer_index, LlamaOp::QueryRope);
            {
                let mut params = RopeParams {
                    input: span(
                        &buffers.hidden_projection,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        query_rope_site,
                    )?,
                    cos: span(
                        &buffers.rope_cos,
                        CudaDType::F32,
                        plan.workspace_spec().rope_cos_bytes(),
                        query_rope_site,
                    )?,
                    sin: span(
                        &buffers.rope_sin,
                        CudaDType::F32,
                        plan.workspace_spec().rope_sin_bytes(),
                        query_rope_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        query_rope_site,
                    )?,
                    token_count: sequence,
                    head_count: query_heads,
                    head_size,
                    rotary_dimension: head_size,
                    table_position_count: sequence,
                    position_offset: 0,
                };
                rope(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(query_rope_site, source))?;
            }
            let key_rope_site = ExecutionSite::layer(layer_index, LlamaOp::KeyRope);
            {
                let mut params = RopeParams {
                    input: span(
                        &buffers.key_raw,
                        CudaDType::BF16,
                        plan.workspace_spec().key_value_buffer_bytes(),
                        key_rope_site,
                    )?,
                    cos: span(
                        &buffers.rope_cos,
                        CudaDType::F32,
                        plan.workspace_spec().rope_cos_bytes(),
                        key_rope_site,
                    )?,
                    sin: span(
                        &buffers.rope_sin,
                        CudaDType::F32,
                        plan.workspace_spec().rope_sin_bytes(),
                        key_rope_site,
                    )?,
                    output: span_mut(
                        &mut buffers.key_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().key_value_buffer_bytes(),
                        key_rope_site,
                    )?,
                    token_count: sequence,
                    head_count: key_value_heads,
                    head_size,
                    rotary_dimension: head_size,
                    table_position_count: sequence,
                    position_offset: 0,
                };
                rope(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(key_rope_site, source))?;
            }

            if let Some(cache) = cache.as_mut() {
                let cache_site = ExecutionSite::layer(layer_index, LlamaOp::KvCacheWrite);
                cache
                    .append_layer(
                        layer_index,
                        &buffers.key_rotary,
                        &buffers.value_raw,
                        sequence,
                        0,
                        stream,
                    )
                    .map_err(|error| error.into_forward_cache_error(cache_site))?;
            }

            let prefill_site = ExecutionSite::layer(layer_index, LlamaOp::PrefillAttention);
            {
                let ForwardBuffers {
                    hidden_rotary,
                    hidden_context,
                    value_raw,
                    key_rotary,
                    attention_workspace,
                    ..
                } = buffers;
                let workspace = attention_workspace
                    .as_mut()
                    .map(|buffer| {
                        span_mut(
                            buffer,
                            CudaDType::BF16,
                            attention.workspace_bytes(),
                            prefill_site,
                        )
                    })
                    .transpose()?;
                let mut params = PrefillAttentionParams {
                    query: span(
                        hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        prefill_site,
                    )?,
                    key: span(
                        key_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().key_value_buffer_bytes(),
                        prefill_site,
                    )?,
                    value: span(
                        value_raw,
                        CudaDType::BF16,
                        plan.workspace_spec().key_value_buffer_bytes(),
                        prefill_site,
                    )?,
                    output: span_mut(
                        hidden_context,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        prefill_site,
                    )?,
                    workspace,
                };
                attention
                    .execute(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(prefill_site, source))?;
            }
            if layer_index == 0 && trace.is_some() {
                let workspace = buffers
                    .attention_workspace
                    .as_mut()
                    .ok_or(LlamaForwardError::TraceRequiresReferenceAttention)?;
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0AttentionProbabilities,
                    workspace,
                    0,
                    io_staging,
                    stream,
                    prefill_site,
                )?;
            }
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0AttentionContext,
                    &mut buffers.hidden_context,
                    0,
                    io_staging,
                    stream,
                    prefill_site,
                )?;
            }

            let output_site = ExecutionSite::layer(layer_index, LlamaOp::OutputProjection);
            let output_weight = weight_span(weights, layer.output_weight(), output_site)?;
            execute_gemm(
                &mut gemms.hidden,
                &buffers.hidden_context,
                output_weight,
                &mut buffers.hidden_projection,
                &mut buffers.gemm_workspace,
                stream,
                output_site,
            )?;
            execute_projection_bias(
                weights,
                layer.output_bias(),
                &mut buffers.hidden_projection,
                sequence,
                hidden,
                stream,
                output_site,
            )?;
            let attention_residual_site =
                ExecutionSite::layer(layer_index, LlamaOp::AttentionResidual);
            {
                let mut params = ResidualAddParams {
                    left: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        attention_residual_site,
                    )?,
                    right: span(
                        &buffers.hidden_projection,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        attention_residual_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        attention_residual_site,
                    )?,
                    element_count: hidden_elements,
                };
                residual_add(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(attention_residual_site, source))?;
            }
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0AfterAttentionResidual,
                    &mut buffers.hidden_rotary,
                    0,
                    io_staging,
                    stream,
                    attention_residual_site,
                )?;
            }

            let post_norm_site = ExecutionSite::layer(layer_index, LlamaOp::PostAttentionNorm);
            let post_norm_weight =
                weight_span(weights, layer.post_attention_norm_weight(), post_norm_site)?;
            {
                let mut params = RmsNormParams {
                    input: span(
                        &buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    weight: post_norm_weight,
                    output: span_mut(
                        &mut buffers.hidden_norm,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    row_count: sequence,
                    hidden_size: hidden,
                    epsilon: layer.post_attention_norm_epsilon(),
                };
                rms_norm(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(post_norm_site, source))?;
            }
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0PostAttentionNorm,
                    &mut buffers.hidden_norm,
                    0,
                    io_staging,
                    stream,
                    post_norm_site,
                )?;
            }

            let gate_site = ExecutionSite::layer(layer_index, LlamaOp::GateProjection);
            let gate_weight = weight_span(weights, layer.gate_weight(), gate_site)?;
            execute_gemm(
                &mut gemms.intermediate,
                &buffers.hidden_norm,
                gate_weight,
                &mut buffers.gate_raw,
                &mut buffers.gemm_workspace,
                stream,
                gate_site,
            )?;
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0GateProjection,
                    &mut buffers.gate_raw,
                    0,
                    io_staging,
                    stream,
                    gate_site,
                )?;
            }
            let up_site = ExecutionSite::layer(layer_index, LlamaOp::UpProjection);
            let up_weight = weight_span(weights, layer.up_weight(), up_site)?;
            execute_gemm(
                &mut gemms.intermediate,
                &buffers.hidden_norm,
                up_weight,
                &mut buffers.up_raw,
                &mut buffers.gemm_workspace,
                stream,
                up_site,
            )?;
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0UpProjection,
                    &mut buffers.up_raw,
                    0,
                    io_staging,
                    stream,
                    up_site,
                )?;
            }
            let silu_site = ExecutionSite::layer(layer_index, LlamaOp::Silu);
            {
                let mut params = SiluParams {
                    input: span(
                        &buffers.gate_raw,
                        CudaDType::BF16,
                        plan.workspace_spec().intermediate_buffer_bytes(),
                        silu_site,
                    )?,
                    output: span_mut(
                        &mut buffers.gate_activated,
                        CudaDType::BF16,
                        plan.workspace_spec().intermediate_buffer_bytes(),
                        silu_site,
                    )?,
                    element_count: intermediate_elements,
                };
                silu(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(silu_site, source))?;
            }
            let gated_site = ExecutionSite::layer(layer_index, LlamaOp::GatedMultiply);
            {
                let mut params = GatedMultiplyParams {
                    activated_gate: span(
                        &buffers.gate_activated,
                        CudaDType::BF16,
                        plan.workspace_spec().intermediate_buffer_bytes(),
                        gated_site,
                    )?,
                    up: span(
                        &buffers.up_raw,
                        CudaDType::BF16,
                        plan.workspace_spec().intermediate_buffer_bytes(),
                        gated_site,
                    )?,
                    output: span_mut(
                        &mut buffers.gated_product,
                        CudaDType::BF16,
                        plan.workspace_spec().intermediate_buffer_bytes(),
                        gated_site,
                    )?,
                    element_count: intermediate_elements,
                };
                gated_multiply(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(gated_site, source))?;
            }
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0Gated,
                    &mut buffers.gated_product,
                    0,
                    io_staging,
                    stream,
                    gated_site,
                )?;
            }
            let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
            let down_weight = weight_span(weights, layer.down_weight(), down_site)?;
            execute_gemm(
                &mut gemms.down,
                &buffers.gated_product,
                down_weight,
                &mut buffers.hidden_current,
                &mut buffers.gemm_workspace,
                stream,
                down_site,
            )?;
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0DownProjection,
                    &mut buffers.hidden_current,
                    0,
                    io_staging,
                    stream,
                    down_site,
                )?;
            }
            let mlp_residual_site = ExecutionSite::layer(layer_index, LlamaOp::MlpResidual);
            {
                let mut params = ResidualAddParams {
                    left: span(
                        &buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        mlp_residual_site,
                    )?,
                    right: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        mlp_residual_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_projection,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        mlp_residual_site,
                    )?,
                    element_count: hidden_elements,
                };
                residual_add(&mut params, stream)
                    .map_err(|source| LlamaForwardError::cuda(mlp_residual_site, source))?;
            }
            if layer_index == 0 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer0Output,
                    &mut buffers.hidden_projection,
                    0,
                    io_staging,
                    stream,
                    mlp_residual_site,
                )?;
            } else if layer_index == 14 {
                capture_trace(
                    &mut trace,
                    LlamaTracePoint::Layer14Output,
                    &mut buffers.hidden_projection,
                    0,
                    io_staging,
                    stream,
                    mlp_residual_site,
                )?;
            }
            mem::swap(&mut buffers.hidden_current, &mut buffers.hidden_projection);
        }

        let final_norm_site = ExecutionSite::global(LlamaOp::FinalNorm);
        capture_trace(
            &mut trace,
            LlamaTracePoint::FinalNormInput,
            &mut buffers.hidden_current,
            0,
            io_staging,
            stream,
            final_norm_site,
        )?;
        let final_norm_weight = weight_span(weights, plan.final_norm_weight(), final_norm_site)?;
        {
            let mut params = RmsNormParams {
                input: span(
                    &buffers.hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    final_norm_site,
                )?,
                weight: final_norm_weight,
                output: span_mut(
                    &mut buffers.hidden_norm,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    final_norm_site,
                )?,
                row_count: sequence,
                hidden_size: hidden,
                epsilon: plan.final_norm_epsilon(),
            };
            rms_norm(&mut params, stream)
                .map_err(|source| LlamaForwardError::cuda(final_norm_site, source))?;
        }
        capture_trace(
            &mut trace,
            LlamaTracePoint::FinalNormOutput,
            &mut buffers.hidden_norm,
            0,
            io_staging,
            stream,
            final_norm_site,
        )?;
        let lm_head_site = ExecutionSite::global(LlamaOp::LmHead);
        let lm_head_weight = weight_span(weights, plan.lm_head_weight(), lm_head_site)?;
        execute_gemm(
            &mut gemms.lm_head,
            &buffers.hidden_norm,
            lm_head_weight,
            &mut buffers.logits,
            &mut buffers.gemm_workspace,
            stream,
            lm_head_site,
        )?;
        let last_logits_bytes = trace_byte_len(plan, LlamaTracePoint::LastLogits)?;
        let last_logits_offset = plan
            .workspace_spec()
            .logits_bytes()
            .checked_sub(last_logits_bytes)
            .ok_or(LlamaForwardError::ArithmeticOverflow {
                resource: LlamaForwardResource::Logits,
            })?;
        capture_trace(
            &mut trace,
            LlamaTracePoint::LastLogits,
            &mut buffers.logits,
            last_logits_offset,
            io_staging,
            stream,
            lm_head_site,
        )?;
        Ok(())
    }
    // HOT_EXECUTE_END
}

#[allow(clippy::cast_precision_loss)]
fn prepare_attention(
    context: &CudaContext,
    plan: &LlamaExecutionPlan,
    preference: AttentionPreference,
) -> LlamaForwardResult<PreparedPrefillAttention> {
    let site = ExecutionSite::layer(0, LlamaOp::PrefillAttention);
    let dimensions = plan.dimensions();
    let head_size = to_u64(
        dimensions.head_dimension(),
        LlamaForwardResource::HiddenRotary,
    )?;
    let request = PrefillAttentionRequest::new(
        1,
        to_u64(plan.sequence_length(), LlamaForwardResource::Attention)?,
        to_u64(dimensions.query_heads(), LlamaForwardResource::Attention)?,
        to_u64(dimensions.key_value_heads(), LlamaForwardResource::KeyRaw)?,
        head_size,
        1.0 / (head_size as f32).sqrt(),
        AttentionMask::Causal,
    );
    PreparedPrefillAttention::select(
        context,
        request,
        preference,
        AttentionBackendAvailability::linked(),
    )
    .map_err(|source| LlamaForwardError::cuda(site, source))
}

fn build_allocation_report(
    weights: &CudaUploadedWeights,
    plan: &LlamaExecutionPlan,
    attention_workspace_bytes: u64,
    gemm_workspace_bytes: u64,
    pinned_host_bytes: u64,
) -> LlamaForwardResult<PreparedLlamaAllocationReport> {
    let graph_bytes = plan
        .workspace_spec()
        .non_attention_planned_bytes()
        .checked_add(attention_workspace_bytes)
        .ok_or(LlamaForwardError::ArithmeticOverflow {
            resource: LlamaForwardResource::Attention,
        })?;
    let total_device_bytes = weights
        .total_physical_bytes()
        .checked_add(graph_bytes)
        .and_then(|bytes| bytes.checked_add(gemm_workspace_bytes))
        .ok_or(LlamaForwardError::ArithmeticOverflow {
            resource: LlamaForwardResource::GemmWorkspace,
        })?;
    let graph_allocations = NON_ATTENTION_GRAPH_ALLOCATION_COUNT
        .checked_add(u64::from(attention_workspace_bytes != 0))
        .and_then(|count| count.checked_add(u64::from(gemm_workspace_bytes != 0)))
        .ok_or(LlamaForwardError::ArithmeticOverflow {
            resource: LlamaForwardResource::Attention,
        })?;
    let physical_allocations = u64::try_from(weights.physical_tensor_count()).map_err(|_| {
        LlamaForwardError::ArithmeticOverflow {
            resource: LlamaForwardResource::UploadedWeights,
        }
    })?;
    Ok(PreparedLlamaAllocationReport {
        weight_bytes: weights.total_physical_bytes(),
        graph_bytes,
        gemm_workspace_bytes,
        total_device_bytes,
        device_allocation_count: physical_allocations.checked_add(graph_allocations).ok_or(
            LlamaForwardError::ArithmeticOverflow {
                resource: LlamaForwardResource::UploadedWeights,
            },
        )?,
        pinned_host_bytes,
        pinned_host_allocation_count: 1,
    })
}

fn prepare_gemms(
    context: &CudaContext,
    plan: &LlamaExecutionPlan,
    workspace_cap: u64,
) -> LlamaForwardResult<GemmPlans> {
    let sequence = to_u64(plan.sequence_length(), LlamaForwardResource::HiddenCurrent)?;
    let dimensions = plan.dimensions();
    let hidden = to_u64(
        dimensions.hidden_size(),
        LlamaForwardResource::HiddenCurrent,
    )?;
    let key_value = to_u64(dimensions.key_value_width(), LlamaForwardResource::KeyRaw)?;
    let intermediate = to_u64(
        dimensions.intermediate_size(),
        LlamaForwardResource::GateRaw,
    )?;
    let vocabulary = to_u64(dimensions.vocabulary_size(), LlamaForwardResource::Logits)?;

    let prepare = |m, n, k, site| -> LlamaForwardResult<CudaPreparedGemm> {
        let config = CudaGemmConfig::new(m, n, k, workspace_cap)
            .map_err(|source| LlamaForwardError::cuda(site, source))?;
        context
            .prepare_gemm(config)
            .map_err(|source| LlamaForwardError::cuda(site, source))
    };
    Ok(GemmPlans {
        hidden: prepare(
            sequence,
            hidden,
            hidden,
            ExecutionSite::layer(0, LlamaOp::QueryProjection),
        )?,
        key_value: prepare(
            sequence,
            key_value,
            hidden,
            ExecutionSite::layer(0, LlamaOp::KeyProjection),
        )?,
        intermediate: prepare(
            sequence,
            intermediate,
            hidden,
            ExecutionSite::layer(0, LlamaOp::GateProjection),
        )?,
        down: prepare(
            sequence,
            hidden,
            intermediate,
            ExecutionSite::layer(0, LlamaOp::DownProjection),
        )?,
        lm_head: prepare(
            sequence,
            vocabulary,
            hidden,
            ExecutionSite::global(LlamaOp::LmHead),
        )?,
    })
}

fn allocate_buffers(
    context: &CudaContext,
    plan: &LlamaExecutionPlan,
    attention_workspace_bytes: u64,
    gemm_workspace_bytes: u64,
) -> LlamaForwardResult<ForwardBuffers> {
    let spec = plan.workspace_spec();
    let allocate = |bytes, site| {
        context
            .allocate_device_buffer(bytes)
            .map_err(|source| LlamaForwardError::cuda(site, source))
    };
    let hidden_site = ExecutionSite::layer(0, LlamaOp::InputNorm);
    let kv_site = ExecutionSite::layer(0, LlamaOp::KeyProjection);
    let intermediate_site = ExecutionSite::layer(0, LlamaOp::GateProjection);
    Ok(ForwardBuffers {
        token_ids: allocate(
            spec.token_ids_bytes(),
            ExecutionSite::global(LlamaOp::Embedding),
        )?,
        hidden_current: allocate(spec.hidden_buffer_bytes(), hidden_site)?,
        hidden_norm: allocate(spec.hidden_buffer_bytes(), hidden_site)?,
        hidden_projection: allocate(spec.hidden_buffer_bytes(), hidden_site)?,
        hidden_rotary: allocate(spec.hidden_buffer_bytes(), hidden_site)?,
        hidden_context: allocate(spec.hidden_buffer_bytes(), hidden_site)?,
        key_raw: allocate(spec.key_value_buffer_bytes(), kv_site)?,
        value_raw: allocate(spec.key_value_buffer_bytes(), kv_site)?,
        key_rotary: allocate(spec.key_value_buffer_bytes(), kv_site)?,
        gate_raw: allocate(spec.intermediate_buffer_bytes(), intermediate_site)?,
        up_raw: allocate(spec.intermediate_buffer_bytes(), intermediate_site)?,
        gate_activated: allocate(spec.intermediate_buffer_bytes(), intermediate_site)?,
        gated_product: allocate(spec.intermediate_buffer_bytes(), intermediate_site)?,
        attention_workspace: if attention_workspace_bytes == 0 {
            None
        } else {
            Some(allocate(
                attention_workspace_bytes,
                ExecutionSite::layer(0, LlamaOp::PrefillAttention),
            )?)
        },
        rope_cos: allocate(
            spec.rope_cos_bytes(),
            ExecutionSite::layer(0, LlamaOp::QueryRope),
        )?,
        rope_sin: allocate(
            spec.rope_sin_bytes(),
            ExecutionSite::layer(0, LlamaOp::QueryRope),
        )?,
        logits: allocate(spec.logits_bytes(), ExecutionSite::global(LlamaOp::LmHead))?,
        embedding_error_scratch: allocate(
            spec.embedding_error_scratch_bytes(),
            ExecutionSite::global(LlamaOp::Embedding),
        )?,
        gemm_workspace: if gemm_workspace_bytes == 0 {
            None
        } else {
            Some(allocate(
                gemm_workspace_bytes,
                ExecutionSite::layer(0, LlamaOp::QueryProjection),
            )?)
        },
    })
}

fn allocate_host_bytes(
    byte_len: u64,
    resource: LlamaForwardResource,
) -> LlamaForwardResult<Box<[u8]>> {
    let length = usize::try_from(byte_len)
        .map_err(|_| LlamaForwardError::ArithmeticOverflow { resource })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| LlamaForwardError::HostAllocation {
            resource,
            requested_bytes: byte_len,
        })?;
    bytes.resize(length, 0);
    Ok(bytes.into_boxed_slice())
}

type RopeTableBytes = (Box<[u8]>, Box<[u8]>);

#[allow(clippy::cast_precision_loss)]
fn build_rope_tables(plan: &LlamaExecutionPlan) -> LlamaForwardResult<RopeTableBytes> {
    let dimensions = plan.dimensions();
    let head_dimension = dimensions.head_dimension();
    let half = head_dimension / 2;
    let mut cos = allocate_host_bytes(
        plan.workspace_spec().rope_cos_bytes(),
        LlamaForwardResource::RopeCos,
    )?;
    let mut sin = allocate_host_bytes(
        plan.workspace_spec().rope_sin_bytes(),
        LlamaForwardResource::RopeSin,
    )?;
    let theta = plan.rope_theta();
    for position in 0..plan.sequence_length() {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dimension as f32;
            let inverse_frequency = 1.0 / theta.powf(exponent);
            let angle = position as f32 * inverse_frequency;
            let (sine, cosine) = angle.sin_cos();
            let element = position
                .checked_mul(half)
                .and_then(|value| value.checked_add(pair))
                .and_then(|value| value.checked_mul(4))
                .ok_or(LlamaForwardError::ArithmeticOverflow {
                    resource: LlamaForwardResource::RopeCos,
                })?;
            cos[element..element + 4].copy_from_slice(&cosine.to_ne_bytes());
            sin[element..element + 4].copy_from_slice(&sine.to_ne_bytes());
        }
    }
    Ok((cos, sin))
}

fn to_u64(value: usize, resource: LlamaForwardResource) -> LlamaForwardResult<u64> {
    u64::try_from(value).map_err(|_| LlamaForwardError::ArithmeticOverflow { resource })
}
