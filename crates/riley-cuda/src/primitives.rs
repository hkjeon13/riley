use std::error;
use std::fmt;
use std::mem::{offset_of, size_of};

use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
use crate::gemm::FIXED37_MAX_REDUCTION_ELEMENTS;
use crate::memory::CudaDeviceBuffer;
use crate::runtime::{CudaExecutionStream, CudaStream, ensure_same_context, execution_stream_mut};

#[cfg(feature = "cuda")]
use crate::ffi;
const EMBEDDING_ERROR_SCRATCH_BYTES: u64 = 32;
#[cfg(feature = "cuda")]
const EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE: u32 = 1;
/// Hidden width certified against Hugging Face `SmolLM2` `RMSNorm`.
pub const HUGGING_FACE_SMOLLM2_RMS_NORM_HIDDEN_SIZE: u64 = 576;
/// Largest reviewed row count for the `SmolLM2` `RMSNorm` topology.
pub const HUGGING_FACE_SMOLLM2_RMS_NORM_MAX_ROWS: u64 = 8_192;
/// Exact FP32 epsilon bits required by the reviewed `SmolLM2` model.
pub const HUGGING_FACE_SMOLLM2_RMS_NORM_EPSILON_BITS: u32 = 0x3727_c5ac;
/// Hidden width certified against the source-bound Hugging Face Qwen P2051
/// cache-free `RMSNorm` diagnostic.
pub const HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE: u64 = 2_048;
/// Exact full-sequence row count certified for the Qwen P2051 diagnostic.
pub const HUGGING_FACE_QWEN_P2051_RMS_NORM_ROW_COUNT: u64 = 2_051;
/// Exact cache-building prefill row count certified for the Qwen P2048
/// cache-on diagnostic.
pub const HUGGING_FACE_QWEN_P2048_CACHE_ON_RMS_NORM_ROW_COUNT: u64 = 2_048;
/// Exact one-token M1 row count admitted only by the source-bound Qwen P2048
/// cache-on diagnostic.
///
/// This is not a generic decode RMSNorm selector. It lets the paired M1 trace
/// exercise the same native reduction topology after its P2048 prefill has
/// populated a contiguous cache; the trace still has to establish BF16
/// equality before any candidate can be promoted.
pub const HUGGING_FACE_QWEN_P2048_CACHE_ON_M1_RMS_NORM_ROW_COUNT: u64 = 1;
/// Exact FP32 epsilon bits required by the Qwen P2051 diagnostic model.
pub const HUGGING_FACE_QWEN_P2051_RMS_NORM_EPSILON_BITS: u32 = 0x3586_37bd;
/// Successful deterministic BF16 argmax row.
pub const BF16_ARGMAX_STATUS_SUCCESS: u32 = 0;
/// At least one logit in the row was NaN or infinity.
pub const BF16_ARGMAX_STATUS_NON_FINITE: u32 = 1;
/// Token sentinel written for a non-finite row.
pub const BF16_ARGMAX_INVALID_TOKEN_ID: u32 = u32::MAX;
/// U32 words in one [`Bf16ArgmaxResult`] device record.
pub const BF16_ARGMAX_RESULT_U32_WORDS: u64 = 2;

/// One fixed-layout deterministic greedy result copied from device storage.
///
/// Device output is exposed as U32 storage, with one record per logits row.
/// Unknown status values must be treated as a native-contract failure by the
/// integration layer rather than as successful sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Bf16ArgmaxResult {
    /// Lowest token id tied for the finite row maximum, or the invalid sentinel.
    pub token_id: u32,
    /// One of the `BF16_ARGMAX_STATUS_*` constants.
    pub status: u32,
}

impl Bf16ArgmaxResult {
    /// Whether this row contains a valid greedy token.
    #[must_use]
    pub const fn is_success(self) -> bool {
        self.status == BF16_ARGMAX_STATUS_SUCCESS
    }
}

const _: () = assert!(size_of::<Bf16ArgmaxResult>() == 8);
const _: () = assert!(offset_of!(Bf16ArgmaxResult, status) == 4);

/// Scalar storage types accepted by the PR 06 CUDA primitive boundary.
///
/// This intentionally stays narrower than general tensor metadata: unsupported
/// integer and floating-point formats cannot cross the native ABI by accident.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CudaDType {
    /// IEEE 754 binary32.
    F32,
    /// Brain floating point, 16-bit.
    BF16,
    /// Unsigned 32-bit integer, used for token ids.
    U32,
    /// Unsigned 16-bit integer, used for paged-cache valid-token counts.
    U16,
    /// Unsigned byte, used for opaque primitive scratch records.
    U8,
}

impl CudaDType {
    /// Storage bytes occupied by one scalar.
    #[must_use]
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::BF16 | Self::U16 => 2,
            Self::U8 => 1,
        }
    }

    /// Stable lowercase diagnostic name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::BF16 => "bf16",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
        }
    }

    const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::BF16)
    }

    #[cfg(feature = "cuda")]
    const fn native_code(self) -> i32 {
        match self {
            Self::F32 => ffi::DTYPE_F32,
            Self::BF16 => ffi::DTYPE_BF16,
            Self::U32 => ffi::DTYPE_U32,
            Self::U16 => ffi::DTYPE_U16,
            Self::U8 => ffi::DTYPE_U8,
        }
    }
}

impl fmt::Display for CudaDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Immutable typed subspan of one opaque device allocation.
///
/// Construction validates allocation bounds and scalar alignment. It never
/// exposes a device address.
#[derive(Clone, Copy)]
pub struct CudaBufferSpan<'a> {
    buffer: &'a CudaDeviceBuffer,
    dtype: CudaDType,
    byte_offset: u64,
    byte_len: u64,
}

impl<'a> CudaBufferSpan<'a> {
    /// Creates a validated immutable typed byte span.
    ///
    /// # Errors
    ///
    /// Returns an alignment or allocation-range error.
    pub fn new(
        buffer: &'a CudaDeviceBuffer,
        dtype: CudaDType,
        byte_offset: u64,
        byte_len: u64,
    ) -> CudaResult<Self> {
        validate_span(
            "CudaBufferSpan::new",
            buffer.byte_len(),
            dtype,
            byte_offset,
            byte_len,
        )?;
        Ok(Self {
            buffer,
            dtype,
            byte_offset,
            byte_len,
        })
    }

    /// Declared scalar dtype.
    #[must_use]
    pub const fn dtype(&self) -> CudaDType {
        self.dtype
    }

    /// Offset from the allocation base in bytes.
    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Accessible bytes from [`Self::byte_offset`].
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Number of whole declared scalar elements in this span.
    #[must_use]
    pub const fn element_capacity(&self) -> u64 {
        self.byte_len / self.dtype.size_bytes()
    }

    /// Returns a typed view strictly contained in this declared span.
    /// Allocation ownership and dtype remain unchanged; no device address is exposed.
    ///
    /// # Errors
    /// Returns a declared-span range, overflow, or scalar-alignment error.
    pub fn subspan(&self, relative_offset:u64, byte_len:u64)->CudaResult<Self> {
        let offset=subspan_absolute_offset(self.byte_offset,self.byte_len,relative_offset,byte_len)?;
        Self::new(self.buffer,self.dtype,offset,byte_len)
    }

    pub(crate) const fn buffer(&self) -> &CudaDeviceBuffer {
        self.buffer
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn raw(&self) -> ffi::RawBufferSpan {
        self.buffer
            .native_handle()
            .span(self.dtype.native_code(), self.byte_offset, self.byte_len)
    }
}

impl fmt::Debug for CudaBufferSpan<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaBufferSpan")
            .field("allocation_byte_len", &self.buffer.byte_len())
            .field("dtype", &self.dtype)
            .field("byte_offset", &self.byte_offset)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

/// Exclusive typed subspan of one opaque device allocation.
///
/// The exclusive owner borrow prevents safe Rust callers from presenting an
/// output allocation as a simultaneous disjoint input. Native exact-in-place
/// entry points remain available internally for a future explicit safe API.
pub struct CudaBufferSpanMut<'a> {
    buffer: &'a mut CudaDeviceBuffer,
    dtype: CudaDType,
    byte_offset: u64,
    byte_len: u64,
}

impl<'a> CudaBufferSpanMut<'a> {
    /// Creates a validated exclusive typed byte span.
    ///
    /// # Errors
    ///
    /// Returns an alignment or allocation-range error.
    pub fn new(
        buffer: &'a mut CudaDeviceBuffer,
        dtype: CudaDType,
        byte_offset: u64,
        byte_len: u64,
    ) -> CudaResult<Self> {
        validate_span(
            "CudaBufferSpanMut::new",
            buffer.byte_len(),
            dtype,
            byte_offset,
            byte_len,
        )?;
        Ok(Self {
            buffer,
            dtype,
            byte_offset,
            byte_len,
        })
    }

    /// Declared scalar dtype.
    #[must_use]
    pub const fn dtype(&self) -> CudaDType {
        self.dtype
    }

    /// Offset from the allocation base in bytes.
    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Accessible bytes from [`Self::byte_offset`].
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Number of whole declared scalar elements in this span.
    #[must_use]
    pub const fn element_capacity(&self) -> u64 {
        self.byte_len / self.dtype.size_bytes()
    }

    /// Reborrows an exclusive typed subspan inside this declared parent range.
    /// # Errors
    /// Returns a parent range, overflow, or scalar alignment error.
    pub fn subspan(&mut self,relative_offset:u64,byte_len:u64)->CudaResult<CudaBufferSpanMut<'_>>{
      let offset=subspan_absolute_offset(self.byte_offset,self.byte_len,relative_offset,byte_len)?;
      CudaBufferSpanMut::new(&mut *self.buffer,self.dtype,offset,byte_len)
    }

    pub(crate) fn buffer(&self) -> &CudaDeviceBuffer {
        self.buffer
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn raw(&self) -> ffi::RawBufferSpan {
        self.buffer
            .native_handle()
            .span(self.dtype.native_code(), self.byte_offset, self.byte_len)
    }
}

impl fmt::Debug for CudaBufferSpanMut<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaBufferSpanMut")
            .field("allocation_byte_len", &self.buffer.byte_len())
            .field("dtype", &self.dtype)
            .field("byte_offset", &self.byte_offset)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

/// Inputs, output, and caller-owned scratch for embedding gather.
#[derive(Debug)]
pub struct EmbeddingParams<'a> {
    /// `[vocabulary_size, hidden_size]`, F32 or BF16.
    pub table: CudaBufferSpan<'a>,
    /// `[token_count]`, U32.
    pub token_ids: CudaBufferSpan<'a>,
    /// `[token_count, hidden_size]`, matching the table dtype.
    pub output: CudaBufferSpanMut<'a>,
    /// At least 32 bytes of U8 device scratch, reusable between calls.
    pub error_scratch: CudaBufferSpanMut<'a>,
    /// Number of token ids to gather.
    pub token_count: u64,
    /// Number of embedding rows.
    pub vocabulary_size: u64,
    /// Elements per embedding row.
    pub hidden_size: u64,
}

/// Structured failure from [`embedding`].
#[derive(Debug)]
pub enum EmbeddingError {
    /// The first invalid token in input order.
    TokenOutOfRange {
        /// Zero-based position in the token input.
        token_position: u64,
        /// Invalid token id read at that position.
        token_id: u64,
        /// Native CUDA error preserving stage and status metadata.
        source: CudaError,
    },
    /// Validation, launch, synchronization, or native-contract failure.
    Cuda(CudaError),
}

impl EmbeddingError {
    /// Underlying uniform CUDA diagnostic.
    #[must_use]
    pub const fn cuda_error(&self) -> &CudaError {
        match self {
            Self::TokenOutOfRange { source, .. } | Self::Cuda(source) => source,
        }
    }
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenOutOfRange {
                token_position,
                token_id,
                source,
            } => write!(
                formatter,
                "embedding token id {token_id} at position {token_position} is out of range: {source}"
            ),
            Self::Cuda(source) => source.fmt(formatter),
        }
    }
}

impl error::Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        Some(self.cuda_error())
    }
}

impl From<CudaError> for EmbeddingError {
    fn from(error: CudaError) -> Self {
        Self::Cuda(error)
    }
}

/// Gathers token rows and synchronously reports the first out-of-range id.
///
/// The explicit stream is synchronized inside the native ABI before this
/// function returns, making every borrowed buffer safe to reuse immediately.
/// The repeated execution path performs no host or device allocation.
///
/// # Errors
///
/// Returns structured token OOB information or a CUDA validation/execution
/// error. All descriptors are validated before any device write.
pub fn embedding<S: CudaExecutionStream + ?Sized>(
    params: &mut EmbeddingParams<'_>,
    stream: &mut S,
) -> Result<(), EmbeddingError> {
    embedding_with_report_mode(params, stream, 0)
}

/// [`embedding`] inside a command batch without the read-back of its device
/// error report, so the gather can be part of a recorded step. The report (a
/// 32-byte `RileyCudaEmbeddingErrorReport`: `u32` size, `u32` code, `u64`
/// token position, `u64` token id, `u64` zero) stays in `error_scratch`; the
/// caller must copy it back and check its code after the batch has completed.
/// Until then an out-of-range token is undetected (the gather leaves its
/// output untouched in that case).
///
/// # Errors
///
/// Returns a CUDA validation or launch error.
pub fn embedding_deferred_report<S: CudaExecutionStream + ?Sized>(
    params: &mut EmbeddingParams<'_>,
    stream: &mut S,
) -> Result<(), EmbeddingError> {
    embedding_with_report_mode(params, stream, 1)
}

fn embedding_with_report_mode<S: CudaExecutionStream + ?Sized>(
    params: &mut EmbeddingParams<'_>,
    stream: &mut S,
    report_mode: u64,
) -> Result<(), EmbeddingError> {
    const OPERATION: &str = "embedding";
    let stream = execution_stream_mut(stream);
    validate_embedding_params(params, stream)?;

    #[cfg(feature = "cuda")]
    {
        let completion = ffi::embedding_execute(
            params.table.raw(),
            params.token_ids.raw(),
            params.output.raw(),
            params.error_scratch.raw(),
            params.token_count,
            params.vocabulary_size,
            params.hidden_size,
            report_mode,
            &mut stream.native,
        );
        match (completion.result, completion.report.code) {
            (Ok(()), 0) => Ok(()),
            (Err(source), EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE)
                if source.kind() == CudaErrorKind::OutOfRange
                    && completion.report.token_position < params.token_count
                    && completion.report.token_id >= params.vocabulary_size =>
            {
                Err(EmbeddingError::TokenOutOfRange {
                    token_position: completion.report.token_position,
                    token_id: completion.report.token_id,
                    source,
                })
            }
            (Err(source), _) => Err(EmbeddingError::Cuda(source)),
            (Ok(()), code) => Err(EmbeddingError::Cuda(native_contract_error(
                OPERATION,
                format!("native embedding succeeded with unexpected report code {code}"),
            ))),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, report_mode);
        Err(CudaError::unavailable(OPERATION).into())
    }
}

/// Inputs and shape for RMS normalization.
#[derive(Debug)]
pub struct RmsNormParams<'a> {
    /// `[row_count, hidden_size]`, F32 or BF16.
    pub input: CudaBufferSpan<'a>,
    /// `[hidden_size]`, matching input dtype.
    pub weight: CudaBufferSpan<'a>,
    /// `[row_count, hidden_size]`, matching input dtype.
    pub output: CudaBufferSpanMut<'a>,
    /// Number of independent rows.
    pub row_count: u64,
    /// Elements reduced per row.
    pub hidden_size: u64,
    /// Positive finite epsilon added before reciprocal square root.
    pub epsilon: f32,
}

/// Executes `RMSNorm` with FP32 accumulation and a synchronously completing
/// explicit CUDA stream.
///
/// # Errors
///
/// Returns a descriptor, dtype, shape, launch, or synchronization error.
pub fn rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "rms_norm";
    let stream = execution_stream_mut(stream);
    require_nonzero(OPERATION, "hidden_size", params.hidden_size)?;
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "epsilon must be finite and greater than zero",
        ));
    }
    require_float(OPERATION, "input", params.input.dtype)?;
    require_dtype(OPERATION, "weight", params.weight.dtype, params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.input.dtype,
    )?;
    require_capacity(OPERATION, "input", params.input.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len, matrix_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.weight.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::rms_norm_execute(
            params.input.raw(),
            params.weight.raw(),
            params.output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes the byte-exact Hugging Face `SmolLM2` BF16 `RMSNorm` topology.
///
/// This additive entry point is intentionally closed over the reviewed
/// `hidden_size=576`, `epsilon=1e-5`, and at most 8192 rows contract. It never
/// selects the generic `RMSNorm` implementation as a fallback.
///
/// # Errors
///
/// Returns not-supported outside the reviewed contract, or a descriptor,
/// shape, overlap, launch, or synchronization error.
pub fn hugging_face_smollm2_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>, stream: &mut S,
) -> CudaResult<()> {
    execute_hugging_face_smollm2_rms_norm(params, stream, false)
}

fn execute_hugging_face_smollm2_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>, stream: &mut S, singleton_rows: bool,
) -> CudaResult<()> {
    const OPERATION: &str = "hugging_face_smollm2_rms_norm";
    let stream = execution_stream_mut(stream);
    validate_hugging_face_smollm2_rms_norm_contract(
        OPERATION,
        params.input.dtype,
        params.row_count,
        params.hidden_size,
        params.epsilon,
    )?;
    require_dtype(OPERATION, "weight", params.weight.dtype, params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.input.dtype,
    )?;
    require_capacity(OPERATION, "input", params.input.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len, matrix_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.weight.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        let execute = if singleton_rows {
            ffi::hugging_face_smollm2_singleton_rms_norm_execute
        } else {
            ffi::hugging_face_smollm2_rms_norm_execute
        };
        execute(
            params.input.raw(),
            params.weight.raw(),
            params.output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, singleton_rows);
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes the source-bound Hugging Face Qwen BF16 `RMSNorm` topology.
///
/// The additive entry point is intentionally closed over three source-bound
/// diagnostic geometries: `row_count=2051` (cache-free), `row_count=2048`
/// (cache-on prefill), or `row_count=1` (the paired cache-on M1 trace), with
/// `hidden_size=2048` and `epsilon=1e-6`. It reproduces the pinned PyTorch
/// F32 `pow(2).mean` reduction order and never falls back to generic RMSNorm.
/// The ABI symbol remains P2051 while the accepted descriptor set is
/// deliberately explicit and finite.
///
/// # Errors
///
/// Returns not-supported outside the reviewed contract, or a descriptor,
/// shape, overlap, launch, or synchronization error.
pub fn hugging_face_qwen_p2051_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "hugging_face_qwen_p2051_rms_norm";
    let stream = execution_stream_mut(stream);
    validate_hugging_face_qwen_p2051_rms_norm_contract(
        OPERATION,
        params.input.dtype,
        params.row_count,
        params.hidden_size,
        params.epsilon,
    )?;
    require_dtype(OPERATION, "weight", params.weight.dtype, params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.input.dtype,
    )?;
    require_capacity(OPERATION, "input", params.input.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len, matrix_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.weight.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::hugging_face_qwen_p2051_rms_norm_execute(
            params.input.raw(),
            params.weight.raw(),
            params.output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes the bounded, source-pinned Qwen head128 float4 mean order.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn hugging_face_qwen_head128_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "hugging_face_qwen_head128_rms_norm";
    let stream = execution_stream_mut(stream);
    validate_hugging_face_qwen_head128_rms_norm_contract(
        OPERATION,
        params.input.dtype,
        params.row_count,
        params.hidden_size,
        params.epsilon,
    )?;
    require_dtype(OPERATION, "weight", params.weight.dtype, params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.input.dtype,
    )?;
    require_capacity(OPERATION, "input", params.input.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len, matrix_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.weight.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::hugging_face_qwen_head128_rms_norm_execute(
            params.input.raw(),
            params.weight.raw(),
            params.output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes bounded Qwen wide RMSNorm with the actual request row geometry.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn hugging_face_qwen_wide_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    execute_hugging_face_qwen_wide_rms_norm(params, stream, false)
}

#[cfg(feature="cuda-cublas-gemm-probe")]
fn hugging_face_qwen_singleton_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>, stream: &mut S,
) -> CudaResult<()> {
    execute_hugging_face_qwen_wide_rms_norm(params, stream, true)
}

#[cfg(feature="cuda-cublas-gemm-probe")]
fn execute_hugging_face_qwen_wide_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>, stream: &mut S, singleton_rows: bool,
) -> CudaResult<()> {
    const OPERATION: &str = "hugging_face_qwen_wide_rms_norm";
    let stream = execution_stream_mut(stream);
    validate_hugging_face_qwen_wide_rms_norm_contract(
        OPERATION,
        params.input.dtype,
        params.row_count,
        params.hidden_size,
        params.epsilon,
    )?;
    require_dtype(OPERATION, "weight", params.weight.dtype, params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.input.dtype,
    )?;
    require_capacity(OPERATION, "input", params.input.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len, matrix_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.weight.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        let execute = if singleton_rows {
            ffi::hugging_face_qwen_singleton_rms_norm_execute
        } else {
            ffi::hugging_face_qwen_wide_rms_norm_execute
        };
        execute(
            params.input.raw(), params.weight.raw(), params.output.raw(),
            params.row_count, params.hidden_size, params.epsilon, &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes `RMSNorm` with the fixed-contiguous-37-balanced-v1 sum-of-squares
/// order.
///
/// Storage rounding, exceptional values, aliases, and synchronous completion
/// are identical to [`rms_norm`]. This additive entry point never selects the
/// canonical kernel as a fallback.
///
/// # Errors
///
/// Returns not-supported if `hidden_size` exceeds the fixed partial capacity,
/// or a descriptor, dtype, shape, launch, or synchronization error.
pub fn fixed37_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut RmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_rms_norm";
    let stream = execution_stream_mut(stream);
    require_nonzero(OPERATION, "hidden_size", params.hidden_size)?;
    validate_fixed37_axis(OPERATION, params.hidden_size)?;
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "epsilon must be finite and greater than zero",
        ));
    }
    require_float(OPERATION, "input", params.input.dtype)?;
    require_dtype(OPERATION, "weight", params.weight.dtype, params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.input.dtype,
    )?;
    require_capacity(OPERATION, "input", params.input.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len, matrix_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.weight.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::fixed37_rms_norm_execute(
            params.input.raw(),
            params.weight.raw(),
            params.output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Inputs for an elementwise residual addition.
#[derive(Debug)]
pub struct ResidualAddParams<'a> {
    /// Left F32 or BF16 vector.
    pub left: CudaBufferSpan<'a>,
    /// Right vector matching `left`.
    pub right: CudaBufferSpan<'a>,
    /// Output vector matching `left`.
    pub output: CudaBufferSpanMut<'a>,
    /// Elements to add.
    pub element_count: u64,
}

/// Executes elementwise residual addition.
///
/// # Errors
///
/// Returns a descriptor, dtype, range, launch, or synchronization error.
pub fn residual_add<S: CudaExecutionStream + ?Sized>(
    params: &mut ResidualAddParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "residual_add";
    let stream = execution_stream_mut(stream);
    validate_matching_elementwise(
        OPERATION,
        params.left,
        params.right,
        &params.output,
        params.element_count,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.left.buffer,
            params.right.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::residual_add_execute(
            params.left.raw(),
            params.right.raw(),
            params.output.raw(),
            params.element_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Inputs and outputs for the exact residual-add plus `RMSNorm` fusion.
#[derive(Debug)]
pub struct ResidualRmsNormParams<'a> {
    /// Left residual input, F32 or BF16.
    pub left: CudaBufferSpan<'a>,
    /// Right residual input matching `left`.
    pub right: CudaBufferSpan<'a>,
    /// Learned norm weight `[hidden_size]` matching `left`.
    pub weight: CudaBufferSpan<'a>,
    /// Stored residual `[row_count, hidden_size]` matching standalone add.
    pub residual_output: CudaBufferSpanMut<'a>,
    /// Normalized output `[row_count, hidden_size]`.
    pub normalized_output: CudaBufferSpanMut<'a>,
    /// Number of independent rows.
    pub row_count: u64,
    /// Elements reduced per row.
    pub hidden_size: u64,
    /// Positive finite epsilon added before reciprocal square root.
    pub epsilon: f32,
}

/// Executes an `E0` fused residual add and `RMSNorm`.
///
/// The residual is rounded to the storage dtype before the norm reduction,
/// preserving the standalone primitive boundary. The native call launches one
/// kernel and synchronizes once; [`residual_add`] plus [`rms_norm`] remain the
/// exact fallback.
///
/// # Errors
///
/// Returns a descriptor, dtype, shape, overlap, launch, or synchronization
/// error.
pub fn residual_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut ResidualRmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "residual_rms_norm";
    let stream = execution_stream_mut(stream);
    require_nonzero(OPERATION, "hidden_size", params.hidden_size)?;
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "epsilon must be finite and greater than zero",
        ));
    }
    require_float(OPERATION, "left", params.left.dtype)?;
    for (field, dtype) in [
        ("right", params.right.dtype),
        ("weight", params.weight.dtype),
        ("residual_output", params.residual_output.dtype),
        ("normalized_output", params.normalized_output.dtype),
    ] {
        require_dtype(OPERATION, field, dtype, params.left.dtype)?;
    }
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.left.dtype,
    )?;
    require_capacity(OPERATION, "left", params.left.byte_len, matrix_bytes)?;
    require_capacity(OPERATION, "right", params.right.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(
        OPERATION,
        "residual_output",
        params.residual_output.byte_len,
        matrix_bytes,
    )?;
    require_capacity(
        OPERATION,
        "normalized_output",
        params.normalized_output.byte_len,
        matrix_bytes,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.left.buffer,
            params.right.buffer,
            params.weight.buffer,
            params.residual_output.buffer,
            params.normalized_output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::residual_rms_norm_execute(
            params.left.raw(),
            params.right.raw(),
            params.weight.raw(),
            params.residual_output.raw(),
            params.normalized_output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes fused residual add plus the reviewed Hugging Face `SmolLM2`
/// `RMSNorm` topology.
///
/// The residual sum is rounded and stored as BF16 before any square is
/// accumulated, matching the standalone Hugging Face staging boundary.
///
/// # Errors
///
/// Returns not-supported outside the reviewed contract, or a descriptor,
/// shape, overlap, launch, or synchronization error.
pub fn hugging_face_smollm2_residual_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut ResidualRmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "hugging_face_smollm2_residual_rms_norm";
    let stream = execution_stream_mut(stream);
    validate_hugging_face_smollm2_rms_norm_contract(
        OPERATION,
        params.left.dtype,
        params.row_count,
        params.hidden_size,
        params.epsilon,
    )?;
    for (field, dtype) in [
        ("right", params.right.dtype),
        ("weight", params.weight.dtype),
        ("residual_output", params.residual_output.dtype),
        ("normalized_output", params.normalized_output.dtype),
    ] {
        require_dtype(OPERATION, field, dtype, params.left.dtype)?;
    }
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.left.dtype,
    )?;
    require_capacity(OPERATION, "left", params.left.byte_len, matrix_bytes)?;
    require_capacity(OPERATION, "right", params.right.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(
        OPERATION,
        "residual_output",
        params.residual_output.byte_len,
        matrix_bytes,
    )?;
    require_capacity(
        OPERATION,
        "normalized_output",
        params.normalized_output.byte_len,
        matrix_bytes,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.left.buffer,
            params.right.buffer,
            params.weight.buffer,
            params.residual_output.buffer,
            params.normalized_output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::hugging_face_smollm2_residual_rms_norm_execute(
            params.left.raw(),
            params.right.raw(),
            params.weight.raw(),
            params.residual_output.raw(),
            params.normalized_output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes fused residual add plus `RMSNorm` with the fixed37 reduction.
///
/// Residual storage is rounded before its sum-of-squares exactly as in
/// [`residual_rms_norm`]. Only the reduction order changes.
///
/// # Errors
///
/// Returns not-supported if `hidden_size` exceeds the fixed partial capacity,
/// or a descriptor, dtype, shape, overlap, launch, or synchronization error.
pub fn fixed37_residual_rms_norm<S: CudaExecutionStream + ?Sized>(
    params: &mut ResidualRmsNormParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_residual_rms_norm";
    let stream = execution_stream_mut(stream);
    require_nonzero(OPERATION, "hidden_size", params.hidden_size)?;
    validate_fixed37_axis(OPERATION, params.hidden_size)?;
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "epsilon must be finite and greater than zero",
        ));
    }
    require_float(OPERATION, "left", params.left.dtype)?;
    for (field, dtype) in [
        ("right", params.right.dtype),
        ("weight", params.weight.dtype),
        ("residual_output", params.residual_output.dtype),
        ("normalized_output", params.normalized_output.dtype),
    ] {
        require_dtype(OPERATION, field, dtype, params.left.dtype)?;
    }
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.left.dtype,
    )?;
    require_capacity(OPERATION, "left", params.left.byte_len, matrix_bytes)?;
    require_capacity(OPERATION, "right", params.right.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(
        OPERATION,
        "residual_output",
        params.residual_output.byte_len,
        matrix_bytes,
    )?;
    require_capacity(
        OPERATION,
        "normalized_output",
        params.normalized_output.byte_len,
        matrix_bytes,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.left.buffer,
            params.right.buffer,
            params.weight.buffer,
            params.residual_output.buffer,
            params.normalized_output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::fixed37_residual_rms_norm_execute(
            params.left.raw(),
            params.right.raw(),
            params.weight.raw(),
            params.residual_output.raw(),
            params.normalized_output.raw(),
            params.row_count,
            params.hidden_size,
            params.epsilon,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// BF16 logits and F32 output for one full-vector fixed37 log-softmax.
#[derive(Debug)]
pub struct Fixed37LogSoftmaxParams<'a> {
    /// BF16 logits `[element_count]`.
    pub logits: CudaBufferSpan<'a>,
    /// F32 log probabilities `[element_count]`.
    pub output: CudaBufferSpanMut<'a>,
    /// Full logical vocabulary length. This must be non-zero.
    pub element_count: u64,
}

/// Computes full-vector log-softmax using fixed37 max and sum reductions.
///
/// Any NaN input, any `+Inf` maximum, or an all-`-Inf` vector produces quiet
/// NaN in every output. With a finite maximum, individual `-Inf` logits map to
/// `-Inf`. CUDA `fmaxf` signed-zero semantics make `+0` win a `+0`/`-0` max
/// pair. No canonical kernel is selected as a fallback.
///
/// # Errors
///
/// Returns not-supported if the vocabulary exceeds the fixed partial
/// capacity, or a dtype, span, context, launch, or synchronization error.
pub fn fixed37_log_softmax<S: CudaExecutionStream + ?Sized>(
    params: &mut Fixed37LogSoftmaxParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_log_softmax";
    let stream = execution_stream_mut(stream);
    require_nonzero(OPERATION, "element_count", params.element_count)?;
    validate_fixed37_axis(OPERATION, params.element_count)?;
    require_dtype(OPERATION, "logits", params.logits.dtype, CudaDType::BF16)?;
    require_dtype(OPERATION, "output", params.output.dtype, CudaDType::F32)?;
    require_capacity(
        OPERATION,
        "logits",
        params.logits.byte_len,
        required_vector_bytes(OPERATION, params.element_count, CudaDType::BF16)?,
    )?;
    require_capacity(
        OPERATION,
        "output",
        params.output.byte_len,
        required_vector_bytes(OPERATION, params.element_count, CudaDType::F32)?,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[params.logits.buffer, params.output.buffer],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::fixed37_log_softmax_execute(
            params.logits.raw(),
            params.output.raw(),
            params.element_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// A contiguous BF16 matrix and per-column BF16 bias for exact in-place add.
#[derive(Debug)]
pub struct RowBiasAddInPlaceParams<'a> {
    /// Mutable `[row_count, column_count]` BF16 matrix and sole output.
    pub matrix: CudaBufferSpanMut<'a>,
    /// Immutable `[column_count]` BF16 bias vector.
    pub bias: CudaBufferSpan<'a>,
    /// Number of matrix rows. Zero rows are a validated no-op.
    pub row_count: u64,
    /// Number of matrix columns. This must be non-zero.
    pub column_count: u64,
}

/// Adds a per-column BF16 bias to a contiguous BF16 matrix in place.
///
/// Each matrix and bias element is expanded to FP32, added once, and narrowed
/// with BF16 round-to-nearest-even. Validation completes before any write and
/// the repeated execution path performs no allocation. Safe Rust ownership
/// makes the matrix exact-in-place and prevents it from aliasing `bias`; the C
/// ABI independently rejects overlapping raw spans.
///
/// # Errors
///
/// Returns a dtype, zero-column, checked shape/byte overflow, span-capacity,
/// context, launch, or synchronization error.
pub fn row_bias_add_in_place<S: CudaExecutionStream + ?Sized>(
    params: &mut RowBiasAddInPlaceParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "row_bias_add_in_place";
    let stream = execution_stream_mut(stream);
    require_nonzero(OPERATION, "column_count", params.column_count)?;
    require_dtype(OPERATION, "matrix", params.matrix.dtype, CudaDType::BF16)?;
    require_dtype(OPERATION, "bias", params.bias.dtype, CudaDType::BF16)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.column_count,
        CudaDType::BF16,
    )?;
    let bias_bytes = required_vector_bytes(OPERATION, params.column_count, CudaDType::BF16)?;
    require_capacity(OPERATION, "matrix", params.matrix.byte_len, matrix_bytes)?;
    require_capacity(OPERATION, "bias", params.bias.byte_len, bias_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[params.matrix.buffer, params.bias.buffer],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::row_bias_add_in_place_execute(
            params.matrix.raw(),
            params.bias.raw(),
            params.row_count,
            params.column_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Sums `data` over every rank of `group` in place (tensor parallelism).
///
/// `data` holds `element_count` BF16 values in this rank's context; afterwards
/// each element is the rank-ordered FP32 sum of all ranks' values, rounded to
/// BF16, identical on every rank. Every rank must make the same call with the
/// same `element_count` on its own stream; the call only enqueues work and may
/// be recorded in a CUDA graph.
///
/// # Errors
///
/// Fails for a non-BF16 or too small span, a rank outside the group, an
/// element count above the group capacity, a stream of another context than
/// the rank's, or a CUDA launch failure.
pub fn peer_all_reduce_bf16_in_place<S: CudaExecutionStream + ?Sized>(
    group: &crate::CudaPeerAllReduceGroup,
    rank: u32,
    data: &mut CudaBufferSpanMut<'_>,
    element_count: u64,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "peer_all_reduce_bf16_in_place";
    let stream = execution_stream_mut(stream);
    require_dtype(OPERATION, "data", data.dtype, CudaDType::BF16)?;
    let data_bytes = required_vector_bytes(OPERATION, element_count, CudaDType::BF16)?;
    require_capacity(OPERATION, "data", data.byte_len, data_bytes)?;
    if element_count > group.max_elements() {
        return Err(CudaError::out_of_range(
            OPERATION,
            "element_count exceeds the group's staging capacity",
        ));
    }
    let context = group
        .context(rank)
        .ok_or_else(|| CudaError::out_of_range(OPERATION, "rank is outside the group"))?;
    crate::runtime::ensure_same_context(context, &stream.context, OPERATION)?;
    validate_resources(OPERATION, stream, &[data.buffer])?;
    #[cfg(feature = "cuda")]
    {
        group
            .native()
            .execute_bf16(rank, data.raw(), element_count, &mut stream.native)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (rank, data);
        Err(CudaError::unavailable(OPERATION))
    }
}

/// A merged BF16 `Q|K|V` projection matrix, its three biases, and the three
/// contiguous outputs it is split into.
#[derive(Debug)]
pub struct QkvSplitBiasParams<'a> {
    /// Immutable `[row_count, query_width + 2 * key_value_width]` BF16 matrix.
    pub merged: CudaBufferSpan<'a>,
    /// Immutable `[query_width]` BF16 bias.
    pub query_bias: CudaBufferSpan<'a>,
    /// Immutable `[key_value_width]` BF16 key bias.
    pub key_bias: CudaBufferSpan<'a>,
    /// Immutable `[key_value_width]` BF16 value bias.
    pub value_bias: CudaBufferSpan<'a>,
    /// `[row_count, query_width]` BF16 output.
    pub query: CudaBufferSpanMut<'a>,
    /// `[row_count, key_value_width]` BF16 key output.
    pub key: CudaBufferSpanMut<'a>,
    /// `[row_count, key_value_width]` BF16 value output.
    pub value: CudaBufferSpanMut<'a>,
    /// Number of matrix rows. Zero rows are a validated no-op.
    pub row_count: u64,
    /// Query projection width. This must be non-zero.
    pub query_width: u64,
    /// Key and value projection width. This must be non-zero.
    pub key_value_width: u64,
}

/// Splits a merged `Q|K|V` BF16 projection into its three outputs while adding
/// each output's per-column BF16 bias.
///
/// Every element is expanded to FP32, added to its bias once, and narrowed
/// with BF16 round-to-nearest-even: the same arithmetic as
/// [`row_bias_add_in_place`] applied to each output separately.
///
/// # Errors
///
/// Returns a dtype, zero-width, checked shape/byte overflow, span-capacity,
/// overlap, context, launch, or synchronization error.
pub fn qkv_split_bias<S: CudaExecutionStream + ?Sized>(
    params: &mut QkvSplitBiasParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    qkv_split_bias_impl(params, None, stream,false)
}

/// `RoPE` inputs for [`qkv_split_bias_rope`].
#[derive(Debug)]
pub struct QkvRopeParams<'a> {
    /// Immutable F32 cosine table `[table_position_count, head_size / 2]`.
    pub cos: CudaBufferSpan<'a>,
    /// Immutable F32 sine table with the cosine table's shape.
    pub sin: CudaBufferSpan<'a>,
    /// Immutable U32 device positions, one per row.
    pub positions: CudaBufferSpan<'a>,
    /// Host copy of the positions of the leading rows that are rotated,
    /// validated against the table. Rows beyond it get only their value output.
    pub positions_host: &'a [u32],
    /// Head width; the whole head is rotary. Must be even and divide both widths.
    pub head_size: u64,
    /// Number of positions in the tables.
    pub table_position_count: u64,
    /// Defers the launch to the `ragged_paged_kv_cache_write` that follows in
    /// the same command batch, which then runs the split and the cache write
    /// in one kernel with the same stored bytes. That write must take this
    /// call's key and value outputs as its sources.
    pub defer_for_kv_write: bool,
    /// Head-norm form for models without Q/K/V bias (Qwen3): `Some(epsilon)`
    /// applies an RMS norm over every query head and key head before the
    /// rotary embedding instead of adding a bias. `query_bias` and `key_bias`
    /// of the split then hold the BF16 `[head_size]` norm scales, `value_bias`
    /// is not read, and `head_size` must be 128.
    pub head_norm_epsilon: Option<f32>,
}

/// [`qkv_split_bias`] with the indexed BF16 `RoPE` of the query and key outputs
/// applied in the same kernel. `query` and `key` receive the rotated values,
/// bit-identical to the split followed by two indexed `RoPE` calls.
///
/// # Errors
///
/// Returns the errors of [`qkv_split_bias`], or an invalid head size, table
/// capacity, or out-of-table position error.
pub fn qkv_split_bias_rope<S: CudaExecutionStream + ?Sized>(
    params: &mut QkvSplitBiasParams<'_>,
    rope: &QkvRopeParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    qkv_split_bias_impl(params, Some(rope), stream,false)
}

/// Diagnostic HF float4 head128 mean fused with split and stored-BF16 RoPE.
/// Existing span, ownership and completion validation remains shared.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn qkv_split_bias_rope_hf_head128<S:CudaExecutionStream+?Sized>(params:&mut QkvSplitBiasParams<'_>,rope:&QkvRopeParams<'_>,stream:&mut S)->CudaResult<()>{
 validate_hf_head_split_contract(params.row_count,rope.head_size,rope.head_norm_epsilon,rope.defer_for_kv_write)?;
 qkv_split_bias_impl(params,Some(rope),stream,true)
}
/// Defers the existing exact HF128 split into the following cache write.
/// Native validation requires one command batch and a matching stream/sources;
/// the batch retains every source/output lease until the fused launch finishes.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn qkv_split_bias_rope_hf_head128_deferred<S:CudaExecutionStream+?Sized>(params:&mut QkvSplitBiasParams<'_>,rope:&QkvRopeParams<'_>,stream:&mut S)->CudaResult<()>{
    validate_hf_head_split_contract(params.row_count,rope.head_size,rope.head_norm_epsilon,false)?;
    if !rope.defer_for_kv_write{return Err(CudaError::invalid_argument("HF128 deferred split","explicit deferred cache-write selection required"));}
    qkv_split_bias_impl(params,Some(rope),stream,true)
}

#[cfg(feature="cuda-cublas-gemm-probe")]
fn validate_hf_head_split_contract(rows:u64,width:u64,eps:Option<f32>,defer:bool)->CudaResult<()>{
 if rows<=8192 && width==128 && eps.map(f32::to_bits)==Some(1e-6f32.to_bits()) && !defer{return Ok(());}
 Err(CudaError::invalid_argument("HF head128 split","non-deferred head128 rows<=8192 epsilon1e-6 required"))
}
#[cfg(all(test,feature="cuda-cublas-gemm-probe"))]
#[test]
fn hf_head_split_contract_preserves_default_and_rejects_deferred(){
 for rows in [0,1,128,397,512,8192]{assert!(validate_hf_head_split_contract(rows,128,Some(1e-6),false).is_ok());}
 for (rows,width,eps,defer) in [(8193,128,Some(1e-6),false),(1,64,Some(1e-6),false),(1,128,None,false),(1,128,Some(1e-5),false),(1,128,Some(1e-6),true)]{assert!(validate_hf_head_split_contract(rows,width,eps,defer).is_err());}
}

fn qkv_split_bias_impl<S: CudaExecutionStream + ?Sized>(
    params: &mut QkvSplitBiasParams<'_>,
    rope: Option<&QkvRopeParams<'_>>,
    stream: &mut S,
    hf_head_norm:bool,
) -> CudaResult<()> {
    const OPERATION: &str = "qkv_split_bias";
    let stream = execution_stream_mut(stream);
    if let Some(rope) = rope {
        require_nonzero(OPERATION, "head_size", rope.head_size)?;
        require_nonzero(OPERATION, "table_position_count", rope.table_position_count)?;
        if rope.head_size % 2 != 0
            || params.query_width % rope.head_size != 0
            || params.key_value_width % rope.head_size != 0
        {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "head_size must be even and divide the query and key widths",
            ));
        }
        if u64::try_from(rope.positions_host.len()).map_or(true, |rows| rows > params.row_count) {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "positions_host must not hold more positions than rows",
            ));
        }
        crate::batch::validate_indexed_positions(rope.positions_host, rope.table_position_count)?;
        require_dtype(OPERATION, "cos", rope.cos.dtype, CudaDType::F32)?;
        require_dtype(OPERATION, "sin", rope.sin.dtype, CudaDType::F32)?;
        require_dtype(OPERATION, "positions", rope.positions.dtype, CudaDType::U32)?;
        let table_bytes = required_matrix_bytes(
            OPERATION,
            rope.table_position_count,
            rope.head_size / 2,
            CudaDType::F32,
        )?;
        require_capacity(OPERATION, "cos", rope.cos.byte_len, table_bytes)?;
        require_capacity(OPERATION, "sin", rope.sin.byte_len, table_bytes)?;
        require_capacity(
            OPERATION,
            "positions",
            rope.positions.byte_len,
            required_vector_bytes(OPERATION, rope.positions_host.len() as u64, CudaDType::U32)?,
        )?;
        validate_resources(
            OPERATION,
            stream,
            &[rope.cos.buffer, rope.sin.buffer, rope.positions.buffer],
        )?;
    }
    require_nonzero(OPERATION, "query_width", params.query_width)?;
    require_nonzero(OPERATION, "key_value_width", params.key_value_width)?;
    for (name, dtype) in [
        ("merged", params.merged.dtype),
        ("query_bias", params.query_bias.dtype),
        ("key_bias", params.key_bias.dtype),
        ("value_bias", params.value_bias.dtype),
        ("query", params.query.dtype),
        ("key", params.key.dtype),
        ("value", params.value.dtype),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
    let merged_width = params
        .key_value_width
        .checked_mul(2)
        .and_then(|width| width.checked_add(params.query_width))
        .ok_or_else(|| CudaError::out_of_range(OPERATION, "merged QKV width overflows u64"))?;
    let merged_bytes =
        required_matrix_bytes(OPERATION, params.row_count, merged_width, CudaDType::BF16)?;
    let query_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.query_width,
        CudaDType::BF16,
    )?;
    let key_value_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.key_value_width,
        CudaDType::BF16,
    )?;
    let head_norm = rope.and_then(|rope| rope.head_norm_epsilon.map(|_| rope.head_size));
    if let Some(head_size) = head_norm {
        if head_size != 128 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "the head-norm split requires a head size of 128",
            ));
        }
    }
    let query_bias_bytes = required_vector_bytes(
        OPERATION,
        head_norm.unwrap_or(params.query_width),
        CudaDType::BF16,
    )?;
    let key_value_bias_bytes = required_vector_bytes(
        OPERATION,
        head_norm.unwrap_or(params.key_value_width),
        CudaDType::BF16,
    )?;
    require_capacity(OPERATION, "merged", params.merged.byte_len, merged_bytes)?;
    require_capacity(
        OPERATION,
        "query_bias",
        params.query_bias.byte_len,
        query_bias_bytes,
    )?;
    require_capacity(
        OPERATION,
        "key_bias",
        params.key_bias.byte_len,
        key_value_bias_bytes,
    )?;
    require_capacity(
        OPERATION,
        "value_bias",
        params.value_bias.byte_len,
        key_value_bias_bytes,
    )?;
    require_capacity(OPERATION, "query", params.query.byte_len, query_bytes)?;
    require_capacity(OPERATION, "key", params.key.byte_len, key_value_bytes)?;
    require_capacity(OPERATION, "value", params.value.byte_len, key_value_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.merged.buffer,
            params.query_bias.buffer,
            params.key_bias.buffer,
            params.value_bias.buffer,
            params.query.buffer,
            params.key.buffer,
            params.value.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::qkv_split_bias_execute(
            params.merged.raw(),
            params.query_bias.raw(),
            params.key_bias.raw(),
            params.value_bias.raw(),
            params.query.raw(),
            params.key.raw(),
            params.value.raw(),
            params.row_count,
            params.query_width,
            params.key_value_width,
            rope.map(|rope| ffi::RawQkvRope {
                cos: rope.cos.raw(),
                sin: rope.sin.raw(),
                positions: rope.positions.raw(),
                head_size: rope.head_size,
                table_position_count: rope.table_position_count,
                rotated_row_count: rope.positions_host.len() as u64,
                defer_for_kv_write: rope.defer_for_kv_write,
                head_norm_epsilon: rope.head_norm_epsilon,
            }),
            hf_head_norm,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, rope);
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Inputs for elementwise `SiLU` activation.
#[derive(Debug)]
pub struct SiluParams<'a> {
    /// Input F32 or BF16 vector.
    pub input: CudaBufferSpan<'a>,
    /// Output matching input dtype.
    pub output: CudaBufferSpanMut<'a>,
    /// Elements to activate.
    pub element_count: u64,
}

/// Executes `x / (1 + exp(-x))` elementwise.
///
/// # Errors
///
/// Returns a descriptor, dtype, range, launch, or synchronization error.
pub fn silu<S: CudaExecutionStream + ?Sized>(
    params: &mut SiluParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "silu";
    let stream = execution_stream_mut(stream);
    require_float(OPERATION, "input", params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let required = required_vector_bytes(OPERATION, params.element_count, params.input.dtype)?;
    require_capacity(OPERATION, "input", params.input.byte_len, required)?;
    require_capacity(OPERATION, "output", params.output.byte_len, required)?;
    validate_resources(
        OPERATION,
        stream,
        &[params.input.buffer, params.output.buffer],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::silu_execute(
            params.input.raw(),
            params.output.raw(),
            params.element_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Inputs for multiplying an already-SiLU-activated gate by the up branch.
#[derive(Debug)]
pub struct GatedMultiplyParams<'a> {
    /// Already activated F32 or BF16 gate.
    pub activated_gate: CudaBufferSpan<'a>,
    /// Up branch matching the gate dtype.
    pub up: CudaBufferSpan<'a>,
    /// Product matching the gate dtype.
    pub output: CudaBufferSpanMut<'a>,
    /// Elements to multiply.
    pub element_count: u64,
}

/// Executes plain `activated_gate * up`; `SiLU` fusion is deliberately deferred.
///
/// # Errors
///
/// Returns a descriptor, dtype, range, launch, or synchronization error.
pub fn gated_multiply<S: CudaExecutionStream + ?Sized>(
    params: &mut GatedMultiplyParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    gated_multiply_impl(params, stream, false)
}

/// Executes `silu(raw_gate) * up` in one BF16 kernel, bit-identical to [`silu`]
/// followed by [`gated_multiply`]; `activated_gate` carries the raw gate.
///
/// # Errors
///
/// Returns a descriptor, dtype, range, launch, or synchronization error.
pub fn silu_gated_multiply<S: CudaExecutionStream + ?Sized>(
    params: &mut GatedMultiplyParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    gated_multiply_impl(params, stream, true)
}

/// Inputs for `silu(gate) * up` over the output of one merged gate|up GEMM.
#[derive(Debug)]
pub struct MergedGatedMultiplyParams<'a> {
    /// BF16 rows of `2 * row_width` values: raw gate, then up.
    pub merged: CudaBufferSpan<'a>,
    /// BF16 product, `row_width` values per row.
    pub output: CudaBufferSpanMut<'a>,
    /// Output elements (`rows * row_width`).
    pub element_count: u64,
    /// Width of the gate (and of the up) projection.
    pub row_width: u64,
}

/// Executes `silu(raw_gate) * up` where both operands sit in one row-merged
/// buffer; the arithmetic per element is that of [`silu_gated_multiply`].
///
/// # Errors
///
/// Returns a descriptor, dtype, range, launch, or synchronization error.
pub fn silu_gated_multiply_merged_rows<S: CudaExecutionStream + ?Sized>(
    params: &mut MergedGatedMultiplyParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "silu_gated_multiply_merged_rows";
    let stream = execution_stream_mut(stream);
    require_dtype(OPERATION, "merged", params.merged.dtype, CudaDType::BF16)?;
    require_dtype(OPERATION, "output", params.output.dtype, CudaDType::BF16)?;
    require_nonzero(OPERATION, "row_width", params.row_width)?;
    if params.element_count % params.row_width != 0 || params.element_count > 0x7fff_ffff {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "element_count must be a multiple of row_width below 2^31",
        ));
    }
    let required = required_vector_bytes(OPERATION, params.element_count, CudaDType::BF16)?;
    require_capacity(OPERATION, "merged", params.merged.byte_len, required * 2)?;
    require_capacity(OPERATION, "output", params.output.byte_len, required)?;
    validate_resources(
        OPERATION,
        stream,
        &[params.merged.buffer, params.output.buffer],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::gated_multiply_execute(
            params.merged.raw(),
            params.merged.raw(),
            params.output.raw(),
            params.element_count,
            true,
            params.row_width,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

fn gated_multiply_impl<S: CudaExecutionStream + ?Sized>(
    params: &mut GatedMultiplyParams<'_>,
    stream: &mut S,
    raw_gate: bool,
) -> CudaResult<()> {
    const OPERATION: &str = "gated_multiply";
    let stream = execution_stream_mut(stream);
    validate_matching_elementwise(
        OPERATION,
        params.activated_gate,
        params.up,
        &params.output,
        params.element_count,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.activated_gate.buffer,
            params.up.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::gated_multiply_execute(
            params.activated_gate.raw(),
            params.up.raw(),
            params.output.raw(),
            params.element_count,
            raw_gate,
            0,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, raw_gate);
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Cold preparation inputs for Llama `RoPE` trigonometric tables.
#[derive(Debug)]
pub struct RopeTableParams<'a> {
    /// F32 angle table replaced in place with its cosine.
    pub angles_cos: CudaBufferSpanMut<'a>,
    /// F32 output receiving the sine of each input angle.
    pub sin: CudaBufferSpanMut<'a>,
    /// Number of logical angle elements in both spans.
    pub element_count: u64,
}

/// Replaces an F32 angle table with CUDA cosine and writes its CUDA sine table.
///
/// This allocation-free cold operation preserves the angle buffer until the
/// sine launch has consumed it, then computes cosine in place. It is intended
/// to reproduce device-side framework trigonometry before repeated `RoPE` use.
///
/// # Errors
///
/// Returns a descriptor, dtype, range, overlap, launch, or synchronization
/// error.
pub fn rope_table<S: CudaExecutionStream + ?Sized>(
    params: &mut RopeTableParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "rope_table";
    let stream = execution_stream_mut(stream);
    require_dtype(
        OPERATION,
        "angles_cos",
        params.angles_cos.dtype,
        CudaDType::F32,
    )?;
    require_dtype(OPERATION, "sin", params.sin.dtype, CudaDType::F32)?;
    let required = required_vector_bytes(OPERATION, params.element_count, CudaDType::F32)?;
    require_capacity(
        OPERATION,
        "angles_cos",
        params.angles_cos.byte_len,
        required,
    )?;
    require_capacity(OPERATION, "sin", params.sin.byte_len, required)?;
    validate_resources(
        OPERATION,
        stream,
        &[params.angles_cos.buffer, params.sin.buffer],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::rope_table_execute(
            params.angles_cos.raw(),
            params.sin.raw(),
            params.element_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Standard non-interleaved Llama `RoPE` invocation.
#[derive(Debug)]
pub struct RopeParams<'a> {
    /// `[token_count, head_count, head_size]`, F32 or BF16.
    pub input: CudaBufferSpan<'a>,
    /// `[table_position_count, rotary_dimension / 2]`, F32.
    pub cos: CudaBufferSpan<'a>,
    /// `[table_position_count, rotary_dimension / 2]`, F32.
    pub sin: CudaBufferSpan<'a>,
    /// Output matching input dtype and shape.
    pub output: CudaBufferSpanMut<'a>,
    /// Number of consecutive positions in the input.
    pub token_count: u64,
    /// Number of query or key heads.
    pub head_count: u64,
    /// Elements per head.
    pub head_size: u64,
    /// Even prefix dimension rotated as two contiguous halves.
    pub rotary_dimension: u64,
    /// Rows in each cosine/sine table.
    pub table_position_count: u64,
    /// First table row used for the input.
    pub position_offset: u64,
}

/// Applies standard non-interleaved Llama `RoPE` with FP32 trigonometric tables.
///
/// # Errors
///
/// Returns a descriptor, dtype, shape, position-range, launch, or
/// synchronization error.
pub fn rope(params: &mut RopeParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "rope";
    require_nonzero(OPERATION, "head_count", params.head_count)?;
    require_nonzero(OPERATION, "head_size", params.head_size)?;
    require_nonzero(OPERATION, "rotary_dimension", params.rotary_dimension)?;
    if params.rotary_dimension > params.head_size || params.rotary_dimension % 2 != 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "rotary_dimension must be even and no larger than head_size",
        ));
    }
    let position_end = params
        .position_offset
        .checked_add(params.token_count)
        .ok_or_else(|| CudaError::out_of_range(OPERATION, "position range overflow"))?;
    if position_end > params.table_position_count {
        return Err(CudaError::out_of_range(
            OPERATION,
            format!(
                "positions {}..{} exceed table_position_count {}",
                params.position_offset, position_end, params.table_position_count
            ),
        ));
    }
    require_float(OPERATION, "input", params.input.dtype)?;
    require_dtype(OPERATION, "cos", params.cos.dtype, CudaDType::F32)?;
    require_dtype(OPERATION, "sin", params.sin.dtype, CudaDType::F32)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let logical_elements = checked_product3(
        OPERATION,
        params.token_count,
        params.head_count,
        params.head_size,
    )?;
    let input_bytes = required_vector_bytes(OPERATION, logical_elements, params.input.dtype)?;
    require_capacity(OPERATION, "input", params.input.byte_len, input_bytes)?;
    require_capacity(OPERATION, "output", params.output.byte_len, input_bytes)?;
    let table_elements = checked_product(
        OPERATION,
        params.table_position_count,
        params.rotary_dimension / 2,
    )?;
    let table_bytes = required_vector_bytes(OPERATION, table_elements, CudaDType::F32)?;
    require_capacity(OPERATION, "cos", params.cos.byte_len, table_bytes)?;
    require_capacity(OPERATION, "sin", params.sin.byte_len, table_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.cos.buffer,
            params.sin.buffer,
            params.output.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::rope_execute(
            params.input.raw(),
            params.cos.raw(),
            params.sin.raw(),
            params.output.raw(),
            params.token_count,
            params.head_count,
            params.head_size,
            params.rotary_dimension,
            params.table_position_count,
            params.position_offset,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Inputs for an explicit BF16/F32 storage conversion.
#[derive(Debug)]
pub struct CastParams<'a> {
    /// BF16 or F32 input.
    pub input: CudaBufferSpan<'a>,
    /// The opposite BF16/F32 dtype.
    pub output: CudaBufferSpanMut<'a>,
    /// Elements to convert.
    pub element_count: u64,
}

/// Converts BF16 to F32 or F32 to BF16 explicitly.
///
/// # Errors
///
/// Returns a descriptor, conversion-pair, range, launch, or synchronization
/// error.
pub fn cast(params: &mut CastParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "cast";
    if !matches!(
        (params.input.dtype, params.output.dtype),
        (CudaDType::BF16, CudaDType::F32) | (CudaDType::F32, CudaDType::BF16)
    ) {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "cast supports only BF16->F32 or F32->BF16",
        ));
    }
    require_capacity(
        OPERATION,
        "input",
        params.input.byte_len,
        required_vector_bytes(OPERATION, params.element_count, params.input.dtype)?,
    )?;
    require_capacity(
        OPERATION,
        "output",
        params.output.byte_len,
        required_vector_bytes(OPERATION, params.element_count, params.output.dtype)?,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[params.input.buffer, params.output.buffer],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::cast_execute(
            params.input.raw(),
            params.output.raw(),
            params.element_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Inputs and row results for deterministic greedy BF16 selection.
#[derive(Debug)]
pub struct Bf16ArgmaxParams<'a> {
    /// Contiguous BF16 `[row_count, vocabulary_size]` logits.
    pub logits: CudaBufferSpan<'a>,
    /// U32 `[row_count, 2]` storage for [`Bf16ArgmaxResult`] records.
    pub results: CudaBufferSpanMut<'a>,
    /// Number of independent logits rows.
    pub row_count: u64,
    /// Non-zero logits count per row, no larger than `u32::MAX`.
    pub vocabulary_size: u64,
}

/// Selects one deterministic greedy token per BF16 logits row.
///
/// All logits in a row must be finite. A row containing any NaN or infinity
/// writes `{ token_id: u32::MAX, status: BF16_ARGMAX_STATUS_NON_FINITE }`.
/// Finite ties select the lower token id. This API has no RNG input or state.
/// Within a [`crate::CudaCommandBatch`], the kernel is enqueued without adding
/// an intermediate stream synchronization.
///
/// # Errors
///
/// Returns a dtype, shape, capacity, context, busy-state, launch, or completion
/// error. Non-finite row values are data-plane statuses, not call failures.
pub fn deterministic_bf16_argmax<S: CudaExecutionStream + ?Sized>(
    params: &mut Bf16ArgmaxParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    bf16_argmax(params, 0, stream)
}

/// [`deterministic_bf16_argmax`] restricted to token IDs below
/// `candidate_count`. Every column of a row is still checked for finiteness;
/// columns at or past the bound can never be selected. This serves models
/// whose logit row is wider than the tokenizer's addressable vocabulary.
///
/// # Errors
///
/// As [`deterministic_bf16_argmax`], plus invalid-argument for a zero bound
/// or one past `vocabulary_size`.
pub fn deterministic_bf16_argmax_bounded<S: CudaExecutionStream + ?Sized>(
    params: &mut Bf16ArgmaxParams<'_>,
    candidate_count: u64,
    stream: &mut S,
) -> CudaResult<()> {
    if candidate_count == 0 || candidate_count > params.vocabulary_size {
        return Err(CudaError::invalid_argument(
            "deterministic_bf16_argmax_bounded",
            "candidate_count must be within 1..=vocabulary_size",
        ));
    }
    bf16_argmax(params, candidate_count, stream)
}

/// Where [`deterministic_bf16_argmax_fp32_ties`] finds each output row's
/// normalized hidden state and the LM head that produced its logits.
#[derive(Debug)]
pub struct Bf16ArgmaxTieSource<'a> {
    /// BF16 `[hidden_rows, hidden_size]` normalized hidden states.
    pub hidden: CudaBufferSpan<'a>,
    /// BF16 `[vocabulary_size, hidden_size]` LM-head weight.
    pub weight: CudaBufferSpan<'a>,
    /// U32 `[row_count]` hidden row of each output row; ignored when
    /// `rows_gathered` (the hidden rows are already in output order) but
    /// still validated.
    pub row_indices: CudaBufferSpan<'a>,
    pub hidden_size: u64,
    pub hidden_rows: u64,
    pub rows_gathered: bool,
    /// Rescore every logit within this many BF16 steps (at the row
    /// maximum's exponent) of the maximum, not only exact ties; at most 16.
    pub window_ulps: u32,
}

/// [`deterministic_bf16_argmax`] (optionally bounded by `candidate_count`,
/// zero for none) whose row-maximum ties are broken in FP32: the tokens
/// sharing a row's largest BF16 logit are rescored as FP32 dot products of
/// the row's hidden state and their LM-head rows, in a fixed order, and the
/// largest wins (lower token id on equal scores). BF16 rounding of the LM
/// head's FP32 accumulators cannot reorder logits, only merge them, so rows
/// without a tie select the same token as the plain argmax. The selection
/// of a row depends only on that row.
///
/// # Errors
///
/// As [`deterministic_bf16_argmax`], plus dtype and capacity errors of the
/// tie source.
pub fn deterministic_bf16_argmax_fp32_ties<S: CudaExecutionStream + ?Sized>(
    params: &mut Bf16ArgmaxParams<'_>,
    source: &Bf16ArgmaxTieSource<'_>,
    candidate_count: u64,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "deterministic_bf16_argmax_fp32_ties";
    if candidate_count > params.vocabulary_size {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "candidate_count must be at most vocabulary_size",
        ));
    }
    if source.hidden_size == 0 || source.window_ulps > 16 || (source.rows_gathered && params.row_count > source.hidden_rows) {
        return Err(CudaError::invalid_argument(OPERATION, "invalid tie source shape"));
    }
    let stream = execution_stream_mut(stream);
    validate_bf16_argmax_descriptor(
        params.logits.dtype,
        params.logits.byte_len,
        params.results.dtype,
        params.results.byte_len,
        params.row_count,
        params.vocabulary_size,
    )?;
    let hidden_bytes = source
        .hidden_rows
        .checked_mul(source.hidden_size)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| CudaError::invalid_argument(OPERATION, "hidden shape overflows"))?;
    let weight_bytes = params
        .vocabulary_size
        .checked_mul(source.hidden_size)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| CudaError::invalid_argument(OPERATION, "weight shape overflows"))?;
    for (name, dtype, want, bytes, need) in [
        ("hidden", source.hidden.dtype, CudaDType::BF16, source.hidden.byte_len, hidden_bytes),
        ("weight", source.weight.dtype, CudaDType::BF16, source.weight.byte_len, weight_bytes),
        ("row_indices", source.row_indices.dtype, CudaDType::U32, source.row_indices.byte_len, params.row_count * 4),
    ] {
        if dtype != want {
            return Err(CudaError::invalid_argument(OPERATION, "tie source dtype mismatch"));
        }
        require_capacity(OPERATION, name, bytes, need)?;
    }
    validate_resources(
        OPERATION,
        stream,
        &[
            params.logits.buffer,
            params.results.buffer,
            source.hidden.buffer,
            source.weight.buffer,
            source.row_indices.buffer,
        ],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::bf16_argmax_fp32_ties_execute(
            params.logits.raw(),
            params.results.raw(),
            source.hidden.raw(),
            source.weight.raw(),
            source.row_indices.raw(),
            params.row_count,
            params.vocabulary_size,
            candidate_count,
            source.hidden_size,
            source.hidden_rows,
            source.rows_gathered,
            source.window_ulps,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, source, candidate_count);
        Err(CudaError::unavailable(OPERATION))
    }
}

fn bf16_argmax<S: CudaExecutionStream + ?Sized>(
    params: &mut Bf16ArgmaxParams<'_>,
    candidate_count: u64,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "deterministic_bf16_argmax";
    let stream = execution_stream_mut(stream);
    validate_bf16_argmax_descriptor(
        params.logits.dtype,
        params.logits.byte_len,
        params.results.dtype,
        params.results.byte_len,
        params.row_count,
        params.vocabulary_size,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[params.logits.buffer, params.results.buffer],
    )?;
    #[cfg(feature = "cuda")]
    {
        ffi::bf16_argmax_execute(
            params.logits.raw(),
            params.results.raw(),
            params.row_count,
            params.vocabulary_size,
            candidate_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, candidate_count);
        Err(CudaError::unavailable(OPERATION))
    }
}

fn validate_bf16_argmax_descriptor(
    logits_dtype: CudaDType,
    logits_byte_len: u64,
    results_dtype: CudaDType,
    results_byte_len: u64,
    row_count: u64,
    vocabulary_size: u64,
) -> CudaResult<()> {
    const OPERATION: &str = "deterministic_bf16_argmax";
    require_dtype(OPERATION, "logits", logits_dtype, CudaDType::BF16)?;
    require_dtype(OPERATION, "results", results_dtype, CudaDType::U32)?;
    require_nonzero(OPERATION, "vocabulary_size", vocabulary_size)?;
    if vocabulary_size > u64::from(u32::MAX) {
        return Err(CudaError::out_of_range(
            OPERATION,
            "vocabulary_size exceeds the U32 token-id contract",
        ));
    }
    let logits_required =
        required_matrix_bytes(OPERATION, row_count, vocabulary_size, CudaDType::BF16)?;
    let result_elements = checked_product(OPERATION, row_count, BF16_ARGMAX_RESULT_U32_WORDS)?;
    let results_required = required_vector_bytes(OPERATION, result_elements, CudaDType::U32)?;
    require_capacity(OPERATION, "logits", logits_byte_len, logits_required)?;
    require_capacity(OPERATION, "results", results_byte_len, results_required)
}

fn validate_embedding_params(params: &EmbeddingParams<'_>, stream: &CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "embedding";
    require_nonzero(OPERATION, "vocabulary_size", params.vocabulary_size)?;
    require_nonzero(OPERATION, "hidden_size", params.hidden_size)?;
    require_float(OPERATION, "table", params.table.dtype)?;
    require_dtype(
        OPERATION,
        "token_ids",
        params.token_ids.dtype,
        CudaDType::U32,
    )?;
    require_dtype(OPERATION, "output", params.output.dtype, params.table.dtype)?;
    require_dtype(
        OPERATION,
        "error_scratch",
        params.error_scratch.dtype,
        CudaDType::U8,
    )?;
    require_capacity(
        OPERATION,
        "table",
        params.table.byte_len,
        required_matrix_bytes(
            OPERATION,
            params.vocabulary_size,
            params.hidden_size,
            params.table.dtype,
        )?,
    )?;
    require_capacity(
        OPERATION,
        "token_ids",
        params.token_ids.byte_len,
        required_vector_bytes(OPERATION, params.token_count, CudaDType::U32)?,
    )?;
    require_capacity(
        OPERATION,
        "output",
        params.output.byte_len,
        required_matrix_bytes(
            OPERATION,
            params.token_count,
            params.hidden_size,
            params.output.dtype,
        )?,
    )?;
    require_capacity(
        OPERATION,
        "error_scratch",
        params.error_scratch.byte_len,
        EMBEDDING_ERROR_SCRATCH_BYTES,
    )?;
    if params.error_scratch.byte_offset % 8 != 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "error_scratch byte offset must be aligned to 8 bytes",
        ));
    }
    validate_resources(
        OPERATION,
        stream,
        &[
            params.table.buffer,
            params.token_ids.buffer,
            params.output.buffer,
            params.error_scratch.buffer,
        ],
    )
}

fn validate_span(
    operation: &'static str,
    allocation_byte_len: u64,
    dtype: CudaDType,
    byte_offset: u64,
    byte_len: u64,
) -> CudaResult<()> {
    let scalar_bytes = dtype.size_bytes();
    if byte_offset % scalar_bytes != 0 || byte_len % scalar_bytes != 0 {
        return Err(CudaError::invalid_argument(
            operation,
            format!("{dtype} span offset and length must be aligned to {scalar_bytes} bytes"),
        ));
    }
    if byte_offset <= allocation_byte_len && byte_len <= allocation_byte_len - byte_offset {
        Ok(())
    } else {
        Err(CudaError::out_of_range(
            operation,
            format!(
                "range offset={byte_offset} byte_len={byte_len} exceeds allocation length {allocation_byte_len}"
            ),
        ))
    }
}

fn validate_resources(
    operation: &'static str,
    stream: &CudaStream,
    buffers: &[&CudaDeviceBuffer],
) -> CudaResult<()> {
    for buffer in buffers {
        ensure_same_context(buffer.context_owner(), &stream.context, operation)?;
        buffer.ensure_idle_for_operation(operation)?;
    }
    Ok(())
}

fn validate_matching_elementwise(
    operation: &'static str,
    left: CudaBufferSpan<'_>,
    right: CudaBufferSpan<'_>,
    output: &CudaBufferSpanMut<'_>,
    element_count: u64,
) -> CudaResult<()> {
    require_float(operation, "left", left.dtype)?;
    require_dtype(operation, "right", right.dtype, left.dtype)?;
    require_dtype(operation, "output", output.dtype, left.dtype)?;
    let required = required_vector_bytes(operation, element_count, left.dtype)?;
    require_capacity(operation, "left", left.byte_len, required)?;
    require_capacity(operation, "right", right.byte_len, required)?;
    require_capacity(operation, "output", output.byte_len, required)
}

fn require_nonzero(operation: &'static str, field: &'static str, value: u64) -> CudaResult<()> {
    if value == 0 {
        Err(CudaError::invalid_argument(
            operation,
            format!("{field} must be greater than zero"),
        ))
    } else {
        Ok(())
    }
}

fn require_float(
    operation: &'static str,
    field: &'static str,
    actual: CudaDType,
) -> CudaResult<()> {
    if actual.is_float() {
        Ok(())
    } else {
        Err(CudaError::invalid_argument(
            operation,
            format!("{field} must be f32 or bf16, got {actual}"),
        ))
    }
}

fn require_dtype(
    operation: &'static str,
    field: &'static str,
    actual: CudaDType,
    expected: CudaDType,
) -> CudaResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(CudaError::invalid_argument(
            operation,
            format!("{field} must be {expected}, got {actual}"),
        ))
    }
}

fn require_capacity(
    operation: &'static str,
    field: &'static str,
    actual: u64,
    required: u64,
) -> CudaResult<()> {
    if actual >= required {
        Ok(())
    } else {
        Err(CudaError::out_of_range(
            operation,
            format!("{field} requires {required} bytes, span exposes {actual}"),
        ))
    }
}

fn required_vector_bytes(
    operation: &'static str,
    element_count: u64,
    dtype: CudaDType,
) -> CudaResult<u64> {
    element_count
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| {
            CudaError::out_of_range(operation, "element byte-length arithmetic overflow")
        })
}

fn required_matrix_bytes(
    operation: &'static str,
    rows: u64,
    columns: u64,
    dtype: CudaDType,
) -> CudaResult<u64> {
    required_vector_bytes(operation, checked_product(operation, rows, columns)?, dtype)
}

fn checked_product(operation: &'static str, left: u64, right: u64) -> CudaResult<u64> {
    left.checked_mul(right)
        .ok_or_else(|| CudaError::out_of_range(operation, "shape arithmetic overflow"))
}

fn checked_product3(
    operation: &'static str,
    first: u64,
    second: u64,
    third: u64,
) -> CudaResult<u64> {
    checked_product(operation, checked_product(operation, first, second)?, third)
}

fn validate_fixed37_axis(operation: &'static str, element_count: u64) -> CudaResult<()> {
    if element_count > FIXED37_MAX_REDUCTION_ELEMENTS {
        return Err(CudaError::new(
            CudaErrorKind::NotSupported,
            CudaErrorDomain::Rust,
            CudaErrorStage::Validation,
            0,
            operation,
            format!(
                "reduction axis {element_count} exceeds the fixed37 limit {FIXED37_MAX_REDUCTION_ELEMENTS}"
            ),
        ));
    }
    Ok(())
}

fn validate_hugging_face_smollm2_rms_norm_contract(
    operation: &'static str,
    dtype: CudaDType,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
) -> CudaResult<()> {
    if dtype == CudaDType::BF16
        && row_count <= HUGGING_FACE_SMOLLM2_RMS_NORM_MAX_ROWS
        && hidden_size == HUGGING_FACE_SMOLLM2_RMS_NORM_HIDDEN_SIZE
        && epsilon.to_bits() == HUGGING_FACE_SMOLLM2_RMS_NORM_EPSILON_BITS
    {
        return Ok(());
    }
    Err(CudaError::new(
        CudaErrorKind::NotSupported,
        CudaErrorDomain::Rust,
        CudaErrorStage::Validation,
        0,
        operation,
        "the reviewed path requires BF16, hidden_size=576, row_count<=8192, and epsilon=1e-5 exactly",
    ))
}

fn validate_hugging_face_qwen_p2051_rms_norm_contract(
    operation: &'static str,
    dtype: CudaDType,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
) -> CudaResult<()> {
    if dtype == CudaDType::BF16
        && matches!(
            row_count,
            HUGGING_FACE_QWEN_P2051_RMS_NORM_ROW_COUNT
                | HUGGING_FACE_QWEN_P2048_CACHE_ON_RMS_NORM_ROW_COUNT
                | HUGGING_FACE_QWEN_P2048_CACHE_ON_M1_RMS_NORM_ROW_COUNT
        )
        && hidden_size == HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE
        && epsilon.to_bits() == HUGGING_FACE_QWEN_P2051_RMS_NORM_EPSILON_BITS
    {
        return Ok(());
    }
    Err(CudaError::new(
        CudaErrorKind::NotSupported,
        CudaErrorDomain::Rust,
        CudaErrorStage::Validation,
        0,
        operation,
        "the Qwen diagnostic path requires BF16, row_count=2051, 2048, or 1, hidden_size=2048, and epsilon=1e-6 exactly",
    ))
}

#[cfg(feature = "cuda")]
fn native_contract_error(operation: &'static str, message: impl Into<String>) -> CudaError {
    CudaError::new(
        CudaErrorKind::Internal,
        CudaErrorDomain::Internal,
        CudaErrorStage::Synchronize,
        0,
        operation,
        message,
    )
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::{
        BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE, BF16_ARGMAX_STATUS_SUCCESS,
        Bf16ArgmaxResult, CudaDType, HUGGING_FACE_QWEN_P2048_CACHE_ON_M1_RMS_NORM_ROW_COUNT,
        HUGGING_FACE_QWEN_P2048_CACHE_ON_RMS_NORM_ROW_COUNT,
        HUGGING_FACE_QWEN_P2051_RMS_NORM_EPSILON_BITS,
        HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE, HUGGING_FACE_QWEN_P2051_RMS_NORM_ROW_COUNT,
        checked_product3, required_matrix_bytes, validate_bf16_argmax_descriptor,
        validate_fixed37_axis, validate_hugging_face_qwen_p2051_rms_norm_contract, validate_span,
    };
    use crate::{CudaErrorKind, CudaErrorStage};

    #[test]
    fn dtype_sizes_are_fixed_for_the_c_abi() {
        assert_eq!(CudaDType::F32.size_bytes(), 4);
        assert_eq!(CudaDType::BF16.size_bytes(), 2);
        assert_eq!(CudaDType::U32.size_bytes(), 4);
        assert_eq!(CudaDType::U16.size_bytes(), 2);
        assert_eq!(CudaDType::U8.size_bytes(), 1);
    }

    #[test]
    fn fixed37_axis_limit_fails_as_not_supported() {
        validate_fixed37_axis("test fixed37 axis", 151_552).expect("maximum axis is supported");
        let error = validate_fixed37_axis("test fixed37 axis", 151_553)
            .expect_err("one element above the maximum must fail");
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert_eq!(error.stage(), CudaErrorStage::Validation);
    }

    #[test]
    fn qwen_rms_norm_profile_is_closed_over_its_source_bound_geometries() {
        let epsilon = f32::from_bits(HUGGING_FACE_QWEN_P2051_RMS_NORM_EPSILON_BITS);
        validate_hugging_face_qwen_p2051_rms_norm_contract(
            "test qwen P2051 RMSNorm",
            CudaDType::BF16,
            HUGGING_FACE_QWEN_P2051_RMS_NORM_ROW_COUNT,
            HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE,
            epsilon,
        )
        .expect("exact Qwen P2051 geometry must be accepted");
        validate_hugging_face_qwen_p2051_rms_norm_contract(
            "test qwen P2048 cache-on RMSNorm",
            CudaDType::BF16,
            HUGGING_FACE_QWEN_P2048_CACHE_ON_RMS_NORM_ROW_COUNT,
            HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE,
            epsilon,
        )
        .expect("exact Qwen P2048 cache-on geometry must be accepted");
        validate_hugging_face_qwen_p2051_rms_norm_contract(
            "test qwen P2048 cache-on M1 RMSNorm",
            CudaDType::BF16,
            HUGGING_FACE_QWEN_P2048_CACHE_ON_M1_RMS_NORM_ROW_COUNT,
            HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE,
            epsilon,
        )
        .expect("exact Qwen P2048 cache-on M1 geometry must be accepted");
        for (dtype, rows, hidden, candidate_epsilon) in [
            (
                CudaDType::F32,
                HUGGING_FACE_QWEN_P2051_RMS_NORM_ROW_COUNT,
                HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE,
                epsilon,
            ),
            (
                CudaDType::BF16,
                HUGGING_FACE_QWEN_P2048_CACHE_ON_RMS_NORM_ROW_COUNT - 1,
                HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE,
                epsilon,
            ),
            (
                CudaDType::BF16,
                HUGGING_FACE_QWEN_P2051_RMS_NORM_ROW_COUNT,
                HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE - 1,
                epsilon,
            ),
            (
                CudaDType::BF16,
                HUGGING_FACE_QWEN_P2051_RMS_NORM_ROW_COUNT,
                HUGGING_FACE_QWEN_P2051_RMS_NORM_HIDDEN_SIZE,
                f32::from_bits(HUGGING_FACE_QWEN_P2051_RMS_NORM_EPSILON_BITS + 1),
            ),
        ] {
            let error = validate_hugging_face_qwen_p2051_rms_norm_contract(
                "test qwen P2051 RMSNorm",
                dtype,
                rows,
                hidden,
                candidate_epsilon,
            )
            .expect_err("every non-exact Qwen P2051 descriptor must fail closed");
            assert_eq!(error.kind(), CudaErrorKind::NotSupported);
            assert_eq!(error.stage(), CudaErrorStage::Validation);
        }
    }

    #[test]
    fn typed_span_validation_checks_alignment_and_half_open_bounds() {
        validate_span("test", 16, CudaDType::BF16, 16, 0).expect("empty tail span is valid");
        let alignment = validate_span("test", 16, CudaDType::F32, 2, 4)
            .expect_err("misaligned offset must fail");
        assert_eq!(alignment.kind(), CudaErrorKind::InvalidArgument);
        assert_eq!(alignment.stage(), CudaErrorStage::Validation);

        let range = validate_span("test", 16, CudaDType::U8, 12, 5)
            .expect_err("range beyond allocation must fail");
        assert_eq!(range.kind(), CudaErrorKind::OutOfRange);
    }

    #[test]
    fn shape_byte_arithmetic_is_checked() {
        assert_eq!(
            required_matrix_bytes("test", 3, 5, CudaDType::BF16).unwrap(),
            30
        );
        assert!(checked_product3("test", u64::MAX, 2, 1).is_err());
    }

    #[test]
    fn bf16_argmax_result_layout_and_status_contract_are_fixed() {
        assert_eq!(size_of::<Bf16ArgmaxResult>(), 8);
        assert_eq!(offset_of!(Bf16ArgmaxResult, token_id), 0);
        assert_eq!(offset_of!(Bf16ArgmaxResult, status), 4);
        assert_eq!(BF16_ARGMAX_STATUS_SUCCESS, 0);
        assert_eq!(BF16_ARGMAX_STATUS_NON_FINITE, 1);
        assert_eq!(BF16_ARGMAX_INVALID_TOKEN_ID, u32::MAX);
    }

    #[test]
    fn bf16_argmax_descriptor_is_bf16_to_two_u32_words_per_row() {
        validate_bf16_argmax_descriptor(CudaDType::BF16, 12, CudaDType::U32, 16, 2, 3)
            .expect("two rows of three BF16 logits have two result records");
        validate_bf16_argmax_descriptor(CudaDType::BF16, 0, CudaDType::U32, 0, 0, 3)
            .expect("zero rows are a validated no-op");

        let wrong_logits =
            validate_bf16_argmax_descriptor(CudaDType::F32, 24, CudaDType::U32, 16, 2, 3)
                .expect_err("F32 logits must fail closed");
        assert_eq!(wrong_logits.kind(), CudaErrorKind::InvalidArgument);
        let wrong_results =
            validate_bf16_argmax_descriptor(CudaDType::BF16, 12, CudaDType::U16, 8, 2, 3)
                .expect_err("result records must use U32 storage");
        assert_eq!(wrong_results.kind(), CudaErrorKind::InvalidArgument);
        let zero_vocabulary =
            validate_bf16_argmax_descriptor(CudaDType::BF16, 0, CudaDType::U32, 8, 1, 0)
                .expect_err("empty rows cannot select a token");
        assert_eq!(zero_vocabulary.kind(), CudaErrorKind::InvalidArgument);
        let token_overflow = validate_bf16_argmax_descriptor(
            CudaDType::BF16,
            u64::MAX,
            CudaDType::U32,
            8,
            1,
            u64::from(u32::MAX) + 1,
        )
        .expect_err("the invalid sentinel cannot also be a valid token id");
        assert_eq!(token_overflow.kind(), CudaErrorKind::OutOfRange);

        for (logits_bytes, results_bytes) in [(11, 16), (12, 15)] {
            let error = validate_bf16_argmax_descriptor(
                CudaDType::BF16,
                logits_bytes,
                CudaDType::U32,
                results_bytes,
                2,
                3,
            )
            .expect_err("short input or output capacity must fail");
            assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
        }
        let shape_overflow = validate_bf16_argmax_descriptor(
            CudaDType::BF16,
            u64::MAX,
            CudaDType::U32,
            u64::MAX,
            u64::MAX,
            u64::from(u32::MAX),
        )
        .expect_err("shape arithmetic must not wrap");
        assert_eq!(shape_overflow.kind(), CudaErrorKind::OutOfRange);
    }

    #[allow(clippy::float_cmp)]
    fn reference_bf16_argmax(bits: &[u16]) -> Bf16ArgmaxResult {
        let mut selected: Option<(f32, u32)> = None;
        for (index, &value_bits) in bits.iter().enumerate() {
            if value_bits & 0x7f80 == 0x7f80 {
                return Bf16ArgmaxResult {
                    token_id: BF16_ARGMAX_INVALID_TOKEN_ID,
                    status: BF16_ARGMAX_STATUS_NON_FINITE,
                };
            }
            let value = f32::from_bits(u32::from(value_bits) << 16);
            let token_id = u32::try_from(index).expect("small CPU fixture");
            if selected.is_none_or(|(maximum, selected_token)| {
                value > maximum || (value == maximum && token_id < selected_token)
            }) {
                selected = Some((value, token_id));
            }
        }
        Bf16ArgmaxResult {
            token_id: selected.expect("non-empty CPU fixture").1,
            status: BF16_ARGMAX_STATUS_SUCCESS,
        }
    }

    #[test]
    fn bf16_argmax_contract_rejects_non_finite_and_uses_lower_tie_id() {
        assert_eq!(
            reference_bf16_argmax(&[0xbf80, 0x4000, 0x4000, 0x3f80]),
            Bf16ArgmaxResult {
                token_id: 1,
                status: BF16_ARGMAX_STATUS_SUCCESS,
            }
        );
        assert_eq!(
            reference_bf16_argmax(&[0xbf80, 0x0000, 0x8000, 0x0000]),
            Bf16ArgmaxResult {
                token_id: 1,
                status: BF16_ARGMAX_STATUS_SUCCESS,
            },
            "+0 and -0 tie at the lower token id"
        );
        for non_finite in [0x7f80, 0xff80, 0x7fc1] {
            assert_eq!(
                reference_bf16_argmax(&[0x3f80, non_finite, 0x4000]),
                Bf16ArgmaxResult {
                    token_id: BF16_ARGMAX_INVALID_TOKEN_ID,
                    status: BF16_ARGMAX_STATUS_NON_FINITE,
                }
            );
        }
    }
}

fn subspan_absolute_offset(base:u64,span_len:u64,relative:u64,len:u64)->CudaResult<u64> {
    const OP:&str="CudaBufferSpan::subspan";
    if relative>span_len || len>span_len-relative {return Err(CudaError::out_of_range(OP,"subspan exceeds its declared parent span"));}
    base.checked_add(relative).ok_or_else(||CudaError::out_of_range(OP,"subspan offset overflow"))
}
#[cfg(test)]
mod subspan_contract_tests {
 use super::*;
 #[test]
 fn subspan_never_expands_declared_span() {
  assert_eq!(subspan_absolute_offset(64,128,128,0).unwrap(),192);
  assert_eq!(subspan_absolute_offset(64,128,127,1).unwrap(),191);
  assert!(subspan_absolute_offset(64,128,128,1).is_err());
  assert!(subspan_absolute_offset(64,128,u64::MAX,2).is_err());
  assert!(subspan_absolute_offset(u64::MAX-1,16,2,0).is_err());
 }
}

#[cfg(feature="cuda-cublas-gemm-probe")]
fn validate_hugging_face_qwen_head128_rms_norm_contract(operation: &'static str,dtype:CudaDType,row_count:u64,hidden_size:u64,epsilon:f32)->CudaResult<()> {
 if dtype==CudaDType::BF16 && hidden_size==128 && row_count<=524288 && epsilon.to_bits()==1e-6f32.to_bits(){return Ok(());}
 Err(CudaError::new(CudaErrorKind::NotSupported,CudaErrorDomain::Rust,CudaErrorStage::Validation,0,operation,"BF16 head128 rows<=524288 epsilon1e-6 required"))
}
#[cfg(all(test,feature="cuda-cublas-gemm-probe"))]
#[test]
fn hf_head128_norm_contract_is_finite_and_strict(){
 for rows in [0,1,8,16,2048,8192,32768,262144,524288]{assert!(validate_hugging_face_qwen_head128_rms_norm_contract("test",CudaDType::BF16,rows,128,1e-6).is_ok());}
 for (dtype,rows,width,eps) in [(CudaDType::F32,16,128,1e-6),(CudaDType::BF16,524289,128,1e-6),(CudaDType::BF16,16,64,1e-6),(CudaDType::BF16,16,128,1e-5)]{assert!(validate_hugging_face_qwen_head128_rms_norm_contract("test",dtype,rows,width,eps).is_err());}
}

#[cfg(feature="cuda-cublas-gemm-probe")]
fn validate_hugging_face_qwen_wide_rms_norm_contract(operation: &'static str,dtype:CudaDType,row_count:u64,hidden_size:u64,epsilon:f32)->CudaResult<()> {
 let supported = (matches!(hidden_size,2048|2560) && epsilon.to_bits()==1e-6f32.to_bits())
     || (hidden_size==4096 && epsilon.to_bits()==1e-5f32.to_bits());
 if dtype==CudaDType::BF16 && supported && row_count<=8192 {return Ok(());}
 Err(CudaError::new(CudaErrorKind::NotSupported,CudaErrorDomain::Rust,CudaErrorStage::Validation,0,operation,"BF16 width2048/2560 epsilon1e-6 or width4096 epsilon1e-5, rows<=8192 required"))
}
#[cfg(all(test,feature="cuda-cublas-gemm-probe"))]
#[test]
fn hf_wide_norm_contract_is_finite_and_strict(){
 for (width,eps) in [(2048,1e-6),(2560,1e-6),(4096,1e-5)]{for rows in [0,1,2,3,8,15,16,128,397,512,1024,2048,8192]{assert!(validate_hugging_face_qwen_wide_rms_norm_contract("test",CudaDType::BF16,rows,width,eps).is_ok());}}
 for (dtype,rows,width,eps) in [(CudaDType::F32,16,2048,1e-6),(CudaDType::BF16,8193,2048,1e-6),(CudaDType::BF16,16,128,1e-6),(CudaDType::BF16,16,2560,1e-5),(CudaDType::BF16,16,4096,1e-6),(CudaDType::BF16,16,4096,f32::NAN)]{assert!(validate_hugging_face_qwen_wide_rms_norm_contract("test",dtype,rows,width,eps).is_err());}
}

/// Applies the bounded wide RMSNorm to each contiguous request row group.
/// Parent capacities and ownership are validated before any group executes.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn hugging_face_qwen_wide_rms_norm_row_groups<S:CudaExecutionStream+?Sized>(params:&mut RmsNormParams<'_>,slots:&[u32],stream:&mut S)->CudaResult<()>{
    hugging_face_rms_norm_row_groups(params, slots, stream, false, None)
}

/// Applies the reviewed SmolLM2 RMSNorm to actual request rows, independently
/// of the owner's padded row capacity and other concurrent requests.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn hugging_face_smollm2_rms_norm_row_groups<S:CudaExecutionStream+?Sized>(params:&mut RmsNormParams<'_>,slots:&[u32],stream:&mut S)->CudaResult<()>{
    hugging_face_rms_norm_row_groups(params, slots, stream, true, None)
}

// The wide HF launch saturates its height at 16 rows: every larger request
// uses exactly 32 threads per row. Joining only these groups changes grid
// extent without changing a row's accumulation or reduction tree.
#[cfg(feature="cuda-cublas-gemm-probe")]
fn merged_wide_prefill_norm_end(
    slots: &[u32], begin: usize, mut end: usize,
    cached: Option<crate::batch::ValidatedRequestRowGroups<'_>>,
) -> CudaResult<usize> {
    if end-begin < 16 { return Ok(end); }
    while end < slots.len() {
        let next = if let Some(groups)=cached { groups.group_end(end)? } else {
            let mut next=end+1;
            while next<slots.len() && slots[next]==slots[end] { next+=1; }
            next
        };
        if next-end < 16 { break; }
        end=next;
    }
    Ok(end)
}

#[cfg(all(test,feature="cuda-cublas-gemm-probe"))]
#[test]
fn wide_prefill_norm_merge_preserves_each_request_reduction_geometry() {
    // Exhaust the threshold on both sides, including a non-eligible group
    // between two large requests. A short request must never inherit the
    // 32-thread tree merely because its neighbours are large.
    for a in 1..=33_usize { for b in 1..=33_usize {
        let lengths=[a,b,15,128,130,1,16];
        let slots:Vec<u32>=lengths.iter().enumerate()
            .flat_map(|(id,&n)| std::iter::repeat_n(id as u32,n)).collect();
        let mut begin=0;
        for (i,&length) in lengths.iter().enumerate() {
            let end=merged_wide_prefill_norm_end(&slots,begin,begin+length,None).unwrap();
            let mut expected=begin+length;
            if length>=16 {
                for &next in &lengths[i+1..] {
                    if next<16 { break; }
                    expected+=next;
                }
            }
            assert_eq!(end,expected);
            let threads=|rows:usize|512/(1_usize << rows.min(16).ilog2());
            if end>begin+length { assert_eq!(threads(length),32); assert_eq!(threads(end-begin),32); }
            begin+=length;
        }
    }}
}

#[cfg(feature="cuda-cublas-gemm-probe")]
fn hugging_face_rms_norm_row_groups<S:CudaExecutionStream+?Sized>(params:&mut RmsNormParams<'_>,slots:&[u32],stream:&mut S,smol:bool,cached:Option<crate::batch::ValidatedRequestRowGroups<'_>>)->CudaResult<()>{
    const OPERATION: &str = "hugging_face_request_rms_norm";
    let stream = execution_stream_mut(stream);
    if smol {
        validate_hugging_face_smollm2_rms_norm_contract(OPERATION,params.input.dtype,params.row_count,params.hidden_size,params.epsilon)?;
    } else {
        validate_hugging_face_qwen_wide_rms_norm_contract(OPERATION,params.input.dtype,params.row_count,params.hidden_size,params.epsilon)?;
    }
    require_dtype(OPERATION, "weight", params.weight.dtype, params.input.dtype)?;
    require_dtype(OPERATION, "output", params.output.dtype, params.input.dtype)?;
    let matrix_bytes = required_matrix_bytes(
        OPERATION,
        params.row_count,
        params.hidden_size,
        params.input.dtype,
    )?;
    require_capacity(OPERATION, "input", params.input.byte_len, matrix_bytes)?;
    require_capacity(
        OPERATION,
        "weight",
        params.weight.byte_len,
        required_vector_bytes(OPERATION, params.hidden_size, params.weight.dtype)?,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len, matrix_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer,
            params.weight.buffer,
            params.output.buffer,
        ],
    )?;

 if slots.len() as u64>params.row_count{return Err(CudaError::invalid_argument(OPERATION,"active rows exceed prepared descriptor"));}
 let row_bytes=params.hidden_size*2;let mut begin=0;
 while begin<slots.len(){let mut end=if let Some(groups)=cached{groups.group_end(begin)?}else{let mut end=begin+1;while end<slots.len()&&slots[end]==slots[begin]{end+=1;}end};
   let singleton = end-begin == 1;
   if singleton {
     while end<slots.len() && (!smol || end-begin<32) {
       let mut next=end+1;while next<slots.len()&&slots[next]==slots[end]{next+=1;}
       if next-end != 1 { break; }
       end=next;
     }
   } else if !smol {
     end=merged_wide_prefill_norm_end(slots,begin,end,cached)?;
   }
   let offset=begin as u64*row_bytes;let bytes=(end-begin) as u64*row_bytes;
   let mut group=RmsNormParams{input:params.input.subspan(offset,bytes)?,weight:params.weight.subspan(0,params.weight.byte_len())?,output:params.output.subspan(offset,bytes)?,row_count:(end-begin) as u64,hidden_size:params.hidden_size,epsilon:params.epsilon};
   if smol {execute_hugging_face_smollm2_rms_norm(&mut group,stream,singleton && end-begin>1)?;}
   else if singleton && end-begin>1 {hugging_face_qwen_singleton_rms_norm(&mut group,stream)?;}
   else {hugging_face_qwen_wide_rms_norm(&mut group,stream)?;}begin=end;
 }
 Ok(())
}

/// Bounded, synchronous, owned bit packing for separately projected Qwen3 QKV.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn hf_pack_separate_qkv<S:CudaExecutionStream+?Sized>(q:CudaBufferSpan<'_>,k:CudaBufferSpan<'_>,v:CudaBufferSpan<'_>,out:CudaBufferSpanMut<'_>,rows:u64,qw:u64,kw:u64,stream:&mut S)->CudaResult<()>{
 const OP:&str="HF pack separate QKV";
 if rows==0||rows>8192||qw!=4096||kw!=1024{return Err(CudaError::invalid_argument(OP,"rows1..8192,Q4096,KV1024 required"));}
 let stream=execution_stream_mut(stream);
 for (name,dtype,bytes,need) in [("query",q.dtype,q.byte_len,rows*qw*2),("key",k.dtype,k.byte_len,rows*kw*2),("value",v.dtype,v.byte_len,rows*kw*2),("merged",out.dtype,out.byte_len,rows*(qw+2*kw)*2)]{
  if dtype!=CudaDType::BF16{return Err(CudaError::invalid_argument(OP,"BF16 required"));}require_capacity(OP,name,bytes,need)?;
 }
 validate_resources(OP,stream,&[q.buffer,k.buffer,v.buffer,out.buffer])?;
 #[cfg(feature="cuda")]{ffi::hf_pack_qkv_execute(q.raw(),k.raw(),v.raw(),out.raw(),rows,qw,kw,&mut stream.native)}
 #[cfg(not(feature="cuda"))]{Err(CudaError::unavailable(OP))}
}

/// Reuses validated request boundaries; the original norm geometry and FP32
/// reduction tree are unchanged. This descriptor owns no GPU allocations.
#[cfg(feature="cuda-cublas-gemm-probe")]
pub fn hugging_face_qwen_wide_rms_norm_row_groups_cached<S:CudaExecutionStream+?Sized>(params:&mut RmsNormParams<'_>,groups:crate::batch::ValidatedRequestRowGroups<'_>,stream:&mut S)->CudaResult<()>{
    hugging_face_rms_norm_row_groups(params,groups.row_sequence_slots(),stream,false,Some(groups))
}
