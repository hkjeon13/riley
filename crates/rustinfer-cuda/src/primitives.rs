use std::error;
use std::fmt;

use crate::error::{CudaError, CudaResult};
use crate::memory::CudaDeviceBuffer;
use crate::runtime::{CudaStream, ensure_same_context};

#[cfg(feature = "cuda")]
use crate::ffi;
#[cfg(feature = "cuda")]
use crate::{CudaErrorDomain, CudaErrorKind, CudaErrorStage};

const EMBEDDING_ERROR_SCRATCH_BYTES: u64 = 32;
#[cfg(feature = "cuda")]
const EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE: u32 = 1;

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
    /// Unsigned byte, used for opaque primitive scratch records.
    U8,
}

impl CudaDType {
    /// Storage bytes occupied by one scalar.
    #[must_use]
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::BF16 => 2,
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
pub fn embedding(
    params: &mut EmbeddingParams<'_>,
    stream: &mut CudaStream,
) -> Result<(), EmbeddingError> {
    const OPERATION: &str = "embedding";
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
        let _ = params;
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
pub fn rms_norm(params: &mut RmsNormParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "rms_norm";
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
pub fn residual_add(params: &mut ResidualAddParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "residual_add";
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
pub fn silu(params: &mut SiluParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "silu";
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
pub fn gated_multiply(
    params: &mut GatedMultiplyParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "gated_multiply";
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
    use super::{CudaDType, checked_product3, required_matrix_bytes, validate_span};
    use crate::{CudaErrorKind, CudaErrorStage};

    #[test]
    fn dtype_sizes_are_fixed_for_the_c_abi() {
        assert_eq!(CudaDType::F32.size_bytes(), 4);
        assert_eq!(CudaDType::BF16.size_bytes(), 2);
        assert_eq!(CudaDType::U32.size_bytes(), 4);
        assert_eq!(CudaDType::U8.size_bytes(), 1);
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
}
