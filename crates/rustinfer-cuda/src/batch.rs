use crate::error::{CudaError, CudaResult};
use crate::memory::CudaDeviceBuffer;
use crate::primitives::{CudaBufferSpan, CudaBufferSpanMut, CudaDType};
use crate::runtime::{CudaStream, ensure_same_context};

#[cfg(feature = "cuda")]
use crate::ffi;

/// Version of the packed multi-sequence metadata contract.
pub const PACKED_BATCH_VERSION: u32 = 1;

/// Fixed token capacity of every physical paged-KV block.
pub const PACKED_BATCH_BLOCK_SIZE: u64 = 16;

const BF16_BYTES: u64 = 2;
const ATTENTION_HEAD_SIZE: u64 = 64;

/// Allocation-free validated host mirror of one packed multi-sequence batch.
///
/// `sequence_block_offsets` is a CSR offset array with `S + 1` entries.
/// `block_ids` and `valid_tokens` are in logical-block order within each
/// sequence. Every active row binds one sequence slot to one zero-based logical
/// token position. The borrowed slices keep the validation evidence alive and
/// immutable for as long as the descriptor can be submitted.
#[derive(Clone, Copy, Debug)]
pub struct PackedBatchHostV1<'a> {
    sequence_block_offsets: &'a [u32],
    block_ids: &'a [u32],
    valid_tokens: &'a [u16],
    row_sequence_slots: &'a [u32],
    row_positions: &'a [u32],
    sequence_count: u64,
    block_count: u64,
    active_row_count: u64,
    physical_block_count: u64,
}

impl<'a> PackedBatchHostV1<'a> {
    /// Validates and borrows version-1 packed metadata without allocating.
    ///
    /// Validation covers the complete safe-boundary contract: CSR start/end
    /// and monotonicity, canonical per-sequence valid-token counts, globally
    /// unique in-pool physical block IDs, row bounds, and unique
    /// `(sequence_slot, position)` pairs.
    ///
    /// # Errors
    ///
    /// Returns before any CUDA call when host metadata is malformed or cannot
    /// be represented by the device's U32 index arrays.
    pub fn new(
        sequence_block_offsets: &'a [u32],
        block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        row_sequence_slots: &'a [u32],
        row_positions: &'a [u32],
        physical_block_count: u64,
    ) -> CudaResult<Self> {
        let dimensions = validate_packed_host(
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
            physical_block_count,
        )?;
        Ok(Self {
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
            sequence_count: dimensions.sequence_count,
            block_count: dimensions.block_count,
            active_row_count: dimensions.active_row_count,
            physical_block_count,
        })
    }

    #[must_use]
    pub const fn format_version(self) -> u32 {
        PACKED_BATCH_VERSION
    }

    #[must_use]
    pub const fn sequence_block_offsets(self) -> &'a [u32] {
        self.sequence_block_offsets
    }

    #[must_use]
    pub const fn block_ids(self) -> &'a [u32] {
        self.block_ids
    }

    #[must_use]
    pub const fn valid_tokens(self) -> &'a [u16] {
        self.valid_tokens
    }

    #[must_use]
    pub const fn row_sequence_slots(self) -> &'a [u32] {
        self.row_sequence_slots
    }

    #[must_use]
    pub const fn row_positions(self) -> &'a [u32] {
        self.row_positions
    }

    #[must_use]
    pub const fn sequence_count(self) -> u64 {
        self.sequence_count
    }

    #[must_use]
    pub const fn block_count(self) -> u64 {
        self.block_count
    }

    #[must_use]
    pub const fn active_row_count(self) -> u64 {
        self.active_row_count
    }

    #[must_use]
    pub const fn physical_block_count(self) -> u64 {
        self.physical_block_count
    }

    #[must_use]
    pub const fn block_size(self) -> u64 {
        PACKED_BATCH_BLOCK_SIZE
    }
}

/// Host-validated packed metadata bound to the exact pre-uploaded device arrays.
///
/// Keeping both views in one immutable value prevents safe callers from
/// submitting dimensions from one batch with device arrays from another.
#[derive(Clone, Copy, Debug)]
pub struct PackedBatchV1<'a> {
    host: PackedBatchHostV1<'a>,
    device_sequence_block_offsets: CudaBufferSpan<'a>,
    device_block_ids: CudaBufferSpan<'a>,
    device_valid_tokens: CudaBufferSpan<'a>,
    device_row_sequence_slots: CudaBufferSpan<'a>,
    device_row_positions: CudaBufferSpan<'a>,
}

impl<'a> PackedBatchV1<'a> {
    /// Binds one validated host mirror to its five device arrays.
    ///
    /// The caller is responsible for uploading byte-for-byte matching array
    /// contents before execution. Dtype and declared span capacity are checked
    /// here; context ownership and idle state are checked on every submission.
    ///
    /// # Errors
    ///
    /// Returns for a wrong device dtype, undersized span, or byte arithmetic
    /// overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: PackedBatchHostV1<'a>,
        device_sequence_block_offsets: CudaBufferSpan<'a>,
        device_block_ids: CudaBufferSpan<'a>,
        device_valid_tokens: CudaBufferSpan<'a>,
        device_row_sequence_slots: CudaBufferSpan<'a>,
        device_row_positions: CudaBufferSpan<'a>,
    ) -> CudaResult<Self> {
        const OPERATION: &str = "PackedBatchV1::new";
        for (name, actual, expected) in [
            (
                "device_sequence_block_offsets",
                device_sequence_block_offsets.dtype(),
                CudaDType::U32,
            ),
            ("device_block_ids", device_block_ids.dtype(), CudaDType::U32),
            (
                "device_valid_tokens",
                device_valid_tokens.dtype(),
                CudaDType::U16,
            ),
            (
                "device_row_sequence_slots",
                device_row_sequence_slots.dtype(),
                CudaDType::U32,
            ),
            (
                "device_row_positions",
                device_row_positions.dtype(),
                CudaDType::U32,
            ),
        ] {
            require_dtype(OPERATION, name, actual, expected)?;
        }
        let offset_count = host
            .sequence_count()
            .checked_add(1)
            .ok_or_else(|| CudaError::out_of_range(OPERATION, "CSR offset count overflows u64"))?;
        for (name, actual, count, dtype) in [
            (
                "device_sequence_block_offsets",
                device_sequence_block_offsets.byte_len(),
                offset_count,
                CudaDType::U32,
            ),
            (
                "device_block_ids",
                device_block_ids.byte_len(),
                host.block_count(),
                CudaDType::U32,
            ),
            (
                "device_valid_tokens",
                device_valid_tokens.byte_len(),
                host.block_count(),
                CudaDType::U16,
            ),
            (
                "device_row_sequence_slots",
                device_row_sequence_slots.byte_len(),
                host.active_row_count(),
                CudaDType::U32,
            ),
            (
                "device_row_positions",
                device_row_positions.byte_len(),
                host.active_row_count(),
                CudaDType::U32,
            ),
        ] {
            require_capacity(
                OPERATION,
                name,
                actual,
                checked_bytes(OPERATION, &[count, dtype.size_bytes()])?,
            )?;
        }
        Ok(Self {
            host,
            device_sequence_block_offsets,
            device_block_ids,
            device_valid_tokens,
            device_row_sequence_slots,
            device_row_positions,
        })
    }

    #[must_use]
    pub const fn host(self) -> PackedBatchHostV1<'a> {
        self.host
    }

    #[must_use]
    pub const fn device_sequence_block_offsets(self) -> CudaBufferSpan<'a> {
        self.device_sequence_block_offsets
    }

    #[must_use]
    pub const fn device_block_ids(self) -> CudaBufferSpan<'a> {
        self.device_block_ids
    }

    #[must_use]
    pub const fn device_valid_tokens(self) -> CudaBufferSpan<'a> {
        self.device_valid_tokens
    }

    #[must_use]
    pub const fn device_row_sequence_slots(self) -> CudaBufferSpan<'a> {
        self.device_row_sequence_slots
    }

    #[must_use]
    pub const fn device_row_positions(self) -> CudaBufferSpan<'a> {
        self.device_row_positions
    }

    #[cfg(feature = "cuda")]
    fn raw(self) -> ffi::PackedBatchRawV1 {
        ffi::PackedBatchRawV1 {
            sequence_block_offsets: self.device_sequence_block_offsets.raw(),
            block_ids: self.device_block_ids.raw(),
            valid_tokens: self.device_valid_tokens.raw(),
            row_sequence_slots: self.device_row_sequence_slots.raw(),
            row_positions: self.device_row_positions.raw(),
            sequence_count: self.host.sequence_count(),
            block_count: self.host.block_count(),
            active_row_count: self.host.active_row_count(),
            physical_block_count: self.host.physical_block_count(),
            block_size: u32::try_from(PACKED_BATCH_BLOCK_SIZE)
                .expect("the fixed packed block size fits u32"),
        }
    }
}

/// Per-row-position `RoPE` over a dense active-row tensor.
#[derive(Debug)]
pub struct IndexedRopeParams<'a> {
    /// `[T,H,D]`, F32 or BF16.
    pub input: CudaBufferSpan<'a>,
    /// `[table_position_count, rotary_dimension / 2]`, F32.
    pub cos: CudaBufferSpan<'a>,
    /// `[table_position_count, rotary_dimension / 2]`, F32.
    pub sin: CudaBufferSpan<'a>,
    /// Pre-uploaded U32 `[T]`, byte-for-byte matching `positions_host`.
    pub positions: CudaBufferSpan<'a>,
    /// Immutable host validation mirror for `positions`.
    pub positions_host: &'a [u32],
    /// `[T,H,D]`, matching `input`.
    pub output: CudaBufferSpanMut<'a>,
    pub head_count: u64,
    pub head_size: u64,
    pub rotary_dimension: u64,
    pub table_position_count: u64,
}

/// Applies non-interleaved Llama `RoPE` at an independent position per row.
///
/// Zero active rows are a validated allocation-free no-op. The native call
/// synchronizes the explicit stream before returning.
///
/// # Errors
///
/// Returns before launch for malformed mirrored positions, dtype/capacity,
/// shape, context, or buffer-idle violations, or for native execution failure.
pub fn indexed_rope(params: &mut IndexedRopeParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "indexed_rope";
    require_nonzero(OPERATION, "head_count", params.head_count)?;
    require_nonzero(OPERATION, "head_size", params.head_size)?;
    require_nonzero(OPERATION, "rotary_dimension", params.rotary_dimension)?;
    require_nonzero(
        OPERATION,
        "table_position_count",
        params.table_position_count,
    )?;
    if params.rotary_dimension > params.head_size || params.rotary_dimension % 2 != 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "rotary_dimension must be even and no larger than head_size",
        ));
    }
    validate_indexed_positions(params.positions_host, params.table_position_count)?;
    require_float(OPERATION, "input", params.input.dtype())?;
    require_dtype(OPERATION, "cos", params.cos.dtype(), CudaDType::F32)?;
    require_dtype(OPERATION, "sin", params.sin.dtype(), CudaDType::F32)?;
    require_dtype(
        OPERATION,
        "positions",
        params.positions.dtype(),
        CudaDType::U32,
    )?;
    require_dtype(
        OPERATION,
        "output",
        params.output.dtype(),
        params.input.dtype(),
    )?;
    let active_row_count = slice_len_u64(OPERATION, "positions_host", params.positions_host)?;
    let tensor_elements = checked_bytes(
        OPERATION,
        &[active_row_count, params.head_count, params.head_size],
    )?;
    let tensor_bytes = checked_bytes(
        OPERATION,
        &[tensor_elements, params.input.dtype().size_bytes()],
    )?;
    let table_elements = checked_bytes(
        OPERATION,
        &[params.table_position_count, params.rotary_dimension / 2],
    )?;
    let table_bytes = checked_bytes(OPERATION, &[table_elements, CudaDType::F32.size_bytes()])?;
    for (name, actual, required) in [
        ("input", params.input.byte_len(), tensor_bytes),
        ("cos", params.cos.byte_len(), table_bytes),
        ("sin", params.sin.byte_len(), table_bytes),
        (
            "positions",
            params.positions.byte_len(),
            checked_bytes(OPERATION, &[active_row_count, CudaDType::U32.size_bytes()])?,
        ),
        ("output", params.output.byte_len(), tensor_bytes),
    ] {
        require_capacity(OPERATION, name, actual, required)?;
    }
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer(),
            params.cos.buffer(),
            params.sin.buffer(),
            params.positions.buffer(),
            params.output.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::indexed_rope_execute(
            params.input.raw(),
            params.cos.raw(),
            params.sin.raw(),
            params.positions.raw(),
            params.output.raw(),
            active_row_count,
            params.head_count,
            params.head_size,
            params.rotary_dimension,
            params.table_position_count,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Allocation-free row selection from one contiguous matrix.
#[derive(Debug)]
pub struct RowGatherParams<'a> {
    /// `[input_row_count,column_count]`, F32 or BF16.
    pub input: CudaBufferSpan<'a>,
    /// Pre-uploaded flattened U32 row indices.
    pub row_indices: CudaBufferSpan<'a>,
    /// Immutable host validation mirror for `row_indices`.
    pub row_indices_host: &'a [u32],
    /// `[row_indices_host.len(),column_count]`, matching `input`.
    pub output: CudaBufferSpanMut<'a>,
    pub input_row_count: u64,
    pub column_count: u64,
}

/// Gathers unique flattened token rows and synchronizes before returning.
///
/// An empty index slice is a validated allocation-free no-op.
///
/// # Errors
///
/// Returns before launch for duplicate/out-of-range mirrored indices,
/// dtype/capacity/context/idle violations, or native execution failure.
pub fn row_gather(params: &mut RowGatherParams<'_>, stream: &mut CudaStream) -> CudaResult<()> {
    const OPERATION: &str = "row_gather";
    require_nonzero(OPERATION, "input_row_count", params.input_row_count)?;
    require_nonzero(OPERATION, "column_count", params.column_count)?;
    validate_gather_indices(params.row_indices_host, params.input_row_count)?;
    require_float(OPERATION, "input", params.input.dtype())?;
    require_dtype(
        OPERATION,
        "row_indices",
        params.row_indices.dtype(),
        CudaDType::U32,
    )?;
    require_dtype(
        OPERATION,
        "output",
        params.output.dtype(),
        params.input.dtype(),
    )?;
    let output_row_count = slice_len_u64(OPERATION, "row_indices_host", params.row_indices_host)?;
    let input_bytes = matrix_bytes(
        OPERATION,
        params.input_row_count,
        params.column_count,
        params.input.dtype(),
    )?;
    let output_bytes = matrix_bytes(
        OPERATION,
        output_row_count,
        params.column_count,
        params.input.dtype(),
    )?;
    for (name, actual, required) in [
        ("input", params.input.byte_len(), input_bytes),
        (
            "row_indices",
            params.row_indices.byte_len(),
            checked_bytes(OPERATION, &[output_row_count, CudaDType::U32.size_bytes()])?,
        ),
        ("output", params.output.byte_len(), output_bytes),
    ] {
        require_capacity(OPERATION, name, actual, required)?;
    }
    validate_resources(
        OPERATION,
        stream,
        &[
            params.input.buffer(),
            params.row_indices.buffer(),
            params.output.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::row_gather_execute(
            params.input.raw(),
            params.row_indices.raw(),
            params.output.raw(),
            params.input_row_count,
            output_row_count,
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

/// Dense active-row K/V scatter into a shared paged cache.
#[derive(Debug)]
pub struct RaggedPagedKvCacheWriteParams<'a> {
    /// BF16 `[T,KVH,D]` post-RoPE keys.
    pub key_source: CudaBufferSpan<'a>,
    /// BF16 `[T,KVH,D]` values.
    pub value_source: CudaBufferSpan<'a>,
    /// BF16 `[physical_block_count,KVH,16,D]` key pool.
    pub key_pool: CudaBufferSpanMut<'a>,
    /// BF16 `[physical_block_count,KVH,16,D]` value pool.
    pub value_pool: CudaBufferSpanMut<'a>,
    /// Host/device-bound post-write address translation.
    pub batch: PackedBatchV1<'a>,
    pub key_value_head_count: u64,
    pub head_size: u64,
}

/// Scatters one packed batch of K/V rows into a shared physical pool.
///
/// Validation completes before either writable pool is modified. Logical
/// scheduler commit remains the caller's responsibility after every layer
/// write succeeds. The native operation is allocation-free and synchronous.
///
/// # Errors
///
/// Returns for dtype/capacity/context/idle violations or native failure.
pub fn ragged_paged_kv_cache_write(
    params: &mut RaggedPagedKvCacheWriteParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "ragged_paged_kv_cache_write";
    require_nonzero(
        OPERATION,
        "key_value_head_count",
        params.key_value_head_count,
    )?;
    require_nonzero(OPERATION, "head_size", params.head_size)?;
    for (name, dtype) in [
        ("key_source", params.key_source.dtype()),
        ("value_source", params.value_source.dtype()),
        ("key_pool", params.key_pool.dtype()),
        ("value_pool", params.value_pool.dtype()),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
    let host = params.batch.host();
    let source_bytes = checked_bytes(
        OPERATION,
        &[
            host.active_row_count(),
            params.key_value_head_count,
            params.head_size,
            BF16_BYTES,
        ],
    )?;
    let pool_bytes = paged_pool_bytes(
        OPERATION,
        host.physical_block_count(),
        params.key_value_head_count,
        params.head_size,
    )?;
    for (name, actual, required) in [
        ("key_source", params.key_source.byte_len(), source_bytes),
        ("value_source", params.value_source.byte_len(), source_bytes),
        ("key_pool", params.key_pool.byte_len(), pool_bytes),
        ("value_pool", params.value_pool.byte_len(), pool_bytes),
    ] {
        require_capacity(OPERATION, name, actual, required)?;
    }
    validate_batch_resources(
        OPERATION,
        stream,
        &params.batch,
        &[
            params.key_source.buffer(),
            params.value_source.buffer(),
            params.key_pool.buffer(),
            params.value_pool.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        let batch = params.batch.raw();
        ffi::ragged_paged_kv_cache_write_execute(
            params.key_source.raw(),
            params.value_source.raw(),
            params.key_pool.raw(),
            params.value_pool.raw(),
            &batch,
            params.key_value_head_count,
            params.head_size,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Ragged causal attention over a shared paged K/V pool.
#[derive(Debug)]
pub struct RaggedPagedAttentionParams<'a> {
    /// BF16 `[T,QH,64]` active queries.
    pub query: CudaBufferSpan<'a>,
    /// BF16 `[physical_block_count,KVH,16,64]` key pool.
    pub key_pool: CudaBufferSpan<'a>,
    /// BF16 `[physical_block_count,KVH,16,64]` value pool.
    pub value_pool: CudaBufferSpan<'a>,
    /// BF16 `[output_row_count,QH,64]`; rows `[T,M)` are zero-filled.
    pub output: CudaBufferSpanMut<'a>,
    /// Host/device-bound post-write address translation.
    pub batch: PackedBatchV1<'a>,
    pub query_head_count: u64,
    pub key_value_head_count: u64,
    pub head_size: u64,
    /// Fixed dense row count `M`, at least the active row count `T`.
    pub output_row_count: u64,
    /// Positive finite attention scale.
    pub scale: f32,
}

/// Executes D64 GQA ragged paged attention and zero-fills inactive output rows.
///
/// Each active row attends through its own logical position, inclusive. The
/// native operation performs no allocation and synchronizes before returning.
///
/// # Errors
///
/// Returns before launch unless D=64, QH is divisible by KVH, scale is finite
/// and positive, `M >= T`, and all spans satisfy dtype, capacity, context, and
/// idle-state requirements.
#[allow(clippy::too_many_lines)]
pub fn ragged_paged_attention(
    params: &mut RaggedPagedAttentionParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "ragged_paged_attention";
    require_nonzero(OPERATION, "query_head_count", params.query_head_count)?;
    require_nonzero(
        OPERATION,
        "key_value_head_count",
        params.key_value_head_count,
    )?;
    if params.head_size != ATTENTION_HEAD_SIZE {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "ragged paged attention requires head_size=64",
        ));
    }
    if params.query_head_count % params.key_value_head_count != 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "key_value_head_count must divide query_head_count",
        ));
    }
    if !params.scale.is_finite() || params.scale <= 0.0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "scale must be finite and greater than zero",
        ));
    }
    let host = params.batch.host();
    if params.output_row_count < host.active_row_count() {
        return Err(CudaError::out_of_range(
            OPERATION,
            "output_row_count must be at least active_row_count",
        ));
    }
    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key_pool", params.key_pool.dtype()),
        ("value_pool", params.value_pool.dtype()),
        ("output", params.output.dtype()),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
    let query_bytes = checked_bytes(
        OPERATION,
        &[
            host.active_row_count(),
            params.query_head_count,
            ATTENTION_HEAD_SIZE,
            BF16_BYTES,
        ],
    )?;
    let output_bytes = checked_bytes(
        OPERATION,
        &[
            params.output_row_count,
            params.query_head_count,
            ATTENTION_HEAD_SIZE,
            BF16_BYTES,
        ],
    )?;
    let pool_bytes = paged_pool_bytes(
        OPERATION,
        host.physical_block_count(),
        params.key_value_head_count,
        ATTENTION_HEAD_SIZE,
    )?;
    for (name, actual, required) in [
        ("query", params.query.byte_len(), query_bytes),
        ("key_pool", params.key_pool.byte_len(), pool_bytes),
        ("value_pool", params.value_pool.byte_len(), pool_bytes),
        ("output", params.output.byte_len(), output_bytes),
    ] {
        require_capacity(OPERATION, name, actual, required)?;
    }
    validate_batch_resources(
        OPERATION,
        stream,
        &params.batch,
        &[
            params.query.buffer(),
            params.key_pool.buffer(),
            params.value_pool.buffer(),
            params.output.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        let batch = params.batch.raw();
        ffi::ragged_paged_attention_execute(
            params.query.raw(),
            params.key_pool.raw(),
            params.value_pool.raw(),
            params.output.raw(),
            &batch,
            params.query_head_count,
            params.key_value_head_count,
            params.head_size,
            params.output_row_count,
            params.scale,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

#[derive(Clone, Copy)]
#[allow(clippy::struct_field_names)]
struct PackedDimensions {
    sequence_count: u64,
    block_count: u64,
    active_row_count: u64,
}

#[allow(clippy::too_many_lines)]
fn validate_packed_host(
    sequence_block_offsets: &[u32],
    block_ids: &[u32],
    valid_tokens: &[u16],
    row_sequence_slots: &[u32],
    row_positions: &[u32],
    physical_block_count: u64,
) -> CudaResult<PackedDimensions> {
    const OPERATION: &str = "PackedBatchHostV1::new";
    let sequence_count_usize = sequence_block_offsets.len().checked_sub(1).ok_or_else(|| {
        CudaError::invalid_argument(OPERATION, "CSR offsets must contain S+1 entries")
    })?;
    let sequence_count = usize_to_u64(OPERATION, "sequence_count", sequence_count_usize)?;
    let block_count = slice_len_u64(OPERATION, "block_ids", block_ids)?;
    let active_row_count = slice_len_u64(OPERATION, "row_sequence_slots", row_sequence_slots)?;
    for (name, value) in [
        ("sequence_count", sequence_count),
        ("block_count", block_count),
        ("active_row_count", active_row_count),
        ("physical_block_count", physical_block_count),
    ] {
        require_nonzero(OPERATION, name, value)?;
        if value > u64::from(u32::MAX) {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!("{name} exceeds the U32 metadata range"),
            ));
        }
    }
    if valid_tokens.len() != block_ids.len() {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "block_ids and valid_tokens lengths differ",
        ));
    }
    if row_positions.len() != row_sequence_slots.len() {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "row_sequence_slots and row_positions lengths differ",
        ));
    }
    if block_count > physical_block_count {
        return Err(CudaError::out_of_range(
            OPERATION,
            "logical block count exceeds physical_block_count",
        ));
    }
    if sequence_block_offsets[0] != 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "CSR offsets must start at zero",
        ));
    }
    let expected_end = u32::try_from(block_count)
        .expect("validated packed block count fits the U32 metadata range");
    if sequence_block_offsets[sequence_count_usize] != expected_end {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "CSR final offset must equal block_count",
        ));
    }
    for pair in sequence_block_offsets.windows(2) {
        if pair[0] > pair[1] {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "CSR offsets must be nondecreasing",
            ));
        }
        if pair[0] == pair[1] {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "every packed sequence must own at least one logical block",
            ));
        }
    }

    for (logical_index, &physical_id) in block_ids.iter().enumerate() {
        if u64::from(physical_id) >= physical_block_count {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!("physical block id {physical_id} is outside the pool"),
            ));
        }
        if block_ids[..logical_index].contains(&physical_id) {
            return Err(CudaError::invalid_argument(
                OPERATION,
                format!("physical block id {physical_id} appears more than once"),
            ));
        }
    }

    for sequence_slot in 0..sequence_count_usize {
        let start = usize::try_from(sequence_block_offsets[sequence_slot])
            .expect("U32 CSR offsets fit every supported host usize");
        let end = usize::try_from(sequence_block_offsets[sequence_slot + 1])
            .expect("U32 CSR offsets fit every supported host usize");
        for (within_sequence, &valid) in valid_tokens[start..end].iter().enumerate() {
            let is_last = within_sequence + 1 == end - start;
            let canonical = if is_last {
                (1..=u16::try_from(PACKED_BATCH_BLOCK_SIZE)
                    .expect("fixed packed block size fits u16"))
                    .contains(&valid)
            } else {
                u64::from(valid) == PACKED_BATCH_BLOCK_SIZE
            };
            if !canonical {
                return Err(CudaError::invalid_argument(
                    OPERATION,
                    format!(
                        "sequence {sequence_slot} logical block {within_sequence} has non-canonical valid_tokens={valid}"
                    ),
                ));
            }
        }
    }

    for row in 0..row_sequence_slots.len() {
        let sequence_slot = usize::try_from(row_sequence_slots[row])
            .expect("U32 sequence slots fit every supported host usize");
        if sequence_slot >= sequence_count_usize {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!("row {row} sequence slot is outside the CSR sequence range"),
            ));
        }
        let logical_length =
            sequence_logical_length(sequence_block_offsets, valid_tokens, sequence_slot);
        if u64::from(row_positions[row]) >= logical_length {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!("row {row} position is outside its post-write sequence length"),
            ));
        }
        for earlier in 0..row {
            if row_sequence_slots[earlier] == row_sequence_slots[row]
                && row_positions[earlier] == row_positions[row]
            {
                return Err(CudaError::invalid_argument(
                    OPERATION,
                    format!(
                        "duplicate row address ({},{})",
                        row_sequence_slots[row], row_positions[row]
                    ),
                ));
            }
        }
    }

    Ok(PackedDimensions {
        sequence_count,
        block_count,
        active_row_count,
    })
}

fn sequence_logical_length(
    sequence_block_offsets: &[u32],
    valid_tokens: &[u16],
    sequence_slot: usize,
) -> u64 {
    let start = usize::try_from(sequence_block_offsets[sequence_slot])
        .expect("U32 CSR offsets fit every supported host usize");
    let end = usize::try_from(sequence_block_offsets[sequence_slot + 1])
        .expect("U32 CSR offsets fit every supported host usize");
    if start == end {
        return 0;
    }
    let full_blocks = u64::try_from(end - start - 1).expect("slice length fits u64");
    full_blocks * PACKED_BATCH_BLOCK_SIZE + u64::from(valid_tokens[end - 1])
}

fn validate_indexed_positions(positions: &[u32], table_position_count: u64) -> CudaResult<()> {
    const OPERATION: &str = "indexed_rope";
    for (row, &position) in positions.iter().enumerate() {
        if u64::from(position) >= table_position_count {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!("position {position} at row {row} exceeds the RoPE table"),
            ));
        }
    }
    Ok(())
}

fn validate_gather_indices(indices: &[u32], input_row_count: u64) -> CudaResult<()> {
    const OPERATION: &str = "row_gather";
    for (row, &index) in indices.iter().enumerate() {
        if u64::from(index) >= input_row_count {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!("flattened row index {index} at output row {row} is out of range"),
            ));
        }
        if indices[..row].contains(&index) {
            return Err(CudaError::invalid_argument(
                OPERATION,
                format!("flattened row index {index} appears more than once"),
            ));
        }
    }
    Ok(())
}

fn validate_batch_resources(
    operation: &'static str,
    stream: &CudaStream,
    batch: &PackedBatchV1<'_>,
    tensors: &[&CudaDeviceBuffer],
) -> CudaResult<()> {
    validate_resources(operation, stream, tensors)?;
    validate_resources(
        operation,
        stream,
        &[
            batch.device_sequence_block_offsets.buffer(),
            batch.device_block_ids.buffer(),
            batch.device_valid_tokens.buffer(),
            batch.device_row_sequence_slots.buffer(),
            batch.device_row_positions.buffer(),
        ],
    )
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

fn require_nonzero(operation: &'static str, name: &'static str, value: u64) -> CudaResult<()> {
    if value == 0 {
        Err(CudaError::invalid_argument(
            operation,
            format!("{name} must be greater than zero"),
        ))
    } else {
        Ok(())
    }
}

fn require_float(operation: &'static str, name: &'static str, dtype: CudaDType) -> CudaResult<()> {
    if matches!(dtype, CudaDType::F32 | CudaDType::BF16) {
        Ok(())
    } else {
        Err(CudaError::invalid_argument(
            operation,
            format!("{name} must be f32 or bf16, got {dtype}"),
        ))
    }
}

fn require_dtype(
    operation: &'static str,
    name: &'static str,
    actual: CudaDType,
    expected: CudaDType,
) -> CudaResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(CudaError::invalid_argument(
            operation,
            format!("{name} must be {expected}, got {actual}"),
        ))
    }
}

fn require_capacity(
    operation: &'static str,
    name: &'static str,
    actual: u64,
    required: u64,
) -> CudaResult<()> {
    if actual >= required {
        Ok(())
    } else {
        Err(CudaError::out_of_range(
            operation,
            format!("{name} requires {required} bytes, span exposes {actual}"),
        ))
    }
}

fn matrix_bytes(
    operation: &'static str,
    rows: u64,
    columns: u64,
    dtype: CudaDType,
) -> CudaResult<u64> {
    checked_bytes(operation, &[rows, columns, dtype.size_bytes()])
}

fn paged_pool_bytes(
    operation: &'static str,
    physical_block_count: u64,
    key_value_head_count: u64,
    head_size: u64,
) -> CudaResult<u64> {
    checked_bytes(
        operation,
        &[
            physical_block_count,
            key_value_head_count,
            PACKED_BATCH_BLOCK_SIZE,
            head_size,
            BF16_BYTES,
        ],
    )
}

fn checked_bytes(operation: &'static str, factors: &[u64]) -> CudaResult<u64> {
    factors.iter().try_fold(1_u64, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            CudaError::out_of_range(operation, "batch shape/byte arithmetic overflow")
        })
    })
}

fn slice_len_u64(
    operation: &'static str,
    name: &'static str,
    slice: &[impl Sized],
) -> CudaResult<u64> {
    usize_to_u64(operation, name, slice.len())
}

fn usize_to_u64(operation: &'static str, name: &'static str, value: usize) -> CudaResult<u64> {
    u64::try_from(value)
        .map_err(|_| CudaError::out_of_range(operation, format!("{name} does not fit u64")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaErrorKind;

    fn valid_host() -> CudaResult<PackedBatchHostV1<'static>> {
        PackedBatchHostV1::new(
            &[0, 2, 3],
            &[2, 0, 4],
            &[16, 3, 16],
            &[0, 0, 1],
            &[0, 18, 15],
            5,
        )
    }

    #[test]
    fn accepts_canonical_ragged_host_metadata() {
        let host = valid_host().expect("canonical metadata");
        assert_eq!(host.sequence_count(), 2);
        assert_eq!(host.block_count(), 3);
        assert_eq!(host.active_row_count(), 3);
        assert_eq!(host.physical_block_count(), 5);
        assert_eq!(host.block_size(), 16);
    }

    #[test]
    fn rejects_malformed_csr_boundaries_and_order() {
        for offsets in [
            &[1, 2, 3][..],
            &[0, 3, 2][..],
            &[0, 1, 2][..],
            &[0, 0, 3][..],
        ] {
            let error = PackedBatchHostV1::new(offsets, &[2, 0, 4], &[16, 3, 16], &[0], &[0], 5)
                .expect_err("malformed CSR must fail");
            assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
        }
    }

    #[test]
    fn rejects_noncanonical_valid_tokens() {
        for valid in [&[15, 3, 16][..], &[16, 0, 16][..], &[16, 17, 16][..]] {
            let error = PackedBatchHostV1::new(&[0, 2, 3], &[2, 0, 4], valid, &[0], &[0], 5)
                .expect_err("noncanonical valid counts must fail");
            assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
        }
    }

    #[test]
    fn rejects_duplicate_and_out_of_pool_blocks() {
        let duplicate = PackedBatchHostV1::new(&[0, 2], &[1, 1], &[16, 1], &[0], &[0], 2)
            .expect_err("duplicate physical block must fail");
        assert_eq!(duplicate.kind(), CudaErrorKind::InvalidArgument);

        let outside = PackedBatchHostV1::new(&[0, 1], &[2], &[1], &[0], &[0], 2)
            .expect_err("out-of-pool physical block must fail");
        assert_eq!(outside.kind(), CudaErrorKind::OutOfRange);
    }

    #[test]
    fn rejects_bad_row_addresses_and_duplicates() {
        let bad_sequence = PackedBatchHostV1::new(&[0, 1], &[0], &[1], &[1], &[0], 1)
            .expect_err("row sequence must be in range");
        assert_eq!(bad_sequence.kind(), CudaErrorKind::OutOfRange);

        let bad_position = PackedBatchHostV1::new(&[0, 1], &[0], &[1], &[0], &[1], 1)
            .expect_err("row position must be in range");
        assert_eq!(bad_position.kind(), CudaErrorKind::OutOfRange);

        let duplicate = PackedBatchHostV1::new(&[0, 1], &[0], &[2], &[0, 0], &[1, 1], 1)
            .expect_err("duplicate row address must fail");
        assert_eq!(duplicate.kind(), CudaErrorKind::InvalidArgument);
    }

    #[test]
    fn rejects_zero_active_rows() {
        let error = PackedBatchHostV1::new(&[0, 1], &[0], &[1], &[], &[], 1)
            .expect_err("native packed kernels require T>0");
        assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
    }

    #[test]
    fn indexed_positions_allow_empty_and_reject_table_oob() {
        validate_indexed_positions(&[], 4).expect("empty indexed RoPE is a no-op");
        let error = validate_indexed_positions(&[0, 4], 4).expect_err("position 4 is OOB");
        assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
    }

    #[test]
    fn gather_indices_allow_empty_and_reject_oob_or_duplicates() {
        validate_gather_indices(&[], 4).expect("empty gather is a no-op");
        let oob = validate_gather_indices(&[4], 4).expect_err("row 4 is OOB");
        assert_eq!(oob.kind(), CudaErrorKind::OutOfRange);
        let duplicate =
            validate_gather_indices(&[3, 1, 3], 4).expect_err("duplicate gather index must fail");
        assert_eq!(duplicate.kind(), CudaErrorKind::InvalidArgument);
    }

    #[test]
    fn public_batch_constants_and_symbols_match_the_native_header_source() {
        let header = include_str!("../../../kernels/include/rustinfer_cuda.h");
        assert!(header.contains("#define RUSTINFER_CUDA_PACKED_BATCH_VERSION 1u"));
        assert!(header.contains("#define RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE 16u"));
        for symbol in [
            "RustInferCudaIndexedRopeParams",
            "RustInferCudaRowGatherParams",
            "RustInferCudaPackedBatchV1",
            "RustInferCudaRaggedPagedKvCacheWriteParams",
            "RustInferCudaRaggedPagedAttentionParams",
            "rustinfer_cuda_indexed_rope_execute",
            "rustinfer_cuda_row_gather_execute",
            "rustinfer_cuda_ragged_paged_kv_cache_write_execute",
            "rustinfer_cuda_ragged_paged_attention_execute",
        ] {
            assert!(header.contains(symbol), "native header is missing {symbol}");
        }
        let attention = header
            .split("typedef struct RustInferCudaRaggedPagedAttentionParams")
            .nth(1)
            .and_then(|tail| {
                tail.split("} RustInferCudaRaggedPagedAttentionParams;")
                    .next()
            })
            .expect("ragged attention declaration");
        assert!(
            attention.find("uint64_t head_size;") < attention.find("uint64_t output_row_count;")
        );
        assert!(attention.find("uint64_t output_row_count;") < attention.find("float scale;"));
    }
}
