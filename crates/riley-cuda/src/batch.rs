use crate::decode::DecodePartialReductionOrder;
use crate::error::{CudaError, CudaResult};
use crate::memory::CudaDeviceBuffer;
use crate::primitives::{CudaBufferSpan, CudaBufferSpanMut, CudaDType};
use crate::runtime::{CudaExecutionStream, CudaStream, ensure_same_context, execution_stream_mut};

#[cfg(feature = "cuda")]
use crate::ffi;

/// Version of the packed multi-sequence metadata contract.
pub const PACKED_BATCH_VERSION: u32 = 1;

/// Fixed token capacity of every physical paged-KV block.
pub const PACKED_BATCH_BLOCK_SIZE: u64 = 16;

/// Largest inclusive logical prefix accepted by fixed37 ragged attention.
pub const FIXED37_RAGGED_MAX_LOGICAL_TOKENS: u64 = 8_192;

const BF16_BYTES: u64 = 2;
const ATTENTION_HEAD_SIZE: u64 = 64;

/// Fixed query-head count of the eager native ragged D128 two-stage path.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT: u64 = 16;
/// Fixed KV-head count of the eager native ragged D128 two-stage path.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_KEY_VALUE_HEAD_COUNT: u64 = 2;
/// Fixed head size of the eager native ragged D128 two-stage path.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE: u64 = 128;
/// F32 words in one `[maximum, denominator, numerator[D]]` partial state.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_PARTIAL_STATE_WIDTH: u64 =
    NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE + 2;
/// F32 words in one V2 transition pair.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_TRANSITION_WIDTH: u64 = 2;

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
pub fn indexed_rope<S: CudaExecutionStream + ?Sized>(
    params: &mut IndexedRopeParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "indexed_rope";
    let stream = execution_stream_mut(stream);
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
pub fn row_gather<S: CudaExecutionStream + ?Sized>(
    params: &mut RowGatherParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "row_gather";
    let stream = execution_stream_mut(stream);
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
pub fn ragged_paged_kv_cache_write<S: CudaExecutionStream + ?Sized>(
    params: &mut RaggedPagedKvCacheWriteParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "ragged_paged_kv_cache_write";
    let stream = execution_stream_mut(stream);
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

/// Byte requirements for the eager native D128 ragged two-stage workspace.
///
/// The layout keeps every fixed output row independent: partial states are
/// `[M,P,16,130]` F32, transition pairs are `[M,P,16,2]` F32, and final
/// normalizers are `[M,16]` F32. `P` is the maximum logical page count of one
/// row, not the aggregate CSR block count of the batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout {
    partial_state_bytes: u64,
    reduction_step_bytes: u64,
    reduction_normalizer_bytes: u64,
    total_bytes: u64,
}

impl NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout {
    /// Calculates exact workspace spans for `M` prepared rows and per-row
    /// page capacity `P`.
    ///
    /// # Errors
    ///
    /// Returns for a zero axis or byte arithmetic overflow.
    pub fn new(output_row_count: u64, partial_state_capacity: u64) -> CudaResult<Self> {
        const OPERATION: &str = "NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new";
        require_nonzero(OPERATION, "output_row_count", output_row_count)?;
        require_nonzero(OPERATION, "partial_state_capacity", partial_state_capacity)?;
        let partial_state_bytes = checked_bytes(
            OPERATION,
            &[
                output_row_count,
                partial_state_capacity,
                NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT,
                NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_PARTIAL_STATE_WIDTH,
                CudaDType::F32.size_bytes(),
            ],
        )?;
        let reduction_step_bytes = checked_bytes(
            OPERATION,
            &[
                output_row_count,
                partial_state_capacity,
                NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT,
                NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_TRANSITION_WIDTH,
                CudaDType::F32.size_bytes(),
            ],
        )?;
        let reduction_normalizer_bytes = checked_bytes(
            OPERATION,
            &[
                output_row_count,
                NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT,
                CudaDType::F32.size_bytes(),
            ],
        )?;
        let total_bytes = partial_state_bytes
            .checked_add(reduction_step_bytes)
            .and_then(|bytes| bytes.checked_add(reduction_normalizer_bytes))
            .ok_or_else(|| CudaError::out_of_range(OPERATION, "workspace byte total overflows"))?;
        Ok(Self {
            partial_state_bytes,
            reduction_step_bytes,
            reduction_normalizer_bytes,
            total_bytes,
        })
    }

    #[must_use]
    pub const fn partial_state_bytes(self) -> u64 {
        self.partial_state_bytes
    }

    #[must_use]
    pub const fn reduction_step_bytes(self) -> u64 {
        self.reduction_step_bytes
    }

    #[must_use]
    pub const fn reduction_normalizer_bytes(self) -> u64 {
        self.reduction_normalizer_bytes
    }

    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Inputs for the eager native BF16 D128 ragged two-stage paged attention
/// control. The native ABI supports only QH=16, KVH=2, D=128, and page=16.
///
/// Each active row uses a private `P`-sized workspace slice. Inactive prepared
/// rows are zeroed only in `output`; their workspace bytes stay untouched so a
/// caller may use sentinel tails to audit active-row boundaries.
#[derive(Debug)]
pub struct NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'a> {
    /// BF16 `[T,16,128]` active queries.
    pub query: CudaBufferSpan<'a>,
    /// BF16 `[physical_block_count,2,16,128]` key pool.
    pub key_pool: CudaBufferSpan<'a>,
    /// BF16 `[physical_block_count,2,16,128]` value pool.
    pub value_pool: CudaBufferSpan<'a>,
    /// F32 `[M,P,16,130]` V1-compatible partial-state prefix.
    pub partial_states: CudaBufferSpanMut<'a>,
    /// F32 `[M,P,16,2]` V2 transition-pair scratch.
    pub reduction_steps: CudaBufferSpanMut<'a>,
    /// F32 `[M,16]` final-normalizer scratch.
    pub reduction_normalizers: CudaBufferSpanMut<'a>,
    /// BF16 `[M,16,128]`; inactive rows `[T,M)` are zero-filled.
    pub output: CudaBufferSpanMut<'a>,
    /// Host/device-bound ragged page-table translation.
    pub batch: PackedBatchV1<'a>,
    pub query_head_count: u64,
    pub key_value_head_count: u64,
    pub head_size: u64,
    /// Prepared output row count `M`, at least active row count `T`.
    pub output_row_count: u64,
    /// Per-row logical-page capacity `P`. The wrapper derives the producer
    /// grid's `L=max_row ceil((position+1)/16)` from [`PackedBatchHostV1`],
    /// so callers cannot accidentally use `P` as a full-grid launch extent.
    pub partial_state_capacity: u64,
    /// Positive finite attention scale.
    pub scale: f32,
    /// Logical partial-state traversal order, replayed exactly by stage 2.
    pub reduction_order: DecodePartialReductionOrder,
}

/// Executes eager-only native BF16 ragged paged D128 attention with V3 ordered
/// transition/replay reduction.
///
/// This explicit control has no graph-capture ABI and never selects or falls
/// back to the D64 ragged implementation. Direct-stream execution synchronizes
/// before returning; command-batch execution retains every resource until its
/// finish boundary.
///
/// # Errors
///
/// Returns before launch unless the fixed D128 geometry, per-row page capacity,
/// packed metadata, workspace spans, ownership, and idle-state contract hold.
#[allow(clippy::too_many_lines)]
pub fn native_bf16_ragged_paged_split_gqa_d128_two_stage<S: CudaExecutionStream + ?Sized>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "native_bf16_ragged_paged_split_gqa_d128_two_stage";
    let stream = execution_stream_mut(stream);
    let launch_partial_state_count =
        validate_native_bf16_ragged_paged_split_gqa_d128_two_stage(OPERATION, params, stream)?;

    #[cfg(feature = "cuda")]
    {
        let batch = params.batch.raw();
        let reduction_order = match params.reduction_order {
            DecodePartialReductionOrder::LogicalAscending => {
                ffi::DECODE_REDUCTION_LOGICAL_ASCENDING
            }
            DecodePartialReductionOrder::LogicalDescending => {
                ffi::DECODE_REDUCTION_LOGICAL_DESCENDING
            }
        };
        ffi::native_bf16_ragged_paged_split_gqa_d128_two_stage_v3_execute(
            params.query.raw(),
            params.key_pool.raw(),
            params.value_pool.raw(),
            params.partial_states.raw(),
            params.reduction_steps.raw(),
            params.reduction_normalizers.raw(),
            params.output.raw(),
            &batch,
            params.query_head_count,
            params.key_value_head_count,
            params.head_size,
            params.output_row_count,
            params.partial_state_capacity,
            launch_partial_state_count,
            params.scale,
            reduction_order,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, launch_partial_state_count);
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Executes D64 GQA ragged paged attention and zero-fills inactive output rows.
///
/// Each active row attends through its own logical position, inclusive. The
/// native operation performs no allocation. Direct-stream execution
/// synchronizes before returning; command-batch execution retains every
/// registered resource until [`crate::CudaCommandBatch::finish`].
///
/// # Errors
///
/// Returns before launch unless D=64, QH is divisible by KVH, scale is finite
/// and positive, `M >= T`, and all spans satisfy dtype, capacity, context, and
/// idle-state requirements.
#[allow(clippy::too_many_lines)]
pub fn ragged_paged_attention<S: CudaExecutionStream + ?Sized>(
    params: &mut RaggedPagedAttentionParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "ragged_paged_attention";
    let stream = execution_stream_mut(stream);
    validate_ragged_paged_attention(OPERATION, params, stream)?;

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

/// Executes canonical D64 GQA ragged paged attention with independent
/// query-head warps grouped into each CUDA thread block.
///
/// The per-head reduction and online-token order match
/// [`ragged_paged_attention`]. This explicit entry point keeps the legacy
/// one-warp launch available for rollout control and paired profiling.
///
/// # Errors
///
/// Returns before launch unless the same shape, span, context, and stream
/// contract as [`ragged_paged_attention`] is satisfied.
#[allow(clippy::too_many_lines)]
pub fn grouped_ragged_paged_attention<S: CudaExecutionStream + ?Sized>(
    params: &mut RaggedPagedAttentionParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "grouped_ragged_paged_attention";
    let stream = execution_stream_mut(stream);
    validate_ragged_paged_attention(OPERATION, params, stream)?;

    #[cfg(feature = "cuda")]
    {
        let batch = params.batch.raw();
        ffi::ragged_paged_attention_grouped_heads_execute(
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

/// Executes fixed37 D64 GQA ragged paged attention without a workspace.
///
/// Each active row attends through `position + 1` logical tokens, inclusive,
/// using the fixed contiguous-37 balanced reduction profile. Rows `[T,M)` are
/// storage-exact zero. Execution performs no CUDA allocation and has no
/// fallback to another numerical profile. Direct-stream execution synchronizes
/// before returning; command-batch execution retains every registered resource
/// until [`crate::CudaCommandBatch::finish`].
///
/// # Errors
///
/// Returns before launch unless the ordinary ragged-attention contract holds
/// and every host-mirrored row has `position + 1 <= 8192`.
#[allow(clippy::too_many_lines)]
pub fn fixed37_ragged_paged_attention<S: CudaExecutionStream + ?Sized>(
    params: &mut RaggedPagedAttentionParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_ragged_paged_attention";
    let stream = execution_stream_mut(stream);
    validate_ragged_paged_attention(OPERATION, params, stream)?;
    let maximum_logical_token_count =
        validate_fixed37_ragged_logical_tokens(OPERATION, params.batch.host().row_positions())?;

    #[cfg(feature = "cuda")]
    {
        let batch = params.batch.raw();
        ffi::fixed37_ragged_paged_attention_two_pass_execute(
            params.query.raw(),
            params.key_pool.raw(),
            params.value_pool.raw(),
            params.output.raw(),
            &batch,
            params.query_head_count,
            params.key_value_head_count,
            params.head_size,
            params.output_row_count,
            maximum_logical_token_count,
            params.scale,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (params, maximum_logical_token_count);
        Err(CudaError::unavailable(OPERATION))
    }
}

fn validate_ragged_paged_attention(
    operation: &'static str,
    params: &RaggedPagedAttentionParams<'_>,
    stream: &CudaStream,
) -> CudaResult<()> {
    require_nonzero(operation, "query_head_count", params.query_head_count)?;
    require_nonzero(
        operation,
        "key_value_head_count",
        params.key_value_head_count,
    )?;
    if params.head_size != ATTENTION_HEAD_SIZE {
        return Err(CudaError::invalid_argument(
            operation,
            "ragged paged attention requires head_size=64",
        ));
    }
    if params.query_head_count % params.key_value_head_count != 0 {
        return Err(CudaError::invalid_argument(
            operation,
            "key_value_head_count must divide query_head_count",
        ));
    }
    if !params.scale.is_finite() || params.scale <= 0.0 {
        return Err(CudaError::invalid_argument(
            operation,
            "scale must be finite and greater than zero",
        ));
    }
    let host = params.batch.host();
    if params.output_row_count < host.active_row_count() {
        return Err(CudaError::out_of_range(
            operation,
            "output_row_count must be at least active_row_count",
        ));
    }
    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key_pool", params.key_pool.dtype()),
        ("value_pool", params.value_pool.dtype()),
        ("output", params.output.dtype()),
    ] {
        require_dtype(operation, name, dtype, CudaDType::BF16)?;
    }
    let query_bytes = checked_bytes(
        operation,
        &[
            host.active_row_count(),
            params.query_head_count,
            ATTENTION_HEAD_SIZE,
            BF16_BYTES,
        ],
    )?;
    let output_bytes = checked_bytes(
        operation,
        &[
            params.output_row_count,
            params.query_head_count,
            ATTENTION_HEAD_SIZE,
            BF16_BYTES,
        ],
    )?;
    let pool_bytes = paged_pool_bytes(
        operation,
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
        require_capacity(operation, name, actual, required)?;
    }
    validate_batch_resources(
        operation,
        stream,
        &params.batch,
        &[
            params.query.buffer(),
            params.key_pool.buffer(),
            params.value_pool.buffer(),
            params.output.buffer(),
        ],
    )
}

#[allow(clippy::too_many_lines)]
fn validate_native_bf16_ragged_paged_split_gqa_d128_two_stage(
    operation: &'static str,
    params: &NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    stream: &CudaStream,
) -> CudaResult<u64> {
    const MAXIMUM_GRID_X: u64 = 2_147_483_647;
    const MAXIMUM_GRID_Y_OR_Z: u64 = 65_535;
    if params.query_head_count != NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT
        || params.key_value_head_count
            != NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_KEY_VALUE_HEAD_COUNT
        || params.head_size != NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE
    {
        return Err(CudaError::invalid_argument(
            operation,
            "native ragged two-stage attention requires QH=16, KVH=2, and head_size=128",
        ));
    }
    require_nonzero(
        operation,
        "partial_state_capacity",
        params.partial_state_capacity,
    )?;
    if !params.scale.is_finite() || params.scale <= 0.0 {
        return Err(CudaError::invalid_argument(
            operation,
            "scale must be finite and greater than zero",
        ));
    }
    let host = params.batch.host();
    if params.output_row_count < host.active_row_count() {
        return Err(CudaError::out_of_range(
            operation,
            "output_row_count must be at least active_row_count",
        ));
    }
    if params.output_row_count > MAXIMUM_GRID_Y_OR_Z {
        return Err(CudaError::out_of_range(
            operation,
            "native ragged two-stage output rows exceed the CUDA grid contract",
        ));
    }

    let launch_partial_state_count = native_bf16_ragged_launch_partial_state_count(
        operation,
        host,
        params.partial_state_capacity,
    )?;
    if launch_partial_state_count > MAXIMUM_GRID_X {
        return Err(CudaError::out_of_range(
            operation,
            "native ragged two-stage producer grid exceeds the CUDA grid contract",
        ));
    }

    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key_pool", params.key_pool.dtype()),
        ("value_pool", params.value_pool.dtype()),
        ("output", params.output.dtype()),
    ] {
        require_dtype(operation, name, dtype, CudaDType::BF16)?;
    }
    for (name, dtype) in [
        ("partial_states", params.partial_states.dtype()),
        ("reduction_steps", params.reduction_steps.dtype()),
        (
            "reduction_normalizers",
            params.reduction_normalizers.dtype(),
        ),
    ] {
        require_dtype(operation, name, dtype, CudaDType::F32)?;
    }

    let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new(
        params.output_row_count,
        params.partial_state_capacity,
    )?;
    let query_bytes = checked_bytes(
        operation,
        &[
            host.active_row_count(),
            NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT,
            NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE,
            BF16_BYTES,
        ],
    )?;
    let output_bytes = checked_bytes(
        operation,
        &[
            params.output_row_count,
            NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT,
            NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE,
            BF16_BYTES,
        ],
    )?;
    let pool_bytes = paged_pool_bytes(
        operation,
        host.physical_block_count(),
        NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_KEY_VALUE_HEAD_COUNT,
        NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE,
    )?;
    for (name, actual, required) in [
        ("query", params.query.byte_len(), query_bytes),
        ("key_pool", params.key_pool.byte_len(), pool_bytes),
        ("value_pool", params.value_pool.byte_len(), pool_bytes),
        (
            "partial_states",
            params.partial_states.byte_len(),
            layout.partial_state_bytes(),
        ),
        (
            "reduction_steps",
            params.reduction_steps.byte_len(),
            layout.reduction_step_bytes(),
        ),
        (
            "reduction_normalizers",
            params.reduction_normalizers.byte_len(),
            layout.reduction_normalizer_bytes(),
        ),
        ("output", params.output.byte_len(), output_bytes),
    ] {
        require_capacity(operation, name, actual, required)?;
    }
    validate_batch_resources(
        operation,
        stream,
        &params.batch,
        &[
            params.query.buffer(),
            params.key_pool.buffer(),
            params.value_pool.buffer(),
            params.partial_states.buffer(),
            params.reduction_steps.buffer(),
            params.reduction_normalizers.buffer(),
            params.output.buffer(),
        ],
    )?;
    Ok(launch_partial_state_count)
}

/// Derives the exact producer-grid `L` from host-validated row prefixes.
///
/// `P` remains a per-row workspace stride. It is intentionally never compared
/// to the aggregate CSR block count: independent rows may collectively own
/// more pages than a single row can address. The returned `L` is in
/// `1..=P`, and each active row's prefix is also checked against its CSR range.
fn native_bf16_ragged_launch_partial_state_count(
    operation: &'static str,
    host: PackedBatchHostV1<'_>,
    partial_state_capacity: u64,
) -> CudaResult<u64> {
    let mut launch_partial_state_count = 0_u64;
    for (row, (&sequence_slot, &position)) in host
        .row_sequence_slots()
        .iter()
        .zip(host.row_positions())
        .enumerate()
    {
        let logical_page_count = u64::from(position) / PACKED_BATCH_BLOCK_SIZE + 1;
        if logical_page_count > partial_state_capacity {
            return Err(CudaError::out_of_range(
                operation,
                format!(
                    "row {row} requires {logical_page_count} logical pages but partial_state_capacity is {partial_state_capacity}",
                ),
            ));
        }
        let sequence = usize::try_from(sequence_slot).map_err(|_| {
            CudaError::out_of_range(
                operation,
                format!("row {row} sequence slot does not fit usize"),
            )
        })?;
        let offsets = host.sequence_block_offsets();
        let block_begin = u64::from(offsets[sequence]);
        let block_end = u64::from(offsets[sequence + 1]);
        let sequence_page_count = block_end.checked_sub(block_begin).ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                format!("row {row} has an invalid CSR page range"),
            )
        })?;
        if logical_page_count > sequence_page_count {
            return Err(CudaError::out_of_range(
                operation,
                format!(
                    "row {row} prefix requires {logical_page_count} pages but its CSR range exposes {sequence_page_count}",
                ),
            ));
        }
        launch_partial_state_count = launch_partial_state_count.max(logical_page_count);
    }
    require_nonzero(
        operation,
        "launch_partial_state_count",
        launch_partial_state_count,
    )?;
    Ok(launch_partial_state_count)
}

fn validate_fixed37_ragged_logical_tokens(
    operation: &'static str,
    row_positions: &[u32],
) -> CudaResult<u64> {
    let maximum_logical_token_count = row_positions
        .iter()
        .map(|&position| u64::from(position) + 1)
        .max()
        .ok_or_else(|| {
            CudaError::invalid_argument(
                operation,
                "fixed37 ragged attention requires at least one active row",
            )
        })?;
    if maximum_logical_token_count > FIXED37_RAGGED_MAX_LOGICAL_TOKENS {
        return Err(CudaError::out_of_range(
            operation,
            "fixed37 ragged attention requires every row position + 1 to be at most 8192",
        ));
    }
    Ok(maximum_logical_token_count)
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

/// Validates the host mirror required by eager and fixed-address graph indexed
/// RoPE before either path enters native CUDA.
pub(crate) fn validate_indexed_positions(
    positions: &[u32],
    table_position_count: u64,
) -> CudaResult<()> {
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

/// Validates the host mirror shared by eager and fixed-address graph row gather.
///
/// The mirror is intentionally validated without allocating or retaining it:
/// callers still own the device-index bytes and their staging policy.
pub(crate) fn validate_gather_indices(indices: &[u32], input_row_count: u64) -> CudaResult<()> {
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
    fn fixed37_ragged_accepts_t8192_and_rejects_t8193() {
        let maximum = validate_fixed37_ragged_logical_tokens(
            "fixed37_ragged_paged_attention",
            &[0, 35, 36, 37, 8_191],
        )
        .expect("fixed37 row prefixes through T=8192 must be accepted");
        assert_eq!(maximum, 8_192);

        let error =
            validate_fixed37_ragged_logical_tokens("fixed37_ragged_paged_attention", &[8_192])
                .expect_err("fixed37 row prefix T=8193 must fail before launch");
        assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
        assert_eq!(FIXED37_RAGGED_MAX_LOGICAL_TOKENS, 8_192);
    }

    #[test]
    fn native_ragged_d128_two_stage_workspace_is_per_prepared_row() {
        let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new(32, 2_048)
            .expect("M32 and per-row P2048 must fit u64 byte arithmetic");
        assert_eq!(
            layout.partial_state_bytes(),
            32 * 2_048 * 16 * 130 * 4,
            "states must be [M,P,QH,D+2] F32"
        );
        assert_eq!(
            layout.reduction_step_bytes(),
            32 * 2_048 * 16 * 2 * 4,
            "steps must be [M,P,QH,2] F32"
        );
        assert_eq!(
            layout.reduction_normalizer_bytes(),
            32 * 16 * 4,
            "normalizers must be [M,QH] F32"
        );
        assert_eq!(
            layout.total_bytes(),
            layout.partial_state_bytes()
                + layout.reduction_step_bytes()
                + layout.reduction_normalizer_bytes()
        );
        for (output_rows, capacity) in [(0, 1), (1, 0)] {
            let error = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new(
                output_rows,
                capacity,
            )
            .expect_err("zero workspace axis must fail closed");
            assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
        }
    }

    #[test]
    fn native_ragged_d128_two_stage_derives_launch_pages_per_row() {
        // Three independent two-page sequences collectively own six blocks.
        // P=2 is valid because workspace capacity is row-local, and the exact
        // producer extent is L=max(1, 2, 2)=2 rather than the aggregate six.
        let host = PackedBatchHostV1::new(
            &[0, 2, 4, 6],
            &[0, 1, 2, 3, 4, 5],
            &[16, 3, 16, 16, 16, 16],
            &[0, 1, 2],
            &[0, 18, 31],
            6,
        )
        .expect("row-local two-page CSR fixture");
        assert!(host.block_count() > 2);
        assert_eq!(
            native_bf16_ragged_launch_partial_state_count("native_ragged_d128_test", host, 2,)
                .expect("each row fits P=2"),
            2,
        );
        let error =
            native_bf16_ragged_launch_partial_state_count("native_ragged_d128_test", host, 1)
                .expect_err("a two-page row must reject P=1 before launch");
        assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
    }

    #[test]
    fn public_batch_constants_and_symbols_match_the_native_header_source() {
        let header = include_str!("../../../kernels/include/riley_cuda.h");
        assert!(header.contains("#define RILEY_CUDA_PACKED_BATCH_VERSION 1u"));
        assert!(header.contains("#define RILEY_CUDA_PAGED_KV_BLOCK_SIZE 16u"));
        for symbol in [
            "RileyCudaIndexedRopeParams",
            "RileyCudaRowGatherParams",
            "RileyCudaPackedBatchV1",
            "RileyCudaRaggedPagedKvCacheWriteParams",
            "RileyCudaRaggedPagedAttentionParams",
            "RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2",
            "RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3",
            "riley_cuda_indexed_rope_execute",
            "riley_cuda_row_gather_execute",
            "riley_cuda_ragged_paged_kv_cache_write_execute",
            "riley_cuda_ragged_paged_attention_execute",
            "riley_cuda_ragged_paged_attention_grouped_heads_execute",
            "riley_cuda_native_bf16_ragged_paged_split_gqa_d128_two_stage_execute",
            "riley_cuda_native_bf16_ragged_paged_split_gqa_d128_two_stage_v3_execute",
            "RileyCudaFixed37RaggedPagedAttentionParams",
            "riley_cuda_fixed37_ragged_paged_attention_two_pass_execute",
        ] {
            assert!(header.contains(symbol), "native header is missing {symbol}");
        }
        let attention = header
            .split("typedef struct RileyCudaRaggedPagedAttentionParams")
            .nth(1)
            .and_then(|tail| tail.split("} RileyCudaRaggedPagedAttentionParams;").next())
            .expect("ragged attention declaration");
        assert!(
            attention.find("uint64_t head_size;") < attention.find("uint64_t output_row_count;")
        );
        assert!(attention.find("uint64_t output_row_count;") < attention.find("float scale;"));
        let native_ragged_v2 = header
            .split("typedef struct RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2")
            .nth(1)
            .and_then(|tail| {
                tail.split("} RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2;")
                    .next()
            })
            .expect("native ragged D128 V2 declaration");
        assert!(
            native_ragged_v2.find("RileyCudaPackedBatchV1 batch;")
                < native_ragged_v2.find("uint64_t query_head_count;")
        );
        assert!(
            native_ragged_v2.find("uint64_t output_row_count;")
                < native_ragged_v2.find("uint64_t partial_state_capacity;")
        );
        assert!(
            native_ragged_v2.find("uint64_t partial_state_capacity;")
                < native_ragged_v2.find("float scale;")
        );
        assert!(!native_ragged_v2.contains("launch_partial_state_count"));
        let native_ragged_v3 = header
            .split("typedef struct RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3")
            .nth(1)
            .and_then(|tail| {
                tail.split("} RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3;")
                    .next()
            })
            .expect("native ragged D128 V3 declaration");
        assert!(
            native_ragged_v3.find("uint64_t partial_state_capacity;")
                < native_ragged_v3.find("uint64_t launch_partial_state_count;")
        );
        assert!(
            native_ragged_v3.find("uint64_t launch_partial_state_count;")
                < native_ragged_v3.find("float scale;")
        );
        let fixed37_attention = header
            .split("typedef struct RileyCudaFixed37RaggedPagedAttentionParams")
            .nth(1)
            .and_then(|tail| {
                tail.split("} RileyCudaFixed37RaggedPagedAttentionParams;")
                    .next()
            })
            .expect("fixed37 ragged attention declaration");
        assert!(
            fixed37_attention.find("uint64_t output_row_count;")
                < fixed37_attention.find("uint64_t maximum_logical_token_count;")
        );
        assert!(
            fixed37_attention.find("uint64_t maximum_logical_token_count;")
                < fixed37_attention.find("float scale;")
        );
    }
}
