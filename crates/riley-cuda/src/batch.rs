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

/// Query-head count of the geometry the native ragged D128 two-stage path
/// was first qualified on (Qwen2.5-3B); the prompt-row producer still needs it.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT: u64 = 16;
/// KV-head count of that geometry.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_KEY_VALUE_HEAD_COUNT: u64 = 2;
/// Most query heads one KV head may serve on the shared-KV and tensor-core
/// producers (one warp, or one MMA column, per query head of a KV head).
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_MAX_GROUP_HEADS: u64 = 8;
/// Fixed head size of the eager native ragged D128 two-stage path.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE: u64 = 128;
/// The second head size of the two-stage attention: the tensor-core producer
/// with the global-max merge only. A state keeps the D128 stride.
pub const NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D64_HEAD_SIZE: u64 = 64;
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
    request_groups:Option<ValidatedRequestRowGroups<'a>>,
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

/// Safe borrowed proof of canonical contiguous request extents.
#[derive(Clone,Copy,Debug)]
pub struct ValidatedRequestRowGroups<'a>{offsets:&'a [u32],slots:&'a [u32]}
impl<'a> ValidatedRequestRowGroups<'a>{
    #[must_use]
    pub const fn row_sequence_slots(self)->&'a [u32]{self.slots}
    pub fn group_end(self,begin:usize)->CudaResult<usize>{
        let slot=self.slots.get(begin).copied().ok_or_else(||CudaError::invalid_argument("request group","start outside active rows"))? as usize;
        if self.offsets.get(slot).copied()!=Some(begin as u32){return Err(CudaError::invalid_argument("request group","start is not a request boundary"));}
        self.offsets.get(slot+1).copied().map(|v|v as usize).ok_or_else(||CudaError::invalid_argument("request group","missing end boundary"))
    }
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
            request_groups:None,
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

    /// Validates the existing input CSR once; device metadata and page ownership
    /// still pass the canonical constructor above. No rows are reordered.
    pub fn with_request_row_groups(mut self,offsets:&'a [u32])->CudaResult<Self>{
        const OP:&str="PackedBatchHostV1::with_request_row_groups";
        if offsets.len()!=self.sequence_count as usize+1 || offsets.first()!=Some(&0)
            || offsets.last().copied().map(u64::from)!=Some(self.active_row_count) {
            return Err(CudaError::invalid_argument(OP,"request CSR bounds differ from packed rows"));
        }
        for (slot,pair) in offsets.windows(2).enumerate(){
            let begin=pair[0] as usize;let end=pair[1] as usize;
            if begin>=end||end>self.row_sequence_slots.len()
                ||self.row_sequence_slots[begin..end].iter().any(|&s|s as usize!=slot)
                ||self.row_positions[begin..end].windows(2).any(|p|p[0].checked_add(1)!=Some(p[1])) {
                return Err(CudaError::invalid_argument(OP,"request rows are not contiguous canonical CSR groups"));
            }
        }
        self.request_groups=Some(ValidatedRequestRowGroups{offsets,slots:self.row_sequence_slots});Ok(self)
    }
    #[must_use]
    pub const fn request_row_groups(self)->Option<ValidatedRequestRowGroups<'a>>{self.request_groups}

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
        Self::with_query_heads(
            output_row_count,
            partial_state_capacity,
            NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_QUERY_HEAD_COUNT,
        )
    }

    /// Calculates the same spans for a model with `query_head_count` query
    /// heads.
    ///
    /// # Errors
    ///
    /// Returns for a zero axis or byte arithmetic overflow.
    pub fn with_query_heads(
        output_row_count: u64,
        partial_state_capacity: u64,
        query_head_count: u64,
    ) -> CudaResult<Self> {
        const OPERATION: &str = "NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new";
        require_nonzero(OPERATION, "output_row_count", output_row_count)?;
        require_nonzero(OPERATION, "partial_state_capacity", partial_state_capacity)?;
        require_nonzero(OPERATION, "query_head_count", query_head_count)?;
        let partial_state_bytes = checked_bytes(
            OPERATION,
            &[
                output_row_count,
                partial_state_capacity,
                query_head_count,
                NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_PARTIAL_STATE_WIDTH,
                CudaDType::F32.size_bytes(),
            ],
        )?;
        let reduction_step_bytes = checked_bytes(
            OPERATION,
            &[
                output_row_count,
                partial_state_capacity,
                query_head_count,
                NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_TRANSITION_WIDTH,
                CudaDType::F32.size_bytes(),
            ],
        )?;
        let reduction_normalizer_bytes = checked_bytes(
            OPERATION,
            &[
                output_row_count,
                query_head_count,
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
    execute_native_bf16_ragged_d128_two_stage(
        params,
        stream,
        D128_PRODUCER_PER_QUERY_HEAD,
        D128_MERGE_ORDERED,
        0,
        0,
    )
}

/// Same operation and numerical contract as
/// [`native_bf16_ragged_paged_split_gqa_d128_two_stage`], with the partial-state
/// producer launched as one CTA per `(page, KV head, row)` whose query-head
/// warps share a single staged K/V tile. Partial states, transitions,
/// normalizers, and output are bit-identical to the per-query-head launch.
///
/// # Errors
///
/// Identical to [`native_bf16_ragged_paged_split_gqa_d128_two_stage`].
pub fn native_bf16_ragged_paged_split_gqa_d128_two_stage_shared_kv<
    S: CudaExecutionStream + ?Sized,
>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    execute_native_bf16_ragged_d128_two_stage(
        params,
        stream,
        D128_PRODUCER_SHARED_KV,
        D128_MERGE_ORDERED,
        0,
        0,
    )
}

/// Shared-KV producer followed by the single global-max page merge instead of
/// the ordered transition + replay pair.
///
/// The merge weights every page by `exp(m_page - max_page m)` and folds pages
/// in ascending order. It is deterministic and depends only on each row's own
/// pages, but it rounds differently from the ordered merge, so its output is
/// not bit-identical to the other entry points. `reduction_steps` and
/// `reduction_normalizers` are left untouched. Requires ascending order.
///
/// # Errors
///
/// As [`native_bf16_ragged_paged_split_gqa_d128_two_stage`], plus an
/// unsupported error for descending order.
pub fn native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge<
    S: CudaExecutionStream + ?Sized,
>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    execute_native_bf16_ragged_d128_two_stage(
        params,
        stream,
        D128_PRODUCER_SHARED_KV,
        D128_MERGE_GLOBAL_MAX,
        0,
        0,
    )
}

/// [`native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge`] with the
/// prompt rows produced by the tensor-core prefill kernel.
///
/// `row_kinds_first_word` is the index of the first U32 row-kind word inside
/// the buffer that holds `batch`'s row positions; there is one word per
/// launched row and the value `1` marks a token of a request's prompt. A
/// prompt row's partial states are folded four pages at a time with
/// tensor-core arithmetic, so its output differs in rounding from the
/// shared-KV producer; it depends only on the row's own query and context,
/// not on the rows it is batched with or the chunk it arrives in. Rows of
/// every other kind are produced exactly as without this option.
///
/// # Errors
///
/// As [`native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge`], plus
/// an out-of-range error when the row kinds lie outside the buffer.
pub fn native_bf16_ragged_paged_split_gqa_d128_global_merge_prompt_mma<
    S: CudaExecutionStream + ?Sized,
>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    row_kinds_first_word: u64,
    stream: &mut S,
) -> CudaResult<()> {
    let prompt_row_kinds = row_kinds_first_word.checked_add(1).ok_or_else(|| {
        CudaError::out_of_range(
            "native_bf16_ragged_paged_split_gqa_d128_two_stage",
            "row kind offset overflows u64",
        )
    })?;
    execute_native_bf16_ragged_d128_two_stage(
        params,
        stream,
        D128_PRODUCER_SHARED_KV,
        D128_MERGE_GLOBAL_MAX,
        prompt_row_kinds,
        0,
    )
}

/// Tensor-core partial-state producer followed by the global-max page merge.
///
/// The producer keeps the shared-KV launch geometry (one CTA per page, KV head
/// and row) and writes the same page-local softmax states, but computes them
/// with tensor-core arithmetic: scores keep the two BF16 roundings, the FP32
/// softmax weights are carried through the tensor cores as three exact BF16
/// pieces. The states therefore differ from the shared-KV producer's in
/// rounding only. An output depends only on the row's own query and context,
/// not on the rows it is batched with or the chunk a prompt token arrives in.
/// Requires ascending order.
///
/// # Errors
///
/// As [`native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge`].
pub fn native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge<
    S: CudaExecutionStream + ?Sized,
>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    stream: &mut S,
) -> CudaResult<()> {
    execute_native_bf16_ragged_d128_two_stage(
        params,
        stream,
        D128_PRODUCER_TENSOR_CORE,
        D128_MERGE_GLOBAL_MAX,
        0,
        0,
    )
}

/// The tensor-core producer for generated rows together with the generic
/// prompt-row tensor-core producer: a prompt row (row-kind word `1`, see
/// [`native_bf16_ragged_paged_split_gqa_d128_global_merge_prompt_mma`]) folds
/// its partial states four pages at a time, sixteen rows of a sequence to a
/// CTA, for any head geometry and head size the tensor-core producer takes. A
/// prompt row's output rounds differently from the per-page producer's; it
/// depends only on the row's own query and context.
///
/// # Errors
///
/// As [`native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge`],
/// plus an out-of-range error when the row kinds lie outside the buffer.
pub fn native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma<
    S: CudaExecutionStream + ?Sized,
>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    row_kinds_first_word: u64,
    generated_rows: Option<(u64, u64)>,
    stream: &mut S,
) -> CudaResult<()> {
    const OPERATION: &str = "native_bf16_ragged_paged_split_gqa_d128_two_stage";
    let prompt_row_kinds = row_kinds_first_word
        .checked_add(1)
        .ok_or_else(|| CudaError::out_of_range(OPERATION, "row kind offset overflows u64"))?;
    // `[count, rows...]` U32 words in the row-position buffer naming the rows
    // that are not prompt rows: first word and row capacity. The per-page
    // producer then launches over the list instead of every dense row.
    let generated_row_list = match generated_rows {
        None => 0,
        Some((first_word, capacity)) => {
            if first_word >= u64::from(u32::MAX) || capacity == 0 || capacity > u64::from(u32::MAX)
            {
                return Err(CudaError::out_of_range(
                    OPERATION,
                    "generated-row list offset or capacity is out of range",
                ));
            }
            (capacity << 32) | (first_word + 1)
        }
    };
    execute_native_bf16_ragged_d128_two_stage(
        params,
        stream,
        D128_PRODUCER_TENSOR_CORE,
        D128_MERGE_GLOBAL_MAX,
        prompt_row_kinds,
        generated_row_list,
    )
}

const D128_PRODUCER_PER_QUERY_HEAD: u64 = 0;
const D128_PRODUCER_SHARED_KV: u64 = 1;
const D128_PRODUCER_TENSOR_CORE: u64 = 2;
const D128_MERGE_ORDERED: u64 = 0;
const D128_MERGE_GLOBAL_MAX: u64 = 1;

#[allow(clippy::too_many_lines)]
fn execute_native_bf16_ragged_d128_two_stage<S: CudaExecutionStream + ?Sized>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    stream: &mut S,
    producer_layout: u64,
    merge_layout: u64,
    prompt_row_kinds: u64,
    generated_row_list: u64,
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
            producer_layout,
            merge_layout,
            prompt_row_kinds,
            generated_row_list,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            params,
            launch_partial_state_count,
            producer_layout,
            merge_layout,
            prompt_row_kinds,
            generated_row_list,
        );
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
    if params.query_head_count == 0
        || params.key_value_head_count == 0
        || params.query_head_count > MAXIMUM_GRID_Y_OR_Z
        || params.query_head_count % params.key_value_head_count != 0
        || (params.head_size != NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D128_HEAD_SIZE
            && params.head_size != NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_D64_HEAD_SIZE)
    {
        return Err(CudaError::invalid_argument(
            operation,
            "native ragged two-stage attention requires head_size=128 or 64 and query heads that are a multiple of the key/value heads",
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

    let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::with_query_heads(
        params.output_row_count,
        params.partial_state_capacity,
        params.query_head_count,
    )?;
    let query_bytes = checked_bytes(
        operation,
        &[
            host.active_row_count(),
            params.query_head_count,
            params.head_size,
            BF16_BYTES,
        ],
    )?;
    let output_bytes = checked_bytes(
        operation,
        &[
            params.output_row_count,
            params.query_head_count,
            params.head_size,
            BF16_BYTES,
        ],
    )?;
    let pool_bytes = paged_pool_bytes(
        operation,
        host.physical_block_count(),
        params.key_value_head_count,
        params.head_size,
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
/// Host-side input checks for a step replayed from a recorded graph.
///
/// A replay launches no per-operation wrapper, so the checks those wrappers
/// run on every eager step are repeated here: every row position must be
/// inside the RoPE table, every output index must name a distinct dense row,
/// and, when the native D128 attention is used, every row's page prefix must
/// fit both its sequence's block range and the per-row workspace capacity.
///
/// # Errors
///
/// Returns the same out-of-range or invalid-argument errors the eager
/// wrappers would have returned.
pub fn validate_replayed_decode_step(
    batch: PackedBatchV1<'_>,
    positions_host: &[u32],
    table_position_count: u64,
    output_indices_host: &[u32],
    input_row_count: u64,
    native_d128_partial_state_capacity: Option<u64>,
) -> CudaResult<()> {
    validate_indexed_positions(positions_host, table_position_count)?;
    validate_gather_indices(output_indices_host, input_row_count)?;
    if let Some(capacity) = native_d128_partial_state_capacity {
        native_bf16_ragged_launch_partial_state_count(
            "replayed native D128 decode step",
            batch.host(),
            capacity,
        )?;
    }
    Ok(())
}

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

/// Eager materialized BF16 reference. Uses caller-owned prepared resources; graph capture remains rejected.
pub fn prepared_materialized_ragged_attention<S: CudaExecutionStream + ?Sized>(
    query: CudaBufferSpan<'_>, key_pool: CudaBufferSpan<'_>, value_pool: CudaBufferSpan<'_>,
    output: &mut CudaDeviceBuffer, batch: PackedBatchV1<'_>,
    reference: &crate::decode::PreparedPagedDecodeAttention, workspace: &mut CudaDeviceBuffer,
    valid_device: &CudaDeviceBuffer,
    valid_host: &mut [u16], duplicate_scratch: &mut [u8],
    stream: &mut S,
) -> CudaResult<()> {
    use crate::decode::{PagedDecodeAttentionParams,PagedKvBlockTableHostV1,PagedKvBlockTableV1};
    let stream=execution_stream_mut(stream);
    let host=batch.host();
    if reference.backend()!=crate::decode::DecodeAttentionBackend::MaterializedReference {
        return Err(CudaError::invalid_argument("prepared_materialized_ragged_attention","an explicitly prepared materialized reference is required"));
    }
    let request=reference.request();
    let row_bytes=request.query_head_count()*request.head_size()*2;
    if query.dtype()!=CudaDType::BF16 || key_pool.dtype()!=CudaDType::BF16 || value_pool.dtype()!=CudaDType::BF16 || host.physical_block_count()!=request.physical_block_count() {
        return Err(CudaError::invalid_argument("prepared_materialized_ragged_attention","BF16 spans and the prepared physical pool are required"));
    }
    if host.row_positions().iter().any(|p|u64::from(*p)+1>request.maximum_sequence_length()) {
        return Err(CudaError::invalid_argument("prepared_materialized_ragged_attention","row exceeds the prepared sequence bound"));
    }
    let active_bytes=host.active_row_count().checked_mul(row_bytes).ok_or_else(||CudaError::invalid_argument("prepared_materialized_ragged_attention","active output byte count overflows"))?;
    if active_bytes>query.byte_len() || active_bytes>output.byte_len() || output.byte_len()%row_bytes!=0 {
        return Err(CudaError::invalid_argument("prepared_materialized_ragged_attention","prepared output or zero scratch is too small"));
    }
    let workspace_len=workspace.byte_len();
    let valid_stride=valid_host.len();
    for row in 0..host.active_row_count() as usize {
        let tokens=u64::from(host.row_positions()[row])+1;
        let slot=host.row_sequence_slots()[row] as usize;
        let begin=host.sequence_block_offsets()[slot] as usize;
        let end=begin+tokens.div_ceil(16) as usize;
        let count=end-begin;
        if count==0 || count>valid_host.len() {
            return Err(CudaError::invalid_argument("prepared_materialized_ragged_attention","prepared host scratch cannot hold the logical prefix"));
        }
        let valid=&mut valid_host[..count];valid.fill(16);
        valid[count-1]=((tokens-1)%16+1) as u16;
        let table=PagedKvBlockTableHostV1::new_with_duplicate_scratch(&host.block_ids()[begin..end],valid,tokens,host.physical_block_count(),duplicate_scratch)?;
        let ids=batch.device_block_ids();
        reference.execute(&mut PagedDecodeAttentionParams{
            query:CudaBufferSpan::new(query.buffer(),CudaDType::BF16,query.byte_offset()+row as u64*row_bytes,row_bytes)?,
            key_pool,value_pool,
            workspace:CudaBufferSpanMut::new(workspace,reference.workspace_dtype(),0,workspace_len)?,
            output:CudaBufferSpanMut::new(output,CudaDType::BF16,row as u64*row_bytes,row_bytes)?,
            block_table:PagedKvBlockTableV1::new(table,CudaBufferSpan::new(ids.buffer(),CudaDType::U32,ids.byte_offset()+begin as u64*4,(end-begin) as u64*4)?,CudaBufferSpan::new(valid_device,CudaDType::U16,row as u64*valid_stride as u64*2,count as u64*2)?)?,
        },stream)?;
    }
    // Direct reference calls complete eagerly; a command batch owns its one fence.
    Ok(())
}

/// Prepares immutable row-prefix metadata before starting an iteration command batch.
/// All storage is caller-owned; no allocation occurs here or in the layer dispatch.
pub fn prepare_materialized_ragged_attention_metadata(
    host: PackedBatchHostV1<'_>, reference: &crate::decode::PreparedPagedDecodeAttention,
    output: &mut CudaDeviceBuffer, valid_device: &mut CudaDeviceBuffer,
    staging: &mut crate::memory::CudaPinnedHostBuffer, valid_host: &mut [u16],
    encoded_valid: &mut [u8], zero_host: &[u8], stream: &mut CudaStream,
) -> CudaResult<()> {
    let op="prepare_materialized_ragged_attention_metadata";
    let row_bytes=reference.request().query_head_count()*reference.request().head_size()*2;
    let active_bytes=host.active_row_count().checked_mul(row_bytes).ok_or_else(||CudaError::invalid_argument(op,"row bytes overflow"))?;
    let stride=valid_host.len();
    let copy_bytes=(host.active_row_count() as usize).checked_mul(stride).and_then(|n|n.checked_mul(2)).ok_or_else(||CudaError::invalid_argument(op,"metadata bytes overflow"))?;
    if stride==0 || copy_bytes>encoded_valid.len() || copy_bytes as u64>valid_device.byte_len() || active_bytes>output.byte_len() || output.byte_len()-active_bytes>zero_host.len() as u64 {
        return Err(CudaError::invalid_argument(op,"prepared scratch cannot hold the iteration"));
    }
    for (row,raw) in encoded_valid[..copy_bytes].chunks_exact_mut(stride*2).enumerate() {
        let tokens=u64::from(host.row_positions()[row])+1;
        let count=tokens.div_ceil(16) as usize;
        if tokens>reference.request().maximum_sequence_length() || count>stride {return Err(CudaError::invalid_argument(op,"prefix exceeds prepared capacity"));}
        raw.fill(0);valid_host[..count].fill(16);valid_host[count-1]=((tokens-1)%16+1) as u16;
        for (bytes,value) in raw[..count*2].chunks_exact_mut(2).zip(valid_host[..count].iter()) {bytes.copy_from_slice(&value.to_le_bytes());}
    }
    valid_device.upload_from_slice(0,&encoded_valid[..copy_bytes],staging,stream)?;
    let inactive=(output.byte_len()-active_bytes) as usize;
    if inactive>0 {output.upload_from_slice(active_bytes,&zero_host[..inactive],staging,stream)?;}
    Ok(())
}

#[cfg(test)]
mod request_row_group_cache_tests {
    use super::*;
    #[test]
    fn request_cache_requires_canonical_boundaries_and_positions(){
        let offsets=[0,1,2];let blocks=[0,1];let valid=[2,1];let slots=[0,0,1];let positions=[0,1,0];
        let host=PackedBatchHostV1::new(&offsets,&blocks,&valid,&slots,&positions,2).unwrap();
        let rows=[0,2,3];let groups=host.with_request_row_groups(&rows).unwrap().request_row_groups().unwrap();
        assert_eq!(groups.group_end(0).unwrap(),2);assert_eq!(groups.group_end(2).unwrap(),3);assert!(groups.group_end(1).is_err());assert!(groups.group_end(3).is_err());
        assert!(host.with_request_row_groups(&[0,1,3]).is_err());assert!(host.with_request_row_groups(&[0,2,2]).is_err());
        let positions=[1,0,0];let host=PackedBatchHostV1::new(&offsets,&blocks,&valid,&slots,&positions,2).unwrap();assert!(host.with_request_row_groups(&rows).is_err());
    }
}
