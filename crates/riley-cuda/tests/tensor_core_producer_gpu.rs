//! Tensor-core page-state producer of the D128 two-stage attention: a row's
//! output must not depend on the rows it is batched with, on its index or on
//! the chunk it arrives in; rows past the active count are zeros; and the
//! result stays within rounding distance of the shared-KV producer.
#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, DecodePartialReductionOrder,
    NativeBf16RaggedPagedSplitGqaD128TwoStageParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout, PackedBatchHostV1, PackedBatchV1,
    native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_SIZE: usize = 128;
const PAGE_SIZE: usize = 16;
const SCALE: f32 = 0.088_388_346;
const SEQUENCES: usize = 6;
const PAGES_PER_SEQUENCE: usize = 20;
const PAGE_CAPACITY: usize = 24;
const ROW_WIDTH: usize = QUERY_HEADS * HEAD_SIZE;
/// Output rows launched past the active rows; they must come back as zeros.
const PADDING_ROWS: usize = 3;

/// One logical row: sequence and position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Row {
    sequence: u32,
    position: u32,
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn pattern(count: usize, multiplier: usize, offset: usize, modulus: usize, scale: f32) -> Vec<u8> {
    (0..count)
        .flat_map(|index| {
            let centered = ((index * multiplier + offset) % modulus) as f32 - modulus as f32 / 2.0;
            bf16_bits(centered / scale).to_ne_bytes()
        })
        .collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn upload(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    bytes: &[u8],
) -> TestResult<CudaDeviceBuffer> {
    let mut buffer = context.allocate_device_buffer(u64::try_from(bytes.len())?)?;
    buffer.upload_from_slice(0, bytes, staging, stream)?;
    Ok(buffer)
}

fn download(
    context: &CudaContext,
    stream: &mut CudaStream,
    buffer: &mut CudaDeviceBuffer,
) -> TestResult<Vec<u8>> {
    let mut staging = context.allocate_pinned_host_buffer(buffer.byte_len())?;
    buffer
        .copy_to_pinned_async(0, &mut staging, 0, buffer.byte_len(), stream)?
        .synchronize()?;
    let bytes = staging.to_vec()?;
    staging.close()?;
    Ok(bytes)
}

/// The query of a logical row is a function of the row alone, so the same row
/// carries the same query in every arrangement.
fn row_query(row: Row) -> Vec<u8> {
    let seed = row.sequence as usize * 7_919 + row.position as usize * 104_729;
    pattern(ROW_WIDTH, 11, seed % 9_973, 29, 9.0)
}

struct Pools {
    key_pool: CudaDeviceBuffer,
    value_pool: CudaDeviceBuffer,
    offsets: CudaDeviceBuffer,
    ids: CudaDeviceBuffer,
    valid: CudaDeviceBuffer,
    sequence_block_offsets: Vec<u32>,
    block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
}

/// Runs the attention over `rows` in the given order and returns each active
/// row's output bytes. `tensor_core` selects the tensor-core producer.
fn run(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    pools: &Pools,
    rows: &[Row],
    tensor_core: bool,
) -> TestResult<Vec<Vec<u8>>> {
    let row_count = rows.len();
    let output_rows = row_count + PADDING_ROWS;
    let slots: Vec<u32> = rows.iter().map(|row| row.sequence).collect();
    let positions: Vec<u32> = rows.iter().map(|row| row.position).collect();
    let host = PackedBatchHostV1::new(
        &pools.sequence_block_offsets,
        &pools.block_ids,
        &pools.valid_tokens,
        &slots,
        &positions,
        u64::try_from(SEQUENCES * PAGES_PER_SEQUENCE)?,
    )?;
    let mut query_bytes: Vec<u8> = rows.iter().flat_map(|row| row_query(*row)).collect();
    query_bytes.resize(output_rows * ROW_WIDTH * 2, 0);
    let query = upload(context, stream, staging, &query_bytes)?;
    let row_slots = upload(context, stream, staging, &encode_u32(&slots))?;
    let position_buffer = upload(context, stream, staging, &encode_u32(&positions))?;
    let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new(
        u64::try_from(output_rows)?,
        u64::try_from(PAGE_CAPACITY)?,
    )?;
    let mut partial_states = context.allocate_device_buffer(layout.partial_state_bytes())?;
    let mut reduction_steps = context.allocate_device_buffer(layout.reduction_step_bytes())?;
    let mut reduction_normalizers =
        context.allocate_device_buffer(layout.reduction_normalizer_bytes())?;
    let mut output = upload(
        context,
        stream,
        staging,
        &vec![0x5a_u8; output_rows * ROW_WIDTH * 2],
    )?;
    {
        let (states_len, steps_len, normalizers_len, output_len) = (
            partial_states.byte_len(),
            reduction_steps.byte_len(),
            reduction_normalizers.byte_len(),
            output.byte_len(),
        );
        let mut params = NativeBf16RaggedPagedSplitGqaD128TwoStageParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(
                &pools.key_pool,
                CudaDType::BF16,
                0,
                pools.key_pool.byte_len(),
            )?,
            value_pool: CudaBufferSpan::new(
                &pools.value_pool,
                CudaDType::BF16,
                0,
                pools.value_pool.byte_len(),
            )?,
            partial_states: CudaBufferSpanMut::new(
                &mut partial_states,
                CudaDType::F32,
                0,
                states_len,
            )?,
            reduction_steps: CudaBufferSpanMut::new(
                &mut reduction_steps,
                CudaDType::F32,
                0,
                steps_len,
            )?,
            reduction_normalizers: CudaBufferSpanMut::new(
                &mut reduction_normalizers,
                CudaDType::F32,
                0,
                normalizers_len,
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: PackedBatchV1::new(
                host,
                CudaBufferSpan::new(&pools.offsets, CudaDType::U32, 0, pools.offsets.byte_len())?,
                CudaBufferSpan::new(&pools.ids, CudaDType::U32, 0, pools.ids.byte_len())?,
                CudaBufferSpan::new(&pools.valid, CudaDType::U16, 0, pools.valid.byte_len())?,
                CudaBufferSpan::new(&row_slots, CudaDType::U32, 0, row_slots.byte_len())?,
                CudaBufferSpan::new(
                    &position_buffer,
                    CudaDType::U32,
                    0,
                    position_buffer.byte_len(),
                )?,
            )?,
            query_head_count: u64::try_from(QUERY_HEADS)?,
            key_value_head_count: u64::try_from(KEY_VALUE_HEADS)?,
            head_size: u64::try_from(HEAD_SIZE)?,
            output_row_count: u64::try_from(output_rows)?,
            partial_state_capacity: u64::try_from(PAGE_CAPACITY)?,
            scale: SCALE,
            reduction_order: DecodePartialReductionOrder::LogicalAscending,
        };
        if tensor_core {
            native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge(&mut params, stream)?;
        } else {
            native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge(&mut params, stream)?;
        }
    }
    stream.synchronize()?;
    let bytes = download(context, stream, &mut output)?;
    for buffer in [
        query,
        row_slots,
        position_buffer,
        partial_states,
        reduction_steps,
        reduction_normalizers,
        output,
    ] {
        buffer.close()?;
    }
    let (active, padding) = bytes.split_at(row_count * ROW_WIDTH * 2);
    assert!(
        padding.iter().all(|byte| *byte == 0),
        "rows past the active count must be zeros"
    );
    Ok(active
        .chunks_exact(ROW_WIDTH * 2)
        .map(<[u8]>::to_vec)
        .collect())
}

fn rows(sequence: u32, positions: impl IntoIterator<Item = u32>) -> Vec<Row> {
    positions
        .into_iter()
        .map(|position| Row { sequence, position })
        .collect()
}

#[test]
#[ignore = "remote GPU"]
fn tensor_core_outputs_are_independent_of_batching_order_and_chunking() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(64 << 20)?;

    let physical_blocks = SEQUENCES * PAGES_PER_SEQUENCE;
    let pool_elements = physical_blocks * KEY_VALUE_HEADS * PAGE_SIZE * HEAD_SIZE;
    let sequence_block_offsets: Vec<u32> = (0..=SEQUENCES)
        .map(|sequence| (sequence * PAGES_PER_SEQUENCE) as u32)
        .collect();
    // Shuffled physical pages.
    let block_ids: Vec<u32> = (0..physical_blocks)
        .map(|block| ((block * 29 + 5) % physical_blocks) as u32)
        .collect();
    let valid_tokens = vec![PAGE_SIZE as u16; physical_blocks];
    let pools = Pools {
        key_pool: upload(
            &context,
            &mut stream,
            &mut staging,
            &pattern(pool_elements, 5, 3, 31, 10.0),
        )?,
        value_pool: upload(
            &context,
            &mut stream,
            &mut staging,
            &pattern(pool_elements, 13, 9, 37, 24.0),
        )?,
        offsets: upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u32(&sequence_block_offsets),
        )?,
        ids: upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?,
        valid: upload(
            &context,
            &mut stream,
            &mut staging,
            &valid_tokens
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<u8>>(),
        )?,
        sequence_block_offsets,
        block_ids,
        valid_tokens,
    };

    // Arrangement A: prompt runs and single rows of every page class: one
    // page, a page boundary on either side, many pages, the last position.
    let mut arrangement_a = rows(0, 0..40);
    arrangement_a.extend(rows(1, [37, 15, 16, 127, 128]));
    arrangement_a.extend(rows(2, 100..131));
    arrangement_a.extend(rows(3, [0, 5, 143, 144, 255, 256, 319]));
    arrangement_a.extend(rows(4, 280..300));
    arrangement_a.extend(rows(5, [160, 63, 64, 65, 271]));
    // Arrangement B: the same rows in another order, next to other neighbours.
    let mut arrangement_b = arrangement_a.clone();
    arrangement_b.reverse();
    arrangement_b.rotate_left(17);
    // Arrangement C: the same rows cut into separate calls.
    let chunks: Vec<&[Row]> = vec![
        &arrangement_a[..1],
        &arrangement_a[1..8],
        &arrangement_a[8..41],
        &arrangement_a[41..90],
        &arrangement_a[90..],
    ];

    let reference = run(&context, &mut stream, &mut staging, &pools, &arrangement_a, false)?;
    let output_a = run(&context, &mut stream, &mut staging, &pools, &arrangement_a, true)?;
    let output_b = run(&context, &mut stream, &mut staging, &pools, &arrangement_b, true)?;
    let mut by_row = std::collections::HashMap::new();
    for (row, bytes) in arrangement_a.iter().zip(&output_a) {
        by_row.insert(*row, bytes.clone());
    }

    let mut elements = 0_usize;
    let mut identical = 0_usize;
    let mut worst = 0.0_f32;
    for (tensor_core, shared_kv) in output_a.iter().zip(&reference) {
        for (left, right) in tensor_core.chunks_exact(2).zip(shared_kv.chunks_exact(2)) {
            let a = bf16_value(u16::from_ne_bytes([left[0], left[1]]));
            let b = bf16_value(u16::from_ne_bytes([right[0], right[1]]));
            assert!(a.is_finite() && b.is_finite(), "non-finite attention output");
            worst = worst.max((a - b).abs());
            elements += 1;
            identical += usize::from(left == right);
        }
    }
    assert!(
        worst <= 0.03125,
        "the tensor-core producer differs from the shared-KV producer by {worst}"
    );
    for (row, bytes) in arrangement_b.iter().zip(&output_b) {
        assert_eq!(
            bytes, &by_row[row],
            "{row:?}: output depends on the batch arrangement"
        );
    }
    let mut chunked_rows = 0_usize;
    for chunk in &chunks {
        let output = run(&context, &mut stream, &mut staging, &pools, chunk, true)?;
        for (row, bytes) in chunk.iter().zip(&output) {
            assert_eq!(bytes, &by_row[row], "{row:?}: output depends on the chunk");
            chunked_rows += 1;
        }
    }
    println!(
        "tensor-core-producer rows={} elements={elements} identical_to_shared_kv={identical} max_abs_difference={worst} rearranged_rows={} chunked_rows={chunked_rows}",
        arrangement_a.len(),
        arrangement_b.len(),
    );

    let Pools {
        key_pool,
        value_pool,
        offsets,
        ids,
        valid,
        ..
    } = pools;
    for buffer in [key_pool, value_pool, offsets, ids, valid] {
        buffer.close()?;
    }
    staging.close()?;
    stream.close()?;
    context.synchronize()?;
    context.close()?;
    Ok(())
}
