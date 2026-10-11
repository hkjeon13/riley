//! Prompt rows of the D128 two-stage attention produced by the tensor-core
//! prefill kernel: a prompt row's output must not depend on the rows it is
//! batched with, on tile alignment or on the chunk it arrives in; every other
//! row must keep the shared-KV producer's bytes.
#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, DecodePartialReductionOrder,
    NativeBf16RaggedPagedSplitGqaD128TwoStageParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout, PackedBatchHostV1, PackedBatchV1,
    native_bf16_ragged_paged_split_gqa_d128_global_merge_prompt_mma,
    native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_SIZE: usize = 128;
const PAGE_SIZE: usize = 16;
const SCALE: f32 = 0.088_388_346;
const SEQUENCES: usize = 6;
const PAGES_PER_SEQUENCE: usize = 12;
const PARTIAL_CAPACITY: usize = 16;
const ROW_WIDTH: usize = QUERY_HEADS * HEAD_SIZE;

/// One logical row: sequence, position and whether it is a prompt token.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Row {
    sequence: u32,
    position: u32,
    prompt: bool,
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

/// Runs the attention over `rows` in the given order and returns each row's
/// output bytes. `prompt_mma` selects the tensor-core producer for prompt rows.
fn run(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    pools: &Pools,
    rows: &[Row],
    prompt_mma: bool,
) -> TestResult<Vec<Vec<u8>>> {
    let row_count = rows.len();
    let slots: Vec<u32> = rows.iter().map(|row| row.sequence).collect();
    let positions: Vec<u32> = rows.iter().map(|row| row.position).collect();
    let kinds: Vec<u32> = rows.iter().map(|row| u32::from(row.prompt)).collect();
    let host = PackedBatchHostV1::new(
        &pools.sequence_block_offsets,
        &pools.block_ids,
        &pools.valid_tokens,
        &slots,
        &positions,
        u64::try_from(SEQUENCES * PAGES_PER_SEQUENCE)?,
    )?;
    let query_bytes: Vec<u8> = rows.iter().flat_map(|row| row_query(*row)).collect();
    let query = upload(context, stream, staging, &query_bytes)?;
    let row_slots = upload(context, stream, staging, &encode_u32(&slots))?;
    // Row positions followed by the row kinds, as the executor lays them out.
    let mut position_words = positions.clone();
    position_words.extend_from_slice(&kinds);
    let position_buffer = upload(context, stream, staging, &encode_u32(&position_words))?;
    let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new(
        u64::try_from(row_count)?,
        u64::try_from(PARTIAL_CAPACITY)?,
    )?;
    let mut partial_states = context.allocate_device_buffer(layout.partial_state_bytes())?;
    let mut reduction_steps = context.allocate_device_buffer(layout.reduction_step_bytes())?;
    let mut reduction_normalizers =
        context.allocate_device_buffer(layout.reduction_normalizer_bytes())?;
    let mut output = upload(
        context,
        stream,
        staging,
        &vec![0x5a_u8; row_count * ROW_WIDTH * 2],
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
                    u64::try_from(row_count * 4)?,
                )?,
            )?,
            query_head_count: u64::try_from(QUERY_HEADS)?,
            key_value_head_count: u64::try_from(KEY_VALUE_HEADS)?,
            head_size: u64::try_from(HEAD_SIZE)?,
            output_row_count: u64::try_from(row_count)?,
            partial_state_capacity: u64::try_from(PARTIAL_CAPACITY)?,
            scale: SCALE,
            reduction_order: DecodePartialReductionOrder::LogicalAscending,
        };
        if prompt_mma {
            native_bf16_ragged_paged_split_gqa_d128_global_merge_prompt_mma(
                &mut params,
                u64::try_from(row_count)?,
                stream,
            )?;
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
    Ok(bytes
        .chunks_exact(ROW_WIDTH * 2)
        .map(<[u8]>::to_vec)
        .collect())
}

fn prompt_rows(sequence: u32, positions: std::ops::Range<u32>) -> Vec<Row> {
    positions
        .map(|position| Row {
            sequence,
            position,
            prompt: true,
        })
        .collect()
}

fn decode_row(sequence: u32, position: u32) -> Row {
    Row {
        sequence,
        position,
        prompt: false,
    }
}

#[test]
#[ignore = "remote GPU"]
fn prompt_rows_are_independent_of_batching_alignment_and_chunking() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(32 << 20)?;

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

    // Arrangement A: prompt runs and decode rows interleaved.
    let mut arrangement_a = prompt_rows(0, 0..40);
    arrangement_a.push(decode_row(1, 37));
    arrangement_a.extend(prompt_rows(2, 100..131));
    arrangement_a.push(decode_row(3, 5));
    arrangement_a.extend(prompt_rows(4, 0..1));
    arrangement_a.push(decode_row(5, 160));
    arrangement_a.extend(prompt_rows(5, 63..66));
    // Arrangement B: the same rows in another order, so every prompt row sits
    // in a different tile slot next to different neighbours.
    let mut arrangement_b = vec![decode_row(3, 5), decode_row(5, 160), decode_row(1, 37)];
    arrangement_b.extend(prompt_rows(5, 63..66));
    arrangement_b.extend(prompt_rows(2, 100..131));
    arrangement_b.extend(prompt_rows(4, 0..1));
    arrangement_b.extend(prompt_rows(0, 0..40));
    // Arrangement C: the prompts cut into chunks that arrive in separate calls.
    let chunks: [Vec<Row>; 4] = [
        prompt_rows(0, 0..7),
        [prompt_rows(0, 7..33), prompt_rows(2, 100..101)].concat(),
        [prompt_rows(2, 101..131), prompt_rows(0, 33..40)].concat(),
        [prompt_rows(5, 63..66), prompt_rows(4, 0..1)].concat(),
    ];

    let reference = run(&context, &mut stream, &mut staging, &pools, &arrangement_a, false)?;
    let output_a = run(&context, &mut stream, &mut staging, &pools, &arrangement_a, true)?;
    let output_b = run(&context, &mut stream, &mut staging, &pools, &arrangement_b, true)?;
    let mut by_row = std::collections::HashMap::new();
    for (row, bytes) in arrangement_a.iter().zip(&output_a) {
        by_row.insert(*row, bytes.clone());
    }

    let mut prompt_elements = 0_usize;
    let mut prompt_identical_to_shared_kv = 0_usize;
    let mut worst = 0.0_f32;
    for ((row, mma), shared) in arrangement_a.iter().zip(&output_a).zip(&reference) {
        if row.prompt {
            for (left, right) in mma.chunks_exact(2).zip(shared.chunks_exact(2)) {
                let a = bf16_value(u16::from_ne_bytes([left[0], left[1]]));
                let b = bf16_value(u16::from_ne_bytes([right[0], right[1]]));
                assert!(a.is_finite() && b.is_finite(), "non-finite attention output");
                worst = worst.max((a - b).abs());
                prompt_elements += 1;
                prompt_identical_to_shared_kv += usize::from(left == right);
            }
        } else {
            assert_eq!(mma, shared, "{row:?}: a non-prompt row changed bytes");
        }
    }
    assert!(
        worst <= 0.03125,
        "prompt rows differ from the shared-KV producer by {worst}"
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
        "prompt-mma rows={} prompt_elements={prompt_elements} identical_to_shared_kv={prompt_identical_to_shared_kv} max_abs_difference={worst} rearranged_rows={} chunked_rows={chunked_rows}",
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
