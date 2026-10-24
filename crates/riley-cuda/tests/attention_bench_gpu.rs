//! Kernel-level timing of the D128 two-stage attention on Qwen3-4B
//! geometry (32 query heads, 8 KV heads, head 128, 16-token pages) for step
//! shapes taken from the natural C32 workload: a decode-only step of 32 rows
//! and a mixed step of 31 decode rows plus a 224-token prompt chunk. Each
//! mode is timed over repeated launches on one stream (P0 harness of
//! docs/plan-fixed-split-varlen-attention-20261023.md). Diagnostic only.
#![allow(clippy::too_many_lines)]

use std::error::Error;
use std::time::Instant;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, DecodePartialReductionOrder,
    NativeBf16RaggedPagedSplitGqaD128TwoStageParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout, PackedBatchHostV1, PackedBatchV1,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma,
    set_native_d128_attention_flashinfer, set_native_d128_prompt_direct,
    set_native_d128_prompt_direct_max_pages,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const QUERY_HEADS: usize = 32;
const KEY_VALUE_HEADS: usize = 8;
const HEAD_SIZE: usize = 128;
const PAGE_SIZE: usize = 16;
const SEQUENCES: usize = 32;
const PAGES_PER_SEQUENCE: usize = 64;
const PAGE_CAPACITY: usize = 64;
const SCALE: f32 = 0.088_388_346;
const WARMUP: usize = 20;
const ITERATIONS: usize = 200;

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
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
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

#[allow(dead_code)]
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


/// Context length of sequence `s` in a natural-workload steady state: the
/// 16-, 128- and 398-token prompts with part of their 32/64/128 outputs.
fn natural_context(sequence: usize) -> u32 {
    let length = match sequence % 3 {
        0 => 16 + (sequence * 7) % 32,
        1 => 128 + (sequence * 13) % 64,
        _ => 398 + (sequence * 17) % 128,
    };
    length as u32
}

/// Rows `(sequence, position, kind)`; kind 1 is a prompt row.
fn step_rows(mixed: bool) -> Vec<(u32, u32, u32)> {
    let mut rows = Vec::new();
    if mixed {
        rows.extend((0..224).map(|position| (0, position, 1)));
    }
    let first = usize::from(mixed);
    for sequence in first..SEQUENCES {
        rows.push((sequence as u32, natural_context(sequence) - 1, 0));
    }
    rows
}

fn time_mode(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    pools: &(CudaDeviceBuffer, CudaDeviceBuffer, CudaDeviceBuffer, CudaDeviceBuffer, CudaDeviceBuffer, Vec<u32>, Vec<u32>, Vec<u16>),
    rows: &[(u32, u32, u32)],
) -> TestResult<f64> {
    let (key_pool, value_pool, offsets, ids, valid, sequence_block_offsets, block_ids, valid_tokens) = pools;
    let row_count = rows.len();
    let slots: Vec<u32> = rows.iter().map(|row| row.0).collect();
    let positions: Vec<u32> = rows.iter().map(|row| row.1).collect();
    let mut words = positions.clone();
    words.extend(rows.iter().map(|row| row.2));
    let host = PackedBatchHostV1::new(
        sequence_block_offsets,
        block_ids,
        valid_tokens,
        &slots,
        &positions,
        u64::try_from(SEQUENCES * PAGES_PER_SEQUENCE)?,
    )?;
    let row_width = QUERY_HEADS * HEAD_SIZE;
    let query = upload(context, stream, staging, &pattern(row_count * row_width, 11, 7, 29, 9.0))?;
    let row_slots = upload(context, stream, staging, &encode_u32(&slots))?;
    let position_buffer = upload(context, stream, staging, &encode_u32(&words))?;
    let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::with_query_heads(
        u64::try_from(row_count)?,
        u64::try_from(PAGE_CAPACITY)?,
        u64::try_from(QUERY_HEADS)?,
    )?;
    let mut partial_states = context.allocate_device_buffer(layout.partial_state_bytes())?;
    let mut reduction_steps = context.allocate_device_buffer(layout.reduction_step_bytes())?;
    let mut reduction_normalizers = context.allocate_device_buffer(layout.reduction_normalizer_bytes())?;
    let mut output = context.allocate_device_buffer(u64::try_from(row_count * row_width * 2)?)?;
    let elapsed;
    {
        let (states_len, steps_len, normalizers_len, output_len) = (
            partial_states.byte_len(),
            reduction_steps.byte_len(),
            reduction_normalizers.byte_len(),
            output.byte_len(),
        );
        let mut params = NativeBf16RaggedPagedSplitGqaD128TwoStageParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(value_pool, CudaDType::BF16, 0, value_pool.byte_len())?,
            partial_states: CudaBufferSpanMut::new(&mut partial_states, CudaDType::F32, 0, states_len)?,
            reduction_steps: CudaBufferSpanMut::new(&mut reduction_steps, CudaDType::F32, 0, steps_len)?,
            reduction_normalizers: CudaBufferSpanMut::new(
                &mut reduction_normalizers,
                CudaDType::F32,
                0,
                normalizers_len,
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: PackedBatchV1::new(
                host,
                CudaBufferSpan::new(offsets, CudaDType::U32, 0, offsets.byte_len())?,
                CudaBufferSpan::new(ids, CudaDType::U32, 0, ids.byte_len())?,
                CudaBufferSpan::new(valid, CudaDType::U16, 0, valid.byte_len())?,
                CudaBufferSpan::new(&row_slots, CudaDType::U32, 0, row_slots.byte_len())?,
                CudaBufferSpan::new(&position_buffer, CudaDType::U32, 0, position_buffer.byte_len())?,
            )?,
            query_head_count: u64::try_from(QUERY_HEADS)?,
            key_value_head_count: u64::try_from(KEY_VALUE_HEADS)?,
            head_size: u64::try_from(HEAD_SIZE)?,
            output_row_count: u64::try_from(row_count)?,
            partial_state_capacity: u64::try_from(PAGE_CAPACITY)?,
            scale: SCALE,
            reduction_order: DecodePartialReductionOrder::LogicalAscending,
        };
        let first_word = u64::try_from(row_count)?;
        for _ in 0..WARMUP {
            native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma(
                &mut params, first_word, None, stream,
            )?;
        }
        stream.synchronize()?;
        let started = Instant::now();
        for _ in 0..ITERATIONS {
            native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma(
                &mut params, first_word, None, stream,
            )?;
        }
        stream.synchronize()?;
        elapsed = started.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64;
    }
    for buffer in [query, row_slots, position_buffer, partial_states, reduction_steps, reduction_normalizers, output] {
        buffer.close()?;
    }
    Ok(elapsed)
}

#[test]
#[ignore = "remote GPU"]
fn attention_step_shapes_timing() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(256 << 20)?;
    let physical_blocks = SEQUENCES * PAGES_PER_SEQUENCE;
    let pool_elements = physical_blocks * KEY_VALUE_HEADS * PAGE_SIZE * HEAD_SIZE;
    let sequence_block_offsets: Vec<u32> = (0..=SEQUENCES)
        .map(|sequence| (sequence * PAGES_PER_SEQUENCE) as u32)
        .collect();
    let block_ids: Vec<u32> = (0..physical_blocks)
        .map(|block| ((block * 29 + 5) % physical_blocks) as u32)
        .collect();
    let valid_tokens = vec![PAGE_SIZE as u16; physical_blocks];
    let pools = (
        upload(&context, &mut stream, &mut staging, &pattern(pool_elements, 5, 3, 31, 10.0))?,
        upload(&context, &mut stream, &mut staging, &pattern(pool_elements, 13, 9, 37, 24.0))?,
        upload(&context, &mut stream, &mut staging, &encode_u32(&sequence_block_offsets))?,
        upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?,
        upload(
            &context,
            &mut stream,
            &mut staging,
            &valid_tokens.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<u8>>(),
        )?,
        sequence_block_offsets,
        block_ids,
        valid_tokens,
    );
    for (mode, direct, max_pages, flashinfer) in [
        ("page", 0, 0, 0),
        ("direct", 1, 0, 0),
        ("flashinfer", 0, 0, 1),
    ] {
        set_native_d128_prompt_direct(direct);
        set_native_d128_prompt_direct_max_pages(max_pages);
        set_native_d128_attention_flashinfer(flashinfer);
        for (shape, mixed) in [("decode32", false), ("mixed31+224", true)] {
            let rows = step_rows(mixed);
            let microseconds = time_mode(&context, &mut stream, &mut staging, &pools, &rows)?;
            println!("ATTN mode={mode} shape={shape} rows={} us_per_layer={microseconds:.1}", rows.len());
        }
    }
    set_native_d128_prompt_direct(0);
    set_native_d128_prompt_direct_max_pages(0);
    set_native_d128_attention_flashinfer(0);
    let (key_pool, value_pool, offsets, ids, valid, ..) = pools;
    for buffer in [key_pool, value_pool, offsets, ids, valid] {
        buffer.close()?;
    }
    staging.close()?;
    stream.close()?;
    context.synchronize()?;
    context.close()?;
    Ok(())
}
