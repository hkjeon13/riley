//! The head-norm form of the merged-QKV split (no bias; an RMS norm over every
//! query head and key head before the rotary embedding, Qwen3) must store the
//! same bytes as the separate launches it replaces: slicing the merged matrix,
//! the BF16 RMS norm on rows of one head, the indexed RoPE and the KV cache
//! write. Checked for the immediate launch and for the launch deferred to the
//! cache write.

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, IndexedRopeParams, PackedBatchHostV1,
    PackedBatchV1, QkvRopeParams, QkvSplitBiasParams, RaggedPagedKvCacheWriteParams, RmsNormParams,
    indexed_rope, qkv_split_bias_rope, ragged_paged_kv_cache_write, rms_norm,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const HEAD: usize = 128;
const BLOCK: usize = 16;

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn encode_bf16(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .flat_map(|value| bf16_bits(value).to_ne_bytes())
        .collect()
}

fn encode_f32(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_ne_bytes).collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn encode_u16(values: &[u16]) -> Vec<u8> {
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

fn span(buffer: &CudaDeviceBuffer, dtype: CudaDType) -> TestResult<CudaBufferSpan<'_>> {
    Ok(CudaBufferSpan::new(buffer, dtype, 0, buffer.byte_len())?)
}

#[test]
#[ignore = "remote GPU"]
fn head_norm_split_matches_slice_norm_rope_and_cache_write() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 23)?;

    // Two sequences; five active rows cross a page boundary and a sixth dense
    // row is padding: it gets a value output but no norm, rotation or cache slot.
    let sequence_block_offsets = [0_u32, 2, 3];
    let block_ids = [4_u32, 1, 3];
    let valid_tokens = [16_u16, 2, 5];
    let row_sequence_slots = [0_u32, 1, 0, 0, 1];
    let row_positions = [16_u32, 4, 15, 17, 0, 0];
    let table_positions = 19_usize;
    let rotated_rows = 5_usize;
    let rows = row_positions.len();
    let physical_blocks = 6_usize;
    // Query wider than a "hidden" multiple on purpose: 6 query heads, 2 KV heads.
    let query_heads = 6_usize;
    let key_value_heads = 2_usize;
    let query_width = query_heads * HEAD;
    let key_value_width = key_value_heads * HEAD;
    let merged_width = query_width + 2 * key_value_width;
    let epsilon = 1.0e-6_f32;
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions[..rotated_rows],
        u64::try_from(physical_blocks)?,
    )?;

    // Values over several binades and both signs, so the norm's reduction
    // order and both BF16 rounding boundaries matter.
    let merged_values: Vec<f32> = (0..rows * merged_width)
        .map(|i| {
            let unit = ((i * 2_654_435_761_usize) >> 7) % 4093;
            let magnitude = [0.031_f32, 0.27, 1.9, 7.3][(i / 5) % 4];
            (unit as f32 / 2046.5 - 1.0) * magnitude
        })
        .collect();
    let slice = |offset: usize, width: usize| -> Vec<u8> {
        encode_bf16((0..rows).flat_map(|row| {
            let begin = row * merged_width + offset;
            merged_values[begin..begin + width].iter().copied()
        }))
    };
    let merged = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(merged_values.iter().copied()),
    )?;
    let query_raw = upload(&context, &mut stream, &mut staging, &slice(0, query_width))?;
    let key_raw = upload(
        &context,
        &mut stream,
        &mut staging,
        &slice(query_width, key_value_width),
    )?;
    let value_raw_bytes = slice(query_width + key_value_width, key_value_width);
    let query_scale = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16((0..HEAD).map(|i| 0.6 + ((i * 29 + 3) % 61) as f32 / 40.0)),
    )?;
    let key_scale = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16((0..HEAD).map(|i| 0.4 + ((i * 17 + 7) % 53) as f32 / 30.0)),
    )?;
    let cos = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_f32((0..table_positions * HEAD / 2).map(|i| (i as f32 * 0.013).cos())),
    )?;
    let sin = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_f32((0..table_positions * HEAD / 2).map(|i| (i as f32 * 0.013).sin())),
    )?;
    let offsets_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let block_ids_device = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
    let valid_tokens_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;

    let sentinel = |elements: usize| encode_bf16(std::iter::repeat_n(-113.0_f32, elements));
    let pool_elements = physical_blocks * key_value_heads * BLOCK * HEAD;
    let mut results = Vec::new();
    // 0: separate launches (reference), 1: head-norm split launched at once,
    // 2: head-norm split deferred to the cache write.
    for variant in 0..3 {
        let mut query = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(rows * query_width),
        )?;
        let mut key = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(rows * key_value_width),
        )?;
        let mut value = upload(
            &context,
            &mut stream,
            &mut staging,
            &if variant == 0 {
                value_raw_bytes.clone()
            } else {
                sentinel(rows * key_value_width)
            },
        )?;
        let mut key_pool = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(pool_elements),
        )?;
        let mut value_pool = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(pool_elements),
        )?;
        let mut query_normed = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(rows * query_width),
        )?;
        let mut key_normed = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(rows * key_value_width),
        )?;
        if variant == 0 {
            for (input, scale, output, heads) in [
                (&query_raw, &query_scale, &mut query_normed, query_heads),
                (&key_raw, &key_scale, &mut key_normed, key_value_heads),
            ] {
                let output_len = output.byte_len();
                rms_norm(
                    &mut RmsNormParams {
                        input: span(input, CudaDType::BF16)?,
                        weight: span(scale, CudaDType::BF16)?,
                        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_len)?,
                        row_count: u64::try_from(rotated_rows * heads)?,
                        hidden_size: u64::try_from(HEAD)?,
                        epsilon,
                    },
                    &mut stream,
                )?;
            }
            for (input, output, heads) in [
                (&query_normed, &mut query, query_heads),
                (&key_normed, &mut key, key_value_heads),
            ] {
                let output_len = output.byte_len();
                indexed_rope(
                    &mut IndexedRopeParams {
                        input: span(input, CudaDType::BF16)?,
                        cos: span(&cos, CudaDType::F32)?,
                        sin: span(&sin, CudaDType::F32)?,
                        positions: span(&row_positions_device, CudaDType::U32)?,
                        positions_host: &row_positions[..rotated_rows],
                        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_len)?,
                        head_count: u64::try_from(heads)?,
                        head_size: u64::try_from(HEAD)?,
                        rotary_dimension: u64::try_from(HEAD)?,
                        table_position_count: u64::try_from(table_positions)?,
                    },
                    &mut stream,
                )?;
            }
        }
        let mut command_batch = stream.begin_command_batch()?;
        {
            let mut commands = command_batch.commands();
            if variant != 0 {
                let (query_len, key_len, value_len) =
                    (query.byte_len(), key.byte_len(), value.byte_len());
                let mut split = QkvSplitBiasParams {
                    merged: span(&merged, CudaDType::BF16)?,
                    query_bias: span(&query_scale, CudaDType::BF16)?,
                    key_bias: span(&key_scale, CudaDType::BF16)?,
                    value_bias: span(&key_scale, CudaDType::BF16)?,
                    query: CudaBufferSpanMut::new(&mut query, CudaDType::BF16, 0, query_len)?,
                    key: CudaBufferSpanMut::new(&mut key, CudaDType::BF16, 0, key_len)?,
                    value: CudaBufferSpanMut::new(&mut value, CudaDType::BF16, 0, value_len)?,
                    row_count: u64::try_from(rows)?,
                    query_width: u64::try_from(query_width)?,
                    key_value_width: u64::try_from(key_value_width)?,
                };
                let rope = QkvRopeParams {
                    cos: span(&cos, CudaDType::F32)?,
                    sin: span(&sin, CudaDType::F32)?,
                    positions: span(&row_positions_device, CudaDType::U32)?,
                    positions_host: &row_positions[..rotated_rows],
                    head_size: u64::try_from(HEAD)?,
                    table_position_count: u64::try_from(table_positions)?,
                    defer_for_kv_write: variant == 2,
                    head_norm_epsilon: Some(epsilon),
                };
                qkv_split_bias_rope(&mut split, &rope, &mut commands)?;
            }
            let (key_pool_len, value_pool_len) = (key_pool.byte_len(), value_pool.byte_len());
            ragged_paged_kv_cache_write(
                &mut RaggedPagedKvCacheWriteParams {
                    key_source: span(&key, CudaDType::BF16)?,
                    value_source: span(&value, CudaDType::BF16)?,
                    key_pool: CudaBufferSpanMut::new(
                        &mut key_pool,
                        CudaDType::BF16,
                        0,
                        key_pool_len,
                    )?,
                    value_pool: CudaBufferSpanMut::new(
                        &mut value_pool,
                        CudaDType::BF16,
                        0,
                        value_pool_len,
                    )?,
                    batch: PackedBatchV1::new(
                        host,
                        span(&offsets_device, CudaDType::U32)?,
                        span(&block_ids_device, CudaDType::U32)?,
                        span(&valid_tokens_device, CudaDType::U16)?,
                        span(&row_slots_device, CudaDType::U32)?,
                        span(&row_positions_device, CudaDType::U32)?,
                    )?,
                    key_value_head_count: u64::try_from(key_value_heads)?,
                    head_size: u64::try_from(HEAD)?,
                },
                &mut commands,
            )?;
        }
        command_batch.finish()?;
        let mut bytes = Vec::new();
        for buffer in [
            &mut query,
            &mut key,
            &mut value,
            &mut key_pool,
            &mut value_pool,
        ] {
            bytes.push(download(&context, &mut stream, buffer)?);
        }
        results.push(bytes);
        for buffer in [
            query,
            key,
            value,
            key_pool,
            value_pool,
            query_normed,
            key_normed,
        ] {
            buffer.close()?;
        }
    }
    for (variant, label) in [(1, "immediate"), (2, "deferred")] {
        for (index, name) in ["query", "key", "value", "key pool", "value pool"]
            .iter()
            .enumerate()
        {
            let differing = results[0][index]
                .chunks_exact(2)
                .zip(results[variant][index].chunks_exact(2))
                .filter(|(left, right)| left != right)
                .count();
            let written = results[0][index]
                .chunks_exact(2)
                .filter(|bits| *bits != bf16_bits(-113.0).to_ne_bytes())
                .count();
            println!(
                "qkv-head-norm-split {label} {name} elements={} written={written} differing={differing}",
                results[0][index].len() / 2
            );
            assert_eq!(
                differing, 0,
                "{label} {name} differs from the separate launches"
            );
            assert!(written > 0, "{name} fixture wrote nothing");
        }
    }
    for buffer in [
        merged,
        query_raw,
        key_raw,
        query_scale,
        key_scale,
        cos,
        sin,
        offsets_device,
        block_ids_device,
        valid_tokens_device,
        row_slots_device,
        row_positions_device,
    ] {
        buffer.close()?;
    }
    staging.close()?;
    stream.close()?;
    context.synchronize()?;
    context.close()?;
    Ok(())
}
