//! The merged-QKV split with RoPE whose launch is deferred to the KV cache
//! write must store the same bytes as the split followed by the write.

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, PackedBatchHostV1, PackedBatchV1, QkvRopeParams,
    QkvSplitBiasParams, RaggedPagedKvCacheWriteParams, qkv_split_bias_rope,
    ragged_paged_kv_cache_write,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const HEAD: usize = 64;
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
fn deferred_qkv_split_and_kv_write_match_the_two_separate_launches() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 22)?;

    // Two sequences; five active rows cross a page boundary (positions 15, 16,
    // 17) and a sixth dense row is padding: it gets a value output but no
    // rotation and no cache slot.
    let sequence_block_offsets = [0_u32, 2, 3];
    let block_ids = [4_u32, 1, 3];
    let valid_tokens = [16_u16, 2, 5];
    let row_sequence_slots = [0_u32, 1, 0, 0, 1];
    let row_positions = [16_u32, 4, 15, 17, 0, 0];
    let table_positions = 19_usize;
    let rotated_rows = 5_usize;
    let rows = row_positions.len();
    let physical_blocks = 6_usize;
    let query_heads = 4_usize;
    let key_value_heads = 2_usize;
    let query_width = query_heads * HEAD;
    let key_value_width = key_value_heads * HEAD;
    let merged_width = query_width + 2 * key_value_width;
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions[..rotated_rows],
        u64::try_from(physical_blocks)?,
    )?;

    let merged = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16((0..rows * merged_width).map(|i| ((i * 37 + 11) % 257) as f32 / 64.0 - 2.0)),
    )?;
    let query_bias = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16((0..query_width).map(|i| ((i * 13 + 5) % 97) as f32 / 128.0 - 0.3)),
    )?;
    let key_bias = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16((0..key_value_width).map(|i| ((i * 7 + 3) % 89) as f32 / 128.0 - 0.4)),
    )?;
    let value_bias = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16((0..key_value_width).map(|i| ((i * 5 + 1) % 83) as f32 / 128.0 - 0.2)),
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
    for deferred in [false, true] {
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
            &sentinel(rows * key_value_width),
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
        let mut command_batch = stream.begin_command_batch()?;
        {
            let mut commands = command_batch.commands();
            let (query_len, key_len, value_len) =
                (query.byte_len(), key.byte_len(), value.byte_len());
            let mut split = QkvSplitBiasParams {
                merged: span(&merged, CudaDType::BF16)?,
                query_bias: span(&query_bias, CudaDType::BF16)?,
                key_bias: span(&key_bias, CudaDType::BF16)?,
                value_bias: span(&value_bias, CudaDType::BF16)?,
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
                defer_for_kv_write: deferred,
                head_norm_epsilon: None,
            };
            qkv_split_bias_rope(&mut split, &rope, &mut commands)?;
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
        for buffer in [query, key, value, key_pool, value_pool] {
            buffer.close()?;
        }
    }
    for (index, name) in ["query", "key", "value", "key pool", "value pool"]
        .iter()
        .enumerate()
    {
        let differing = results[0][index]
            .chunks_exact(2)
            .zip(results[1][index].chunks_exact(2))
            .filter(|(left, right)| left != right)
            .count();
        let written = results[0][index]
            .chunks_exact(2)
            .filter(|bits| *bits != bf16_bits(-113.0).to_ne_bytes())
            .count();
        println!(
            "qkv-split-kv-write {name} elements={} written={written} differing={differing}",
            results[0][index].len() / 2
        );
        assert_eq!(differing, 0, "deferred {name} differs from split + write");
        assert!(written > 0, "{name} fixture wrote nothing");
    }
    for buffer in [
        merged,
        query_bias,
        key_bias,
        value_bias,
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
