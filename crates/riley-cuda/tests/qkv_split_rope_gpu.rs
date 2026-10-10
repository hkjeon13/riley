use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, IndexedRopeParams, QkvRopeParams,
    QkvSplitBiasParams, indexed_rope, qkv_split_bias, qkv_split_bias_rope,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn first_device() -> TestResult<(CudaRuntime, CudaDevice)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    Ok((runtime, device))
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
    byte_len: u64,
) -> TestResult<Vec<u8>> {
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;
    buffer
        .copy_to_pinned_async(0, &mut staging, 0, byte_len, stream)?
        .synchronize()?;
    let output = staging.to_vec()?;
    staging.close()?;
    Ok(output)
}

fn mixed(seed: u64) -> u64 {
    let mut value = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 29;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^ (value >> 32)
}

// Finite BF16 values spread over several binades, both signs.
fn bf16_fixture(element_count: usize, salt: u64) -> Vec<u8> {
    (0..element_count)
        .flat_map(|index| {
            let bits = mixed(index as u64 ^ (salt << 40));
            let exponent = 0x76 + (bits >> 8) % 0x12;
            let value = ((bits >> 63) << 15) | (exponent << 7) | (bits & 0x7f);
            u16::try_from(value).unwrap_or(0).to_ne_bytes()
        })
        .collect()
}

fn f32_fixture(element_count: usize, salt: u64) -> Vec<u8> {
    (0..element_count)
        .flat_map(|index| {
            let bits = mixed(index as u64 ^ (salt << 40));
            let unit = (bits >> 11) as f64 / (1_u64 << 53) as f64;
            #[allow(clippy::cast_possible_truncation)]
            let value = (unit * 2.0 - 1.0) as f32;
            value.to_ne_bytes()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    rows: usize,
    positions_host: &[u32],
    query_heads: usize,
    key_value_heads: usize,
    head_size: usize,
    table_positions: usize,
) -> TestResult {
    let query_width = query_heads * head_size;
    let key_value_width = key_value_heads * head_size;
    let merged_width = query_width + 2 * key_value_width;
    let merged = upload(
        context,
        stream,
        staging,
        &bf16_fixture(rows * merged_width, 1),
    )?;
    let query_bias = upload(context, stream, staging, &bf16_fixture(query_width, 2))?;
    let key_bias = upload(context, stream, staging, &bf16_fixture(key_value_width, 3))?;
    let value_bias = upload(context, stream, staging, &bf16_fixture(key_value_width, 4))?;
    let table = table_positions * head_size / 2;
    let cos = upload(context, stream, staging, &f32_fixture(table, 5))?;
    let sin = upload(context, stream, staging, &f32_fixture(table, 6))?;
    let position_bytes: Vec<u8> = positions_host
        .iter()
        .flat_map(|position| position.to_ne_bytes())
        .collect();
    let positions = upload(context, stream, staging, &position_bytes)?;
    let query_bytes = u64::try_from(rows * query_width * 2)?;
    let key_value_bytes = u64::try_from(rows * key_value_width * 2)?;
    let mut query_raw = context.allocate_device_buffer(query_bytes)?;
    let mut key_raw = context.allocate_device_buffer(key_value_bytes)?;
    let mut value_separate = context.allocate_device_buffer(key_value_bytes)?;
    let mut query_separate = context.allocate_device_buffer(query_bytes)?;
    let mut key_separate = context.allocate_device_buffer(key_value_bytes)?;
    let mut query_fused = context.allocate_device_buffer(query_bytes)?;
    let mut key_fused = context.allocate_device_buffer(key_value_bytes)?;
    let mut value_fused = context.allocate_device_buffer(key_value_bytes)?;
    let rows_u64 = u64::try_from(rows)?;
    let head_size_u64 = u64::try_from(head_size)?;
    let table_u64 = u64::try_from(table_positions)?;
    let split = |query: &mut CudaDeviceBuffer,
                 key: &mut CudaDeviceBuffer,
                 value: &mut CudaDeviceBuffer,
                 stream: &mut CudaStream,
                 rope: Option<&QkvRopeParams<'_>>|
     -> TestResult {
        let mut params = QkvSplitBiasParams {
            merged: CudaBufferSpan::new(&merged, CudaDType::BF16, 0, merged.byte_len())?,
            query_bias: CudaBufferSpan::new(
                &query_bias,
                CudaDType::BF16,
                0,
                query_bias.byte_len(),
            )?,
            key_bias: CudaBufferSpan::new(&key_bias, CudaDType::BF16, 0, key_bias.byte_len())?,
            value_bias: CudaBufferSpan::new(
                &value_bias,
                CudaDType::BF16,
                0,
                value_bias.byte_len(),
            )?,
            query: CudaBufferSpanMut::new(query, CudaDType::BF16, 0, query_bytes)?,
            key: CudaBufferSpanMut::new(key, CudaDType::BF16, 0, key_value_bytes)?,
            value: CudaBufferSpanMut::new(value, CudaDType::BF16, 0, key_value_bytes)?,
            row_count: rows_u64,
            query_width: u64::try_from(query_width)?,
            key_value_width: u64::try_from(key_value_width)?,
        };
        match rope {
            Some(rope) => qkv_split_bias_rope(&mut params, rope, stream)?,
            None => qkv_split_bias(&mut params, stream)?,
        }
        Ok(())
    };
    split(
        &mut query_raw,
        &mut key_raw,
        &mut value_separate,
        stream,
        None,
    )?;
    for (input, output, heads) in [
        (&query_raw, &mut query_separate, query_heads),
        (&key_raw, &mut key_separate, key_value_heads),
    ] {
        let rotated_bytes = u64::try_from(positions_host.len() * heads * head_size * 2)?;
        let mut params = IndexedRopeParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, rotated_bytes)?,
            cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
            sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
            positions: CudaBufferSpan::new(&positions, CudaDType::U32, 0, positions.byte_len())?,
            positions_host,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, rotated_bytes)?,
            head_count: u64::try_from(heads)?,
            head_size: head_size_u64,
            rotary_dimension: head_size_u64,
            table_position_count: table_u64,
        };
        indexed_rope(&mut params, stream)?;
    }
    {
        let rope = QkvRopeParams {
            cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
            sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
            positions: CudaBufferSpan::new(&positions, CudaDType::U32, 0, positions.byte_len())?,
            positions_host,
            head_size: head_size_u64,
            table_position_count: table_u64,
        };
        split(
            &mut query_fused,
            &mut key_fused,
            &mut value_fused,
            stream,
            Some(&rope),
        )?;
    }
    let rotated_query = u64::try_from(positions_host.len() * query_width * 2)?;
    let rotated_key = u64::try_from(positions_host.len() * key_value_width * 2)?;
    for (name, separate, fused, byte_len) in [
        (
            "query",
            &mut query_separate,
            &mut query_fused,
            rotated_query,
        ),
        ("key", &mut key_separate, &mut key_fused, rotated_key),
        (
            "value",
            &mut value_separate,
            &mut value_fused,
            key_value_bytes,
        ),
    ] {
        let expected = download(context, stream, separate, byte_len)?;
        let actual = download(context, stream, fused, byte_len)?;
        let differing = expected
            .chunks_exact(2)
            .zip(actual.chunks_exact(2))
            .filter(|(left, right)| left != right)
            .count();
        println!(
            "qkv-split-rope rows={rows} rotated={} {name} elements={} differing={differing}",
            positions_host.len(),
            expected.len() / 2
        );
        assert_eq!(differing, 0, "fused {name} differs from split + RoPE");
    }
    for buffer in [
        merged,
        query_bias,
        key_bias,
        value_bias,
        cos,
        sin,
        positions,
        query_raw,
        key_raw,
        value_separate,
        query_separate,
        key_separate,
        query_fused,
        key_fused,
        value_fused,
    ] {
        buffer.close()?;
    }
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn bf16_qkv_split_rope_matches_split_then_indexed_rope() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let baseline = context.allocation_stats()?;

    run_case(
        &context,
        &mut stream,
        &mut staging,
        5,
        &[0, 7, 3],
        4,
        2,
        8,
        9,
    )?;
    run_case(
        &context,
        &mut stream,
        &mut staging,
        1,
        &[11],
        16,
        2,
        128,
        12,
    )?;
    let positions: Vec<u32> = (0..33_u32).map(|row| (row * 37) % 300).collect();
    run_case(
        &context,
        &mut stream,
        &mut staging,
        40,
        &positions,
        16,
        2,
        128,
        300,
    )?;
    let positions: Vec<u32> = (0..48_u32).map(|row| (row * 5) % 64).collect();
    run_case(
        &context,
        &mut stream,
        &mut staging,
        48,
        &positions,
        16,
        2,
        128,
        64,
    )?;
    assert_eq!(context.allocation_stats()?, baseline);

    staging.close()?;
    stream.close()?;
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
