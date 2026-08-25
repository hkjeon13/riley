#![allow(clippy::too_many_lines)]

use std::error::Error;

use rustinfer_cuda::{
    CastParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice,
    CudaDeviceBuffer, CudaErrorKind, CudaPinnedHostBuffer, CudaRuntime, CudaStream, EmbeddingError,
    EmbeddingParams, GatedMultiplyParams, ResidualAddParams, RmsNormParams, RopeParams, SiluParams,
    cast, embedding, gated_multiply, residual_add, rms_norm, rope, silu,
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

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

fn upload(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    bytes: &[u8],
) -> TestResult<CudaDeviceBuffer> {
    let byte_len = u64::try_from(bytes.len())?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;
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
    let output = staging.to_vec()?;
    staging.close()?;
    Ok(output)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let is_nan = bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0;
    let rounded = if is_nan {
        0x7fff
    } else {
        let tie = (bits >> 16) & 1;
        bits.wrapping_add(0x7fff + tie) >> 16
    };
    u16::try_from(rounded).unwrap_or(0x7fff)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32_to_bf16_bits(value).to_ne_bytes())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

#[test]
#[ignore = "remote GPU"]
fn embedding_bf16_reports_oob_and_repeats_without_allocating() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let table_host: Vec<f32> = (0_u16..15).map(f32::from).collect();
    let table = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&table_host),
    )?;
    let mut token_ids = upload(&context, &mut stream, &mut staging, &u32_bytes(&[4, 0, 2]))?;
    let mut output = context.allocate_device_buffer(18)?;
    let mut scratch = context.allocate_device_buffer(32)?;

    let before = context.allocation_stats()?;
    let mut params = EmbeddingParams {
        table: CudaBufferSpan::new(&table, CudaDType::BF16, 0, table.byte_len())?,
        token_ids: CudaBufferSpan::new(&token_ids, CudaDType::U32, 0, token_ids.byte_len())?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, 18)?,
        error_scratch: CudaBufferSpanMut::new(&mut scratch, CudaDType::U8, 0, 32)?,
        token_count: 3,
        vocabulary_size: 5,
        hidden_size: 3,
    };
    for _ in 0..16 {
        embedding(&mut params, &mut stream)?;
    }
    let after = context.allocation_stats()?;
    assert_eq!(
        before, after,
        "primitive execution changed allocation accounting"
    );
    let actual = decode_bf16(&download(&context, &mut stream, &mut output)?);
    assert_close(
        &actual,
        &[12.0, 13.0, 14.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0],
        0.0,
    );

    token_ids.upload_from_slice(0, &u32_bytes(&[0, 5, 1]), &mut staging, &mut stream)?;
    let mut oob_params = EmbeddingParams {
        table: CudaBufferSpan::new(&table, CudaDType::BF16, 0, table.byte_len())?,
        token_ids: CudaBufferSpan::new(&token_ids, CudaDType::U32, 0, token_ids.byte_len())?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, 18)?,
        error_scratch: CudaBufferSpanMut::new(&mut scratch, CudaDType::U8, 0, 32)?,
        token_count: 3,
        vocabulary_size: 5,
        hidden_size: 3,
    };
    let error = embedding(&mut oob_params, &mut stream)
        .expect_err("invalid token id must produce structured OOB information");
    match error {
        EmbeddingError::TokenOutOfRange {
            token_position,
            token_id,
            source,
        } => {
            assert_eq!(token_position, 1);
            assert_eq!(token_id, 5);
            assert_eq!(source.kind(), CudaErrorKind::OutOfRange);
        }
        EmbeddingError::Cuda(error) => panic!("expected token OOB, got {error}"),
    }
    assert_eq!(
        decode_bf16(&download(&context, &mut stream, &mut output)?),
        actual,
        "embedding OOB must leave the complete output untouched"
    );

    let mut zero_params = EmbeddingParams {
        table: CudaBufferSpan::new(&table, CudaDType::BF16, 0, table.byte_len())?,
        token_ids: CudaBufferSpan::new(&token_ids, CudaDType::U32, 0, 0)?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, 0)?,
        error_scratch: CudaBufferSpanMut::new(&mut scratch, CudaDType::U8, 0, 32)?,
        token_count: 0,
        vocabulary_size: 5,
        hidden_size: 3,
    };
    embedding(&mut zero_params, &mut stream)?;

    table.close()?;
    token_ids.close()?;
    output.close()?;
    scratch.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn norm_and_elementwise_bf16_match_f32_reference() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let left_host = [-2.0_f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let right_host = [0.5_f32, 2.0, -4.0, 1.5, -0.25, 0.75];
    let weight_host = [1.0_f32, 0.5, 2.0];
    let left_bytes = bf16_bytes(&left_host);
    let left = upload(&context, &mut stream, &mut staging, &left_bytes)?;
    let right = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&right_host),
    )?;
    let weight = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&weight_host),
    )?;
    let mut output = context.allocate_device_buffer(12)?;

    let mut norm = RmsNormParams {
        input: CudaBufferSpan::new(&left, CudaDType::BF16, 0, 12)?,
        weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, 6)?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, 12)?,
        row_count: 2,
        hidden_size: 3,
        epsilon: 1.0e-5,
    };
    rms_norm(&mut norm, &mut stream)?;
    let actual_norm = decode_bf16(&download(&context, &mut stream, &mut output)?);
    let mut expected_norm = Vec::with_capacity(6);
    for row in left_host.chunks_exact(3) {
        let square_sum = row.iter().fold(0.0_f32, |sum, value| sum + value * value);
        let inverse = (square_sum / 3.0 + 1.0e-5).sqrt().recip();
        expected_norm.extend(
            row.iter()
                .zip(weight_host)
                .map(|(&value, scale)| value * inverse * scale),
        );
    }
    assert_close(&actual_norm, &expected_norm, 0.02);

    let mut residual = ResidualAddParams {
        left: CudaBufferSpan::new(&left, CudaDType::BF16, 0, 12)?,
        right: CudaBufferSpan::new(&right, CudaDType::BF16, 0, 12)?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, 12)?,
        element_count: 6,
    };
    residual_add(&mut residual, &mut stream)?;
    let actual_sum = decode_bf16(&download(&context, &mut stream, &mut output)?);
    let expected_sum: Vec<_> = left_host
        .iter()
        .zip(right_host)
        .map(|(&left, right)| left + right)
        .collect();
    assert_close(&actual_sum, &expected_sum, 0.02);

    let mut activation = SiluParams {
        input: CudaBufferSpan::new(&left, CudaDType::BF16, 0, 12)?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, 12)?,
        element_count: 6,
    };
    silu(&mut activation, &mut stream)?;
    let actual_silu = decode_bf16(&download(&context, &mut stream, &mut output)?);
    let expected_silu: Vec<_> = left_host
        .iter()
        .map(|&value| value / (1.0 + (-value).exp()))
        .collect();
    assert_close(&actual_silu, &expected_silu, 0.01);

    let mut product = context.allocate_device_buffer(12)?;
    let before = context.allocation_stats()?;
    let mut gated = GatedMultiplyParams {
        activated_gate: CudaBufferSpan::new(&output, CudaDType::BF16, 0, 12)?,
        up: CudaBufferSpan::new(&right, CudaDType::BF16, 0, 12)?,
        output: CudaBufferSpanMut::new(&mut product, CudaDType::BF16, 0, 12)?,
        element_count: 6,
    };
    for _ in 0..16 {
        gated_multiply(&mut gated, &mut stream)?;
    }
    assert_eq!(before, context.allocation_stats()?);
    let actual_product = decode_bf16(&download(&context, &mut stream, &mut product)?);
    let expected_product: Vec<_> = actual_silu
        .iter()
        .zip(right_host)
        .map(|(&gate, up)| gate * up)
        .collect();
    assert_close(&actual_product, &expected_product, 0.02);

    left.close()?;
    right.close()?;
    weight.close()?;
    output.close()?;
    product.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn rope_and_explicit_cast_match_reference_at_long_position() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let input_host = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
    let table_positions = 8_192_u64;
    let mut cos_host = Vec::with_capacity(16_384);
    let mut sin_host = Vec::with_capacity(16_384);
    for position in 0_u16..8_192 {
        let position = f32::from(position);
        for exponent in [0.0_f32, 0.5] {
            let angle = position / 10_000.0_f32.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            cos_host.push(cos);
            sin_host.push(sin);
        }
    }
    let input = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&input_host),
    )?;
    let cos = upload(&context, &mut stream, &mut staging, &f32_bytes(&cos_host))?;
    let sin = upload(&context, &mut stream, &mut staging, &f32_bytes(&sin_host))?;
    let mut rotated = context.allocate_device_buffer(16)?;

    let mut rope_params = RopeParams {
        input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, 16)?,
        cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
        sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
        output: CudaBufferSpanMut::new(&mut rotated, CudaDType::BF16, 0, 16)?,
        token_count: 2,
        head_count: 1,
        head_size: 4,
        rotary_dimension: 4,
        table_position_count: table_positions,
        position_offset: 8_190,
    };
    rope(&mut rope_params, &mut stream)?;
    let actual = decode_bf16(&download(&context, &mut stream, &mut rotated)?);
    let mut expected = vec![0.0_f32; input_host.len()];
    for token in 0..2 {
        for pair in 0..2 {
            let table_index = (8_190 + token) * 2 + pair;
            let first_index = token * 4 + pair;
            let second_index = first_index + 2;
            let first = input_host[first_index];
            let second = input_host[second_index];
            let cosine = cos_host[table_index];
            let sine = sin_host[table_index];
            expected[first_index] = first * cosine - second * sine;
            expected[second_index] = second * cosine + first * sine;
        }
    }
    assert_close(&actual, &expected, 0.02);

    let mut f32_output = context.allocate_device_buffer(32)?;
    let mut expand = CastParams {
        input: CudaBufferSpan::new(&rotated, CudaDType::BF16, 0, 16)?,
        output: CudaBufferSpanMut::new(&mut f32_output, CudaDType::F32, 0, 32)?,
        element_count: 8,
    };
    cast(&mut expand, &mut stream)?;
    let expanded = decode_f32(&download(&context, &mut stream, &mut f32_output)?);
    assert_close(&expanded, &actual, 0.0);

    let mut roundtrip = context.allocate_device_buffer(16)?;
    let mut narrow = CastParams {
        input: CudaBufferSpan::new(&f32_output, CudaDType::F32, 0, 32)?,
        output: CudaBufferSpanMut::new(&mut roundtrip, CudaDType::BF16, 0, 16)?,
        element_count: 8,
    };
    cast(&mut narrow, &mut stream)?;
    assert_eq!(
        download(&context, &mut stream, &mut rotated)?,
        download(&context, &mut stream, &mut roundtrip)?,
        "BF16->F32->BF16 must preserve every storage bit"
    );

    let rounding_host = [
        f32::from_bits(0x3f80_7fff),
        f32::from_bits(0x3f80_8000),
        f32::from_bits(0x3f80_8001),
        f32::from_bits(0x3f81_8000),
        f32::from_bits(0xbf80_8001),
        f32::from_bits(0x0000_8001),
        f32::from_bits(0x7f80_0000),
        f32::from_bits(0x7f80_0001),
    ];
    f32_output.upload_from_slice(0, &f32_bytes(&rounding_host), &mut staging, &mut stream)?;
    let mut rounding = CastParams {
        input: CudaBufferSpan::new(&f32_output, CudaDType::F32, 0, 32)?,
        output: CudaBufferSpanMut::new(&mut roundtrip, CudaDType::BF16, 0, 16)?,
        element_count: 8,
    };
    cast(&mut rounding, &mut stream)?;
    assert_eq!(
        download(&context, &mut stream, &mut roundtrip)?,
        bf16_bytes(&rounding_host),
        "F32->BF16 must use round-to-nearest-even and CUDA's canonical NaN"
    );

    input.close()?;
    cos.close()?;
    sin.close()?;
    rotated.close()?;
    f32_output.close()?;
    roundtrip.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}
