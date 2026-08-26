#![allow(clippy::too_many_lines)]

use std::error::Error;

use rustinfer_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice, CudaDeviceBuffer,
    CudaErrorKind, CudaFixed37GemmMetadata, CudaGemmConfig, CudaRuntime, CudaStream,
    FIXED37_CHUNK_ELEMENTS, FIXED37_MAX_CHUNK_COUNT, FIXED37_REDUCTION_VERSION, Fixed37GemmParams,
    Fixed37LogSoftmaxParams, ResidualRmsNormParams, RmsNormParams, fixed37_log_softmax,
    fixed37_residual_rms_norm, fixed37_rms_norm, rms_norm,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const REPEAT_COUNT: usize = 100;

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
    bytes: &[u8],
) -> TestResult<CudaDeviceBuffer> {
    let byte_len = u64::try_from(bytes.len())?;
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;
    buffer.upload_from_slice(0, bytes, &mut staging, stream)?;
    staging.close()?;
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

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
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

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn round_bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_bits(value))
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32_to_bf16_bits(value).to_ne_bytes())
        .collect()
}

fn bf16_bits_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
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

fn balanced_sum(mut partials: Vec<f32>) -> f32 {
    assert!(!partials.is_empty());
    while partials.len() > 1 {
        let mut merged = partials
            .chunks_exact(2)
            .map(|pair| pair[0] + pair[1])
            .collect::<Vec<_>>();
        if partials.len() % 2 != 0 {
            merged.push(*partials.last().expect("non-empty partials"));
        }
        partials = merged;
    }
    partials[0]
}

fn fixed37_sum(values: &[f32]) -> f32 {
    balanced_sum(
        values
            .chunks(usize::try_from(FIXED37_CHUNK_ELEMENTS).expect("chunk fits usize"))
            .map(|chunk| {
                chunk
                    .iter()
                    .copied()
                    .fold(0.0_f32, |sum, value| sum + value)
            })
            .collect(),
    )
}

fn flat_sum(values: &[f32]) -> f32 {
    values
        .iter()
        .copied()
        .fold(0.0_f32, |sum, value| sum + value)
}

fn fixed37_sumsq(values: &[f32]) -> f32 {
    balanced_sum(
        values
            .chunks(usize::try_from(FIXED37_CHUNK_ELEMENTS).expect("chunk fits usize"))
            .map(|chunk| {
                chunk
                    .iter()
                    .copied()
                    .fold(0.0_f32, |sum, value| value.mul_add(value, sum))
            })
            .collect(),
    )
}

fn fixed37_dot(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    balanced_sum(
        left.chunks(usize::try_from(FIXED37_CHUNK_ELEMENTS).expect("chunk fits usize"))
            .zip(right.chunks(usize::try_from(FIXED37_CHUNK_ELEMENTS).expect("chunk fits usize")))
            .map(|(left_chunk, right_chunk)| {
                left_chunk
                    .iter()
                    .zip(right_chunk)
                    .fold(0.0_f32, |sum, (&left, &right)| left.mul_add(right, sum))
            })
            .collect(),
    )
}

fn patterned_bf16(element_count: usize, seed: usize) -> Vec<f32> {
    (0..element_count)
        .map(|index| {
            let bucket = (index.wrapping_mul(29).wrapping_add(seed.wrapping_mul(17))) % 41;
            let centered = i16::try_from(bucket).expect("modulo 41 fits i16") - 20;
            round_bf16(f32::from(centered) / 11.0)
        })
        .collect()
}

fn expected_fixed37_gemm(input: &[f32], weight: &[f32], m: usize, n: usize, k: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(m * n * 2);
    for row in 0..m {
        for column in 0..n {
            let dot = fixed37_dot(
                &input[row * k..(row + 1) * k],
                &weight[column * k..(column + 1) * k],
            );
            output.extend_from_slice(&f32_to_bf16_bits(dot).to_ne_bytes());
        }
    }
    output
}

fn run_gemm_case(context: &CudaContext, stream: &mut CudaStream, k: usize) -> TestResult {
    let (m, n) = (2_usize, 3_usize);
    let config = CudaGemmConfig::new(u64::try_from(m)?, u64::try_from(n)?, u64::try_from(k)?, 0)?;
    let mut plan = context.prepare_fixed37_gemm(config)?;
    let metadata = plan.metadata();
    assert_eq!(metadata.backend_id(), CudaFixed37GemmMetadata::BACKEND_ID);
    assert_eq!(metadata.reduction_version(), FIXED37_REDUCTION_VERSION);
    assert_eq!(metadata.chunk_elements(), FIXED37_CHUNK_ELEMENTS);
    assert_eq!(metadata.accumulator_dtype(), CudaDType::F32);
    assert_eq!(metadata.output_dtype(), CudaDType::BF16);
    assert_eq!(metadata.threads_per_block(), 256);
    assert!(metadata.deterministic());
    assert_eq!(metadata.workspace_bytes(), 0);
    assert_eq!(
        metadata.dimensions(),
        (u64::try_from(m)?, u64::try_from(n)?, u64::try_from(k)?)
    );
    let chunks = u64::try_from(k)?.div_ceil(u64::from(FIXED37_CHUNK_ELEMENTS));
    assert_eq!(metadata.dynamic_shared_memory_bytes(), chunks * 2 * 4);

    let input_host = patterned_bf16(m * k, k.wrapping_add(1));
    let weight_host = patterned_bf16(n * k, k.wrapping_add(2));
    let expected = expected_fixed37_gemm(&input_host, &weight_host, m, n, k);
    let input = upload(context, stream, &bf16_bytes(&input_host))?;
    let weight = upload(context, stream, &bf16_bytes(&weight_host))?;
    let mut output = context.allocate_device_buffer(config.output_bytes())?;

    // A safe validation failure must not poison the plan. The following
    // successful batch and explicit closes also regress native use counts.
    {
        let mut invalid = Fixed37GemmParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(
                &mut output,
                CudaDType::BF16,
                0,
                config.output_bytes() - 2,
            )?,
        };
        let error = plan
            .execute(&mut invalid, stream)
            .expect_err("short output must fail before native execution");
        assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
        assert!(!plan.is_poisoned());
    }

    let stable_allocations = context.allocation_stats()?;
    let mut batch = stream.begin_command_batch()?;
    {
        let mut commands = batch.commands();
        let mut params = Fixed37GemmParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, config.output_bytes())?,
        };
        for _ in 0..REPEAT_COUNT {
            plan.execute(&mut params, &mut commands)?;
        }
    }
    batch.finish()?;
    assert_eq!(context.allocation_stats()?, stable_allocations);
    assert_eq!(download(context, stream, &mut output)?, expected);

    plan.close()?;
    input.close()?;
    weight.close()?;
    output.close()?;
    Ok(())
}

fn run_gemm_order_witness(context: &CudaContext, stream: &mut CudaStream) -> TestResult {
    // Chunk zero starts at 2^24, so all 36 following ones are individually
    // lost. Chunk one accumulates 36 ones before cancelling 2^24. A flat
    // left-fold loses all 72 ones before the cancellation and returns zero;
    // fixed37 returns 36, which remains distinct after BF16 storage rounding.
    let mut input_host = vec![1.0_f32; 74];
    input_host[0] = 16_777_216.0;
    input_host[73] = -16_777_216.0;
    let weight_host = vec![1.0_f32; input_host.len()];
    let fixed = fixed37_dot(&input_host, &weight_host);
    let flat = input_host
        .iter()
        .zip(&weight_host)
        .fold(0.0_f32, |sum, (&left, &right)| left.mul_add(right, sum));
    assert_eq!(fixed.to_bits(), 36.0_f32.to_bits());
    assert_eq!(flat.to_bits(), 0.0_f32.to_bits());
    assert_ne!(f32_to_bf16_bits(fixed), f32_to_bf16_bits(flat));

    let config = CudaGemmConfig::new(1, 1, 74, 0)?;
    let mut plan = context.prepare_fixed37_gemm(config)?;
    let input = upload(context, stream, &bf16_bytes(&input_host))?;
    let weight = upload(context, stream, &bf16_bytes(&weight_host))?;
    let mut output = context.allocate_device_buffer(config.output_bytes())?;
    {
        let mut params = Fixed37GemmParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, config.output_bytes())?,
        };
        plan.execute(&mut params, stream)?;
    }
    assert_eq!(
        download(context, stream, &mut output)?,
        f32_to_bf16_bits(fixed).to_ne_bytes(),
        "the GPU result must preserve the fixed37 order through BF16 storage"
    );
    plan.close()?;
    input.close()?;
    weight.close()?;
    output.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn fixed37_gemm_k576_k1536_repeats_and_releases_every_use() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    run_gemm_case(&context, &mut stream, 576)?;
    run_gemm_case(&context, &mut stream, 1_536)?;
    run_gemm_order_witness(&context, &mut stream)?;
    assert!(context.allocation_stats()?.is_zero());
    stream.close()?;
    close_context(context)
}

fn run_rms_axis(context: &CudaContext, stream: &mut CudaStream, hidden_size: usize) -> TestResult {
    let input_host: Vec<f32> = (0..hidden_size)
        .map(|index| {
            let centered = i16::try_from(index % 17).expect("modulo fits") - 8;
            f32::from(centered) / 8.0
        })
        .collect();
    let weight_host: Vec<f32> = (0..hidden_size)
        .map(|index| 0.75 + f32::from(u8::try_from(index % 7).expect("modulo fits")) / 16.0)
        .collect();
    let input = upload(context, stream, &f32_bytes(&input_host))?;
    let weight = upload(context, stream, &f32_bytes(&weight_host))?;
    let byte_len = u64::try_from(hidden_size)?
        .checked_mul(4)
        .ok_or("byte overflow")?;
    let mut output = context.allocate_device_buffer(byte_len)?;
    let epsilon = 1.0e-5_f32;
    {
        let mut params = RmsNormParams {
            input: CudaBufferSpan::new(&input, CudaDType::F32, 0, byte_len)?,
            weight: CudaBufferSpan::new(&weight, CudaDType::F32, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::F32, 0, byte_len)?,
            row_count: 1,
            hidden_size: u64::try_from(hidden_size)?,
            epsilon,
        };
        fixed37_rms_norm(&mut params, stream)?;
    }
    let actual = decode_f32(&download(context, stream, &mut output)?);
    let hidden_size_f32 = f32::from(u16::try_from(hidden_size)?);
    let inverse = 1.0 / (fixed37_sumsq(&input_host) / hidden_size_f32 + epsilon).sqrt();
    for (index, ((&input, &weight), &actual)) in
        input_host.iter().zip(&weight_host).zip(&actual).enumerate()
    {
        let expected = input * inverse * weight;
        assert!(
            (actual - expected).abs() <= 2.0e-5_f32.max(expected.abs() * 2.0e-5),
            "fixed37 RMS axis {hidden_size} output[{index}] expected {expected}, got {actual}"
        );
    }
    input.close()?;
    weight.close()?;
    output.close()?;
    Ok(())
}

fn canonical_fixed_order_witness() -> Vec<f32> {
    vec![
        4096.0, 256.0, 32.0, 64.0, 512.0, 1024.0, 256.0, 4096.0, 128.0, 2.0, 512.0, 4096.0, 128.0,
        4.0, 256.0, 256.0, 4096.0, 16.0, 8.0, 2.0, 64.0, 1.0, 128.0, 1024.0, 64.0, 2.0, 8.0, 16.0,
        512.0, 2.0, 16.0, 1024.0, 2.0, 2.0, 4.0, 1024.0, 1024.0, 2.0, 1024.0, 1.0, 4.0, 2.0, 128.0,
        2.0, 32.0, 128.0, 128.0, 64.0, 128.0, 16.0, 512.0, 16.0, 64.0, 128.0, 32.0, 8.0, 64.0, 2.0,
        4.0, 8.0, 2.0, 8.0, 8.0, 512.0, 512.0, 256.0, 256.0, 512.0, 1024.0, 1024.0, 4096.0, 8.0,
        256.0, 8.0,
    ]
}

fn canonical_rms_sum(values: &[f32]) -> f32 {
    let mut partials = vec![0.0_f32; 256];
    for (thread, partial) in partials.iter_mut().enumerate() {
        for &value in values.iter().skip(thread).step_by(256) {
            *partial = value.mul_add(value, *partial);
        }
    }
    let mut offset = 128;
    while offset != 0 {
        for index in 0..offset {
            partials[index] += partials[index + offset];
        }
        offset /= 2;
    }
    partials[0]
}

fn run_order_witness(context: &CudaContext, stream: &mut CudaStream) -> TestResult {
    let input_host = canonical_fixed_order_witness();
    assert_eq!(input_host.len(), 74);
    let canonical_sum = canonical_rms_sum(&input_host);
    let fixed_sum = fixed37_sumsq(&input_host);
    assert_eq!(canonical_sum.to_bits(), 94_729_072.0_f32.to_bits());
    assert_eq!(fixed_sum.to_bits(), 94_729_040.0_f32.to_bits());
    assert_ne!(canonical_sum.to_bits(), fixed_sum.to_bits());

    let weight_host = vec![1024.0_f32; input_host.len()];
    let byte_len = u64::try_from(input_host.len() * 4)?;
    let input = upload(context, stream, &f32_bytes(&input_host))?;
    let weight = upload(context, stream, &f32_bytes(&weight_host))?;
    let mut canonical_output = context.allocate_device_buffer(byte_len)?;
    let mut fixed_output = context.allocate_device_buffer(byte_len)?;
    {
        let mut params = RmsNormParams {
            input: CudaBufferSpan::new(&input, CudaDType::F32, 0, byte_len)?,
            weight: CudaBufferSpan::new(&weight, CudaDType::F32, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut canonical_output, CudaDType::F32, 0, byte_len)?,
            row_count: 1,
            hidden_size: 74,
            epsilon: 1.0e-12,
        };
        rms_norm(&mut params, stream)?;
    }
    {
        let mut params = RmsNormParams {
            input: CudaBufferSpan::new(&input, CudaDType::F32, 0, byte_len)?,
            weight: CudaBufferSpan::new(&weight, CudaDType::F32, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut fixed_output, CudaDType::F32, 0, byte_len)?,
            row_count: 1,
            hidden_size: 74,
            epsilon: 1.0e-12,
        };
        fixed37_rms_norm(&mut params, stream)?;
    }
    let canonical_bytes = download(context, stream, &mut canonical_output)?;
    let fixed_bytes = download(context, stream, &mut fixed_output)?;
    assert_ne!(
        canonical_bytes, fixed_bytes,
        "the reviewed order-sensitive witness must distinguish canonical and fixed37 RMSNorm"
    );
    input.close()?;
    weight.close()?;
    canonical_output.close()?;
    fixed_output.close()?;
    Ok(())
}

fn run_fused_equivalence(context: &CudaContext, stream: &mut CudaStream) -> TestResult {
    let (row_count, hidden_size) = (2_usize, 38_usize);
    let elements = row_count * hidden_size;
    let left_host = patterned_bf16(elements, 101);
    let right_host = patterned_bf16(elements, 202);
    let weight_host: Vec<f32> = patterned_bf16(hidden_size, 303)
        .into_iter()
        .map(|value| round_bf16(value.abs() + 0.5))
        .collect();
    let expected_residual: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| round_bf16(left + right))
        .collect();
    let byte_len = u64::try_from(elements * 2)?;
    let weight_bytes = u64::try_from(hidden_size * 2)?;
    let left = upload(context, stream, &bf16_bytes(&left_host))?;
    let right = upload(context, stream, &bf16_bytes(&right_host))?;
    let weight = upload(context, stream, &bf16_bytes(&weight_host))?;
    let mut residual = context.allocate_device_buffer(byte_len)?;
    let mut fused_normalized = context.allocate_device_buffer(byte_len)?;
    let mut standalone_normalized = context.allocate_device_buffer(byte_len)?;
    {
        let mut params = ResidualRmsNormParams {
            left: CudaBufferSpan::new(&left, CudaDType::BF16, 0, byte_len)?,
            right: CudaBufferSpan::new(&right, CudaDType::BF16, 0, byte_len)?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, weight_bytes)?,
            residual_output: CudaBufferSpanMut::new(&mut residual, CudaDType::BF16, 0, byte_len)?,
            normalized_output: CudaBufferSpanMut::new(
                &mut fused_normalized,
                CudaDType::BF16,
                0,
                byte_len,
            )?,
            row_count: u64::try_from(row_count)?,
            hidden_size: u64::try_from(hidden_size)?,
            epsilon: 1.0e-5,
        };
        fixed37_residual_rms_norm(&mut params, stream)?;
    }
    assert_eq!(
        download(context, stream, &mut residual)?,
        bf16_bytes(&expected_residual),
        "fused residual storage must match standalone BF16 rounding"
    );
    {
        let mut params = RmsNormParams {
            input: CudaBufferSpan::new(&residual, CudaDType::BF16, 0, byte_len)?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, weight_bytes)?,
            output: CudaBufferSpanMut::new(
                &mut standalone_normalized,
                CudaDType::BF16,
                0,
                byte_len,
            )?,
            row_count: u64::try_from(row_count)?,
            hidden_size: u64::try_from(hidden_size)?,
            epsilon: 1.0e-5,
        };
        fixed37_rms_norm(&mut params, stream)?;
    }
    assert_eq!(
        download(context, stream, &mut fused_normalized)?,
        download(context, stream, &mut standalone_normalized)?,
        "fixed37 fused and standalone RMSNorm outputs must be bit-identical"
    );
    left.close()?;
    right.close()?;
    weight.close()?;
    residual.close()?;
    fused_normalized.close()?;
    standalone_normalized.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn fixed37_rms_chunk_boundaries_fused_and_canonical_witness() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    // Pin both sides of early chunk boundaries, adjacent merges, and odd
    // carries at two, three, and four partials.
    for hidden_size in [36, 37, 38, 73, 74, 75, 111, 112, 148, 149] {
        run_rms_axis(&context, &mut stream, hidden_size)?;
    }
    run_fused_equivalence(&context, &mut stream)?;
    run_order_witness(&context, &mut stream)?;
    assert!(context.allocation_stats()?.is_zero());
    stream.close()?;
    close_context(context)
}

fn finite_log_softmax_reference(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted_exponentials = logits
        .iter()
        .map(|&value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let logarithm = fixed37_sum(&shifted_exponentials).ln();
    logits
        .iter()
        .map(|&value| (value - maximum) - logarithm)
        .collect()
}

fn run_special_log_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    bits: &[u16],
) -> TestResult<Vec<f32>> {
    let logits = upload(context, stream, &bf16_bits_bytes(bits))?;
    let output_bytes = u64::try_from(bits.len())?
        .checked_mul(4)
        .ok_or("byte overflow")?;
    let mut output = context.allocate_device_buffer(output_bytes)?;
    {
        let mut params = Fixed37LogSoftmaxParams {
            logits: CudaBufferSpan::new(
                &logits,
                CudaDType::BF16,
                0,
                u64::try_from(bits.len() * 2)?,
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::F32, 0, output_bytes)?,
            element_count: u64::try_from(bits.len())?,
        };
        fixed37_log_softmax(&mut params, stream)?;
    }
    let result = decode_f32(&download(context, stream, &mut output)?);
    logits.close()?;
    output.close()?;
    Ok(result)
}

fn assert_canonical_nan(values: &[f32]) {
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_nan()));
    assert!(
        values.iter().all(|value| value.to_bits() == 0x7fff_ffff),
        "every exceptional log-softmax output must use CUDART_NAN_F"
    );
}

#[test]
#[ignore = "remote GPU"]
fn fixed37_log_softmax_vocab49152_repeats_and_pins_special_values() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;

    let vocabulary_size = 49_152_usize;
    // Each exp(-17) is below half an F32 ulp at 1.0. With the maximum first,
    // a flat fold loses every tail contribution, while fixed37 first combines
    // the all-small chunks and retains a denominator correction around 0.2%.
    let mut logits_host = vec![-17.0_f32; vocabulary_size];
    logits_host[0] = 0.0;
    let shifted_exponentials = logits_host
        .iter()
        .map(|&value| value.exp())
        .collect::<Vec<_>>();
    let fixed_denominator = fixed37_sum(&shifted_exponentials);
    let flat_denominator = flat_sum(&shifted_exponentials);
    assert_eq!(flat_denominator.to_bits(), 1.0_f32.to_bits());
    assert!(
        (fixed_denominator.ln() - flat_denominator.ln()).abs() > 1.0e-3,
        "the vocabulary fixture must distinguish fixed37 from a flat sum beyond GPU tolerance"
    );
    let reference = finite_log_softmax_reference(&logits_host);
    let logits_bytes = u64::try_from(vocabulary_size * 2)?;
    let output_bytes = u64::try_from(vocabulary_size * 4)?;
    let logits = upload(&context, &mut stream, &bf16_bytes(&logits_host))?;
    let mut output = context.allocate_device_buffer(output_bytes)?;
    {
        let mut params = Fixed37LogSoftmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits_bytes)?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::F32, 0, output_bytes)?,
            element_count: u64::try_from(vocabulary_size)?,
        };
        fixed37_log_softmax(&mut params, &mut stream)?;
    }
    let first_bytes = download(&context, &mut stream, &mut output)?;
    let actual = decode_f32(&first_bytes);
    for index in (0..vocabulary_size).step_by(97) {
        assert!(
            (actual[index] - reference[index]).abs() <= 3.0e-4,
            "vocab log-softmax[{index}] expected {}, got {}",
            reference[index],
            actual[index]
        );
    }
    let stable_allocations = context.allocation_stats()?;
    {
        let mut params = Fixed37LogSoftmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits_bytes)?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::F32, 0, output_bytes)?,
            element_count: u64::try_from(vocabulary_size)?,
        };
        for _ in 0..REPEAT_COUNT {
            fixed37_log_softmax(&mut params, &mut stream)?;
        }
    }
    assert_eq!(context.allocation_stats()?, stable_allocations);
    assert_eq!(download(&context, &mut stream, &mut output)?, first_bytes);
    logits.close()?;
    output.close()?;

    let nan = run_special_log_case(&context, &mut stream, &[0x7fff, 0x0000])?;
    assert_canonical_nan(&nan);
    let positive_infinity = run_special_log_case(&context, &mut stream, &[0x0000, 0x7f80])?;
    assert_canonical_nan(&positive_infinity);
    let all_negative_infinity = run_special_log_case(&context, &mut stream, &[0xff80, 0xff80])?;
    assert_canonical_nan(&all_negative_infinity);
    let finite_with_negative_infinity =
        run_special_log_case(&context, &mut stream, &[0xff80, 0x0000])?;
    assert!(finite_with_negative_infinity[0].is_infinite());
    assert!(finite_with_negative_infinity[0].is_sign_negative());
    assert_eq!(
        finite_with_negative_infinity[1].to_bits(),
        0.0_f32.to_bits()
    );
    let negative_zero = run_special_log_case(&context, &mut stream, &[0x8000])?;
    assert_eq!(negative_zero[0].to_bits(), 0.0_f32.to_bits());
    let signed_zero_pair = run_special_log_case(&context, &mut stream, &[0x8000, 0x0000])?;
    assert_eq!(signed_zero_pair[0].to_bits(), signed_zero_pair[1].to_bits());
    assert!((signed_zero_pair[0] + std::f32::consts::LN_2).abs() <= 2.0e-6);

    assert!(context.allocation_stats()?.is_zero());
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn fixed37_maximum_chunk_boundary_succeeds_and_plus_one_is_rejected() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let maximum = u64::from(FIXED37_CHUNK_ELEMENTS)
        .checked_mul(u64::from(FIXED37_MAX_CHUNK_COUNT))
        .ok_or("test axis overflow")?;
    let supported_config = CudaGemmConfig::new(1, 1, maximum, 0)?;
    let supported_plan = context.prepare_fixed37_gemm(supported_config)?;
    assert_eq!(supported_plan.metadata().dimensions(), (1, 1, maximum));
    assert_eq!(
        supported_plan.metadata().dynamic_shared_memory_bytes(),
        u64::from(FIXED37_MAX_CHUNK_COUNT) * 2 * 4
    );
    supported_plan.close()?;

    let over_limit = maximum.checked_add(1).ok_or("test axis overflow")?;
    let config = CudaGemmConfig::new(1, 1, over_limit, 0)?;
    let error = context
        .prepare_fixed37_gemm(config)
        .expect_err("more than 4096 chunks must fail before a CUDA launch");
    assert_eq!(error.kind(), CudaErrorKind::NotSupported);
    close_context(context)
}
