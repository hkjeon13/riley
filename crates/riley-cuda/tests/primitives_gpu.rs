#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CastParams, CudaBufferSpan, CudaBufferSpanMut, CudaCommandBatch, CudaContext, CudaDType,
    CudaDevice, CudaDeviceBuffer, CudaError, CudaErrorKind, CudaPinnedHostBuffer, CudaResult,
    CudaRuntime, CudaStream, EmbeddingError, EmbeddingParams, GatedMultiplyParams,
    ResidualAddParams, ResidualRmsNormParams, RmsNormParams, RopeParams, RopeTableParams,
    SiluParams, cast, embedding, gated_multiply, hugging_face_smollm2_residual_rms_norm,
    hugging_face_smollm2_rms_norm, residual_add, residual_rms_norm, rms_norm, rope, rope_table,
    silu,
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

fn lcg_bf16_bytes(
    count: usize,
    multiplier: u32,
    increment: u32,
    exponent_base: u32,
    exponent_span: u32,
    positive_only: bool,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 2);
    for index in 0..count {
        let index = u32::try_from(index).expect("test fixture index fits u32");
        let value = index.wrapping_mul(multiplier).wrapping_add(increment);
        let sign = u16::try_from((value >> 31) << 15).expect("sign fits u16");
        let exponent = u16::try_from(exponent_base + ((value >> 24) % exponent_span))
            .expect("fixture exponent fits u16");
        let mantissa = u16::try_from(value & 0x7f).expect("mantissa fits u16");
        let mut bits = sign | (exponent << 7) | mantissa;
        if positive_only {
            bits &= 0x7fff;
        }
        bytes.extend_from_slice(&bits.to_ne_bytes());
    }
    bytes
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
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

#[derive(Clone, Copy, Debug)]
enum ExactFloat {
    F32,
    Bf16,
}

impl ExactFloat {
    const fn dtype(self) -> CudaDType {
        match self {
            Self::F32 => CudaDType::F32,
            Self::Bf16 => CudaDType::BF16,
        }
    }

    fn bytes(self, values: &[f32]) -> Vec<u8> {
        match self {
            Self::F32 => f32_bytes(values),
            Self::Bf16 => bf16_bytes(values),
        }
    }
}

fn assert_error_kind(error: &riley_cuda::CudaError, expected: CudaErrorKind, label: &str) {
    assert_eq!(error.kind(), expected, "{label}: {error}");
}

fn enqueue_command_batch_chain_with_validation_error(
    batch: &mut CudaCommandBatch<'_>,
    left: &CudaDeviceBuffer,
    right: &CudaDeviceBuffer,
    intermediate: &mut CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
) -> CudaResult<CudaError> {
    let mut commands = batch.commands();
    {
        let mut residual = ResidualAddParams {
            left: CudaBufferSpan::new(left, CudaDType::BF16, 0, 12)?,
            right: CudaBufferSpan::new(right, CudaDType::BF16, 0, 12)?,
            output: CudaBufferSpanMut::new(intermediate, CudaDType::BF16, 0, 12)?,
            element_count: 6,
        };
        residual_add(&mut residual, &mut commands)?;
    }
    {
        let mut activation = SiluParams {
            input: CudaBufferSpan::new(intermediate, CudaDType::BF16, 0, 12)?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, 12)?,
            element_count: 6,
        };
        silu(&mut activation, &mut commands)?;
    }
    let mut invalid = ResidualAddParams {
        left: CudaBufferSpan::new(left, CudaDType::BF16, 0, 12)?,
        right: CudaBufferSpan::new(right, CudaDType::BF16, 0, 12)?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, 12)?,
        element_count: 7,
    };
    Ok(residual_add(&mut invalid, &mut commands)
        .expect_err("oversized command must fail validation without ending the batch"))
}

#[derive(Clone, Copy)]
struct Bf16FusedDescriptor {
    matrix_span_bytes: u64,
    right_dtype: CudaDType,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
}

#[allow(clippy::too_many_arguments)]
fn run_bf16_fused_descriptor(
    left: &CudaDeviceBuffer,
    right: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    residual_output: &mut CudaDeviceBuffer,
    normalized_output: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
    descriptor: Bf16FusedDescriptor,
) -> riley_cuda::CudaResult<()> {
    let mut params = ResidualRmsNormParams {
        left: CudaBufferSpan::new(left, CudaDType::BF16, 0, descriptor.matrix_span_bytes)?,
        right: CudaBufferSpan::new(
            right,
            descriptor.right_dtype,
            0,
            descriptor.matrix_span_bytes,
        )?,
        weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, 6)?,
        residual_output: CudaBufferSpanMut::new(
            residual_output,
            CudaDType::BF16,
            0,
            descriptor.matrix_span_bytes,
        )?,
        normalized_output: CudaBufferSpanMut::new(
            normalized_output,
            CudaDType::BF16,
            0,
            descriptor.matrix_span_bytes,
        )?,
        row_count: descriptor.row_count,
        hidden_size: descriptor.hidden_size,
        epsilon: descriptor.epsilon,
    };
    residual_rms_norm(&mut params, stream)
}

fn residual_rms_norm_exact_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    storage: ExactFloat,
    row_count: u64,
    hidden_size: u64,
) -> TestResult {
    let element_count =
        usize::try_from(row_count.checked_mul(hidden_size).ok_or("shape overflow")?)?;
    let hidden_size_usize = usize::try_from(hidden_size)?;
    let left_host: Vec<_> = (0..element_count)
        .map(|index| {
            let bounded =
                i16::try_from(index.wrapping_mul(17) % 37).expect("modulo 37 always fits i16");
            f32::from(bounded - 18) / 9.0
        })
        .collect();
    let right_host: Vec<_> = (0..element_count)
        .map(|index| {
            let bounded =
                i16::try_from(index.wrapping_mul(11) % 29).expect("modulo 29 always fits i16");
            f32::from(bounded - 14) / 13.0
        })
        .collect();
    let weight_host: Vec<_> = (0..hidden_size_usize)
        .map(|index| {
            let bounded =
                u8::try_from(index.wrapping_mul(7) % 19).expect("modulo 19 always fits u8");
            0.625 + f32::from(bounded) / 16.0
        })
        .collect();
    let left_bytes = storage.bytes(&left_host);
    let right_bytes = storage.bytes(&right_host);
    let weight_bytes = storage.bytes(&weight_host);
    let matrix_bytes = u64::try_from(left_bytes.len())?;
    let weight_byte_len = u64::try_from(weight_bytes.len())?;
    let dtype = storage.dtype();

    let left = upload(context, stream, staging, &left_bytes)?;
    let right = upload(context, stream, staging, &right_bytes)?;
    let weight = upload(context, stream, staging, &weight_bytes)?;
    let mut standalone_residual = context.allocate_device_buffer(matrix_bytes)?;
    let mut standalone_norm = context.allocate_device_buffer(matrix_bytes)?;
    let mut fused_residual = context.allocate_device_buffer(matrix_bytes)?;
    let mut fused_norm = context.allocate_device_buffer(matrix_bytes)?;

    let mut residual = ResidualAddParams {
        left: CudaBufferSpan::new(&left, dtype, 0, matrix_bytes)?,
        right: CudaBufferSpan::new(&right, dtype, 0, matrix_bytes)?,
        output: CudaBufferSpanMut::new(&mut standalone_residual, dtype, 0, matrix_bytes)?,
        element_count: u64::try_from(element_count)?,
    };
    residual_add(&mut residual, stream)?;
    let expected_residual = download(context, stream, &mut standalone_residual)?;

    let mut norm = RmsNormParams {
        input: CudaBufferSpan::new(&standalone_residual, dtype, 0, matrix_bytes)?,
        weight: CudaBufferSpan::new(&weight, dtype, 0, weight_byte_len)?,
        output: CudaBufferSpanMut::new(&mut standalone_norm, dtype, 0, matrix_bytes)?,
        row_count,
        hidden_size,
        epsilon: 1.0e-5,
    };
    rms_norm(&mut norm, stream)?;
    let expected_norm = download(context, stream, &mut standalone_norm)?;

    let allocation_stats = context.allocation_stats()?;
    let mut fused = ResidualRmsNormParams {
        left: CudaBufferSpan::new(&left, dtype, 0, matrix_bytes)?,
        right: CudaBufferSpan::new(&right, dtype, 0, matrix_bytes)?,
        weight: CudaBufferSpan::new(&weight, dtype, 0, weight_byte_len)?,
        residual_output: CudaBufferSpanMut::new(&mut fused_residual, dtype, 0, matrix_bytes)?,
        normalized_output: CudaBufferSpanMut::new(&mut fused_norm, dtype, 0, matrix_bytes)?,
        row_count,
        hidden_size,
        epsilon: 1.0e-5,
    };
    for _ in 0..4 {
        residual_rms_norm(&mut fused, stream)?;
    }
    assert_eq!(allocation_stats, context.allocation_stats()?);
    assert_eq!(
        download(context, stream, &mut fused_residual)?,
        expected_residual,
        "{storage:?} fused residual differs at rows={row_count}, hidden={hidden_size}"
    );
    assert_eq!(
        download(context, stream, &mut fused_norm)?,
        expected_norm,
        "{storage:?} fused norm differs at rows={row_count}, hidden={hidden_size}"
    );

    left.close()?;
    right.close()?;
    weight.close()?;
    standalone_residual.close()?;
    standalone_norm.close()?;
    fused_residual.close()?;
    fused_norm.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn hugging_face_smollm2_rms_norm_matches_pytorch_2_13_cuda_oracles() -> TestResult {
    // Generated by torch 2.13.0.dev20260824+cu130 on RTX 4090 from the same
    // raw-BF16 LCG fixtures. Each tuple is
    // (rows, plain norm, materialized residual, fused norm).
    const ORACLE_HASHES: &[(u64, u64, u64, u64)] = &[
        (
            1,
            0x3c53_5f73_33f4_9c72,
            0xb783_430d_8661_38df,
            0x855b_c553_6cfe_4682,
        ),
        (
            2,
            0xffa6_13fa_8ee8_ac5d,
            0x3f5a_cd2d_5d5c_6a9c,
            0xe6d0_9692_7ba8_7e65,
        ),
        (
            3,
            0x8ee3_0642_03dd_3dd8,
            0x94cb_f1bb_4420_2695,
            0x3c40_d7c6_92f9_3243,
        ),
        (
            4,
            0x7ee7_df2c_4ec0_5efc,
            0xd93f_527e_80e0_e6f9,
            0xf3e5_dea5_fa15_41e0,
        ),
        (
            7,
            0x1d7c_25ef_c5b9_8904,
            0x78e5_76b0_4eb8_1245,
            0xce15_aab8_ec56_81c5,
        ),
        (
            8,
            0x33a0_281f_aa1e_9176,
            0x2a6d_7581_cfe2_1f06,
            0x60f1_d4f7_f015_8efb,
        ),
        (
            10,
            0x47ca_7869_36f5_3a9b,
            0xebcb_3770_4abd_14db,
            0xe87e_83fc_8ecc_5a00,
        ),
        (
            15,
            0xa446_f79d_9e51_11cb,
            0xeb00_76d1_b886_f546,
            0xc715_9f99_63ae_7e12,
        ),
        (
            16,
            0xb426_1a60_54ea_6773,
            0xd6e3_90b6_eb19_3705,
            0x4930_c228_3e01_8ee1,
        ),
        (
            17,
            0x8526_ddfd_dd91_3bfa,
            0xb0b4_1192_d808_92ed,
            0x5560_449f_8847_83b5,
        ),
        (
            31,
            0x621c_ac67_abad_eae8,
            0xaf3c_0cb7_c55d_9813,
            0x1a80_dc5d_55c6_77a1,
        ),
        (
            128,
            0x2e13_2ef9_b111_75b3,
            0x0917_3ba8_6aca_ec40,
            0xaa06_33fa_a940_b19b,
        ),
    ];
    const HIDDEN_SIZE: u64 = 576;
    const MAX_ROWS: u64 = 128;

    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let element_count = usize::try_from(MAX_ROWS * HIDDEN_SIZE)?;
    let matrix_bytes = u64::try_from(element_count * 2)?;
    let weight_byte_len = HIDDEN_SIZE * 2;
    let mut staging = context.allocate_pinned_host_buffer(matrix_bytes)?;

    let left_bytes = lcg_bf16_bytes(element_count, 1_664_525, 1_013_904_223, 120, 16, false);
    let right_bytes = lcg_bf16_bytes(element_count, 22_695_477, 1, 118, 15, false);
    let weight_bytes = lcg_bf16_bytes(
        usize::try_from(HIDDEN_SIZE)?,
        1_103_515_245,
        12_345,
        126,
        2,
        true,
    );
    let left = upload(&context, &mut stream, &mut staging, &left_bytes)?;
    let right = upload(&context, &mut stream, &mut staging, &right_bytes)?;
    let weight = upload(&context, &mut stream, &mut staging, &weight_bytes)?;
    let mut plain_output = context.allocate_device_buffer(matrix_bytes)?;
    let mut residual_output = context.allocate_device_buffer(matrix_bytes)?;
    let mut fused_output = context.allocate_device_buffer(matrix_bytes)?;
    let allocation_stats = context.allocation_stats()?;

    for &(row_count, plain_hash, residual_hash, fused_hash) in ORACLE_HASHES {
        let case_bytes = row_count * HIDDEN_SIZE * 2;
        let case_len = usize::try_from(case_bytes)?;
        {
            let mut params = RmsNormParams {
                input: CudaBufferSpan::new(&left, CudaDType::BF16, 0, case_bytes)?,
                weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, weight_byte_len)?,
                output: CudaBufferSpanMut::new(&mut plain_output, CudaDType::BF16, 0, case_bytes)?,
                row_count,
                hidden_size: HIDDEN_SIZE,
                epsilon: 1.0e-5,
            };
            hugging_face_smollm2_rms_norm(&mut params, &mut stream)?;
        }
        let actual_plain = download(&context, &mut stream, &mut plain_output)?;
        assert_eq!(
            fnv1a64(&actual_plain[..case_len]),
            plain_hash,
            "plain Hugging Face RMSNorm differs at rows={row_count}"
        );

        {
            let mut params = ResidualRmsNormParams {
                left: CudaBufferSpan::new(&left, CudaDType::BF16, 0, case_bytes)?,
                right: CudaBufferSpan::new(&right, CudaDType::BF16, 0, case_bytes)?,
                weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, weight_byte_len)?,
                residual_output: CudaBufferSpanMut::new(
                    &mut residual_output,
                    CudaDType::BF16,
                    0,
                    case_bytes,
                )?,
                normalized_output: CudaBufferSpanMut::new(
                    &mut fused_output,
                    CudaDType::BF16,
                    0,
                    case_bytes,
                )?,
                row_count,
                hidden_size: HIDDEN_SIZE,
                epsilon: 1.0e-5,
            };
            // Repetition is deliberate: the x=32 topology materializes each
            // vec4 residual with different ownership than the later
            // column-stride read. The first submission also proves that the
            // exact path preserves command-batch ownership and completion.
            let mut batch = stream.begin_command_batch()?;
            let execute_result = {
                let mut commands = batch.commands();
                hugging_face_smollm2_residual_rms_norm(&mut params, &mut commands)
            };
            let finish_result = batch.finish();
            execute_result?;
            finish_result?;
            for _ in 1..16 {
                hugging_face_smollm2_residual_rms_norm(&mut params, &mut stream)?;
            }
        }
        let actual_residual = download(&context, &mut stream, &mut residual_output)?;
        let actual_fused = download(&context, &mut stream, &mut fused_output)?;
        assert_eq!(
            fnv1a64(&actual_residual[..case_len]),
            residual_hash,
            "fused residual materialization differs at rows={row_count}"
        );
        assert_eq!(
            fnv1a64(&actual_fused[..case_len]),
            fused_hash,
            "fused Hugging Face RMSNorm differs at rows={row_count}"
        );
    }
    assert_eq!(allocation_stats, context.allocation_stats()?);

    left.close()?;
    right.close()?;
    weight.close()?;
    plain_output.close()?;
    residual_output.close()?;
    fused_output.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
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
fn command_batch_releases_multi_primitive_resource_ledger_after_validation_error() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let left_host = [1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let right_host = [-1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
    let left = upload(&context, &mut stream, &mut staging, &bf16_bytes(&left_host))?;
    let right = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&right_host),
    )?;
    let mut intermediate = context.allocate_device_buffer(12)?;
    let mut output = context.allocate_device_buffer(12)?;
    let stable = context.allocation_stats()?;

    let mut batch = stream.begin_command_batch()?;
    let body_result = enqueue_command_batch_chain_with_validation_error(
        &mut batch,
        &left,
        &right,
        &mut intermediate,
        &mut output,
    );
    let finish_result = batch.finish();
    finish_result?;
    let validation_error = body_result?;
    assert_eq!(validation_error.kind(), CudaErrorKind::OutOfRange);

    let zeros = bf16_bytes(&[0.0; 6]);
    let queued_chain_output = download(&context, &mut stream, &mut output)?;
    let queued_chain_raw_byte_mismatches = queued_chain_output
        .iter()
        .zip(&zeros)
        .filter(|(actual, expected)| actual != expected)
        .count()
        + queued_chain_output.len().abs_diff(zeros.len());
    assert_eq!(
        queued_chain_output, zeros,
        "the queued residual-add and SiLU chain must complete exactly"
    );
    let after_queued_chain = context.allocation_stats()?;
    assert_eq!(after_queued_chain, stable);

    {
        let mut residual = ResidualAddParams {
            left: CudaBufferSpan::new(&left, CudaDType::BF16, 0, 12)?,
            right: CudaBufferSpan::new(&right, CudaDType::BF16, 0, 12)?,
            output: CudaBufferSpanMut::new(&mut intermediate, CudaDType::BF16, 0, 12)?,
            element_count: 6,
        };
        residual_add(&mut residual, &mut stream)?;
    }
    assert_eq!(
        download(&context, &mut stream, &mut intermediate)?,
        zeros,
        "finished command batch must release resource leases for immediate reuse"
    );
    let after_reuse = context.allocation_stats()?;
    assert_eq!(after_reuse, stable);
    let cuda_live_allocation_delta = i128::from(
        after_reuse
            .device_live_allocations()
            .checked_add(after_reuse.pinned_host_live_allocations())
            .ok_or("CUDA live allocation count overflow")?,
    ) - i128::from(
        stable
            .device_live_allocations()
            .checked_add(stable.pinned_host_live_allocations())
            .ok_or("CUDA live allocation count overflow")?,
    );

    left.close()?;
    right.close()?;
    intermediate.close()?;
    output.close()?;
    staging.close()?;
    let after_owner_close = context.allocation_stats()?;
    let owner_close_live_allocation_count = after_owner_close
        .device_live_allocations()
        .checked_add(after_owner_close.pinned_host_live_allocations())
        .ok_or("CUDA live allocation count overflow")?;
    assert!(after_owner_close.is_zero());
    stream.close()?;
    close_context(context)?;
    println!(
        "pr16-command-batch-resource-ledger schema_version=1 \
validation_fail_closed=true \
queued_chain_raw_byte_mismatches={queued_chain_raw_byte_mismatches} \
cuda_live_allocation_delta={cuda_live_allocation_delta} stream_reuse_after_finish=true \
owner_close_live_allocation_count={owner_close_live_allocation_count} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn residual_rms_norm_matches_standalone_raw_bytes_at_reduction_edges() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(8_192)?;

    for storage in [ExactFloat::F32, ExactFloat::Bf16] {
        for (row_count, hidden_size) in [(5, 1), (4, 3), (2, 255), (2, 256), (2, 257)] {
            residual_rms_norm_exact_case(
                &context,
                &mut stream,
                &mut staging,
                storage,
                row_count,
                hidden_size,
            )?;
        }
    }

    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn residual_rms_norm_validation_is_fail_closed_and_zero_rows_are_a_noop() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let left = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
    )?;
    let right = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&[0.5, 2.0, -4.0, 1.5, -0.25, 0.75]),
    )?;
    let weight = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes(&[1.0, 0.5, 2.0]),
    )?;
    let mut residual_output = context.allocate_device_buffer(12)?;
    let mut normalized_output = context.allocate_device_buffer(12)?;
    let mut standalone_residual = context.allocate_device_buffer(12)?;
    let mut standalone_norm = context.allocate_device_buffer(12)?;
    let sentinel = [0x5a_u8; 12];
    residual_output.upload_from_slice(0, &sentinel, &mut staging, &mut stream)?;
    normalized_output.upload_from_slice(0, &sentinel, &mut staging, &mut stream)?;
    let allocation_stats = context.allocation_stats()?;
    let valid_descriptor = Bf16FusedDescriptor {
        matrix_span_bytes: 12,
        right_dtype: CudaDType::BF16,
        row_count: 2,
        hidden_size: 3,
        epsilon: 1.0e-5,
    };

    for (epsilon, label) in [
        (0.0, "zero epsilon"),
        (-1.0e-5, "negative epsilon"),
        (f32::NAN, "NaN epsilon"),
        (f32::INFINITY, "infinite epsilon"),
    ] {
        let error = run_bf16_fused_descriptor(
            &left,
            &right,
            &weight,
            &mut residual_output,
            &mut normalized_output,
            &mut stream,
            Bf16FusedDescriptor {
                epsilon,
                ..valid_descriptor
            },
        )
        .expect_err(label);
        assert_error_kind(&error, CudaErrorKind::InvalidArgument, label);
    }

    let zero_hidden_error = run_bf16_fused_descriptor(
        &left,
        &right,
        &weight,
        &mut residual_output,
        &mut normalized_output,
        &mut stream,
        Bf16FusedDescriptor {
            hidden_size: 0,
            ..valid_descriptor
        },
    )
    .expect_err("zero hidden size must fail");
    assert_error_kind(
        &zero_hidden_error,
        CudaErrorKind::InvalidArgument,
        "zero hidden size",
    );

    let dtype_error = run_bf16_fused_descriptor(
        &left,
        &right,
        &weight,
        &mut residual_output,
        &mut normalized_output,
        &mut stream,
        Bf16FusedDescriptor {
            right_dtype: CudaDType::F32,
            ..valid_descriptor
        },
    )
    .expect_err("mixed dtypes must fail");
    assert_error_kind(&dtype_error, CudaErrorKind::InvalidArgument, "mixed dtypes");

    let capacity_error = run_bf16_fused_descriptor(
        &left,
        &right,
        &weight,
        &mut residual_output,
        &mut normalized_output,
        &mut stream,
        Bf16FusedDescriptor {
            matrix_span_bytes: 10,
            ..valid_descriptor
        },
    )
    .expect_err("short matrix span must fail");
    assert_error_kind(
        &capacity_error,
        CudaErrorKind::OutOfRange,
        "short matrix span",
    );

    assert_eq!(
        download(&context, &mut stream, &mut residual_output)?,
        sentinel,
        "validation errors must not write the residual output"
    );
    assert_eq!(
        download(&context, &mut stream, &mut normalized_output)?,
        sentinel,
        "validation errors must not write the normalized output"
    );

    run_bf16_fused_descriptor(
        &left,
        &right,
        &weight,
        &mut residual_output,
        &mut normalized_output,
        &mut stream,
        Bf16FusedDescriptor {
            matrix_span_bytes: 0,
            row_count: 0,
            ..valid_descriptor
        },
    )?;
    assert_eq!(
        download(&context, &mut stream, &mut residual_output)?,
        sentinel,
        "zero rows must be a no-op"
    );
    assert_eq!(
        download(&context, &mut stream, &mut normalized_output)?,
        sentinel,
        "zero rows must be a no-op"
    );

    run_bf16_fused_descriptor(
        &left,
        &right,
        &weight,
        &mut residual_output,
        &mut normalized_output,
        &mut stream,
        valid_descriptor,
    )?;
    {
        let mut residual = ResidualAddParams {
            left: CudaBufferSpan::new(&left, CudaDType::BF16, 0, 12)?,
            right: CudaBufferSpan::new(&right, CudaDType::BF16, 0, 12)?,
            output: CudaBufferSpanMut::new(&mut standalone_residual, CudaDType::BF16, 0, 12)?,
            element_count: 6,
        };
        residual_add(&mut residual, &mut stream)?;
    }
    {
        let mut norm = RmsNormParams {
            input: CudaBufferSpan::new(&standalone_residual, CudaDType::BF16, 0, 12)?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, 6)?,
            output: CudaBufferSpanMut::new(&mut standalone_norm, CudaDType::BF16, 0, 12)?,
            row_count: 2,
            hidden_size: 3,
            epsilon: 1.0e-5,
        };
        rms_norm(&mut norm, &mut stream)?;
    }
    assert_eq!(
        download(&context, &mut stream, &mut residual_output)?,
        download(&context, &mut stream, &mut standalone_residual)?,
        "resources must remain usable after rejected calls"
    );
    assert_eq!(
        download(&context, &mut stream, &mut normalized_output)?,
        download(&context, &mut stream, &mut standalone_norm)?,
        "resources must remain usable after rejected calls"
    );
    assert_eq!(allocation_stats, context.allocation_stats()?);

    // Exact in-place and overlapping descriptors cannot be expressed through
    // the public safe wrapper: immutable inputs and mutable outputs borrow the
    // owning allocation incompatibly. Native ABI overlap checks therefore
    // require a private-FFI test or a future explicit safe in-place entry point.
    left.close()?;
    right.close()?;
    weight.close()?;
    residual_output.close()?;
    normalized_output.close()?;
    standalone_residual.close()?;
    standalone_norm.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn rope_table_matches_hugging_face_cuda_bf16_boundaries() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    const POSITION_COUNT: u16 = 8_192;
    const HALF_DIMENSION: u16 = 32;
    let element_count = usize::from(POSITION_COUNT) * usize::from(HALF_DIMENSION);
    let mut angles = Vec::with_capacity(element_count);
    let mut cpu_cos_bf16 = Vec::with_capacity(element_count);
    let mut cpu_sin_bf16 = Vec::with_capacity(element_count);
    for position in 0..POSITION_COUNT {
        for pair in 0..HALF_DIMENSION {
            let exponent = f32::from(2 * pair) / 64.0;
            let inverse_frequency = 1.0 / 100_000.0_f32.powf(exponent);
            let angle = f32::from(position) * inverse_frequency;
            let (sine, cosine) = angle.sin_cos();
            angles.push(angle);
            cpu_cos_bf16.push(f32_to_bf16_bits(cosine));
            cpu_sin_bf16.push(f32_to_bf16_bits(sine));
        }
    }
    let table_bytes = u64::try_from(element_count)? * 4;
    let mut staging = context.allocate_pinned_host_buffer(table_bytes)?;
    let mut angles_cos = upload(&context, &mut stream, &mut staging, &f32_bytes(&angles))?;
    let mut sin = context.allocate_device_buffer(table_bytes)?;
    let mut params = RopeTableParams {
        angles_cos: CudaBufferSpanMut::new(&mut angles_cos, CudaDType::F32, 0, table_bytes)?,
        sin: CudaBufferSpanMut::new(&mut sin, CudaDType::F32, 0, table_bytes)?,
        element_count: u64::try_from(element_count)?,
    };
    rope_table(&mut params, &mut stream)?;

    let cos_values = decode_f32(&download(&context, &mut stream, &mut angles_cos)?);
    let sin_values = decode_f32(&download(&context, &mut stream, &mut sin)?);
    let cos_mismatches: Vec<_> = cos_values
        .into_iter()
        .map(f32_to_bf16_bits)
        .zip(cpu_cos_bf16)
        .enumerate()
        .filter_map(|(index, (cuda, cpu))| (cuda != cpu).then_some((index, cpu, cuda)))
        .collect();
    let sin_mismatches: Vec<_> = sin_values
        .into_iter()
        .map(f32_to_bf16_bits)
        .zip(cpu_sin_bf16)
        .enumerate()
        .filter_map(|(index, (cuda, cpu))| (cuda != cpu).then_some((index, cpu, cuda)))
        .collect();
    assert_eq!(
        cos_mismatches,
        [(usize::from(5_972_u16) * 32 + 17, 0x3f51, 0x3f52)]
    );
    assert_eq!(
        sin_mismatches,
        [
            (usize::from(746_u16) * 32 + 16, 0x3f34, 0x3f35),
            (usize::from(2_402_u16) * 32 + 9, 0x3c39, 0x3c38),
        ]
    );

    angles_cos.close()?;
    sin.close()?;
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
