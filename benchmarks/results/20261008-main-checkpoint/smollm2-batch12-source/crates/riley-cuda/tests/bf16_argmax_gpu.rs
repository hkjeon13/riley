use std::error::Error;

use riley_cuda::{
    BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE, BF16_ARGMAX_STATUS_SUCCESS,
    Bf16ArgmaxParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice,
    CudaDeviceBuffer, CudaErrorKind, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    ResidualAddParams, deterministic_bf16_argmax, residual_add,
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

fn bf16_bit_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[allow(clippy::float_cmp)]
fn cpu_bf16_argmax_bytes(logits: &[u16], row_count: usize, vocabulary_size: usize) -> Vec<u8> {
    assert_ne!(vocabulary_size, 0);
    assert_eq!(logits.len(), row_count * vocabulary_size);
    let mut words = Vec::with_capacity(row_count * 2);
    for row in logits.chunks_exact(vocabulary_size) {
        let mut selected: Option<(f32, u32)> = None;
        let mut non_finite = false;
        for (index, &bits) in row.iter().enumerate() {
            if bits & 0x7f80 == 0x7f80 {
                non_finite = true;
                continue;
            }
            let value = f32::from_bits(u32::from(bits) << 16);
            let token_id = u32::try_from(index).expect("small CPU fixture token id");
            if selected.is_none_or(|(maximum, selected_token)| {
                value > maximum || (value == maximum && token_id < selected_token)
            }) {
                selected = Some((value, token_id));
            }
        }
        if non_finite {
            words.extend_from_slice(&[BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE]);
        } else {
            words.extend_from_slice(&[
                selected.expect("finite non-empty row").1,
                BF16_ARGMAX_STATUS_SUCCESS,
            ]);
        }
    }
    u32_bytes(&words)
}

#[test]
#[ignore = "remote GPU"]
fn deterministic_bf16_argmax_matches_cpu_bytes_for_odd_multi_row_edges() -> TestResult {
    const ROW_COUNT: usize = 7;
    const VOCABULARY_SIZE: usize = 257;

    // -8.0 gives every row a finite baseline. The selected witnesses exercise
    // the last odd column, cross-warp ties, signed-zero equality, all-negative
    // maxima, and each BF16 non-finite class.
    let mut logits_bits = vec![0xc100; ROW_COUNT * VOCABULARY_SIZE];
    let index = |row: usize, token: usize| row * VOCABULARY_SIZE + token;
    logits_bits[index(0, 256)] = 0x40a0; // 5.0
    logits_bits[index(1, 3)] = 0x4040; // 3.0, lower-id tie winner
    logits_bits[index(1, 193)] = 0x4040;
    logits_bits[index(2, 17)] = 0x8000; // -0.0, lower-id tie winner
    logits_bits[index(2, 201)] = 0x0000; // +0.0
    logits_bits[index(3, 7)] = 0xbf80; // -1.0
    logits_bits[index(3, 251)] = 0xbf00; // -0.5, finite negative maximum
    logits_bits[index(4, 29)] = 0x7fc1; // NaN
    logits_bits[index(5, 127)] = 0x7f80; // +infinity
    logits_bits[index(6, 255)] = 0xff80; // -infinity

    let cpu_expected = cpu_bf16_argmax_bytes(&logits_bits, ROW_COUNT, VOCABULARY_SIZE);
    assert_eq!(
        cpu_expected,
        u32_bytes(&[
            256,
            BF16_ARGMAX_STATUS_SUCCESS,
            3,
            BF16_ARGMAX_STATUS_SUCCESS,
            17,
            BF16_ARGMAX_STATUS_SUCCESS,
            251,
            BF16_ARGMAX_STATUS_SUCCESS,
            BF16_ARGMAX_INVALID_TOKEN_ID,
            BF16_ARGMAX_STATUS_NON_FINITE,
            BF16_ARGMAX_INVALID_TOKEN_ID,
            BF16_ARGMAX_STATUS_NON_FINITE,
            BF16_ARGMAX_INVALID_TOKEN_ID,
            BF16_ARGMAX_STATUS_NON_FINITE,
        ]),
        "the independent CPU oracle must pin every token/status word"
    );

    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let logits_bytes = bf16_bit_bytes(&logits_bits);
    let mut staging = context.allocate_pinned_host_buffer(u64::try_from(logits_bytes.len())?)?;
    let logits = upload(&context, &mut stream, &mut staging, &logits_bytes)?;
    let result_byte_len = u64::try_from(ROW_COUNT * 8)?;
    let sentinel = vec![0xa5; usize::try_from(result_byte_len)?];
    let mut results = upload(&context, &mut stream, &mut staging, &sentinel)?;
    let stable_allocations = context.allocation_stats()?;

    for _ in 0..64 {
        let mut params = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits.byte_len())?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, result_byte_len)?,
            row_count: u64::try_from(ROW_COUNT)?,
            vocabulary_size: u64::try_from(VOCABULARY_SIZE)?,
        };
        deterministic_bf16_argmax(&mut params, &mut stream)?;
    }
    assert_eq!(
        stable_allocations,
        context.allocation_stats()?,
        "repeated greedy iterations must not change CUDA allocation accounting"
    );
    assert_eq!(
        download(&context, &mut stream, &mut results)?,
        cpu_expected,
        "device token/status bytes differ from the BF16 CPU oracle"
    );

    results.upload_from_slice(0, &sentinel, &mut staging, &mut stream)?;
    {
        let mut zero_rows = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, 0)?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, 0)?,
            row_count: 0,
            vocabulary_size: u64::try_from(VOCABULARY_SIZE)?,
        };
        deterministic_bf16_argmax(&mut zero_rows, &mut stream)?;
    }
    assert_eq!(
        download(&context, &mut stream, &mut results)?,
        sentinel,
        "zero rows must not mutate result storage"
    );

    logits.close()?;
    results.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn deterministic_bf16_argmax_observes_command_batch_order_without_allocations() -> TestResult {
    const ROW_COUNT: usize = 2;
    const VOCABULARY_SIZE: usize = 257;
    let element_count = ROW_COUNT * VOCABULARY_SIZE;
    let left_bits = vec![0xc100; element_count]; // -8.0
    let mut right_bits = vec![0x0000; element_count]; // +0.0
    right_bits[17] = 0x4100; // row 0: -8 + 8 = 0 at token 17
    right_bits[VOCABULARY_SIZE + 256] = 0x4180; // row 1: -8 + 16 = 8 at token 256
    // If argmax were launched before the preceding residual add, both rows
    // would select token 0 from this initial intermediate value instead.
    let mut intermediate_bits = vec![0xc100; element_count];
    intermediate_bits[0] = 0x4080;
    intermediate_bits[VOCABULARY_SIZE] = 0x4080;
    let expected = u32_bytes(&[
        17,
        BF16_ARGMAX_STATUS_SUCCESS,
        256,
        BF16_ARGMAX_STATUS_SUCCESS,
    ]);

    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let matrix_byte_len = u64::try_from(element_count * 2)?;
    let mut staging = context.allocate_pinned_host_buffer(matrix_byte_len)?;
    let left = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bit_bytes(&left_bits),
    )?;
    let right = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bit_bytes(&right_bits),
    )?;
    let mut intermediate = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bit_bytes(&intermediate_bits),
    )?;
    let mut results = upload(&context, &mut stream, &mut staging, &[0xa5; 16])?;
    let stable_allocations = context.allocation_stats()?;

    for _ in 0..64 {
        let mut batch = stream.begin_command_batch()?;
        {
            let mut commands = batch.commands();
            {
                let mut add = ResidualAddParams {
                    left: CudaBufferSpan::new(&left, CudaDType::BF16, 0, matrix_byte_len)?,
                    right: CudaBufferSpan::new(&right, CudaDType::BF16, 0, matrix_byte_len)?,
                    output: CudaBufferSpanMut::new(
                        &mut intermediate,
                        CudaDType::BF16,
                        0,
                        matrix_byte_len,
                    )?,
                    element_count: u64::try_from(element_count)?,
                };
                residual_add(&mut add, &mut commands)?;
            }
            {
                let mut argmax = Bf16ArgmaxParams {
                    logits: CudaBufferSpan::new(
                        &intermediate,
                        CudaDType::BF16,
                        0,
                        matrix_byte_len,
                    )?,
                    results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, 16)?,
                    row_count: u64::try_from(ROW_COUNT)?,
                    vocabulary_size: u64::try_from(VOCABULARY_SIZE)?,
                };
                deterministic_bf16_argmax(&mut argmax, &mut commands)?;
            }
        }
        batch.finish()?;
    }
    assert_eq!(
        stable_allocations,
        context.allocation_stats()?,
        "command-batch greedy iterations must reuse caller-owned storage"
    );
    assert_eq!(
        download(&context, &mut stream, &mut results)?,
        expected,
        "argmax must observe the residual-add output queued before it"
    );

    left.close()?;
    right.close()?;
    intermediate.close()?;
    results.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn deterministic_bf16_argmax_validation_fails_before_result_mutation() -> TestResult {
    const ROW_COUNT: u64 = 2;
    const VOCABULARY_SIZE: u64 = 5;
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let foreign_context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut foreign_stream = foreign_context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(64)?;
    let logits_bits = [
        0xbf80, 0x4000, 0x3f80, 0x4040, 0xc000, 0x4080, 0xbf00, 0xc040, 0x0000, 0x3f00,
    ];
    let logits = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bit_bytes(&logits_bits),
    )?;
    let sentinel_words = [0x1122_3344, 0x5566_7788, 0x99aa_bbcc, 0xddee_ff00];
    let sentinel = u32_bytes(&sentinel_words);
    let mut results = upload(&context, &mut stream, &mut staging, &sentinel)?;

    let misaligned = CudaBufferSpan::new(&logits, CudaDType::BF16, 1, 2)
        .expect_err("misaligned BF16 logits span must fail");
    assert_eq!(misaligned.kind(), CudaErrorKind::InvalidArgument);

    {
        let mut short_logits = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, 18)?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, 16)?,
            row_count: ROW_COUNT,
            vocabulary_size: VOCABULARY_SIZE,
        };
        let error = deterministic_bf16_argmax(&mut short_logits, &mut stream)
            .expect_err("short logits capacity must fail before launch");
        assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
    }
    {
        let mut short_results = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, 20)?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, 12)?,
            row_count: ROW_COUNT,
            vocabulary_size: VOCABULARY_SIZE,
        };
        let error = deterministic_bf16_argmax(&mut short_results, &mut stream)
            .expect_err("short result capacity must fail before launch");
        assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
    }
    {
        let mut foreign_owner = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, 20)?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, 16)?,
            row_count: ROW_COUNT,
            vocabulary_size: VOCABULARY_SIZE,
        };
        let error = deterministic_bf16_argmax(&mut foreign_owner, &mut foreign_stream)
            .expect_err("foreign-context stream must fail before launch");
        assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    }
    assert_eq!(
        download(&context, &mut stream, &mut results)?,
        sentinel,
        "span and context validation failures must preserve every result byte"
    );

    logits.close()?;
    results.close()?;
    staging.close()?;
    stream.close()?;
    foreign_stream.close()?;
    close_context(context)?;
    close_context(foreign_context)
}
