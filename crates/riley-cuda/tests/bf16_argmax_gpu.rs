use std::error::Error;

use riley_cuda::{
    BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE, BF16_ARGMAX_STATUS_SUCCESS,
    Bf16ArgmaxParams, Bf16ArgmaxTieSource, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice,
    CudaDeviceBuffer, CudaErrorKind, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    ResidualAddParams, deterministic_bf16_argmax, deterministic_bf16_argmax_bounded,
    deterministic_bf16_argmax_fp32_ties, residual_add,
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

/// Independent oracle for the bounded contract: any non-finite column fails
/// the row, and only token IDs below `bound` can win (lowest ID on ties).
fn cpu_bounded_argmax(bits: &[u16], vocabulary: usize, bound: usize) -> Vec<u8> {
    let mut words = Vec::new();
    for row in bits.chunks_exact(vocabulary) {
        let value = |bits: u16| f32::from_bits(u32::from(bits) << 16);
        if row.iter().any(|&bits| !value(bits).is_finite()) {
            words.extend([BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE]);
            continue;
        }
        let mut winner = 0;
        for token in 1..bound {
            if value(row[token]) > value(row[winner]) {
                winner = token;
            }
        }
        words.extend([u32::try_from(winner).unwrap(), BF16_ARGMAX_STATUS_SUCCESS]);
    }
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

#[test]
#[ignore = "remote GPU"]
fn bounded_argmax_matches_cpu_oracle_and_ignores_columns_past_the_bound() -> TestResult {
    const ROW_COUNT: usize = 6;
    const VOCABULARY_SIZE: usize = 1_031;
    const BOUND: usize = 1_000;
    let mut logits_bits = vec![0xbf80_u16; ROW_COUNT * VOCABULARY_SIZE]; // -1.0
    let index = |row: usize, token: usize| row * VOCABULARY_SIZE + token;
    // Row 0: the global maximum lies past the bound and must not win.
    logits_bits[index(0, 1_020)] = 0x4120; // 10.0
    logits_bits[index(0, 37)] = 0x4000; // 2.0
    // Row 1: the best in-bound value is the last candidate.
    logits_bits[index(1, BOUND - 1)] = 0x4040; // 3.0
    logits_bits[index(1, BOUND)] = 0x4080; // 4.0, first excluded column
    // Row 2: ties inside the bound keep the lowest ID; signed zeros are equal.
    logits_bits[index(2, 400)] = 0x0000;
    logits_bits[index(2, 90)] = 0x8000;
    logits_bits[index(2, 1_005)] = 0x3f80;
    // Row 3: a non-finite value past the bound still fails the row.
    logits_bits[index(3, 5)] = 0x4000;
    logits_bits[index(3, 1_030)] = 0x7fc1; // NaN
    // Row 4: a non-finite value inside the bound fails the row.
    logits_bits[index(4, 2)] = 0x7f80;
    // Row 5: every candidate equal; token 0 wins.
    let expected = cpu_bounded_argmax(&logits_bits, VOCABULARY_SIZE, BOUND);
    let word = |token: u32, status: u32| [token, status];
    let pinned: Vec<u8> = [
        word(37, BF16_ARGMAX_STATUS_SUCCESS),
        word(999, BF16_ARGMAX_STATUS_SUCCESS),
        word(90, BF16_ARGMAX_STATUS_SUCCESS),
        word(BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE),
        word(BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE),
        word(0, BF16_ARGMAX_STATUS_SUCCESS),
    ]
    .iter()
    .flatten()
    .flat_map(|value| value.to_ne_bytes())
    .collect();
    assert_eq!(
        expected, pinned,
        "the CPU oracle must pin every token/status word"
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
    for bound in [0_u64, u64::try_from(VOCABULARY_SIZE)? + 1] {
        let mut params = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits.byte_len())?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, result_byte_len)?,
            row_count: u64::try_from(ROW_COUNT)?,
            vocabulary_size: u64::try_from(VOCABULARY_SIZE)?,
        };
        let error = deterministic_bf16_argmax_bounded(&mut params, bound, &mut stream)
            .expect_err("a bound outside 1..=vocabulary must be rejected");
        assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
    }
    assert_eq!(download(&context, &mut stream, &mut results)?, sentinel);
    {
        let mut params = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits.byte_len())?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, result_byte_len)?,
            row_count: u64::try_from(ROW_COUNT)?,
            vocabulary_size: u64::try_from(VOCABULARY_SIZE)?,
        };
        deterministic_bf16_argmax_bounded(&mut params, u64::try_from(BOUND)?, &mut stream)?;
    }
    assert_eq!(
        download(&context, &mut stream, &mut results)?,
        expected,
        "bounded device token/status bytes differ from the CPU oracle"
    );
    // A bound equal to the vocabulary is the unbounded contract.
    {
        let mut params = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits.byte_len())?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, result_byte_len)?,
            row_count: u64::try_from(ROW_COUNT)?,
            vocabulary_size: u64::try_from(VOCABULARY_SIZE)?,
        };
        deterministic_bf16_argmax_bounded(
            &mut params,
            u64::try_from(VOCABULARY_SIZE)?,
            &mut stream,
        )?;
    }
    let full = download(&context, &mut stream, &mut results)?;
    {
        let mut params = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits.byte_len())?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, result_byte_len)?,
            row_count: u64::try_from(ROW_COUNT)?,
            vocabulary_size: u64::try_from(VOCABULARY_SIZE)?,
        };
        deterministic_bf16_argmax(&mut params, &mut stream)?;
    }
    assert_eq!(download(&context, &mut stream, &mut results)?, full);
    assert_eq!(
        full,
        cpu_bounded_argmax(&logits_bits, VOCABULARY_SIZE, VOCABULARY_SIZE)
    );

    logits.close()?;
    results.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
/// Model-sized vocabularies: each thread of the row's block holds several
/// 8-logit groups in flight, and odd sizes leave rows that start off a
/// 16-byte boundary. Planted equal maxima far apart (lowest ID wins) and a
/// maximum in the unaligned head or tail are compared with the CPU oracle.
fn deterministic_bf16_argmax_matches_cpu_bytes_for_model_vocabularies() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    for (vocabulary_size, row_count) in [
        (151_936_usize, 3_usize),
        (152_064, 2),
        (49_152, 4),
        (128_256, 2),
        (32_768 + 3, 5),
        (8_192 * 4 + 1, 3),
    ] {
        // Pseudo-random finite logits in [-8, 8) as BF16 bits.
        let mut bits: Vec<u16> = (0..row_count * vocabulary_size)
            .map(|i| {
                let unit = ((i.wrapping_mul(2_654_435_761) >> 7) % 4096) as f32 / 256.0 - 8.0;
                (unit.to_bits() >> 16) as u16
            })
            .collect();
        for row in 0..row_count {
            let at = |token: usize| row * vocabulary_size + token;
            match row % 4 {
                // Equal maxima: the lower ID in the last load slot of one
                // thread, the higher in the first group.
                0 => {
                    bits[at(vocabulary_size - 9)] = 0x4180;
                    bits[at(5)] = 0x4180;
                }
                // Maximum in the last (possibly unaligned) column.
                1 => bits[at(vocabulary_size - 1)] = 0x4190,
                // Maximum in the first column; an equal value much later.
                2 => {
                    bits[at(0)] = 0x41a0;
                    bits[at(vocabulary_size / 2 + 3)] = 0x41a0;
                }
                // Equal maxima inside the same 8-logit group.
                _ => {
                    bits[at(4096 * 8 + 2)] = 0x41b0;
                    bits[at(4096 * 8 + 1)] = 0x41b0;
                }
            }
        }
        let expected = cpu_bf16_argmax_bytes(&bits, row_count, vocabulary_size);
        let logits_bytes = bf16_bit_bytes(&bits);
        let mut staging =
            context.allocate_pinned_host_buffer(u64::try_from(logits_bytes.len())?)?;
        let logits = upload(&context, &mut stream, &mut staging, &logits_bytes)?;
        let result_byte_len = u64::try_from(row_count * 8)?;
        let mut results = upload(
            &context,
            &mut stream,
            &mut staging,
            &vec![0xa5; usize::try_from(result_byte_len)?],
        )?;
        // Every row on its own (one row per call, as at one client) and all
        // rows together.
        let mut single_rows = Vec::new();
        for row in 0..row_count {
            let row_bytes = u64::try_from(vocabulary_size * 2)?;
            let mut params = Bf16ArgmaxParams {
                logits: CudaBufferSpan::new(
                    &logits,
                    CudaDType::BF16,
                    u64::try_from(row)? * row_bytes,
                    row_bytes,
                )?,
                results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, 8)?,
                row_count: 1,
                vocabulary_size: u64::try_from(vocabulary_size)?,
            };
            deterministic_bf16_argmax(&mut params, &mut stream)?;
            single_rows.extend_from_slice(&download(&context, &mut stream, &mut results)?[..8]);
        }
        let mut params = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&logits, CudaDType::BF16, 0, logits.byte_len())?,
            results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, result_byte_len)?,
            row_count: u64::try_from(row_count)?,
            vocabulary_size: u64::try_from(vocabulary_size)?,
        };
        deterministic_bf16_argmax(&mut params, &mut stream)?;
        let together = download(&context, &mut stream, &mut results)?;
        eprintln!(
            "argmax vocabulary={vocabulary_size} rows={row_count} single-row match={} all-rows match={}",
            single_rows == expected,
            together == expected
        );
        assert_eq!(
            single_rows, expected,
            "vocabulary {vocabulary_size}: single-row calls differ"
        );
        assert_eq!(
            together, expected,
            "vocabulary {vocabulary_size}: multi-row call differs"
        );
        logits.close()?;
        results.close()?;
        staging.close()?;
    }
    stream.close()?;
    close_context(context)
}

fn bf16_of(value: f32) -> u16 {
    let bits = value.to_bits();
    assert_eq!(bits & 0xffff, 0, "fixture values must be exact in BF16");
    u16::try_from(bits >> 16).expect("upper half")
}

/// Tie-refined selection of one row: the plain argmax, unless the row's
/// maximum is shared by 2..=64 selectable tokens, which are then ranked by
/// their (integer-exact) FP32 dot product, lowest token on equal scores.
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
fn cpu_fp32_tie_token(
    row: &[u16],
    hidden: &[f32],
    weight: &[f32],
    candidate_count: usize,
) -> u32 {
    let value = |token: usize| f32::from_bits(u32::from(row[token]) << 16);
    let maximum = (0..candidate_count).map(value).fold(f32::NEG_INFINITY, f32::max);
    let tied: Vec<usize> = (0..candidate_count).filter(|&t| value(t) == maximum).collect();
    let pick = if tied.len() < 2 || tied.len() > 64 {
        tied[0]
    } else {
        let hidden_size = hidden.len();
        let score = |t: usize| -> f32 {
            (0..hidden_size).map(|k| hidden[k] * weight[t * hidden_size + k]).sum()
        };
        let mut best = tied[0];
        for &t in &tied[1..] {
            if score(t) > score(best) {
                best = t;
            }
        }
        best
    };
    u32::try_from(pick).expect("small fixture token")
}

#[test]
#[ignore = "remote GPU"]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn fp32_tie_break_rescores_only_tied_maxima_per_row() -> TestResult {
    const ROWS: usize = 6;
    const VOCABULARY: usize = 2053;
    const HIDDEN: usize = 96;
    const BOUND: usize = 2000;

    // Small integers keep every FP32 dot product exact, so the oracle does
    // not depend on summation order.
    let weight: Vec<f32> = (0..VOCABULARY * HIDDEN)
        .map(|i| ((i / HIDDEN * 7 + i % HIDDEN * 3) % 5) as f32 - 2.0)
        .collect();
    let mut weight = weight;
    // Rows 1/2: a later tied token scores strictly higher; row 4: equal rows.
    for k in 0..HIDDEN {
        weight[1500 * HIDDEN + k] = 2.0;
        weight[5 * HIDDEN + k] = -1.0;
        weight[700 * HIDDEN + k] = 2.0;
        weight[10 * HIDDEN + k] = 1.0;
        weight[2052 * HIDDEN + k] = 0.0;
        weight[40 * HIDDEN + k] = 1.0;
        weight[41 * HIDDEN + k] = 1.0;
    }
    let hidden: Vec<f32> = (0..ROWS * HIDDEN).map(|i| ((i % 7) as f32) - 1.0).collect();
    let mut logits = vec![bf16_of(-8.0); ROWS * VOCABULARY];
    let at = |row: usize, token: usize| row * VOCABULARY + token;
    logits[at(0, 900)] = bf16_of(4.0); // no tie
    logits[at(0, 901)] = bf16_of(3.0);
    logits[at(1, 5)] = bf16_of(3.0);
    logits[at(1, 1500)] = bf16_of(3.0);
    logits[at(2, 10)] = bf16_of(6.0);
    logits[at(2, 700)] = bf16_of(6.0);
    logits[at(2, 2052)] = bf16_of(6.0); // past the bound in the bounded run
    logits[at(3, 100)] = bf16_of(5.0);
    logits[at(3, 2052)] = bf16_of(5.0); // only tie partner is past the bound
    logits[at(4, 41)] = bf16_of(2.0);
    logits[at(4, 40)] = bf16_of(2.0);
    for token in 200..270 {
        logits[at(5, token)] = bf16_of(1.0); // 70 tied: over capacity
    }
    let weight_bits: Vec<u16> = weight.iter().map(|&v| bf16_of(v)).collect();
    let hidden_bits: Vec<u16> = hidden.iter().map(|&v| bf16_of(v)).collect();

    let expected = |candidate_count: usize, hidden_of: &dyn Fn(usize) -> usize| -> Vec<u32> {
        (0..ROWS)
            .map(|row| {
                let h = hidden_of(row);
                cpu_fp32_tie_token(
                    &logits[row * VOCABULARY..(row + 1) * VOCABULARY],
                    &hidden[h * HIDDEN..(h + 1) * HIDDEN],
                    &weight,
                    candidate_count,
                )
            })
            .collect()
    };
    let unbounded = expected(VOCABULARY, &|row| row);
    // The plain argmax would pick 900, 5, 10, 100, 40, 200.
    assert_eq!(unbounded, vec![900, 1500, 700, 2052, 40, 200], "oracle fixture");
    // Bounded, row 3 keeps one selectable token and is not rescored.
    assert_eq!(expected(BOUND, &|row| row), vec![900, 1500, 700, 100, 40, 200]);

    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let weight_bytes = bf16_bit_bytes(&weight_bits);
    let mut staging = context.allocate_pinned_host_buffer(u64::try_from(weight_bytes.len())?)?;
    let logits_buffer = upload(&context, &mut stream, &mut staging, &bf16_bit_bytes(&logits))?;
    let weight_buffer = upload(&context, &mut stream, &mut staging, &weight_bytes)?;
    // Gathered hidden rows in output order, and the same rows reversed behind
    // an index map.
    let gathered = upload(&context, &mut stream, &mut staging, &bf16_bit_bytes(&hidden_bits))?;
    let reversed_bits: Vec<u16> = (0..ROWS)
        .rev()
        .flat_map(|row| hidden_bits[row * HIDDEN..(row + 1) * HIDDEN].to_vec())
        .collect();
    let reversed = upload(&context, &mut stream, &mut staging, &bf16_bit_bytes(&reversed_bits))?;
    let indices: Vec<u32> = (0..ROWS).map(|row| u32::try_from(ROWS - 1 - row).expect("small")).collect();
    let indices_buffer = upload(&context, &mut stream, &mut staging, &u32_bytes(&indices))?;
    let mut results = context.allocate_device_buffer(u64::try_from(ROWS * 8)?)?;

    let mut run = |hidden: &CudaDeviceBuffer,
                   gathered_rows: bool,
                   row_count: usize,
                   logits_offset: usize,
                   candidate_count: usize,
                   stream: &mut CudaStream,
                   results: &mut CudaDeviceBuffer|
     -> TestResult<Vec<u32>> {
        {
            let mut params = Bf16ArgmaxParams {
                logits: CudaBufferSpan::new(
                    &logits_buffer,
                    CudaDType::BF16,
                    u64::try_from(logits_offset * VOCABULARY * 2)?,
                    u64::try_from(row_count * VOCABULARY * 2)?,
                )?,
                results: CudaBufferSpanMut::new(results, CudaDType::U32, 0, u64::try_from(row_count * 8)?)?,
                row_count: u64::try_from(row_count)?,
                vocabulary_size: u64::try_from(VOCABULARY)?,
            };
            let source = Bf16ArgmaxTieSource {
                hidden: CudaBufferSpan::new(
                    hidden,
                    CudaDType::BF16,
                    u64::try_from(logits_offset * HIDDEN * 2)?,
                    u64::try_from(row_count * HIDDEN * 2)?,
                )?,
                weight: CudaBufferSpan::new(&weight_buffer, CudaDType::BF16, 0, weight_buffer.byte_len())?,
                row_indices: CudaBufferSpan::new(&indices_buffer, CudaDType::U32, 0, u64::try_from(row_count * 4)?)?,
                hidden_size: u64::try_from(HIDDEN)?,
                hidden_rows: u64::try_from(row_count)?,
                rows_gathered: gathered_rows,
                window_ulps: 0,
            };
            deterministic_bf16_argmax_fp32_ties(
                &mut params,
                &source,
                u64::try_from(candidate_count)?,
                stream,
            )?;
        }
        let bytes = download(&context, stream, results)?;
        Ok(bytes
            .chunks_exact(8)
            .take(row_count)
            .map(|word| {
                let status = u32::from_ne_bytes(word[4..8].try_into().expect("word"));
                assert_eq!(status, BF16_ARGMAX_STATUS_SUCCESS);
                u32::from_ne_bytes(word[0..4].try_into().expect("word"))
            })
            .collect())
    };

    assert_eq!(run(&gathered, true, ROWS, 0, 0, &mut stream, &mut results)?, unbounded);
    assert_eq!(
        run(&gathered, true, ROWS, 0, BOUND, &mut stream, &mut results)?,
        expected(BOUND, &|row| row)
    );
    // Hidden rows found through the index map select the same tokens.
    assert_eq!(run(&reversed, false, ROWS, 0, 0, &mut stream, &mut results)?, unbounded);
    // A row alone selects what it selects inside the batch.
    for row in 0..ROWS {
        assert_eq!(
            run(&gathered, true, 1, row, 0, &mut stream, &mut results)?,
            vec![unbounded[row]],
            "row {row} alone"
        );
    }

    logits_buffer.close()?;
    weight_buffer.close()?;
    gathered.close()?;
    reversed.close()?;
    indices_buffer.close()?;
    results.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

/// Timing screen, not a correctness check: wall time per launch of the plain
/// and the FP32-tie argmax at a Qwen3 vocabulary, with and without a tied
/// row maximum. Run with `--nocapture` to read the numbers.
#[test]
#[ignore = "remote GPU"]
#[allow(clippy::cast_precision_loss)]
fn fp32_tie_break_launch_cost_screen() -> TestResult {
    const VOCABULARY: usize = 151_936;
    const HIDDEN: usize = 1024;
    const ITERATIONS: u32 = 2000;
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    for rows in [1usize, 32] {
        for tie in [false, true] {
            let mut logits: Vec<u16> = (0..rows * VOCABULARY)
                .map(|i| bf16_of(((i * 2_654_435_761) % 64) as f32 / 16.0))
                .collect();
            for row in 0..rows {
                logits[row * VOCABULARY + 77_777] = bf16_of(20.0);
                if tie {
                    logits[row * VOCABULARY + 99] = bf16_of(20.0);
                }
            }
            let weight_bits = vec![bf16_of(0.015_625); VOCABULARY * HIDDEN];
            let hidden_bits = vec![bf16_of(1.0); rows * HIDDEN];
            let weight_bytes = bf16_bit_bytes(&weight_bits);
            let mut staging =
                context.allocate_pinned_host_buffer(u64::try_from(weight_bytes.len())?)?;
            let logits_buffer =
                upload(&context, &mut stream, &mut staging, &bf16_bit_bytes(&logits))?;
            let weight_buffer = upload(&context, &mut stream, &mut staging, &weight_bytes)?;
            let hidden_buffer =
                upload(&context, &mut stream, &mut staging, &bf16_bit_bytes(&hidden_bits))?;
            let indices: Vec<u32> = (0..rows).map(|r| u32::try_from(r).expect("small")).collect();
            let indices_buffer = upload(&context, &mut stream, &mut staging, &u32_bytes(&indices))?;
            let mut results = context.allocate_device_buffer(u64::try_from(rows * 8)?)?;
            for refine in [false, true] {
                let mut launch = |stream: &mut CudaStream| -> TestResult {
                    let mut params = Bf16ArgmaxParams {
                        logits: CudaBufferSpan::new(
                            &logits_buffer,
                            CudaDType::BF16,
                            0,
                            logits_buffer.byte_len(),
                        )?,
                        results: CudaBufferSpanMut::new(
                            &mut results,
                            CudaDType::U32,
                            0,
                            u64::try_from(rows * 8)?,
                        )?,
                        row_count: u64::try_from(rows)?,
                        vocabulary_size: u64::try_from(VOCABULARY)?,
                    };
                    if refine {
                        let source = Bf16ArgmaxTieSource {
                            hidden: CudaBufferSpan::new(
                                &hidden_buffer,
                                CudaDType::BF16,
                                0,
                                hidden_buffer.byte_len(),
                            )?,
                            weight: CudaBufferSpan::new(
                                &weight_buffer,
                                CudaDType::BF16,
                                0,
                                weight_buffer.byte_len(),
                            )?,
                            row_indices: CudaBufferSpan::new(
                                &indices_buffer,
                                CudaDType::U32,
                                0,
                                indices_buffer.byte_len(),
                            )?,
                            hidden_size: u64::try_from(HIDDEN)?,
                            hidden_rows: u64::try_from(rows)?,
                            rows_gathered: true,
                            window_ulps: 0,
                        };
                        deterministic_bf16_argmax_fp32_ties(&mut params, &source, 0, stream)?;
                    } else {
                        deterministic_bf16_argmax(&mut params, stream)?;
                    }
                    Ok(())
                };
                for _ in 0..100 {
                    launch(&mut stream)?;
                }
                context.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..ITERATIONS {
                    launch(&mut stream)?;
                }
                context.synchronize()?;
                let micros = start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERATIONS);
                println!("ARGMAX-COST rows={rows} tie={tie} refine={refine} us/launch={micros:.2}");
            }
            logits_buffer.close()?;
            weight_buffer.close()?;
            hidden_buffer.close()?;
            indices_buffer.close()?;
            results.close()?;
            staging.close()?;
        }
    }
    stream.close()?;
    close_context(context)
}

/// Window mode: a logit one BF16 step below the row maximum is rescored and
/// wins when its FP32 score is larger; two steps below stays out of a
/// one-step window. Exact mode keeps the maximum.
#[test]
#[ignore = "remote GPU"]
fn fp32_window_rescores_logits_near_the_maximum() -> TestResult {
    const VOCABULARY: usize = 300;
    const HIDDEN: usize = 32;
    let mut weight = vec![0.0f32; VOCABULARY * HIDDEN];
    for k in 0..HIDDEN {
        weight[10 * HIDDEN + k] = 1.0; // row max, FP32 score 32
        weight[20 * HIDDEN + k] = 2.0; // one step below, score 64
        weight[30 * HIDDEN + k] = 3.0; // two steps below, score 96
    }
    let hidden = vec![1.0f32; HIDDEN];
    // At exponent 2 (values in [4, 8)) one BF16 step is 1/32.
    let mut logits = vec![bf16_of(-8.0); VOCABULARY];
    logits[10] = bf16_of(4.5);
    logits[20] = bf16_of(4.5 - 1.0 / 32.0);
    logits[30] = bf16_of(4.5 - 2.0 / 32.0);
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let weight_bits: Vec<u16> = weight.iter().map(|&v| bf16_of(v)).collect();
    let weight_bytes = bf16_bit_bytes(&weight_bits);
    let mut staging = context.allocate_pinned_host_buffer(u64::try_from(weight_bytes.len())?)?;
    let logits_buffer = upload(&context, &mut stream, &mut staging, &bf16_bit_bytes(&logits))?;
    let weight_buffer = upload(&context, &mut stream, &mut staging, &weight_bytes)?;
    let hidden_bits: Vec<u16> = hidden.iter().map(|&v| bf16_of(v)).collect();
    let hidden_buffer = upload(&context, &mut stream, &mut staging, &bf16_bit_bytes(&hidden_bits))?;
    let indices_buffer = upload(&context, &mut stream, &mut staging, &u32_bytes(&[0]))?;
    let mut results = context.allocate_device_buffer(8)?;
    let mut picks = Vec::new();
    for window_ulps in [0u32, 1, 2] {
        {
            let mut params = Bf16ArgmaxParams {
                logits: CudaBufferSpan::new(&logits_buffer, CudaDType::BF16, 0, logits_buffer.byte_len())?,
                results: CudaBufferSpanMut::new(&mut results, CudaDType::U32, 0, 8)?,
                row_count: 1,
                vocabulary_size: u64::try_from(VOCABULARY)?,
            };
            let source = Bf16ArgmaxTieSource {
                hidden: CudaBufferSpan::new(&hidden_buffer, CudaDType::BF16, 0, hidden_buffer.byte_len())?,
                weight: CudaBufferSpan::new(&weight_buffer, CudaDType::BF16, 0, weight_buffer.byte_len())?,
                row_indices: CudaBufferSpan::new(&indices_buffer, CudaDType::U32, 0, 4)?,
                hidden_size: u64::try_from(HIDDEN)?,
                hidden_rows: 1,
                rows_gathered: true,
                window_ulps,
            };
            deterministic_bf16_argmax_fp32_ties(&mut params, &source, 0, &mut stream)?;
        }
        let bytes = download(&context, &mut stream, &mut results)?;
        picks.push(u32::from_ne_bytes(bytes[0..4].try_into()?));
    }
    assert_eq!(picks, vec![10, 20, 30]);
    logits_buffer.close()?;
    weight_buffer.close()?;
    hidden_buffer.close()?;
    indices_buffer.close()?;
    results.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}
