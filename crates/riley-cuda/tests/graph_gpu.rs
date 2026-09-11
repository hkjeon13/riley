use std::error::Error;

use riley_cuda::{
    BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE, BF16_ARGMAX_STATUS_SUCCESS,
    Bf16ArgmaxParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice,
    CudaErrorKind, CudaGraphCaptureMode, CudaResult, CudaRuntime, CudaStream, GatedMultiplyParams,
    IndexedRopeParams, ResidualAddParams, RmsNormParams, RowGatherParams, SiluParams,
    deterministic_bf16_argmax, gated_multiply, indexed_rope, residual_add, rms_norm, row_gather,
    silu,
};

fn all_f32_bits_equal(values: &[f32], expected: f32) -> bool {
    let expected_bits = expected.to_bits();
    values.iter().all(|value| value.to_bits() == expected_bits)
}

fn first_device() -> Result<CudaDevice, Box<dyn Error>> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    Ok(runtime.device(0)?)
}

fn close_context(context: CudaContext) -> Result<(), Box<dyn Error>> {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

fn assert_invalid_state<T>(result: CudaResult<T>, operation: &str) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("{operation} unexpectedly succeeded during capture"));
    assert_eq!(
        error.kind(),
        CudaErrorKind::InvalidState,
        "{operation} must reject before issuing a prohibited CUDA call"
    );
}

fn assert_eager_fill_after_recovery(
    context: &CudaContext,
    stream: &mut CudaStream,
    value: f32,
) -> Result<(), Box<dyn Error>> {
    let kernel = context.kernel();
    let values = kernel.launch_fill(stream, 4_096, value)?.finish()?;
    assert!(all_f32_bits_equal(&values, value));
    drop(kernel);
    Ok(())
}

fn download_f32_buffer(
    context: &CudaContext,
    buffer: &mut riley_cuda::CudaDeviceBuffer,
    stream: &mut CudaStream,
    element_count: u64,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let byte_len = element_count
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("f32 capture output byte length overflow")?;
    let host_len = usize::try_from(byte_len)?;
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;
    let mut bytes = vec![0_u8; host_len];
    buffer.download_to_slice(0, &mut bytes, &mut staging, stream)?;
    staging.close()?;

    let values = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(values)
}

fn move_into_cold_owner(exec: riley_cuda::OwnedGraphExec) -> riley_cuda::OwnedGraphExec {
    exec
}

fn h2d_payload(byte_len: usize, replay: usize) -> Vec<u8> {
    (0..byte_len)
        .map(|index| {
            let mixed = index
                .wrapping_mul(29)
                .wrapping_add(replay.wrapping_mul(71))
                .wrapping_add(replay.rotate_left(3));
            (mixed & 0xff) as u8
        })
        .collect()
}

fn graph_silu_bf16_fixture_bytes(element_count: usize) -> Vec<u8> {
    // Keep signed zero, normal finite values, infinities, and a payload-bearing
    // NaN in the fixed device input. The graph implementation is required to
    // match eager SiLU's BF16 storage result byte-for-byte, rather than merely
    // satisfy a tolerance after conversion back to f32.
    const PATTERN: [u16; 12] = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4080, 0xc080, 0x3d00, 0xbd00, 0x7f80, 0xff80, 0x7fc1,
        0xffc1,
    ];
    (0..element_count)
        .flat_map(|index| PATTERN[index % PATTERN.len()].to_ne_bytes())
        .collect()
}

fn graph_gated_multiply_bf16_fixture_bytes(element_count: usize, branch: usize) -> Vec<u8> {
    // Use different BF16 inputs for the activated-gate and up branches, while
    // retaining signed zero, finite values, infinities, and NaNs. Eager and
    // graph execution must agree on the exact BF16 storage-rounding result.
    const GATE: [u16; 12] = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4080, 0xc080, 0x3d00, 0xbd00, 0x7f80, 0xff80, 0x7fc1,
        0xffc1,
    ];
    const UP: [u16; 12] = [
        0x3f80, 0xbf80, 0x4000, 0xc000, 0x3f00, 0xbf00, 0x4040, 0xc040, 0x0000, 0x8000, 0x7fc5,
        0xffc5,
    ];
    let pattern = if branch == 0 { &GATE } else { &UP };
    (0..element_count)
        .flat_map(|index| pattern[index % pattern.len()].to_ne_bytes())
        .collect()
}

fn graph_residual_add_bf16_fixture_bytes(element_count: usize, branch: usize) -> Vec<u8> {
    // Keep the exact BF16 edge values relevant to add: signed zero, opposite
    // finite values, infinities, and distinct NaN payloads. The expected
    // bytes come from the eager primitive so graph parity includes its storage
    // rounding behavior rather than an f32 tolerance.
    const LEFT: [u16; 12] = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4080, 0xc080, 0x3d00, 0xbd00, 0x7f80, 0xff80, 0x7fc1,
        0xffc1,
    ];
    const RIGHT: [u16; 12] = [
        0x8000, 0x0000, 0x3f80, 0x3f80, 0xc080, 0x4080, 0x3d00, 0x3d00, 0xff80, 0x7f80, 0x7fc5,
        0xffc5,
    ];
    let pattern = if branch == 0 { &LEFT } else { &RIGHT };
    (0..element_count)
        .flat_map(|index| pattern[index % pattern.len()].to_ne_bytes())
        .collect()
}

fn graph_canonical_rms_norm_bf16_fixture_bytes(element_count: usize, branch: usize) -> Vec<u8> {
    // These deliberately finite, non-profile-specific BF16 patterns exercise
    // the generic canonical primitive's per-row reduction and its BF16 round
    // of the normalized activation before learned-weight multiplication. They
    // are not a Hugging Face SmolLM2 or Fixed37 fixture.
    const INPUT: [u16; 12] = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x3e80, 0xbe80, 0x4080, 0xc080, 0x3f00,
        0xbf00,
    ];
    const WEIGHT: [u16; 12] = [
        0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x4000, 0xc000, 0x3e80, 0xbe80, 0x3f40, 0xbf40, 0x3fc0,
        0xbfc0,
    ];
    let pattern = if branch == 0 { &INPUT } else { &WEIGHT };
    (0..element_count)
        .flat_map(|index| pattern[index % pattern.len()].to_ne_bytes())
        .collect()
}

fn graph_bf16_argmax_edge_fixture_bytes() -> Vec<u8> {
    const ROW_COUNT: usize = 7;
    const VOCABULARY_SIZE: usize = 257;

    // Keep the 7x257 edge corpus from `bf16_argmax_gpu.rs`: last odd column,
    // cross-warp equal maxima, signed-zero equality, negative maxima, and all
    // BF16 non-finite classes. Its U32 token/status layout is the semantic
    // contract for this fixed-address graph node.
    let mut logits_bits: Vec<u16> = vec![0xc100; ROW_COUNT * VOCABULARY_SIZE]; // -8.0
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
    logits_bits
        .iter()
        .flat_map(|&bits| bits.to_ne_bytes())
        .collect()
}

fn graph_bf16_argmax_edge_expected_result_bytes() -> Vec<u8> {
    [
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
    ]
    .iter()
    .flat_map(|&word| word.to_ne_bytes())
    .collect()
}

fn graph_bf16_row_gather_fixture_bytes(input_row_count: usize, column_count: usize) -> Vec<u8> {
    // Gathering is a byte-preserving BF16 copy. Keep unusual BF16 payloads in
    // every row so a replay that merely has the right numerical values but
    // corrupts storage bits cannot pass this parity test.
    const PATTERN: [u16; 16] = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x3d00, 0xbd00, 0x7f80, 0xff80, 0x7fc1,
        0xffc1, 0x7f81, 0xff81, 0x3f40, 0xbf40,
    ];
    (0..input_row_count)
        .flat_map(|row| {
            (0..column_count).map(move |column| {
                PATTERN[(row.wrapping_mul(11).wrapping_add(column.wrapping_mul(7))) % PATTERN.len()]
                    .to_ne_bytes()
            })
        })
        .flatten()
        .collect()
}

fn u32_words_to_ne_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn bf16_words_to_ne_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn f32_words_to_ne_bytes(words: &[f32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn graph_indexed_rope_bf16_fixture_bytes() -> Vec<u8> {
    // Three rows, one head, six BF16 channels. The final two channels are
    // outside a four-wide rotary dimension and must remain byte-identical.
    const WORDS: [u16; 18] = [
        0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, // row 0
        0xbf80, 0xc000, 0xc040, 0xc080, 0xc0a0, 0xc0c0, // row 1
        0x3e80, 0xbf00, 0x3f40, 0xbf80, 0x4000, 0xc000, // row 2
    ];
    bf16_words_to_ne_bytes(&WORDS)
}

fn graph_indexed_rope_table_bytes() -> (Vec<u8>, Vec<u8>) {
    // Five positions × two rotary pairs. Tables are F32 by the eager/native
    // indexed-RoPE ABI, while output comparisons remain exact BF16 bytes.
    const COS: [f32; 10] = [1.0, 1.0, 0.5, 0.25, 0.0, -0.5, -0.5, -1.0, 0.25, 0.75];
    const SIN: [f32; 10] = [0.0, 0.0, 0.5, 0.75, 1.0, 0.5, 0.75, 0.0, -0.5, 0.25];
    (f32_words_to_ne_bytes(&COS), f32_words_to_ne_bytes(&SIN))
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_silu_graph_replays_fixed_input_byte_exact_against_eager() -> Result<(), Box<dyn Error>>
{
    const ELEMENT_COUNT: u64 = 4_096;
    const REPLAYS: usize = 32;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<u16>())?)
        .ok_or("BF16 SiLU graph byte length overflow")?;
    let host_input = graph_silu_bf16_fixture_bytes(usize::try_from(ELEMENT_COUNT)?);
    assert_eq!(u64::try_from(host_input.len())?, byte_len);
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;

    let mut eager_input = context.allocate_device_buffer(byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    let graph_input = {
        let mut input = context.allocate_device_buffer(byte_len)?;
        input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        input
    };
    let mut eager_output = context.allocate_device_buffer(byte_len)?;
    let graph_output = context.allocate_device_buffer(byte_len)?;

    {
        let mut eager = SiluParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, byte_len)?,
            element_count: ELEMENT_COUNT,
        };
        silu(&mut eager, &mut eager_stream)?;
    }

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_silu_bf16_capture(
        graph_input,
        graph_output,
        ELEMENT_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_silu_bf16()?;
    // The local one-node rejection must leave the first node/capture usable.
    assert_invalid_state(
        capture.enqueue_silu_bf16(),
        "second fixed BF16 SiLU graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // The fixed-address graph transition must not allocate another tracked
    // device or pinned allocation. This is a lifecycle assertion, not a
    // performance claim about CUDA driver's internal graph bookkeeping.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }

    let resources = exec.close()?;
    let (capture_stream, mut graph_input, mut graph_output) = resources.into_parts();
    let mut eager_bytes = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_bytes = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_input_after = vec![0_u8; usize::try_from(byte_len)?];
    eager_output.download_to_slice(0, &mut eager_bytes, &mut staging, &mut transfer_stream)?;
    graph_output.download_to_slice(0, &mut graph_bytes, &mut staging, &mut transfer_stream)?;
    graph_input.download_to_slice(
        0,
        &mut graph_input_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_bytes, eager_bytes,
        "fixed graph BF16 SiLU output must match eager SiLU bit-for-bit"
    );
    assert_eq!(
        graph_input_after, host_input,
        "fixed graph replay must not mutate its retained input allocation"
    );

    graph_output.close()?;
    graph_input.close()?;
    eager_output.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-8-owned-bf16-silu-fixed-address replays={REPLAYS} elements={ELEMENT_COUNT} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_silu_graph_preflight_and_abort_recover_every_resource() -> Result<(), Box<dyn Error>>
{
    const ELEMENT_COUNT: u64 = 128;
    const BYTE_LEN: u64 = ELEMENT_COUNT * 2;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let input = context.allocate_device_buffer(BYTE_LEN)?;
    let output = context.allocate_device_buffer(BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let error = match stream.begin_owned_graph_silu_bf16_capture(
        input,
        output,
        0,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("zero-element owned BF16 SiLU graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("Rust BF16 SiLU preflight must return all untouched resources");
    let (stream, input, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = stream
        .begin_owned_graph_silu_bf16_capture(
            input,
            output,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    let (stream, input, output) = resources.into_parts();
    let error = match stream.begin_owned_graph_silu_bf16_capture(
        input,
        output,
        ELEMENT_COUNT + 1,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("oversized owned BF16 SiLU graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("oversized BF16 SiLU preflight must preserve the three resources")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-8-owned-bf16-silu-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_gated_multiply_graph_replays_fixed_inputs_byte_exact_against_eager()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 4_096;
    const REPLAYS: usize = 32;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<u16>())?)
        .ok_or("BF16 gated-multiply graph byte length overflow")?;
    let host_activated_gate =
        graph_gated_multiply_bf16_fixture_bytes(usize::try_from(ELEMENT_COUNT)?, 0);
    let host_up = graph_gated_multiply_bf16_fixture_bytes(usize::try_from(ELEMENT_COUNT)?, 1);
    assert_eq!(u64::try_from(host_activated_gate.len())?, byte_len);
    assert_eq!(u64::try_from(host_up.len())?, byte_len);
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;

    let mut eager_activated_gate = context.allocate_device_buffer(byte_len)?;
    eager_activated_gate.upload_from_slice(
        0,
        &host_activated_gate,
        &mut staging,
        &mut eager_stream,
    )?;
    let mut eager_up = context.allocate_device_buffer(byte_len)?;
    eager_up.upload_from_slice(0, &host_up, &mut staging, &mut eager_stream)?;
    let graph_activated_gate = {
        let mut buffer = context.allocate_device_buffer(byte_len)?;
        buffer.upload_from_slice(0, &host_activated_gate, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_up = {
        let mut buffer = context.allocate_device_buffer(byte_len)?;
        buffer.upload_from_slice(0, &host_up, &mut staging, &mut eager_stream)?;
        buffer
    };
    let mut eager_output = context.allocate_device_buffer(byte_len)?;
    let graph_output = context.allocate_device_buffer(byte_len)?;

    {
        let mut eager = GatedMultiplyParams {
            activated_gate: CudaBufferSpan::new(
                &eager_activated_gate,
                CudaDType::BF16,
                0,
                byte_len,
            )?,
            up: CudaBufferSpan::new(&eager_up, CudaDType::BF16, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, byte_len)?,
            element_count: ELEMENT_COUNT,
        };
        gated_multiply(&mut eager, &mut eager_stream)?;
    }

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_gated_multiply_bf16_capture(
        graph_activated_gate,
        graph_up,
        graph_output,
        ELEMENT_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_gated_multiply_bf16()?;
    // The local one-node rejection must leave the first node/capture usable.
    assert_invalid_state(
        capture.enqueue_gated_multiply_bf16(),
        "second fixed BF16 gated-multiply graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // This is a lifecycle assertion, not a CUDA Graph performance claim.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (capture_stream, mut graph_activated_gate, mut graph_up, mut graph_output) =
        resources.into_parts();
    let mut eager_bytes = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_bytes = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_activated_gate_after = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_up_after = vec![0_u8; usize::try_from(byte_len)?];
    eager_output.download_to_slice(0, &mut eager_bytes, &mut staging, &mut transfer_stream)?;
    graph_output.download_to_slice(0, &mut graph_bytes, &mut staging, &mut transfer_stream)?;
    graph_activated_gate.download_to_slice(
        0,
        &mut graph_activated_gate_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_up.download_to_slice(0, &mut graph_up_after, &mut staging, &mut transfer_stream)?;
    assert_eq!(
        graph_bytes, eager_bytes,
        "fixed graph BF16 gated-multiply output must match eager output bit-for-bit"
    );
    assert_eq!(
        graph_activated_gate_after, host_activated_gate,
        "fixed graph replay must not mutate its retained activated-gate allocation"
    );
    assert_eq!(
        graph_up_after, host_up,
        "fixed graph replay must not mutate its retained up allocation"
    );

    graph_output.close()?;
    graph_up.close()?;
    graph_activated_gate.close()?;
    eager_output.close()?;
    eager_up.close()?;
    eager_activated_gate.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-10-owned-bf16-gated-multiply-fixed-address replays={REPLAYS} elements={ELEMENT_COUNT} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_gated_multiply_graph_preflight_and_abort_recover_every_resource()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 128;
    const BYTE_LEN: u64 = ELEMENT_COUNT * 2;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let activated_gate = context.allocate_device_buffer(BYTE_LEN)?;
    let up = context.allocate_device_buffer(BYTE_LEN)?;
    let output = context.allocate_device_buffer(BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let error = match stream.begin_owned_graph_gated_multiply_bf16_capture(
        activated_gate,
        up,
        output,
        0,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => {
            panic!("zero-element owned BF16 gated-multiply graph preflight unexpectedly succeeded")
        }
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("Rust BF16 gated-multiply preflight must return all untouched resources");
    let (stream, activated_gate, up, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = stream
        .begin_owned_graph_gated_multiply_bf16_capture(
            activated_gate,
            up,
            output,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    let (stream, activated_gate, up, output) = resources.into_parts();
    let error = match stream.begin_owned_graph_gated_multiply_bf16_capture(
        activated_gate,
        up,
        output,
        ELEMENT_COUNT + 1,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => {
            panic!("oversized owned BF16 gated-multiply graph preflight unexpectedly succeeded")
        }
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("oversized BF16 gated-multiply preflight must preserve all resources")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-10-owned-bf16-gated-multiply-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_residual_add_graph_replays_fixed_inputs_byte_exact_against_eager()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 4_096;
    const REPLAYS: usize = 32;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<u16>())?)
        .ok_or("BF16 residual-add graph byte length overflow")?;
    let host_left = graph_residual_add_bf16_fixture_bytes(usize::try_from(ELEMENT_COUNT)?, 0);
    let host_right = graph_residual_add_bf16_fixture_bytes(usize::try_from(ELEMENT_COUNT)?, 1);
    assert_eq!(u64::try_from(host_left.len())?, byte_len);
    assert_eq!(u64::try_from(host_right.len())?, byte_len);
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;

    let mut eager_left = context.allocate_device_buffer(byte_len)?;
    eager_left.upload_from_slice(0, &host_left, &mut staging, &mut eager_stream)?;
    let mut eager_right = context.allocate_device_buffer(byte_len)?;
    eager_right.upload_from_slice(0, &host_right, &mut staging, &mut eager_stream)?;
    let graph_left = {
        let mut buffer = context.allocate_device_buffer(byte_len)?;
        buffer.upload_from_slice(0, &host_left, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_right = {
        let mut buffer = context.allocate_device_buffer(byte_len)?;
        buffer.upload_from_slice(0, &host_right, &mut staging, &mut eager_stream)?;
        buffer
    };
    let mut eager_output = context.allocate_device_buffer(byte_len)?;
    let graph_output = context.allocate_device_buffer(byte_len)?;

    {
        let mut eager = ResidualAddParams {
            left: CudaBufferSpan::new(&eager_left, CudaDType::BF16, 0, byte_len)?,
            right: CudaBufferSpan::new(&eager_right, CudaDType::BF16, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, byte_len)?,
            element_count: ELEMENT_COUNT,
        };
        residual_add(&mut eager, &mut eager_stream)?;
    }

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_residual_add_bf16_capture(
        graph_left,
        graph_right,
        graph_output,
        ELEMENT_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_residual_add_bf16()?;
    // The local one-node rejection must leave the first node/capture usable.
    assert_invalid_state(
        capture.enqueue_residual_add_bf16(),
        "second fixed BF16 residual-add graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // This is a lifecycle assertion, not a CUDA Graph performance claim.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (capture_stream, mut graph_left, mut graph_right, mut graph_output) =
        resources.into_parts();
    let mut eager_bytes = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_bytes = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_left_after = vec![0_u8; usize::try_from(byte_len)?];
    let mut graph_right_after = vec![0_u8; usize::try_from(byte_len)?];
    eager_output.download_to_slice(0, &mut eager_bytes, &mut staging, &mut transfer_stream)?;
    graph_output.download_to_slice(0, &mut graph_bytes, &mut staging, &mut transfer_stream)?;
    graph_left.download_to_slice(0, &mut graph_left_after, &mut staging, &mut transfer_stream)?;
    graph_right.download_to_slice(
        0,
        &mut graph_right_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_bytes, eager_bytes,
        "fixed graph BF16 residual-add output must match eager output bit-for-bit"
    );
    assert_eq!(
        graph_left_after, host_left,
        "fixed graph replay must not mutate its retained left allocation"
    );
    assert_eq!(
        graph_right_after, host_right,
        "fixed graph replay must not mutate its retained right allocation"
    );

    graph_output.close()?;
    graph_right.close()?;
    graph_left.close()?;
    eager_output.close()?;
    eager_right.close()?;
    eager_left.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-11-owned-bf16-residual-add-fixed-address replays={REPLAYS} elements={ELEMENT_COUNT} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_residual_add_graph_preflight_and_abort_recover_every_resource()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 128;
    const BYTE_LEN: u64 = ELEMENT_COUNT * 2;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let left = context.allocate_device_buffer(BYTE_LEN)?;
    let right = context.allocate_device_buffer(BYTE_LEN)?;
    let output = context.allocate_device_buffer(BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let error = match stream.begin_owned_graph_residual_add_bf16_capture(
        left,
        right,
        output,
        0,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => {
            panic!("zero-element owned BF16 residual-add graph preflight unexpectedly succeeded")
        }
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("Rust BF16 residual-add preflight must return all untouched resources");
    let (stream, left, right, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = stream
        .begin_owned_graph_residual_add_bf16_capture(
            left,
            right,
            output,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    let (stream, left, right, output) = resources.into_parts();
    let error = match stream.begin_owned_graph_residual_add_bf16_capture(
        left,
        right,
        output,
        ELEMENT_COUNT + 1,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("oversized owned BF16 residual-add graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("oversized BF16 residual-add preflight must preserve all resources")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-11-owned-bf16-residual-add-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_canonical_bf16_rms_norm_graph_replays_fixed_inputs_byte_exact_against_eager()
-> Result<(), Box<dyn Error>> {
    // This deliberately uses a generic, non-model geometry. It must remain
    // byte-exact with the canonical eager `rms_norm` primitive, rather than
    // with a Hugging Face SmolLM2, Fixed37, or residual-fused variant.
    const ROW_COUNT: u64 = 17;
    const HIDDEN_SIZE: u64 = 769;
    const REPLAYS: usize = 32;
    const EPSILON: f32 = 1.0e-5;
    const ELEMENT_COUNT: u64 = ROW_COUNT * HIDDEN_SIZE;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let tensor_byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<u16>())?)
        .ok_or("canonical BF16 RMSNorm graph tensor byte length overflow")?;
    let weight_byte_len = HIDDEN_SIZE
        .checked_mul(u64::try_from(std::mem::size_of::<u16>())?)
        .ok_or("canonical BF16 RMSNorm graph weight byte length overflow")?;
    let host_input =
        graph_canonical_rms_norm_bf16_fixture_bytes(usize::try_from(ELEMENT_COUNT)?, 0);
    let host_weight = graph_canonical_rms_norm_bf16_fixture_bytes(usize::try_from(HIDDEN_SIZE)?, 1);
    assert_eq!(u64::try_from(host_input.len())?, tensor_byte_len);
    assert_eq!(u64::try_from(host_weight.len())?, weight_byte_len);
    let mut staging = context.allocate_pinned_host_buffer(tensor_byte_len)?;

    let mut eager_input = context.allocate_device_buffer(tensor_byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    let mut eager_weight = context.allocate_device_buffer(weight_byte_len)?;
    eager_weight.upload_from_slice(0, &host_weight, &mut staging, &mut eager_stream)?;
    let graph_input = {
        let mut buffer = context.allocate_device_buffer(tensor_byte_len)?;
        buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_weight = {
        let mut buffer = context.allocate_device_buffer(weight_byte_len)?;
        buffer.upload_from_slice(0, &host_weight, &mut staging, &mut eager_stream)?;
        buffer
    };
    let mut eager_output = context.allocate_device_buffer(tensor_byte_len)?;
    let graph_output = context.allocate_device_buffer(tensor_byte_len)?;

    {
        let mut eager = RmsNormParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, tensor_byte_len)?,
            weight: CudaBufferSpan::new(&eager_weight, CudaDType::BF16, 0, weight_byte_len)?,
            output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, tensor_byte_len)?,
            row_count: ROW_COUNT,
            hidden_size: HIDDEN_SIZE,
            epsilon: EPSILON,
        };
        rms_norm(&mut eager, &mut eager_stream)?;
    }

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_canonical_rms_norm_bf16_capture(
        graph_input,
        graph_weight,
        graph_output,
        ROW_COUNT,
        HIDDEN_SIZE,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_canonical_rms_norm_bf16()?;
    // The local one-node rejection must leave the first node/capture usable.
    assert_invalid_state(
        capture.enqueue_canonical_rms_norm_bf16(),
        "second fixed canonical BF16 RMSNorm graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // This guards the owned lifecycle's resource accounting only; it is not a
    // CUDA Graph performance claim.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (capture_stream, mut graph_input, mut graph_weight, mut graph_output) =
        resources.into_parts();
    let mut eager_bytes = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_bytes = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_input_after = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_weight_after = vec![0_u8; usize::try_from(weight_byte_len)?];
    eager_output.download_to_slice(0, &mut eager_bytes, &mut staging, &mut transfer_stream)?;
    graph_output.download_to_slice(0, &mut graph_bytes, &mut staging, &mut transfer_stream)?;
    graph_input.download_to_slice(
        0,
        &mut graph_input_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_weight.download_to_slice(
        0,
        &mut graph_weight_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_bytes, eager_bytes,
        "fixed graph canonical BF16 RMSNorm output must match eager output bit-for-bit"
    );
    assert_eq!(
        graph_input_after, host_input,
        "fixed graph replay must not mutate its retained canonical RMSNorm input"
    );
    assert_eq!(
        graph_weight_after, host_weight,
        "fixed graph replay must not mutate its retained canonical RMSNorm weight"
    );

    graph_output.close()?;
    graph_weight.close()?;
    graph_input.close()?;
    eager_output.close()?;
    eager_weight.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-12-owned-canonical-bf16-rms-norm-fixed-address replays={REPLAYS} rows={ROW_COUNT} hidden={HIDDEN_SIZE} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_canonical_bf16_rms_norm_graph_preflight_and_abort_recover_every_resource()
-> Result<(), Box<dyn Error>> {
    const ROW_COUNT: u64 = 4;
    const HIDDEN_SIZE: u64 = 32;
    const EPSILON: f32 = 1.0e-5;
    const TENSOR_BYTE_LEN: u64 = ROW_COUNT * HIDDEN_SIZE * 2;
    const WEIGHT_BYTE_LEN: u64 = HIDDEN_SIZE * 2;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let input = context.allocate_device_buffer(TENSOR_BYTE_LEN)?;
    let weight = context.allocate_device_buffer(WEIGHT_BYTE_LEN)?;
    let output = context.allocate_device_buffer(TENSOR_BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let error = match stream.begin_owned_graph_canonical_rms_norm_bf16_capture(
        input,
        weight,
        output,
        0,
        HIDDEN_SIZE,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => {
            panic!("zero-row owned canonical BF16 RMSNorm graph preflight unexpectedly succeeded")
        }
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("zero-row canonical RMSNorm preflight must return all untouched resources");
    let (stream, input, weight, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let error = match stream.begin_owned_graph_canonical_rms_norm_bf16_capture(
        input,
        weight,
        output,
        ROW_COUNT,
        HIDDEN_SIZE,
        0.0,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!(
            "zero-epsilon owned canonical BF16 RMSNorm graph preflight unexpectedly succeeded"
        ),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::InvalidArgument);
    let resources = error
        .into_resources()
        .expect("invalid-epsilon canonical RMSNorm preflight must return all untouched resources");
    let (stream, input, weight, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = stream
        .begin_owned_graph_canonical_rms_norm_bf16_capture(
            input,
            weight,
            output,
            ROW_COUNT,
            HIDDEN_SIZE,
            EPSILON,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    let (stream, input, weight, output) = resources.into_parts();
    let error = match stream.begin_owned_graph_canonical_rms_norm_bf16_capture(
        input,
        weight,
        output,
        ROW_COUNT + 1,
        HIDDEN_SIZE,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => {
            panic!("oversized owned canonical BF16 RMSNorm graph preflight unexpectedly succeeded")
        }
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("oversized canonical RMSNorm preflight must preserve all four resources")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-12-owned-canonical-bf16-rms-norm-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_argmax_graph_replays_edge_fixture_byte_exact_against_eager()
-> Result<(), Box<dyn Error>> {
    const ROW_COUNT: u64 = 7;
    const VOCABULARY_SIZE: u64 = 257;
    const RESULT_U32_WORDS_PER_ROW: u64 = 2;
    const REPLAYS: usize = 64;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let bf16_byte_width = u64::try_from(std::mem::size_of::<u16>())?;
    let u32_byte_width = u64::try_from(std::mem::size_of::<u32>())?;
    let logits_byte_len = ROW_COUNT
        .checked_mul(VOCABULARY_SIZE)
        .and_then(|element_count| element_count.checked_mul(bf16_byte_width))
        .ok_or("BF16 argmax graph logits byte length overflow")?;
    let results_byte_len = ROW_COUNT
        .checked_mul(RESULT_U32_WORDS_PER_ROW)
        .and_then(|word_count| word_count.checked_mul(u32_byte_width))
        .ok_or("BF16 argmax graph result byte length overflow")?;
    let host_logits = graph_bf16_argmax_edge_fixture_bytes();
    let expected_result_bytes = graph_bf16_argmax_edge_expected_result_bytes();
    assert_eq!(u64::try_from(host_logits.len())?, logits_byte_len);
    assert_eq!(
        u64::try_from(expected_result_bytes.len())?,
        results_byte_len
    );
    let sentinel_results = vec![0xa5; usize::try_from(results_byte_len)?];
    let mut staging = context.allocate_pinned_host_buffer(logits_byte_len)?;

    let mut eager_logits = context.allocate_device_buffer(logits_byte_len)?;
    eager_logits.upload_from_slice(0, &host_logits, &mut staging, &mut eager_stream)?;
    let mut eager_results = context.allocate_device_buffer(results_byte_len)?;
    eager_results.upload_from_slice(0, &sentinel_results, &mut staging, &mut eager_stream)?;
    let graph_logits = {
        let mut buffer = context.allocate_device_buffer(logits_byte_len)?;
        buffer.upload_from_slice(0, &host_logits, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_results = {
        let mut buffer = context.allocate_device_buffer(results_byte_len)?;
        buffer.upload_from_slice(0, &sentinel_results, &mut staging, &mut eager_stream)?;
        buffer
    };

    {
        let mut eager = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&eager_logits, CudaDType::BF16, 0, logits_byte_len)?,
            results: CudaBufferSpanMut::new(
                &mut eager_results,
                CudaDType::U32,
                0,
                results_byte_len,
            )?,
            row_count: ROW_COUNT,
            vocabulary_size: VOCABULARY_SIZE,
        };
        deterministic_bf16_argmax(&mut eager, &mut eager_stream)?;
    }

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_bf16_argmax_capture(
        graph_logits,
        graph_results,
        ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_bf16_argmax()?;
    // The local one-node rejection must leave the admitted node/capture usable.
    assert_invalid_state(
        capture.enqueue_bf16_argmax(),
        "second fixed BF16 argmax graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // This asserts owned lifecycle accounting, not an inference-performance
    // claim about CUDA Graph driver internals.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (capture_stream, mut graph_logits, mut graph_results) = resources.into_parts();
    let mut eager_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    let mut graph_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    let mut graph_logits_after = vec![0_u8; usize::try_from(logits_byte_len)?];
    eager_results.download_to_slice(
        0,
        &mut eager_result_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_results.download_to_slice(
        0,
        &mut graph_result_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_logits.download_to_slice(
        0,
        &mut graph_logits_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        eager_result_bytes, expected_result_bytes,
        "eager deterministic BF16 argmax must retain the 7x257 token/status contract"
    );
    assert_eq!(
        graph_result_bytes, eager_result_bytes,
        "fixed graph BF16 argmax U32 token/status bytes must match eager output"
    );
    assert_eq!(
        graph_logits_after, host_logits,
        "fixed graph replay must not mutate its retained BF16 logits allocation"
    );

    graph_results.close()?;
    graph_logits.close()?;
    eager_results.close()?;
    eager_logits.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-13-owned-bf16-argmax-fixed-address replays={REPLAYS} rows={ROW_COUNT} vocabulary={VOCABULARY_SIZE} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_argmax_graph_preflight_and_abort_recover_every_resource() -> Result<(), Box<dyn Error>>
{
    const ROW_COUNT: u64 = 7;
    const VOCABULARY_SIZE: u64 = 257;
    const LOGITS_BYTE_LEN: u64 = ROW_COUNT * VOCABULARY_SIZE * 2;
    const RESULTS_BYTE_LEN: u64 = ROW_COUNT * 2 * 4;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let logits = context.allocate_device_buffer(LOGITS_BYTE_LEN)?;
    let results = context.allocate_device_buffer(RESULTS_BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let error = match stream.begin_owned_graph_bf16_argmax_capture(
        logits,
        results,
        0,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("zero-row owned BF16 argmax graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("zero-row BF16 argmax preflight must return all untouched resources");
    let (stream, logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let error = match stream.begin_owned_graph_bf16_argmax_capture(
        logits,
        results,
        ROW_COUNT,
        0,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("zero-vocabulary owned BF16 argmax graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::InvalidArgument);
    let resources = error
        .into_resources()
        .expect("zero-vocabulary BF16 argmax preflight must return all untouched resources");
    let (stream, logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let error = match stream.begin_owned_graph_bf16_argmax_capture(
        logits,
        results,
        ROW_COUNT,
        u64::from(u32::MAX) + 1,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("U32-overflow owned BF16 argmax graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("U32-overflow BF16 argmax preflight must return all untouched resources");
    let (stream, logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = stream
        .begin_owned_graph_bf16_argmax_capture(
            logits,
            results,
            ROW_COUNT,
            VOCABULARY_SIZE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    let (stream, logits, results) = resources.into_parts();
    let error = match stream.begin_owned_graph_bf16_argmax_capture(
        logits,
        results,
        ROW_COUNT + 1,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("oversized owned BF16 argmax graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("oversized BF16 argmax preflight must preserve all three resources")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-13-owned-bf16-argmax-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_row_gather_graph_replays_unique_permutation_byte_exact_against_eager()
-> Result<(), Box<dyn Error>> {
    const INPUT_ROW_COUNT: u64 = 13;
    const COLUMN_COUNT: u64 = 257;
    const REPLAYS: usize = 64;

    let eager_row_indices = vec![12_u32, 0, 7, 3, 10, 1, 5];
    let graph_row_indices_host = eager_row_indices.clone();
    let output_row_count = u64::try_from(graph_row_indices_host.len())?;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let bf16_byte_width = u64::try_from(std::mem::size_of::<u16>())?;
    let u32_byte_width = u64::try_from(std::mem::size_of::<u32>())?;
    let input_byte_len = INPUT_ROW_COUNT
        .checked_mul(COLUMN_COUNT)
        .and_then(|element_count| element_count.checked_mul(bf16_byte_width))
        .ok_or("BF16 row-gather graph input byte length overflow")?;
    let row_indices_byte_len = output_row_count
        .checked_mul(u32_byte_width)
        .ok_or("BF16 row-gather graph row-index byte length overflow")?;
    let output_byte_len = output_row_count
        .checked_mul(COLUMN_COUNT)
        .and_then(|element_count| element_count.checked_mul(bf16_byte_width))
        .ok_or("BF16 row-gather graph output byte length overflow")?;
    let host_input = graph_bf16_row_gather_fixture_bytes(
        usize::try_from(INPUT_ROW_COUNT)?,
        usize::try_from(COLUMN_COUNT)?,
    );
    let host_row_indices = u32_words_to_ne_bytes(&eager_row_indices);
    assert_eq!(u64::try_from(host_input.len())?, input_byte_len);
    assert_eq!(u64::try_from(host_row_indices.len())?, row_indices_byte_len);
    let sentinel_output = vec![0xa5; usize::try_from(output_byte_len)?];
    let mut staging = context.allocate_pinned_host_buffer(input_byte_len)?;

    let mut eager_input = context.allocate_device_buffer(input_byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    let mut eager_row_indices_buffer = context.allocate_device_buffer(row_indices_byte_len)?;
    eager_row_indices_buffer.upload_from_slice(
        0,
        &host_row_indices,
        &mut staging,
        &mut eager_stream,
    )?;
    let mut eager_output = context.allocate_device_buffer(output_byte_len)?;
    eager_output.upload_from_slice(0, &sentinel_output, &mut staging, &mut eager_stream)?;

    let graph_input = {
        let mut buffer = context.allocate_device_buffer(input_byte_len)?;
        buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_row_indices = {
        let mut buffer = context.allocate_device_buffer(row_indices_byte_len)?;
        buffer.upload_from_slice(0, &host_row_indices, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_output = {
        let mut buffer = context.allocate_device_buffer(output_byte_len)?;
        buffer.upload_from_slice(0, &sentinel_output, &mut staging, &mut eager_stream)?;
        buffer
    };

    {
        let mut eager = RowGatherParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, input_byte_len)?,
            row_indices: CudaBufferSpan::new(
                &eager_row_indices_buffer,
                CudaDType::U32,
                0,
                row_indices_byte_len,
            )?,
            row_indices_host: &eager_row_indices,
            output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, output_byte_len)?,
            input_row_count: INPUT_ROW_COUNT,
            column_count: COLUMN_COUNT,
        };
        row_gather(&mut eager, &mut eager_stream)?;
    }

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_bf16_row_gather_capture(
        graph_input,
        graph_row_indices,
        graph_output,
        &graph_row_indices_host,
        INPUT_ROW_COUNT,
        COLUMN_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    // The graph owner validates the host mirror at admission and must not
    // retain it through capture, graph, executable, or replay ownership.
    drop(graph_row_indices_host);
    capture.enqueue_bf16_row_gather()?;
    assert_invalid_state(
        capture.enqueue_bf16_row_gather(),
        "second fixed BF16 row-gather graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // This asserts owned lifecycle accounting, not a CUDA Graph performance
    // claim about driver internals.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (capture_stream, mut graph_input, mut graph_row_indices, mut graph_output) =
        resources.into_parts();
    let mut eager_output_bytes = vec![0_u8; usize::try_from(output_byte_len)?];
    let mut graph_output_bytes = vec![0_u8; usize::try_from(output_byte_len)?];
    let mut graph_input_after = vec![0_u8; usize::try_from(input_byte_len)?];
    let mut graph_row_indices_after = vec![0_u8; usize::try_from(row_indices_byte_len)?];
    eager_output.download_to_slice(
        0,
        &mut eager_output_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_output.download_to_slice(
        0,
        &mut graph_output_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_input.download_to_slice(
        0,
        &mut graph_input_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_row_indices.download_to_slice(
        0,
        &mut graph_row_indices_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_output_bytes, eager_output_bytes,
        "fixed graph BF16 row-gather output must match eager output byte-for-byte"
    );
    assert_eq!(
        graph_input_after, host_input,
        "fixed graph replay must not mutate its retained BF16 input allocation"
    );
    assert_eq!(
        graph_row_indices_after, host_row_indices,
        "fixed graph replay must not mutate its retained U32 row-index allocation"
    );

    graph_output.close()?;
    graph_row_indices.close()?;
    graph_input.close()?;
    eager_output.close()?;
    eager_row_indices_buffer.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-14-owned-bf16-row-gather-fixed-address replays={REPLAYS} input_rows={INPUT_ROW_COUNT} output_rows={output_row_count} columns={COLUMN_COUNT} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_row_gather_graph_preflight_and_abort_recover_every_resource()
-> Result<(), Box<dyn Error>> {
    const INPUT_ROW_COUNT: u64 = 8;
    const COLUMN_COUNT: u64 = 32;
    const OUTPUT_ROW_COUNT: u64 = 4;
    const INPUT_BYTE_LEN: u64 = INPUT_ROW_COUNT * COLUMN_COUNT * 2;
    const ROW_INDICES_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * 4;
    const OUTPUT_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * COLUMN_COUNT * 2;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let input = context.allocate_device_buffer(INPUT_BYTE_LEN)?;
    let row_indices = context.allocate_device_buffer(ROW_INDICES_BYTE_LEN)?;
    let output = context.allocate_device_buffer(OUTPUT_BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let empty_host_mirror: [u32; 0] = [];
    let error = match stream.begin_owned_graph_bf16_row_gather_capture(
        input,
        row_indices,
        output,
        &empty_host_mirror,
        INPUT_ROW_COUNT,
        COLUMN_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("empty row-index mirror graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("empty row-index mirror preflight must return all untouched resources");
    let (stream, input, row_indices, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let valid_host_mirror = [0_u32, 1, 2, 3];
    let error = match stream.begin_owned_graph_bf16_row_gather_capture(
        input,
        row_indices,
        output,
        &valid_host_mirror,
        0,
        COLUMN_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("zero-input-row BF16 row-gather graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("zero-input-row preflight must return all untouched resources");
    let (stream, input, row_indices, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let error = match stream.begin_owned_graph_bf16_row_gather_capture(
        input,
        row_indices,
        output,
        &valid_host_mirror,
        INPUT_ROW_COUNT,
        0,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("zero-column BF16 row-gather graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("zero-column preflight must return all untouched resources");
    let (stream, input, row_indices, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let duplicate_host_mirror = [0_u32, 1, 1, 3];
    let error = match stream.begin_owned_graph_bf16_row_gather_capture(
        input,
        row_indices,
        output,
        &duplicate_host_mirror,
        INPUT_ROW_COUNT,
        COLUMN_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("duplicate row-index mirror graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::InvalidArgument);
    let resources = error
        .into_resources()
        .expect("duplicate row-index mirror preflight must return all untouched resources");
    let (stream, input, row_indices, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let out_of_range_host_mirror = [0_u32, 1, 2, INPUT_ROW_COUNT as u32];
    let error = match stream.begin_owned_graph_bf16_row_gather_capture(
        input,
        row_indices,
        output,
        &out_of_range_host_mirror,
        INPUT_ROW_COUNT,
        COLUMN_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("out-of-range row-index mirror graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("out-of-range row-index mirror preflight must return all untouched resources");
    let (stream, input, row_indices, output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = stream
        .begin_owned_graph_bf16_row_gather_capture(
            input,
            row_indices,
            output,
            &valid_host_mirror,
            INPUT_ROW_COUNT,
            COLUMN_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    let (stream, input, row_indices, output) = resources.into_parts();
    let oversized_host_mirror = [0_u32, 1, 2, 3, 4];
    let error = match stream.begin_owned_graph_bf16_row_gather_capture(
        input,
        row_indices,
        output,
        &oversized_host_mirror,
        INPUT_ROW_COUNT,
        COLUMN_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("oversized BF16 row-gather graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("oversized BF16 row-gather preflight must preserve all four resources")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-14-owned-bf16-row-gather-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_row_gather_argmax_graph_replays_unique_permutation_byte_exact_against_eager()
-> Result<(), Box<dyn Error>> {
    const INPUT_ROW_COUNT: u64 = 13;
    const OUTPUT_ROW_COUNT: u64 = 7;
    const VOCABULARY_SIZE: u64 = 257;
    const REPLAYS: usize = 64;

    let eager_row_indices = vec![12_u32, 0, 7, 3, 10, 1, 5];
    let graph_row_indices_host = eager_row_indices.clone();
    assert_eq!(
        u64::try_from(graph_row_indices_host.len())?,
        OUTPUT_ROW_COUNT
    );

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let bf16_byte_width = u64::try_from(std::mem::size_of::<u16>())?;
    let u32_byte_width = u64::try_from(std::mem::size_of::<u32>())?;
    let input_byte_len = INPUT_ROW_COUNT
        .checked_mul(VOCABULARY_SIZE)
        .and_then(|element_count| element_count.checked_mul(bf16_byte_width))
        .ok_or("BF16 row-gather -> argmax graph input byte length overflow")?;
    let row_indices_byte_len = OUTPUT_ROW_COUNT
        .checked_mul(u32_byte_width)
        .ok_or("BF16 row-gather -> argmax graph row-index byte length overflow")?;
    let gathered_byte_len = OUTPUT_ROW_COUNT
        .checked_mul(VOCABULARY_SIZE)
        .and_then(|element_count| element_count.checked_mul(bf16_byte_width))
        .ok_or("BF16 row-gather -> argmax graph gathered byte length overflow")?;
    let results_byte_len = OUTPUT_ROW_COUNT
        .checked_mul(2)
        .and_then(|word_count| word_count.checked_mul(u32_byte_width))
        .ok_or("BF16 row-gather -> argmax graph result byte length overflow")?;

    // Reuse the C05-13 semantic edge corpus, but scatter its seven logits
    // rows into a larger input so graph parity proves the gather-to-argmax
    // dependency rather than merely a direct argmax replay.
    let selected_logits = graph_bf16_argmax_edge_fixture_bytes();
    assert_eq!(u64::try_from(selected_logits.len())?, gathered_byte_len);
    let row_byte_len = usize::try_from(VOCABULARY_SIZE * bf16_byte_width)?;
    let mut host_input = vec![0xc1_u8; usize::try_from(input_byte_len)?];
    for (output_row, &input_row) in eager_row_indices.iter().enumerate() {
        let source_start = output_row * row_byte_len;
        let source_end = source_start + row_byte_len;
        let destination_start = usize::try_from(input_row)? * row_byte_len;
        let destination_end = destination_start + row_byte_len;
        host_input[destination_start..destination_end]
            .copy_from_slice(&selected_logits[source_start..source_end]);
    }
    let host_row_indices = u32_words_to_ne_bytes(&eager_row_indices);
    assert_eq!(u64::try_from(host_input.len())?, input_byte_len);
    assert_eq!(u64::try_from(host_row_indices.len())?, row_indices_byte_len);
    let gathered_sentinel = vec![0xa5; usize::try_from(gathered_byte_len)?];
    let results_sentinel = vec![0xa5; usize::try_from(results_byte_len)?];
    let mut staging = context.allocate_pinned_host_buffer(input_byte_len)?;

    let mut eager_input = context.allocate_device_buffer(input_byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    let mut eager_row_indices_buffer = context.allocate_device_buffer(row_indices_byte_len)?;
    eager_row_indices_buffer.upload_from_slice(
        0,
        &host_row_indices,
        &mut staging,
        &mut eager_stream,
    )?;
    let mut eager_gathered = context.allocate_device_buffer(gathered_byte_len)?;
    eager_gathered.upload_from_slice(0, &gathered_sentinel, &mut staging, &mut eager_stream)?;
    let mut eager_results = context.allocate_device_buffer(results_byte_len)?;
    eager_results.upload_from_slice(0, &results_sentinel, &mut staging, &mut eager_stream)?;

    let graph_input = {
        let mut buffer = context.allocate_device_buffer(input_byte_len)?;
        buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_row_indices = {
        let mut buffer = context.allocate_device_buffer(row_indices_byte_len)?;
        buffer.upload_from_slice(0, &host_row_indices, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_gathered = {
        let mut buffer = context.allocate_device_buffer(gathered_byte_len)?;
        buffer.upload_from_slice(0, &gathered_sentinel, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_results = {
        let mut buffer = context.allocate_device_buffer(results_byte_len)?;
        buffer.upload_from_slice(0, &results_sentinel, &mut staging, &mut eager_stream)?;
        buffer
    };

    {
        let mut eager = RowGatherParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, input_byte_len)?,
            row_indices: CudaBufferSpan::new(
                &eager_row_indices_buffer,
                CudaDType::U32,
                0,
                row_indices_byte_len,
            )?,
            row_indices_host: &eager_row_indices,
            output: CudaBufferSpanMut::new(
                &mut eager_gathered,
                CudaDType::BF16,
                0,
                gathered_byte_len,
            )?,
            input_row_count: INPUT_ROW_COUNT,
            column_count: VOCABULARY_SIZE,
        };
        row_gather(&mut eager, &mut eager_stream)?;
    }
    {
        let mut eager = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&eager_gathered, CudaDType::BF16, 0, gathered_byte_len)?,
            results: CudaBufferSpanMut::new(
                &mut eager_results,
                CudaDType::U32,
                0,
                results_byte_len,
            )?,
            row_count: OUTPUT_ROW_COUNT,
            vocabulary_size: VOCABULARY_SIZE,
        };
        deterministic_bf16_argmax(&mut eager, &mut eager_stream)?;
    }

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        graph_input,
        graph_row_indices,
        graph_gathered,
        graph_results,
        &graph_row_indices_host,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    // The owner validates this temporary mirror at admission and must not
    // retain it through capture, graph, executable, or replay ownership.
    drop(graph_row_indices_host);
    capture.enqueue_bf16_row_gather_argmax()?;
    assert_invalid_state(
        capture.enqueue_bf16_row_gather_argmax(),
        "second fixed BF16 row-gather -> argmax graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // This asserts owned lifecycle accounting and exact primitive parity, not
    // an inference-performance claim about CUDA Graph driver internals.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (
        capture_stream,
        mut graph_input,
        mut graph_row_indices,
        mut graph_gathered,
        mut graph_results,
    ) = resources.into_parts();
    let mut eager_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
    let mut graph_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
    let mut eager_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    let mut graph_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    let mut graph_input_after = vec![0_u8; usize::try_from(input_byte_len)?];
    let mut graph_row_indices_after = vec![0_u8; usize::try_from(row_indices_byte_len)?];
    eager_gathered.download_to_slice(
        0,
        &mut eager_gathered_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_gathered.download_to_slice(
        0,
        &mut graph_gathered_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    eager_results.download_to_slice(
        0,
        &mut eager_result_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_results.download_to_slice(
        0,
        &mut graph_result_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_input.download_to_slice(
        0,
        &mut graph_input_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_row_indices.download_to_slice(
        0,
        &mut graph_row_indices_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_gathered_bytes, eager_gathered_bytes,
        "fixed graph gathered BF16 bytes must match eager row-gather output"
    );
    assert_eq!(
        eager_result_bytes,
        graph_bf16_argmax_edge_expected_result_bytes(),
        "eager gathered edge corpus must retain the deterministic argmax token/status contract"
    );
    assert_eq!(
        graph_result_bytes, eager_result_bytes,
        "fixed graph deterministic argmax result bytes must match the eager two-kernel chain"
    );
    assert_eq!(
        graph_input_after, host_input,
        "fixed graph replay must not mutate its retained BF16 input allocation"
    );
    assert_eq!(
        graph_row_indices_after, host_row_indices,
        "fixed graph replay must not mutate its retained U32 row-index allocation"
    );

    graph_results.close()?;
    graph_gathered.close()?;
    graph_row_indices.close()?;
    graph_input.close()?;
    eager_results.close()?;
    eager_gathered.close()?;
    eager_row_indices_buffer.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-15-owned-bf16-row-gather-argmax-fixed-address replays={REPLAYS} input_rows={INPUT_ROW_COUNT} output_rows={OUTPUT_ROW_COUNT} vocabulary={VOCABULARY_SIZE} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_row_gather_argmax_graph_preflight_and_abort_recover_every_resource()
-> Result<(), Box<dyn Error>> {
    const INPUT_ROW_COUNT: u64 = 8;
    const OUTPUT_ROW_COUNT: u64 = 4;
    const VOCABULARY_SIZE: u64 = 32;
    const INPUT_BYTE_LEN: u64 = INPUT_ROW_COUNT * VOCABULARY_SIZE * 2;
    const ROW_INDICES_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * 4;
    const GATHERED_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * VOCABULARY_SIZE * 2;
    const RESULTS_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * 2 * 4;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let input = context.allocate_device_buffer(INPUT_BYTE_LEN)?;
    let row_indices = context.allocate_device_buffer(ROW_INDICES_BYTE_LEN)?;
    let gathered_logits = context.allocate_device_buffer(GATHERED_BYTE_LEN)?;
    let results = context.allocate_device_buffer(RESULTS_BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let empty_host_mirror: [u32; 0] = [];
    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        &empty_host_mirror,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("empty row-index mirror composite graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("empty composite host-mirror preflight must return all untouched resources");
    let (stream, input, row_indices, gathered_logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let valid_host_mirror = [0_u32, 1, 2, 3];
    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        &valid_host_mirror,
        0,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("zero-input-row composite graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("zero-input-row composite preflight must return all untouched resources");
    let (stream, input, row_indices, gathered_logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        &valid_host_mirror,
        INPUT_ROW_COUNT,
        0,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("zero-vocabulary composite graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::InvalidArgument);
    let resources = error
        .into_resources()
        .expect("zero-vocabulary composite preflight must return all untouched resources");
    let (stream, input, row_indices, gathered_logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        &valid_host_mirror,
        INPUT_ROW_COUNT,
        u64::from(u32::MAX) + 1,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("U32-overflow composite graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("U32-overflow composite preflight must return all untouched resources");
    let (stream, input, row_indices, gathered_logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let duplicate_host_mirror = [0_u32, 1, 1, 3];
    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        &duplicate_host_mirror,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("duplicate composite graph host-mirror preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::InvalidArgument);
    let resources = error
        .into_resources()
        .expect("duplicate composite preflight must return all untouched resources");
    let (stream, input, row_indices, gathered_logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let out_of_range_host_mirror = [0_u32, 1, 2, INPUT_ROW_COUNT as u32];
    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        &out_of_range_host_mirror,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => {
            panic!("out-of-range composite graph host-mirror preflight unexpectedly succeeded")
        }
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("out-of-range composite preflight must return all untouched resources");
    let (stream, input, row_indices, gathered_logits, results) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = stream
        .begin_owned_graph_bf16_row_gather_argmax_capture(
            input,
            row_indices,
            gathered_logits,
            results,
            &valid_host_mirror,
            INPUT_ROW_COUNT,
            VOCABULARY_SIZE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    let (stream, input, row_indices, gathered_logits, results) = resources.into_parts();
    let oversized_host_mirror = [0_u32, 1, 2, 3, 4];
    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        &oversized_host_mirror,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("oversized composite graph preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("oversized composite preflight must preserve all five resources")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-15-owned-bf16-row-gather-argmax-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_row_gather_argmax_d2h_graph_replays_raw_result_bytes_exactly_against_eager()
-> Result<(), Box<dyn Error>> {
    const INPUT_ROW_COUNT: u64 = 13;
    const OUTPUT_ROW_COUNT: u64 = 7;
    const VOCABULARY_SIZE: u64 = 257;
    const REPLAYS: usize = 64;

    let eager_row_indices = vec![12_u32, 0, 7, 3, 10, 1, 5];
    let graph_row_indices_host = eager_row_indices.clone();
    assert_eq!(
        u64::try_from(graph_row_indices_host.len())?,
        OUTPUT_ROW_COUNT
    );

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let bf16_byte_width = u64::try_from(std::mem::size_of::<u16>())?;
    let u32_byte_width = u64::try_from(std::mem::size_of::<u32>())?;
    let input_byte_len = INPUT_ROW_COUNT
        .checked_mul(VOCABULARY_SIZE)
        .and_then(|element_count| element_count.checked_mul(bf16_byte_width))
        .ok_or("C05-16 BF16 row-gather -> argmax -> D2H input byte length overflow")?;
    let row_indices_byte_len = OUTPUT_ROW_COUNT
        .checked_mul(u32_byte_width)
        .ok_or("C05-16 BF16 row-gather -> argmax -> D2H row-index byte length overflow")?;
    let gathered_byte_len = OUTPUT_ROW_COUNT
        .checked_mul(VOCABULARY_SIZE)
        .and_then(|element_count| element_count.checked_mul(bf16_byte_width))
        .ok_or("C05-16 BF16 row-gather -> argmax -> D2H gathered byte length overflow")?;
    let results_byte_len = OUTPUT_ROW_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<
            riley_cuda::Bf16ArgmaxResult,
        >())?)
        .ok_or("C05-16 BF16 row-gather -> argmax -> D2H result byte length overflow")?;

    let selected_logits = graph_bf16_argmax_edge_fixture_bytes();
    assert_eq!(u64::try_from(selected_logits.len())?, gathered_byte_len);
    let row_byte_len = usize::try_from(VOCABULARY_SIZE * bf16_byte_width)?;
    let mut host_input = vec![0xc1_u8; usize::try_from(input_byte_len)?];
    for (output_row, &input_row) in eager_row_indices.iter().enumerate() {
        let source_start = output_row * row_byte_len;
        let source_end = source_start + row_byte_len;
        let destination_start = usize::try_from(input_row)? * row_byte_len;
        let destination_end = destination_start + row_byte_len;
        host_input[destination_start..destination_end]
            .copy_from_slice(&selected_logits[source_start..source_end]);
    }
    let host_row_indices = u32_words_to_ne_bytes(&eager_row_indices);
    let gathered_sentinel = vec![0xa5; usize::try_from(gathered_byte_len)?];
    let results_sentinel = vec![0xa5; usize::try_from(results_byte_len)?];
    let mut staging = context.allocate_pinned_host_buffer(input_byte_len)?;

    let mut eager_input = context.allocate_device_buffer(input_byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    let mut eager_row_indices_buffer = context.allocate_device_buffer(row_indices_byte_len)?;
    eager_row_indices_buffer.upload_from_slice(
        0,
        &host_row_indices,
        &mut staging,
        &mut eager_stream,
    )?;
    let mut eager_gathered = context.allocate_device_buffer(gathered_byte_len)?;
    eager_gathered.upload_from_slice(0, &gathered_sentinel, &mut staging, &mut eager_stream)?;
    let mut eager_results = context.allocate_device_buffer(results_byte_len)?;
    eager_results.upload_from_slice(0, &results_sentinel, &mut staging, &mut eager_stream)?;

    let graph_input = {
        let mut buffer = context.allocate_device_buffer(input_byte_len)?;
        buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_row_indices = {
        let mut buffer = context.allocate_device_buffer(row_indices_byte_len)?;
        buffer.upload_from_slice(0, &host_row_indices, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_gathered = {
        let mut buffer = context.allocate_device_buffer(gathered_byte_len)?;
        buffer.upload_from_slice(0, &gathered_sentinel, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_results = {
        let mut buffer = context.allocate_device_buffer(results_byte_len)?;
        buffer.upload_from_slice(0, &results_sentinel, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_pinned_results = context.allocate_pinned_host_buffer(results_byte_len)?;

    {
        let mut eager = RowGatherParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, input_byte_len)?,
            row_indices: CudaBufferSpan::new(
                &eager_row_indices_buffer,
                CudaDType::U32,
                0,
                row_indices_byte_len,
            )?,
            row_indices_host: &eager_row_indices,
            output: CudaBufferSpanMut::new(
                &mut eager_gathered,
                CudaDType::BF16,
                0,
                gathered_byte_len,
            )?,
            input_row_count: INPUT_ROW_COUNT,
            column_count: VOCABULARY_SIZE,
        };
        row_gather(&mut eager, &mut eager_stream)?;
    }
    {
        let mut eager = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&eager_gathered, CudaDType::BF16, 0, gathered_byte_len)?,
            results: CudaBufferSpanMut::new(
                &mut eager_results,
                CudaDType::U32,
                0,
                results_byte_len,
            )?,
            row_count: OUTPUT_ROW_COUNT,
            vocabulary_size: VOCABULARY_SIZE,
        };
        deterministic_bf16_argmax(&mut eager, &mut eager_stream)?;
    }
    eager_stream.synchronize()?;
    let mut eager_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    eager_results.download_to_slice(
        0,
        &mut eager_result_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        eager_result_bytes,
        graph_bf16_argmax_edge_expected_result_bytes(),
        "C05-16 eager chain must retain deterministic result-record semantics"
    );

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_bf16_row_gather_argmax_d2h_capture(
        graph_input,
        graph_row_indices,
        graph_gathered,
        graph_results,
        graph_pinned_results,
        &graph_row_indices_host,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    // Admission uses this safe mirror only for preflight. It must never be
    // retained by capture, graph, exec, or the completion-scoped raw receipt.
    drop(graph_row_indices_host);
    capture.enqueue_bf16_row_gather_argmax_d2h()?;
    assert_invalid_state(
        capture.enqueue_bf16_row_gather_argmax_d2h(),
        "second C05-16 fixed three-node graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let mut graph_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    for _ in 0..REPLAYS {
        let mut completion = exec.launch()?.finish()?;
        let mut short_destination = vec![0_u8; graph_result_bytes.len() - 1];
        let error = completion
            .read_result_bytes(&mut short_destination)
            .expect_err(
                "a short C05-16 raw result destination must be rejected before native read",
            );
        assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
        completion.read_result_bytes(&mut graph_result_bytes)?;
        assert_eq!(
            graph_result_bytes, eager_result_bytes,
            "each C05-16 graph replay must D2H exact eager result-record bytes"
        );
        drop(completion);
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (
        capture_stream,
        mut graph_input,
        mut graph_row_indices,
        mut graph_gathered,
        mut graph_results,
        graph_pinned_results,
    ) = resources.into_parts();
    let mut eager_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
    let mut graph_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
    let mut graph_device_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    let mut graph_input_after = vec![0_u8; usize::try_from(input_byte_len)?];
    let mut graph_row_indices_after = vec![0_u8; usize::try_from(row_indices_byte_len)?];
    eager_gathered.download_to_slice(
        0,
        &mut eager_gathered_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_gathered.download_to_slice(
        0,
        &mut graph_gathered_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_results.download_to_slice(
        0,
        &mut graph_device_result_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_input.download_to_slice(
        0,
        &mut graph_input_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_row_indices.download_to_slice(
        0,
        &mut graph_row_indices_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_gathered_bytes, eager_gathered_bytes,
        "C05-16 gathered BF16 device bytes must match the eager first node"
    );
    assert_eq!(
        graph_device_result_bytes, graph_result_bytes,
        "C05-16 pinned D2H bytes must match the graph's retained result records"
    );
    assert_eq!(graph_input_after, host_input);
    assert_eq!(graph_row_indices_after, host_row_indices);

    graph_pinned_results.close()?;
    graph_results.close()?;
    graph_gathered.close()?;
    graph_row_indices.close()?;
    graph_input.close()?;
    eager_results.close()?;
    eager_gathered.close()?;
    eager_row_indices_buffer.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-16-owned-bf16-row-gather-argmax-d2h-fixed-address replays={REPLAYS} input_rows={INPUT_ROW_COUNT} output_rows={OUTPUT_ROW_COUNT} vocabulary={VOCABULARY_SIZE} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_row_gather_argmax_d2h_graph_requires_exact_pinned_results_and_recovers_every_resource()
-> Result<(), Box<dyn Error>> {
    const INPUT_ROW_COUNT: u64 = 8;
    const OUTPUT_ROW_COUNT: u64 = 4;
    const VOCABULARY_SIZE: u64 = 32;
    const INPUT_BYTE_LEN: u64 = INPUT_ROW_COUNT * VOCABULARY_SIZE * 2;
    const ROW_INDICES_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * 4;
    const GATHERED_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * VOCABULARY_SIZE * 2;
    const RESULTS_BYTE_LEN: u64 = OUTPUT_ROW_COUNT * 8;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let valid_host_mirror = [0_u32, 1, 2, 3];

    let stream = context.create_stream()?;
    let input = context.allocate_device_buffer(INPUT_BYTE_LEN)?;
    let row_indices = context.allocate_device_buffer(ROW_INDICES_BYTE_LEN)?;
    let gathered_logits = context.allocate_device_buffer(GATHERED_BYTE_LEN)?;
    let results = context.allocate_device_buffer(RESULTS_BYTE_LEN)?;
    let short_pinned_results = context.allocate_pinned_host_buffer(RESULTS_BYTE_LEN - 1)?;
    let allocation_with_resources = context.allocation_stats()?;
    let error = match stream.begin_owned_graph_bf16_row_gather_argmax_d2h_capture(
        input,
        row_indices,
        gathered_logits,
        results,
        short_pinned_results,
        &valid_host_mirror,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("short pinned C05-16 result buffer preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("pinned-result preflight must return every untouched C05-16 resource");
    let (stream, input, row_indices, gathered_logits, results, short_pinned_results) =
        resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    short_pinned_results.close()?;
    let exact_pinned_results = context.allocate_pinned_host_buffer(RESULTS_BYTE_LEN)?;
    let resources = stream
        .begin_owned_graph_bf16_row_gather_argmax_d2h_capture(
            input,
            row_indices,
            gathered_logits,
            results,
            exact_pinned_results,
            &valid_host_mirror,
            INPUT_ROW_COUNT,
            VOCABULARY_SIZE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    resources.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-16-owned-bf16-row-gather-argmax-d2h-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_row_gather_argmax_d2h_graph_preserves_device_index_oob_raw_result_bytes()
-> Result<(), Box<dyn Error>> {
    const INPUT_ROW_COUNT: u64 = 4;
    const OUTPUT_ROW_COUNT: u64 = 3;
    const VOCABULARY_SIZE: u64 = 5;
    const BF16_BYTES: u64 = 2;
    const U32_BYTES: u64 = 4;

    // The safe host mirror deliberately remains in range. It proves only
    // shape/admission; device bytes remain the captured truth, and the middle
    // device index is deliberately OOB to exercise the raw CUDA primitive's
    // NaN -> non-finite argmax record behavior through C05-16 D2H.
    let host_mirror = [2_u32, 3, 0];
    let device_indices = [2_u32, INPUT_ROW_COUNT as u32, 0];
    let host_input_bits = [
        0x3f80_u16, 0x4000, 0x4040, 0x4080, 0x40a0, // row 0: token 4
        0xbf80, 0xc000, 0xc040, 0xc080, 0xc0a0, // row 1: token 0
        0x0000, 0x3f80, 0x4000, 0x4040, 0x4080, // row 2: token 4
        0x3f00, 0x3f80, 0x4000, 0x4040, 0x4080, // row 3: unused
    ];
    let host_input: Vec<u8> = host_input_bits
        .iter()
        .flat_map(|bits| bits.to_ne_bytes())
        .collect();
    let host_device_index_bytes = u32_words_to_ne_bytes(&device_indices);
    let input_byte_len = INPUT_ROW_COUNT * VOCABULARY_SIZE * BF16_BYTES;
    let row_indices_byte_len = OUTPUT_ROW_COUNT * U32_BYTES;
    let gathered_byte_len = OUTPUT_ROW_COUNT * VOCABULARY_SIZE * BF16_BYTES;
    let results_byte_len =
        OUTPUT_ROW_COUNT * u64::try_from(std::mem::size_of::<riley_cuda::Bf16ArgmaxResult>())?;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(input_byte_len)?;

    let mut eager_input = context.allocate_device_buffer(input_byte_len)?;
    let mut eager_indices = context.allocate_device_buffer(row_indices_byte_len)?;
    let mut eager_gathered = context.allocate_device_buffer(gathered_byte_len)?;
    let mut eager_results = context.allocate_device_buffer(results_byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    eager_indices.upload_from_slice(
        0,
        &host_device_index_bytes,
        &mut staging,
        &mut eager_stream,
    )?;
    {
        let mut eager = RowGatherParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, input_byte_len)?,
            row_indices: CudaBufferSpan::new(
                &eager_indices,
                CudaDType::U32,
                0,
                row_indices_byte_len,
            )?,
            row_indices_host: &host_mirror,
            output: CudaBufferSpanMut::new(
                &mut eager_gathered,
                CudaDType::BF16,
                0,
                gathered_byte_len,
            )?,
            input_row_count: INPUT_ROW_COUNT,
            column_count: VOCABULARY_SIZE,
        };
        row_gather(&mut eager, &mut eager_stream)?;
    }
    {
        let mut eager = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(&eager_gathered, CudaDType::BF16, 0, gathered_byte_len)?,
            results: CudaBufferSpanMut::new(
                &mut eager_results,
                CudaDType::U32,
                0,
                results_byte_len,
            )?,
            row_count: OUTPUT_ROW_COUNT,
            vocabulary_size: VOCABULARY_SIZE,
        };
        deterministic_bf16_argmax(&mut eager, &mut eager_stream)?;
    }
    eager_stream.synchronize()?;
    let mut eager_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    eager_results.download_to_slice(
        0,
        &mut eager_result_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;

    let graph_input = {
        let mut buffer = context.allocate_device_buffer(input_byte_len)?;
        buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_indices = {
        let mut buffer = context.allocate_device_buffer(row_indices_byte_len)?;
        buffer.upload_from_slice(0, &host_device_index_bytes, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_gathered = context.allocate_device_buffer(gathered_byte_len)?;
    let graph_results = context.allocate_device_buffer(results_byte_len)?;
    let graph_pinned_results = context.allocate_pinned_host_buffer(results_byte_len)?;
    let capture = capture_stream.begin_owned_graph_bf16_row_gather_argmax_d2h_capture(
        graph_input,
        graph_indices,
        graph_gathered,
        graph_results,
        graph_pinned_results,
        &host_mirror,
        INPUT_ROW_COUNT,
        VOCABULARY_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    let mut capture = capture;
    capture.enqueue_bf16_row_gather_argmax_d2h()?;
    let mut exec = capture.end()?.instantiate()?;
    let mut graph_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
    exec.launch()?
        .finish()?
        .read_result_bytes(&mut graph_result_bytes)?;
    assert_eq!(
        graph_result_bytes, eager_result_bytes,
        "C05-16 D2H must preserve raw OOB result bytes from the eager chain"
    );

    let invalid_result_start = std::mem::size_of::<riley_cuda::Bf16ArgmaxResult>();
    let invalid_result_end =
        invalid_result_start + std::mem::size_of::<riley_cuda::Bf16ArgmaxResult>();
    let invalid_result = &graph_result_bytes[invalid_result_start..invalid_result_end];
    let invalid_token_id = u32::from_ne_bytes(
        invalid_result[..std::mem::size_of::<u32>()]
            .try_into()
            .expect("one result token field must occupy four bytes"),
    );
    let invalid_status = u32::from_ne_bytes(
        invalid_result[std::mem::size_of::<u32>()..]
            .try_into()
            .expect("one result status field must occupy four bytes"),
    );
    assert_eq!(invalid_token_id, BF16_ARGMAX_INVALID_TOKEN_ID);
    assert_eq!(invalid_status, BF16_ARGMAX_STATUS_NON_FINITE);

    let resources = exec.close()?;
    let (
        capture_stream,
        graph_input,
        graph_indices,
        mut graph_gathered,
        graph_results,
        graph_pinned_results,
    ) = resources.into_parts();
    let mut eager_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
    let mut graph_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
    eager_gathered.download_to_slice(
        0,
        &mut eager_gathered_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_gathered.download_to_slice(
        0,
        &mut graph_gathered_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(graph_gathered_bytes, eager_gathered_bytes);
    let invalid_row_start = usize::try_from(VOCABULARY_SIZE * BF16_BYTES)?;
    let invalid_row_end = invalid_row_start + usize::try_from(VOCABULARY_SIZE * BF16_BYTES)?;
    assert!(
        graph_gathered_bytes[invalid_row_start..invalid_row_end]
            .chunks_exact(usize::try_from(BF16_BYTES)?)
            .all(|bytes| {
                let bits = u16::from_ne_bytes([bytes[0], bytes[1]]);
                (bits & 0x7f80) == 0x7f80 && (bits & 0x007f) != 0
            }),
        "the OOB graph-gather row must contain BF16 NaNs before D2H result capture"
    );

    graph_pinned_results.close()?;
    graph_results.close()?;
    graph_gathered.close()?;
    graph_indices.close()?;
    graph_input.close()?;
    eager_results.close()?;
    eager_gathered.close()?;
    eager_indices.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-16-owned-bf16-row-gather-argmax-d2h-oob-result-bytes status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_indexed_rope_bf16_graph_replays_byte_exact_against_eager() -> Result<(), Box<dyn Error>> {
    const ACTIVE_ROW_COUNT: u64 = 3;
    const HEAD_COUNT: u64 = 1;
    const HEAD_SIZE: u64 = 6;
    const ROTARY_DIMENSION: u64 = 4;
    const TABLE_POSITION_COUNT: u64 = 5;
    const REPLAYS: usize = 64;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let bf16_byte_width = u64::try_from(std::mem::size_of::<u16>())?;
    let f32_byte_width = u64::try_from(std::mem::size_of::<f32>())?;
    let u32_byte_width = u64::try_from(std::mem::size_of::<u32>())?;
    let tensor_byte_len = ACTIVE_ROW_COUNT
        .checked_mul(HEAD_COUNT)
        .and_then(|value| value.checked_mul(HEAD_SIZE))
        .and_then(|value| value.checked_mul(bf16_byte_width))
        .ok_or("C05-17 BF16 indexed-RoPE tensor byte length overflow")?;
    let table_byte_len = TABLE_POSITION_COUNT
        .checked_mul(ROTARY_DIMENSION / 2)
        .and_then(|value| value.checked_mul(f32_byte_width))
        .ok_or("C05-17 indexed-RoPE table byte length overflow")?;
    let positions_byte_len = ACTIVE_ROW_COUNT
        .checked_mul(u32_byte_width)
        .ok_or("C05-17 indexed-RoPE positions byte length overflow")?;
    let host_input = graph_indexed_rope_bf16_fixture_bytes();
    let (host_cos, host_sin) = graph_indexed_rope_table_bytes();
    let eager_positions_host = [4_u32, 0, 2];
    let graph_positions_host = vec![4_u32, 0, 2];
    let host_positions = u32_words_to_ne_bytes(&eager_positions_host);
    let output_sentinel = vec![0xa5; usize::try_from(tensor_byte_len)?];
    assert_eq!(u64::try_from(host_input.len())?, tensor_byte_len);
    assert_eq!(u64::try_from(host_cos.len())?, table_byte_len);
    assert_eq!(u64::try_from(host_sin.len())?, table_byte_len);
    assert_eq!(u64::try_from(host_positions.len())?, positions_byte_len);
    let mut staging = context
        .allocate_pinned_host_buffer(tensor_byte_len.max(table_byte_len).max(positions_byte_len))?;

    let mut eager_input = context.allocate_device_buffer(tensor_byte_len)?;
    let mut eager_cos = context.allocate_device_buffer(table_byte_len)?;
    let mut eager_sin = context.allocate_device_buffer(table_byte_len)?;
    let mut eager_positions = context.allocate_device_buffer(positions_byte_len)?;
    let mut eager_output = context.allocate_device_buffer(tensor_byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    eager_cos.upload_from_slice(0, &host_cos, &mut staging, &mut eager_stream)?;
    eager_sin.upload_from_slice(0, &host_sin, &mut staging, &mut eager_stream)?;
    eager_positions.upload_from_slice(0, &host_positions, &mut staging, &mut eager_stream)?;
    eager_output.upload_from_slice(0, &output_sentinel, &mut staging, &mut eager_stream)?;

    let graph_input = {
        let mut buffer = context.allocate_device_buffer(tensor_byte_len)?;
        buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_cos = {
        let mut buffer = context.allocate_device_buffer(table_byte_len)?;
        buffer.upload_from_slice(0, &host_cos, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_sin = {
        let mut buffer = context.allocate_device_buffer(table_byte_len)?;
        buffer.upload_from_slice(0, &host_sin, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_positions = {
        let mut buffer = context.allocate_device_buffer(positions_byte_len)?;
        buffer.upload_from_slice(0, &host_positions, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_output = {
        let mut buffer = context.allocate_device_buffer(tensor_byte_len)?;
        buffer.upload_from_slice(0, &output_sentinel, &mut staging, &mut eager_stream)?;
        buffer
    };

    {
        let mut eager = IndexedRopeParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, tensor_byte_len)?,
            cos: CudaBufferSpan::new(&eager_cos, CudaDType::F32, 0, table_byte_len)?,
            sin: CudaBufferSpan::new(&eager_sin, CudaDType::F32, 0, table_byte_len)?,
            positions: CudaBufferSpan::new(
                &eager_positions,
                CudaDType::U32,
                0,
                positions_byte_len,
            )?,
            positions_host: &eager_positions_host,
            output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, tensor_byte_len)?,
            head_count: HEAD_COUNT,
            head_size: HEAD_SIZE,
            rotary_dimension: ROTARY_DIMENSION,
            table_position_count: TABLE_POSITION_COUNT,
        };
        indexed_rope(&mut eager, &mut eager_stream)?;
    }
    eager_stream.synchronize()?;

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_indexed_rope_bf16_capture(
        graph_input,
        graph_cos,
        graph_sin,
        graph_positions,
        graph_output,
        &graph_positions_host,
        HEAD_COUNT,
        HEAD_SIZE,
        ROTARY_DIMENSION,
        TABLE_POSITION_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    // The safe host mirror is admission-only; the graph retains no host
    // pointer and replays the fixed device positions allocation.
    drop(graph_positions_host);
    capture.enqueue_indexed_rope_bf16()?;
    assert_invalid_state(
        capture.enqueue_indexed_rope_bf16(),
        "second C05-17 fixed one-node indexed-RoPE graph enqueue",
    );
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (
        capture_stream,
        mut graph_input,
        mut graph_cos,
        mut graph_sin,
        mut graph_positions,
        mut graph_output,
    ) = resources.into_parts();
    let mut eager_output_bytes = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_output_bytes = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_input_after = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_cos_after = vec![0_u8; usize::try_from(table_byte_len)?];
    let mut graph_sin_after = vec![0_u8; usize::try_from(table_byte_len)?];
    let mut graph_positions_after = vec![0_u8; usize::try_from(positions_byte_len)?];
    eager_output.download_to_slice(
        0,
        &mut eager_output_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_output.download_to_slice(
        0,
        &mut graph_output_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_input.download_to_slice(
        0,
        &mut graph_input_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_cos.download_to_slice(0, &mut graph_cos_after, &mut staging, &mut transfer_stream)?;
    graph_sin.download_to_slice(0, &mut graph_sin_after, &mut staging, &mut transfer_stream)?;
    graph_positions.download_to_slice(
        0,
        &mut graph_positions_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_output_bytes, eager_output_bytes,
        "C05-17 fixed graph output must match eager indexed-RoPE bytes exactly"
    );
    assert_eq!(graph_input_after, host_input);
    assert_eq!(graph_cos_after, host_cos);
    assert_eq!(graph_sin_after, host_sin);
    assert_eq!(graph_positions_after, host_positions);

    graph_output.close()?;
    graph_positions.close()?;
    graph_sin.close()?;
    graph_cos.close()?;
    graph_input.close()?;
    eager_output.close()?;
    eager_positions.close()?;
    eager_sin.close()?;
    eager_cos.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-17-owned-indexed-rope-bf16-fixed-address replays={REPLAYS} rows={ACTIVE_ROW_COUNT} heads={HEAD_COUNT} head_size={HEAD_SIZE} rotary={ROTARY_DIMENSION} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_indexed_rope_bf16_graph_preserves_device_position_oob_nan_bytes()
-> Result<(), Box<dyn Error>> {
    const ACTIVE_ROW_COUNT: u64 = 3;
    const HEAD_COUNT: u64 = 1;
    const HEAD_SIZE: u64 = 6;
    const ROTARY_DIMENSION: u64 = 4;
    const TABLE_POSITION_COUNT: u64 = 5;
    const BF16_BYTES: u64 = 2;
    const U32_BYTES: u64 = 4;

    // The mirror is in range and therefore approves capture admission. The
    // second fixed device position is exactly table_position_count and tests
    // the eager primitive's raw OOB sentinel path instead.
    let eager_positions_host = [4_u32, 0, 2];
    let graph_positions_host = vec![4_u32, 0, 2];
    let device_positions = [4_u32, TABLE_POSITION_COUNT as u32, 2];
    let host_input = graph_indexed_rope_bf16_fixture_bytes();
    let (host_cos, host_sin) = graph_indexed_rope_table_bytes();
    let host_device_positions = u32_words_to_ne_bytes(&device_positions);
    let tensor_byte_len = ACTIVE_ROW_COUNT * HEAD_COUNT * HEAD_SIZE * BF16_BYTES;
    let table_byte_len =
        TABLE_POSITION_COUNT * (ROTARY_DIMENSION / 2) * u64::try_from(std::mem::size_of::<f32>())?;
    let positions_byte_len = ACTIVE_ROW_COUNT * U32_BYTES;
    assert_eq!(u64::try_from(host_input.len())?, tensor_byte_len);
    assert_eq!(
        u64::try_from(host_device_positions.len())?,
        positions_byte_len
    );

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let mut staging = context
        .allocate_pinned_host_buffer(tensor_byte_len.max(table_byte_len).max(positions_byte_len))?;

    let mut eager_input = context.allocate_device_buffer(tensor_byte_len)?;
    let mut eager_cos = context.allocate_device_buffer(table_byte_len)?;
    let mut eager_sin = context.allocate_device_buffer(table_byte_len)?;
    let mut eager_positions = context.allocate_device_buffer(positions_byte_len)?;
    let mut eager_output = context.allocate_device_buffer(tensor_byte_len)?;
    eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
    eager_cos.upload_from_slice(0, &host_cos, &mut staging, &mut eager_stream)?;
    eager_sin.upload_from_slice(0, &host_sin, &mut staging, &mut eager_stream)?;
    eager_positions.upload_from_slice(
        0,
        &host_device_positions,
        &mut staging,
        &mut eager_stream,
    )?;

    let graph_input = {
        let mut buffer = context.allocate_device_buffer(tensor_byte_len)?;
        buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_cos = {
        let mut buffer = context.allocate_device_buffer(table_byte_len)?;
        buffer.upload_from_slice(0, &host_cos, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_sin = {
        let mut buffer = context.allocate_device_buffer(table_byte_len)?;
        buffer.upload_from_slice(0, &host_sin, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_positions = {
        let mut buffer = context.allocate_device_buffer(positions_byte_len)?;
        buffer.upload_from_slice(0, &host_device_positions, &mut staging, &mut eager_stream)?;
        buffer
    };
    let graph_output = context.allocate_device_buffer(tensor_byte_len)?;

    {
        let mut eager = IndexedRopeParams {
            input: CudaBufferSpan::new(&eager_input, CudaDType::BF16, 0, tensor_byte_len)?,
            cos: CudaBufferSpan::new(&eager_cos, CudaDType::F32, 0, table_byte_len)?,
            sin: CudaBufferSpan::new(&eager_sin, CudaDType::F32, 0, table_byte_len)?,
            positions: CudaBufferSpan::new(
                &eager_positions,
                CudaDType::U32,
                0,
                positions_byte_len,
            )?,
            positions_host: &eager_positions_host,
            output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, tensor_byte_len)?,
            head_count: HEAD_COUNT,
            head_size: HEAD_SIZE,
            rotary_dimension: ROTARY_DIMENSION,
            table_position_count: TABLE_POSITION_COUNT,
        };
        indexed_rope(&mut eager, &mut eager_stream)?;
    }
    eager_stream.synchronize()?;

    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = capture_stream.begin_owned_graph_indexed_rope_bf16_capture(
        graph_input,
        graph_cos,
        graph_sin,
        graph_positions,
        graph_output,
        &graph_positions_host,
        HEAD_COUNT,
        HEAD_SIZE,
        ROTARY_DIMENSION,
        TABLE_POSITION_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    drop(graph_positions_host);
    capture.enqueue_indexed_rope_bf16()?;
    let mut exec = capture.end()?.instantiate()?;
    exec.launch()?.finish()?;
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (capture_stream, mut graph_input, graph_cos, graph_sin, graph_positions, mut graph_output) =
        resources.into_parts();
    let mut eager_output_bytes = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_output_bytes = vec![0_u8; usize::try_from(tensor_byte_len)?];
    let mut graph_input_after = vec![0_u8; usize::try_from(tensor_byte_len)?];
    eager_output.download_to_slice(
        0,
        &mut eager_output_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_output.download_to_slice(
        0,
        &mut graph_output_bytes,
        &mut staging,
        &mut transfer_stream,
    )?;
    graph_input.download_to_slice(
        0,
        &mut graph_input_after,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_output_bytes, eager_output_bytes,
        "C05-17 graph must preserve eager raw device-position OOB bytes"
    );
    assert_eq!(graph_input_after, host_input);

    let oob_row = 1_usize;
    for channel in 0..usize::try_from(ROTARY_DIMENSION)? {
        let element = oob_row * usize::try_from(HEAD_SIZE)? + channel;
        let byte_offset = element * usize::try_from(BF16_BYTES)?;
        let bits = u16::from_ne_bytes([
            graph_output_bytes[byte_offset],
            graph_output_bytes[byte_offset + 1],
        ]);
        assert!(
            (bits & 0x7f80) == 0x7f80 && (bits & 0x007f) != 0,
            "C05-17 device-OOB row rotary channel {channel} must retain the eager BF16 NaN sentinel"
        );
    }
    for channel in usize::try_from(ROTARY_DIMENSION)?..usize::try_from(HEAD_SIZE)? {
        let element = oob_row * usize::try_from(HEAD_SIZE)? + channel;
        let byte_offset = element * usize::try_from(BF16_BYTES)?;
        assert_eq!(
            &graph_output_bytes[byte_offset..byte_offset + usize::try_from(BF16_BYTES)?],
            &host_input[byte_offset..byte_offset + usize::try_from(BF16_BYTES)?],
            "C05-17 device-OOB row non-rotary tail channel {channel} must remain an exact input copy"
        );
    }

    graph_output.close()?;
    graph_positions.close()?;
    graph_sin.close()?;
    graph_cos.close()?;
    graph_input.close()?;
    eager_output.close()?;
    eager_positions.close()?;
    eager_sin.close()?;
    eager_cos.close()?;
    eager_input.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-17-owned-indexed-rope-bf16-device-position-oob status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_indexed_rope_bf16_graph_preflight_and_abort_recover_every_resource()
-> Result<(), Box<dyn Error>> {
    const ACTIVE_ROW_COUNT: u64 = 3;
    const HEAD_COUNT: u64 = 1;
    const HEAD_SIZE: u64 = 6;
    const ROTARY_DIMENSION: u64 = 4;
    const TABLE_POSITION_COUNT: u64 = 5;
    const TENSOR_BYTE_LEN: u64 = ACTIVE_ROW_COUNT * HEAD_COUNT * HEAD_SIZE * 2;
    const TABLE_BYTE_LEN: u64 = TABLE_POSITION_COUNT * (ROTARY_DIMENSION / 2) * 4;
    const POSITIONS_BYTE_LEN: u64 = ACTIVE_ROW_COUNT * 4;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let input = context.allocate_device_buffer(TENSOR_BYTE_LEN)?;
    let cos = context.allocate_device_buffer(TABLE_BYTE_LEN)?;
    let sin = context.allocate_device_buffer(TABLE_BYTE_LEN)?;
    let positions = context.allocate_device_buffer(POSITIONS_BYTE_LEN)?;
    let short_output = context.allocate_device_buffer(TENSOR_BYTE_LEN - 2)?;
    let allocation_with_resources = context.allocation_stats()?;
    let valid_positions_host = [0_u32, 1, 2];
    let oob_positions_host = [0_u32, 1, TABLE_POSITION_COUNT as u32];

    let error = match stream.begin_owned_graph_indexed_rope_bf16_capture(
        input,
        cos,
        sin,
        positions,
        short_output,
        &oob_positions_host,
        HEAD_COUNT,
        HEAD_SIZE,
        ROTARY_DIMENSION,
        TABLE_POSITION_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("C05-17 out-of-range host position preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("C05-17 host-mirror preflight must return every untouched resource");
    let (stream, input, cos, sin, positions, short_output) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let error = match stream.begin_owned_graph_indexed_rope_bf16_capture(
        input,
        cos,
        sin,
        positions,
        short_output,
        &valid_positions_host,
        HEAD_COUNT,
        HEAD_SIZE,
        ROTARY_DIMENSION,
        TABLE_POSITION_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("C05-17 short-output preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("C05-17 short-output preflight must return every untouched resource");
    let (stream, input, cos, sin, positions, short_output) = resources.into_parts();
    short_output.close()?;
    let output = context.allocate_device_buffer(TENSOR_BYTE_LEN)?;

    let resources = stream
        .begin_owned_graph_indexed_rope_bf16_capture(
            input,
            cos,
            sin,
            positions,
            output,
            &valid_positions_host,
            HEAD_COUNT,
            HEAD_SIZE,
            ROTARY_DIMENSION,
            TABLE_POSITION_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    resources.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-17-owned-indexed-rope-bf16-preflight-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_fixed_fill_exec_moves_replays_and_returns_resources() -> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 1_024;
    const REPLAYS: usize = 32;
    const FINAL_VALUE: f32 = -13.5;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let capture_stream = context.create_stream()?;
    let mut download_stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("owned graph capture allocation byte length overflow")?;
    let buffer = context.allocate_device_buffer(byte_len)?;

    let mut capture = capture_stream.begin_owned_graph_fill_capture(
        buffer,
        ELEMENT_COUNT,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_fill(2.0)?;
    capture.enqueue_fill(FINAL_VALUE)?;
    let captured = capture.end()?;
    // This value move is the C05-6 boundary: the executable has no lexical
    // borrow into its stream/buffer pair and can live in a cold owner.
    let mut exec = move_into_cold_owner(captured.instantiate()?);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }

    let resources = exec.close()?;
    let (capture_stream, mut buffer) = resources.into_parts();
    let values = download_f32_buffer(&context, &mut buffer, &mut download_stream, ELEMENT_COUNT)?;
    assert_eq!(values.len(), usize::try_from(ELEMENT_COUNT)?);
    assert!(all_f32_bits_equal(&values, FINAL_VALUE));

    // A known exec close releases both Rust and native graph leases. The same
    // moved resources are therefore admissible for a fresh value-owning
    // capture, not just for independent D2H.
    let resources = capture_stream
        .begin_owned_graph_fill_capture(buffer, ELEMENT_COUNT, CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    let (capture_stream, buffer) = resources.into_parts();
    buffer.close()?;
    capture_stream.close()?;
    download_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!(
        "c05-6-owned-fill-replay replays={REPLAYS} elements={ELEMENT_COUNT} final_value={FINAL_VALUE} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_h2d_graph_replays_fresh_exact_payloads_and_recovers_every_resource()
-> Result<(), Box<dyn Error>> {
    const BYTE_LEN: u64 = 4_096;
    const REPLAYS: usize = 32;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let capture_stream = context.create_stream()?;
    let source = context.allocate_pinned_host_buffer(BYTE_LEN)?;
    let destination = context.allocate_device_buffer(BYTE_LEN)?;
    let allocation_with_resources = context.allocation_stats()?;

    let mut capture = capture_stream.begin_owned_graph_h2d_capture(
        source,
        destination,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_h2d()?;
    let captured = capture.end()?;
    let mut exec = captured.instantiate()?;

    // Capture, instantiate, and the graph lifecycle itself must not allocate
    // another tracked device or pinned-host buffer. This intentionally does
    // not claim that CUDA graph launch has no native completion allocation.
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let byte_len = usize::try_from(BYTE_LEN)?;
    let wrong_length_error = match exec.launch_with_source(&h2d_payload(byte_len - 1, 0)) {
        Ok(_) => panic!("short H2D graph payload unexpectedly launched"),
        Err(error) => error,
    };
    assert_eq!(wrong_length_error.kind(), CudaErrorKind::OutOfRange);
    // A Rust-only length rejection has not staged or launched anything, so a
    // later exact whole-slab replay remains admissible.
    let mut expected = Vec::new();
    for replay in 0..REPLAYS {
        let payload = h2d_payload(byte_len, replay);
        exec.launch_with_source(&payload)?.finish()?;
        expected = payload;
    }

    let resources = exec.close()?;
    let (mut capture_stream, mut source, mut destination) = resources.into_parts();
    let mut actual = vec![0_u8; byte_len];
    destination.download_to_slice(0, &mut actual, &mut source, &mut capture_stream)?;
    assert_eq!(
        actual, expected,
        "last staged H2D payload must replay byte-exactly"
    );

    source.close()?;
    destination.close()?;
    capture_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-7-owned-h2d-replay replays={REPLAYS} bytes={BYTE_LEN} status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_h2d_graph_preflight_errors_return_untouched_three_resource_bundles()
-> Result<(), Box<dyn Error>> {
    const BYTE_LEN: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    for (case, source_len, destination_len) in
        [("zero-sized", 0, 0), ("mismatched", BYTE_LEN, BYTE_LEN + 1)]
    {
        let stream = context.create_stream()?;
        let source = context.allocate_pinned_host_buffer(source_len)?;
        let destination = context.allocate_device_buffer(destination_len)?;
        let error = match stream.begin_owned_graph_h2d_capture(
            source,
            destination,
            CudaGraphCaptureMode::ThreadLocal,
        ) {
            Ok(_) => panic!("{case} owned H2D graph preflight unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
        let resources = error
            .into_resources()
            .expect("Rust H2D preflight must return all untouched moved resources");
        let (stream, source, destination) = resources.into_parts();
        destination.close()?;
        source.close()?;
        stream.close()?;
        assert_eq!(
            context.allocation_stats()?,
            allocation_baseline,
            "{case} H2D preflight recovery must restore the exact allocation baseline"
        );
    }

    close_context(context)?;
    println!("c05-7-owned-h2d-preflight-resource-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_zero_enqueue_end_drop_recovers_and_releases_cold_resources() -> Result<(), Box<dyn Error>>
{
    const ELEMENT_COUNT: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("owned zero-enqueue capture allocation byte length overflow")?;
    let buffer = context.allocate_device_buffer(byte_len)?;

    let error = match stream
        .begin_owned_graph_fill_capture(buffer, ELEMENT_COUNT, CudaGraphCaptureMode::ThreadLocal)?
        .end()
    {
        Ok(_) => panic!("owned zero-enqueue graph capture unexpectedly ended"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    // `end` consumed the by-value capture. Its Drop must abort natively before
    // the moved stream/buffer destructors run, leaving no allocation ledger
    // residue and no resource available for accidental reuse.
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-6-owned-zero-enqueue-drop-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_fill_capture_preflight_error_returns_untouched_resources() -> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 128;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("owned preflight capture allocation byte length overflow")?;
    let buffer = context.allocate_device_buffer(byte_len)?;

    let error = match stream.begin_owned_graph_fill_capture(
        buffer,
        ELEMENT_COUNT + 1,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("oversized owned capture preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("Rust-side preflight must return its untouched moved resources");
    let (mut stream, buffer) = resources.into_parts();
    assert_eager_fill_after_recovery(&context, &mut stream, 9.25)?;

    buffer.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-6-owned-preflight-resource-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn fixed_buffer_fill_graph_replays_exactly_and_releases_every_lease() -> Result<(), Box<dyn Error>>
{
    const ELEMENT_COUNT: u64 = 4_096;
    const REPLAYS: usize = 1_000;
    const FINAL_VALUE: f32 = -7.25;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut capture_stream = context.create_stream()?;
    let mut download_stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("f32 capture allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    let captured = {
        let mut capture = capture_stream.begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        // The final node makes every replay's expected output unambiguous.
        capture.enqueue_fill(1.5)?;
        capture.enqueue_fill(-3.0)?;
        capture.enqueue_fill(FINAL_VALUE)?;
        capture.end()?
    };
    let mut exec = captured.instantiate()?;
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }

    // Safe Rust keeps the capture stream and buffer mutably borrowed until
    // this close succeeds. The native graph exec retains matching raw leases
    // too, so only a known close may return them for this independent D2H.
    exec.close()?;
    let values = download_f32_buffer(&context, &mut buffer, &mut download_stream, ELEMENT_COUNT)?;
    assert_eq!(values.len(), usize::try_from(ELEMENT_COUNT)?);
    assert!(all_f32_bits_equal(&values, FINAL_VALUE));

    buffer.close()?;
    capture_stream.close()?;
    download_stream.close()?;
    assert_eq!(
        context.allocation_stats()?,
        allocation_baseline,
        "successful graph/exec close plus D2H staging close must restore the exact allocation baseline"
    );
    close_context(context)?;
    println!(
        "c05-5-fixed-fill-replay replays={REPLAYS} elements={ELEMENT_COUNT} final_value={FINAL_VALUE} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn captured_graph_close_releases_stream_and_buffer_for_reuse() -> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("captured graph close allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    let captured = {
        let mut capture = stream.begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        capture.enqueue_fill(4.0)?;
        capture.end()?
    };
    // This exercises graph destruction without instantiation. Success must
    // return both the native graph leases and the Rust mutable borrows.
    captured.close()?;
    assert_eager_fill_after_recovery(&context, &mut stream, 4.0)?;

    buffer.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-5-captured-graph-close-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn fixed_buffer_fill_capture_abort_releases_stream_and_buffer_for_reuse()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("f32 abort-recovery allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    for _ in 0..2 {
        stream
            .begin_graph_fill_capture(
                &mut buffer,
                ELEMENT_COUNT,
                CudaGraphCaptureMode::ThreadLocal,
            )?
            .abort()?;
    }
    assert_eager_fill_after_recovery(&context, &mut stream, 2.5)?;

    // The buffer comes back from the capture borrow only after native abort
    // has released its active-use lease; explicit close is the observable
    // proof that capture did not strand it busy.
    buffer.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-5-fixed-fill-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn zero_enqueue_fill_end_rejects_before_native_end_and_abort_restores_reuse()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("zero-enqueue capture allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    // `end` rejects the missing admitted node in safe Rust, before calling
    // native end. Its consuming error path drops the still-active capture and
    // performs the one-shot abort/recovery instead of leaving either lease
    // stranded.
    let error = {
        let capture = stream.begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        match capture.end() {
            Ok(_) => panic!("zero-enqueue graph capture unexpectedly ended"),
            Err(error) => error,
        }
    };
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);

    // Both the same preallocated buffer and stream must be immediately
    // admissible for a fresh capture after that automatic abort.
    stream
        .begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    assert_eager_fill_after_recovery(&context, &mut stream, -1.5)?;

    buffer.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-5-zero-enqueue-end-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn explicit_abort_restores_stream_for_eager_work() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;

    stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    assert_eager_fill_after_recovery(&context, &mut stream, 3.25)?;

    stream.close()?;
    close_context(context)?;
    println!("c05-4-explicit-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn drop_abort_restores_stream_for_eager_work() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;

    {
        let _capture = stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    }
    assert_eager_fill_after_recovery(&context, &mut stream, -7.5)?;

    stream.close()?;
    close_context(context)?;
    println!("c05-4-drop-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn repeated_abort_releases_stream_and_context_leases() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;

    for _ in 0..8 {
        stream
            .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
            .abort()?;
    }
    assert_eager_fill_after_recovery(&context, &mut stream, 0.125)?;

    stream.close()?;
    close_context(context)?;
    println!("c05-4-repeated-abort-recovery iterations=8 status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn pending_fills_block_capture_until_both_complete() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut first_pending_stream = context.create_stream()?;
    let mut second_pending_stream = context.create_stream()?;
    let mut captured_stream = context.create_stream()?;
    let kernel = context.kernel();

    // Every pending fill reserves the same primary-context admission domain
    // before enqueue. Capture must stay out until all of their native buffers
    // are consumed, not merely until the first stream reports completion.
    let first = kernel.launch_fill(&mut first_pending_stream, 4_096, 6.75)?;
    let second = kernel.launch_fill(&mut second_pending_stream, 4_096, -6.75)?;
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while two pending smoke fills are live",
    );
    assert!(all_f32_bits_equal(&first.finish()?, 6.75));
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while one pending smoke fill remains live",
    );
    assert!(all_f32_bits_equal(&second.finish()?, -6.75));
    drop(kernel);

    captured_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    first_pending_stream.close()?;
    second_pending_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-pending-fill-admission-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn same_context_resource_drops_and_closes_are_deferred_until_abort() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut captured_stream = context.create_stream()?;
    let spare_stream = context.create_stream()?;
    let event = context.create_event()?;
    let device_buffer = context.allocate_device_buffer(256)?;
    let pinned_buffer = context.allocate_pinned_host_buffer(256)?;

    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;

    // The owning capture stream stays exclusively borrowed, but independent
    // native owners on the same thread must transfer their close into abort
    // cleanup rather than issue CUDA destruction during capture. Cover both
    // implicit Drop and the explicit close API.
    drop(device_buffer);
    drop(pinned_buffer);
    event.close()?;
    spare_stream.close()?;

    capture.abort()?;
    assert!(
        context.allocation_stats()?.is_zero(),
        "abort must drain every deferred same-context allocation"
    );

    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-deferred-same-context-resources status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn foreign_context_resource_drops_and_closes_survive_abort() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let capture_context = device.create_context()?;
    let mut captured_stream = capture_context.create_stream()?;
    let foreign_context = device.create_context()?;
    let foreign_stream = foreign_context.create_stream()?;
    let foreign_event = foreign_context.create_event()?;
    let foreign_device_buffer = foreign_context.allocate_device_buffer(256)?;
    let foreign_pinned_buffer = foreign_context.allocate_pinned_host_buffer(256)?;

    // Leave the resources as the only Rust owners of their foreign context.
    // Their destruction during capture must retain that Rust lease until the
    // native abort callback has destroyed the queued native owners.
    drop(foreign_context);
    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    drop(foreign_device_buffer);
    foreign_pinned_buffer.close()?;
    foreign_event.close()?;
    foreign_stream.close()?;
    capture.abort()?;

    captured_stream.close()?;
    close_context(capture_context)?;

    // A fresh primary-context lease proves the foreign-owner cleanup did not
    // leave the CUDA runtime in a capture-poisoned or child-leaked state.
    let recovery_context = device.create_context()?;
    let mut recovery_stream = recovery_context.create_stream()?;
    assert_eager_fill_after_recovery(&recovery_context, &mut recovery_stream, 1.75)?;
    recovery_stream.close()?;
    close_context(recovery_context)?;
    println!("c05-4-deferred-foreign-context-resources status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn bare_foreign_context_close_is_deferred_until_abort() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let capture_context = device.create_context()?;
    let mut captured_stream = capture_context.create_stream()?;
    let foreign_context = device.create_context()?;

    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    foreign_context.close()?;
    capture.abort()?;

    captured_stream.close()?;
    close_context(capture_context)?;

    let recovery_context = device.create_context()?;
    let mut recovery_stream = recovery_context.create_stream()?;
    assert_eager_fill_after_recovery(&recovery_context, &mut recovery_stream, -1.75)?;
    recovery_stream.close()?;
    close_context(recovery_context)?;
    println!("c05-4-deferred-foreign-context-close status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn pending_copy_blocks_capture_until_consumed() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut copy_stream = context.create_stream()?;
    let mut captured_stream = context.create_stream()?;
    let mut device_buffer = context.allocate_device_buffer(256)?;
    let mut pinned_buffer = context.allocate_pinned_host_buffer(256)?;
    pinned_buffer.write(0, &[0x5a; 256])?;

    let pending =
        device_buffer.copy_from_pinned_async(0, &mut pinned_buffer, 0, 256, &mut copy_stream)?;
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while a pending pinned-host copy is live",
    );
    pending.synchronize()?;

    captured_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    device_buffer.close()?;
    pinned_buffer.close()?;
    copy_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-pending-copy-admission-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn zero_element_pending_fill_blocks_capture_until_consumed() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut pending_stream = context.create_stream()?;
    let mut captured_stream = context.create_stream()?;
    let kernel = context.kernel();

    let pending = kernel.launch_fill(&mut pending_stream, 0, 0.0)?;
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while a zero-element pending fill is live",
    );
    assert!(
        pending.finish()?.is_empty(),
        "zero-element fill must preserve its empty output contract"
    );
    drop(kernel);

    captured_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    pending_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-zero-element-fill-admission-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn same_thread_capture_blocks_context_and_foreign_stream_cuda_work() -> Result<(), Box<dyn Error>> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut captured_stream = context.create_stream()?;
    let mut foreign_stream = context.create_stream()?;
    let mut third_stream = context.create_stream()?;

    // Nested batches remain supported across streams, but their thread-local
    // count must keep any third-stream capture out until every batch ends.
    let first_batch = foreign_stream.begin_command_batch()?;
    let second_batch = captured_stream.begin_command_batch()?;
    assert_invalid_state(
        third_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while two foreign stream command batches are active",
    );
    second_batch.finish()?;
    assert_invalid_state(
        third_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while one foreign stream command batch remains active",
    );
    drop(first_batch);
    third_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;

    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;

    // ThreadLocal capture must reject all potentially unsafe CUDA calls issued
    // from its owning host thread, even when they target a distinct stream.
    assert_invalid_state(CudaRuntime::initialize(), "runtime device-count query");
    assert_invalid_state(runtime.device(0), "device-properties query");
    assert_invalid_state(context.synchronize(), "context synchronize");
    assert_invalid_state(context.memory_info(), "context memory-info query");
    assert_invalid_state(context.create_stream(), "secondary stream create");
    assert_invalid_state(context.create_event(), "event create");
    assert_invalid_state(
        context.allocate_device_buffer(256),
        "device-buffer allocation",
    );
    assert_invalid_state(device.create_context(), "secondary context create");
    assert_invalid_state(
        foreign_stream.begin_command_batch(),
        "foreign stream command-batch begin",
    );
    assert_invalid_state(
        foreign_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "foreign stream capture begin",
    );
    let kernel = context.kernel();
    assert_invalid_state(
        kernel.launch_fill(&mut foreign_stream, 256, 1.5),
        "foreign stream eager fill",
    );
    drop(kernel);

    capture.abort()?;
    assert_eager_fill_after_recovery(&context, &mut captured_stream, 4.5)?;
    assert_eager_fill_after_recovery(&context, &mut foreign_stream, -4.5)?;

    third_stream.close()?;
    foreign_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-thread-local-gate-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn cross_thread_context_controls_are_rejected_while_capturing() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    // A second ABI wrapper retains the same CUDA primary context. It proves
    // the gate is primary-context scoped instead of just another TLS check on
    // the stream-owning wrapper.
    let secondary_context = device.create_context()?;
    let mut stream = context.create_stream()?;

    let capture = stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            // Stream creation remains independent across host threads, but a
            // diagnostic pending fill carries a future synchronization/free
            // obligation. Its admission must reject before CUDA enqueue while
            // the primary context is capturing.
            let mut worker_stream = secondary_context
                .create_stream()
                .expect("cross-thread independent stream creation must succeed");
            let kernel = secondary_context.kernel();
            assert_invalid_state(
                kernel.launch_fill(&mut worker_stream, 4_096, -2.25),
                "cross-thread pending smoke-fill launch",
            );
            drop(kernel);
            worker_stream
                .close()
                .expect("cross-thread independent stream close must succeed");

            assert_invalid_state(
                secondary_context.synchronize(),
                "cross-thread context synchronize",
            );
            assert_invalid_state(
                secondary_context.memory_info(),
                "cross-thread context memory-info query",
            );
            assert_invalid_state(
                device.create_context(),
                "cross-thread primary-context retain",
            );
        });
        worker
            .join()
            .expect("cross-thread context-control test worker must not panic");
    });

    capture.abort()?;
    let (_free_bytes, total_bytes) = secondary_context.memory_info()?;
    assert!(
        total_bytes > 0,
        "recovered primary context must be queryable"
    );
    secondary_context.synchronize()?;
    assert_eager_fill_after_recovery(&context, &mut stream, 8.5)?;
    let mut recovery_stream = secondary_context.create_stream()?;
    assert_eager_fill_after_recovery(&secondary_context, &mut recovery_stream, -2.25)?;
    recovery_stream.close()?;

    stream.close()?;
    secondary_context.close()?;
    close_context(context)?;
    println!("c05-4-cross-thread-context-gate-recovery status=passed");
    Ok(())
}
