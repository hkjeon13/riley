use std::error::Error;

use riley_cuda::{
    BF16_LOGPROB_SUMMARY_WORDS, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaPinnedHostBuffer, CudaRuntime, CudaStream, bf16_logprob_summary,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

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

/// The host's `approximate_exp` (riley-runtime sampling.rs), operation for operation.
fn approximate_exp(x: f32) -> f32 {
    let scaled = x.max(-87.0) * std::f32::consts::LOG2_E;
    let whole = scaled.floor();
    let part = scaled - whole;
    let mantissa = 1.0
        + part
            * (0.693_147_2
                + part * (0.240_226_5 + part * (0.055_504_11 + part * (0.009_618_13 + part * 0.001_333_55))));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let power = f32::from_bits((((whole as i32) + 127) as u32) << 23);
    mantissa * power
}

fn expected_record(row: &[u16], addressable: usize, selected: u32, top_count: usize) -> Vec<u32> {
    let values: Vec<f32> = row[..addressable].iter().map(|&bits| f32::from_bits(u32::from(bits) << 16)).collect();
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut lanes = [0.0_f32; 16];
    for chunk in values.chunks(16) {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane += approximate_exp(*value - maximum);
        }
    }
    let mut order: Vec<u32> = (0..u32::try_from(addressable).unwrap()).collect();
    order.sort_by(|&a, &b| {
        values[b as usize].partial_cmp(&values[a as usize]).unwrap().then(a.cmp(&b))
    });
    let in_range = (selected as usize) < addressable;
    let mut words = vec![u32::from(in_range), maximum.to_bits()];
    words.extend(lanes.iter().map(|lane| lane.to_bits()));
    words.push(if in_range { values[selected as usize].to_bits() } else { 0 });
    for &token in &order[..top_count] {
        words.extend([token, values[token as usize].to_bits()]);
    }
    words
}

#[test]
#[ignore = "remote GPU"]
fn logprob_summaries_reproduce_the_host_lane_sums_and_top_tokens() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    for (rows, vocabulary, addressable, top_count) in
        [(5_usize, 151_936_usize, 151_665_usize, 5_usize), (3, 4099, 4099, 20), (2, 257, 250, 0), (33, 49_152, 49_152, 1)]
    {
        // Coarse values give many exact ties, including ±0.
        let mut state = 0x9e37_79b9_u32 ^ u32::try_from(vocabulary)?;
        let mut logits = Vec::with_capacity(rows * vocabulary);
        for _ in 0..rows * vocabulary {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = (f64::from(state % 2001) - 1000.0) * 0.02;
            logits.push(((value as f32).to_bits() >> 16) as u16);
        }
        logits[3] = 0x8000;
        logits[4] = 0x0000;
        let selected: Vec<u32> = (0..rows).map(|row| u32::try_from((row * 7919 + 11) % vocabulary).unwrap()).collect();
        let greedy: Vec<u32> = selected.iter().flat_map(|&token| [token, 0]).collect();
        let logit_bytes: Vec<u8> = logits.iter().flat_map(|bits| bits.to_ne_bytes()).collect();
        let greedy_bytes: Vec<u8> = greedy.iter().flat_map(|word| word.to_ne_bytes()).collect();
        let record_bytes = BF16_LOGPROB_SUMMARY_WORDS * 4;
        let mut staging = context.allocate_pinned_host_buffer(u64::try_from(logit_bytes.len())?)?;
        let logits_device = upload(&context, &mut stream, &mut staging, &logit_bytes)?;
        let greedy_device = upload(&context, &mut stream, &mut staging, &greedy_bytes)?;
        // One spare record checks that nothing is written past row_count.
        let mut summaries = upload(&context, &mut stream, &mut staging, &vec![0xa5; (rows + 1) * record_bytes])?;
        {
            let logits_span = CudaBufferSpan::new(&logits_device, CudaDType::BF16, 0, logits_device.byte_len())?;
            let greedy_span = CudaBufferSpan::new(&greedy_device, CudaDType::U32, 0, greedy_device.byte_len())?;
            let mut summary_span =
                CudaBufferSpanMut::new(&mut summaries, CudaDType::U32, 0, u64::try_from(rows * record_bytes)?)?;
            bf16_logprob_summary(
                &logits_span,
                &greedy_span,
                &mut summary_span,
                u64::try_from(rows)?,
                u64::try_from(vocabulary)?,
                u64::try_from(addressable)?,
                u32::try_from(top_count)?,
                &mut stream,
            )?;
        }
        let mut output = vec![0_u8; (rows + 1) * record_bytes];
        summaries.download_to_slice(0, &mut output, &mut staging, &mut stream)?;
        for row in 0..rows {
            let expected = expected_record(
                &logits[row * vocabulary..(row + 1) * vocabulary],
                addressable,
                selected[row],
                top_count,
            );
            let actual: Vec<u32> = output[row * record_bytes..row * record_bytes + expected.len() * 4]
                .chunks_exact(4)
                .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(actual, expected, "rows {rows} vocabulary {vocabulary} row {row}");
        }
        assert!(output[rows * record_bytes..].iter().all(|&byte| byte == 0xa5));
        staging.close()?;
        logits_device.close()?;
        greedy_device.close()?;
        summaries.close()?;
    }
    stream.close()?;
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
