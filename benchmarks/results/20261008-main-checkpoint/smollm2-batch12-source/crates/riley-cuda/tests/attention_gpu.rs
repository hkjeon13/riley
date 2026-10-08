#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    AvGqaParams, CausalSoftmaxInPlaceParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext,
    CudaDType, CudaDevice, CudaDeviceBuffer, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    QkGqaParams, ScaleCausalMaskInPlaceParams, av_gqa, causal_softmax_in_place, qk_gqa,
    scale_causal_mask_in_place,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const EXACT_SEQUENCE_LENGTH: usize = 5;
const CAUSAL_SEQUENCE_LENGTHS: &[usize] = &[1, 2, 7, 31, 32, 33];
const QH: usize = 6;
const KVH: usize = 2;
const D: usize = 4;
const SCALE: f32 = 0.125;
const FUTURE_MASK_BITS: u16 = 0xff7f;

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

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32_to_bf16_bits(value).to_ne_bytes())
        .collect()
}

fn decode_bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn bf16_ulp_key(bits: u16) -> i32 {
    if bits & 0x8000 == 0 {
        i32::from(bits) + 0x8000
    } else {
        i32::from(!bits)
    }
}

fn assert_bf16_ulps(actual: &[u16], expected: &[u16], maximum_ulps: i32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let distance = (bf16_ulp_key(actual) - bf16_ulp_key(expected)).abs();
        assert!(
            distance <= maximum_ulps,
            "{label}[{index}] differs by {distance} BF16 ULPs: expected 0x{expected:04x} ({:?}), got 0x{actual:04x} ({:?})",
            bf16_to_f32(expected),
            bf16_to_f32(actual)
        );
    }
}

fn q_index(token: usize, head: usize, depth: usize) -> usize {
    (token * QH + head) * D + depth
}

fn kv_index(token: usize, head: usize, depth: usize) -> usize {
    (token * KVH + head) * D + depth
}

fn score_index(token_count: usize, head: usize, query_token: usize, key_token: usize) -> usize {
    (head * token_count + query_token) * token_count + key_token
}

fn cpu_qk(query: &[u16], key: &[u16], token_count: usize) -> Vec<u16> {
    let mut scores = vec![0_u16; QH * token_count * token_count];
    let group_size = QH / KVH;
    for query_head in 0..QH {
        let key_value_head = query_head / group_size;
        for query_token in 0..token_count {
            for key_token in 0..token_count {
                let mut accumulator = 0.0_f32;
                for depth in 0..D {
                    accumulator = bf16_to_f32(query[q_index(query_token, query_head, depth)])
                        .mul_add(
                            bf16_to_f32(key[kv_index(key_token, key_value_head, depth)]),
                            accumulator,
                        );
                }
                scores[score_index(token_count, query_head, query_token, key_token)] =
                    f32_to_bf16_bits(accumulator);
            }
        }
    }
    scores
}

fn cpu_scale_mask(scores: &mut [u16], token_count: usize) {
    for head in 0..QH {
        for query_token in 0..token_count {
            for key_token in 0..token_count {
                let index = score_index(token_count, head, query_token, key_token);
                let scaled = f32_to_bf16_bits(bf16_to_f32(scores[index]) * SCALE);
                let mask = if key_token > query_token {
                    bf16_to_f32(FUTURE_MASK_BITS)
                } else {
                    0.0
                };
                scores[index] = f32_to_bf16_bits(bf16_to_f32(scaled) + mask);
            }
        }
    }
}

fn cpu_softmax(scores: &mut [u16], token_count: usize) {
    for head in 0..QH {
        for query_token in 0..token_count {
            let base = score_index(token_count, head, query_token, 0);
            let mut maximum = f32::NEG_INFINITY;
            for key_token in 0..token_count {
                maximum = maximum.max(bf16_to_f32(scores[base + key_token]));
            }
            let mut denominator = 0.0_f32;
            for key_token in 0..token_count {
                denominator += (bf16_to_f32(scores[base + key_token]) - maximum).exp();
            }
            for key_token in 0..token_count {
                let numerator = (bf16_to_f32(scores[base + key_token]) - maximum).exp();
                scores[base + key_token] = f32_to_bf16_bits(numerator / denominator);
            }
        }
    }
}

fn cpu_av(probabilities: &[u16], value: &[u16], token_count: usize) -> Vec<u16> {
    let mut output = vec![0_u16; token_count * QH * D];
    let group_size = QH / KVH;
    for query_token in 0..token_count {
        for query_head in 0..QH {
            let key_value_head = query_head / group_size;
            for depth in 0..D {
                let mut accumulator = 0.0_f32;
                for key_token in 0..token_count {
                    accumulator = bf16_to_f32(
                        probabilities[score_index(token_count, query_head, query_token, key_token)],
                    )
                    .mul_add(
                        bf16_to_f32(value[kv_index(key_token, key_value_head, depth)]),
                        accumulator,
                    );
                }
                output[q_index(query_token, query_head, depth)] = f32_to_bf16_bits(accumulator);
            }
        }
    }
    output
}

fn execute_attention(
    token_count: usize,
    query: &CudaDeviceBuffer,
    key: &CudaDeviceBuffer,
    value: &CudaDeviceBuffer,
    scores: &mut CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let score_bytes = scores.byte_len();
    let output_bytes = output.byte_len();
    {
        let mut params = QkGqaParams {
            query: CudaBufferSpan::new(query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(key, CudaDType::BF16, 0, key.byte_len())?,
            output: CudaBufferSpanMut::new(scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            key_value_head_count: u64::try_from(KVH)?,
            head_size: u64::try_from(D)?,
        };
        qk_gqa(&mut params, stream)?;
    }
    {
        let mut params = ScaleCausalMaskInPlaceParams {
            scores: CudaBufferSpanMut::new(scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            scale: SCALE,
        };
        scale_causal_mask_in_place(&mut params, stream)?;
    }
    {
        let mut params = CausalSoftmaxInPlaceParams {
            scores: CudaBufferSpanMut::new(scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
        };
        causal_softmax_in_place(&mut params, stream)?;
    }
    {
        let mut params = AvGqaParams {
            probabilities: CudaBufferSpan::new(scores, CudaDType::BF16, 0, score_bytes)?,
            value: CudaBufferSpan::new(value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            key_value_head_count: u64::try_from(KVH)?,
            head_size: u64::try_from(D)?,
        };
        av_gqa(&mut params, stream)?;
    }
    Ok(())
}

fn assert_causal_length_matches_cpu_reference(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    token_count: usize,
) -> TestResult {
    let query_host: Vec<f32> = (0..token_count * QH * D)
        .map(|index| (f32::from(u8::try_from((index * 11 + 5) % 29).unwrap_or(0)) - 14.0) * 0.0625)
        .collect();
    let key_host: Vec<f32> = (0..token_count * KVH * D)
        .map(|index| (f32::from(u8::try_from((index * 7 + 3) % 17).unwrap_or(0)) - 8.0) * 0.125)
        .collect();
    let value_host: Vec<f32> = (0..token_count * KVH * D)
        .map(|index| (f32::from(u8::try_from((index * 5 + 1) % 23).unwrap_or(0)) - 11.0) * 0.25)
        .collect();
    let query_bytes = encode_bf16(&query_host);
    let key_bytes = encode_bf16(&key_host);
    let value_bytes = encode_bf16(&value_host);
    let query_bits = decode_bf16_bits(&query_bytes);
    let key_bits = decode_bf16_bits(&key_bytes);
    let value_bits = decode_bf16_bits(&value_bytes);

    let query = upload(context, stream, staging, &query_bytes)?;
    let key = upload(context, stream, staging, &key_bytes)?;
    let value = upload(context, stream, staging, &value_bytes)?;
    let score_bytes = u64::try_from(QH * token_count * token_count * 2)?;
    let output_bytes = u64::try_from(token_count * QH * D * 2)?;
    let mut scores = context.allocate_device_buffer(score_bytes)?;
    let mut output = context.allocate_device_buffer(output_bytes)?;

    let expected_qk = cpu_qk(&query_bits, &key_bits, token_count);
    {
        let mut params = QkGqaParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            output: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            key_value_head_count: u64::try_from(KVH)?,
            head_size: u64::try_from(D)?,
        };
        qk_gqa(&mut params, stream)?;
    }
    let actual_qk = decode_bf16_bits(&download(context, stream, &mut scores)?);
    assert_eq!(
        actual_qk, expected_qk,
        "raw QK BF16 checkpoint at S={token_count}"
    );

    let mut expected_probabilities = expected_qk;
    cpu_scale_mask(&mut expected_probabilities, token_count);
    {
        let mut params = ScaleCausalMaskInPlaceParams {
            scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            scale: SCALE,
        };
        scale_causal_mask_in_place(&mut params, stream)?;
    }
    let actual_masked = decode_bf16_bits(&download(context, stream, &mut scores)?);
    assert_eq!(
        actual_masked, expected_probabilities,
        "scale/mask BF16 checkpoint at S={token_count}"
    );
    for head in 0..QH {
        for query_token in 0..token_count {
            for key_token in query_token + 1..token_count {
                assert_eq!(
                    actual_masked[score_index(token_count, head, query_token, key_token)],
                    FUTURE_MASK_BITS,
                    "future mask bits at S={token_count}, h={head}, q={query_token}, k={key_token}"
                );
            }
        }
    }

    cpu_softmax(&mut expected_probabilities, token_count);
    {
        let mut params = CausalSoftmaxInPlaceParams {
            scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
        };
        causal_softmax_in_place(&mut params, stream)?;
    }
    let actual_probabilities = decode_bf16_bits(&download(context, stream, &mut scores)?);
    assert_bf16_ulps(
        &actual_probabilities,
        &expected_probabilities,
        1,
        &format!("softmax probability at S={token_count}"),
    );
    for head in 0..QH {
        for query_token in 0..token_count {
            for key_token in query_token + 1..token_count {
                assert_eq!(
                    actual_probabilities[score_index(token_count, head, query_token, key_token,)],
                    0,
                    "future probability at S={token_count}, h={head}, q={query_token}, k={key_token}"
                );
            }
        }
    }

    let expected_output = cpu_av(&actual_probabilities, &value_bits, token_count);
    {
        let mut params = AvGqaParams {
            probabilities: CudaBufferSpan::new(&scores, CudaDType::BF16, 0, score_bytes)?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            key_value_head_count: u64::try_from(KVH)?,
            head_size: u64::try_from(D)?,
        };
        av_gqa(&mut params, stream)?;
    }
    let actual_output = decode_bf16_bits(&download(context, stream, &mut output)?);
    assert_eq!(
        actual_output, expected_output,
        "AV BF16 checkpoint at S={token_count}"
    );

    query.close()?;
    key.close()?;
    value.close()?;
    scores.close()?;
    output.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn materialized_gqa_matches_the_staged_bf16_contract_without_allocating() -> TestResult {
    let token_count = EXACT_SEQUENCE_LENGTH;
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let query_host: Vec<f32> = (0..token_count * QH * D)
        .map(|index| (f32::from(u8::try_from(index % 19).unwrap_or(0)) - 9.0) * 0.125)
        .collect();
    let key_host: Vec<f32> = (0..token_count * KVH * D)
        .map(|index| (f32::from(u8::try_from((index * 7 + 3) % 17).unwrap_or(0)) - 8.0) * 0.0625)
        .collect();
    let value_host: Vec<f32> = (0..token_count * KVH * D)
        .map(|index| (f32::from(u8::try_from((index * 5 + 1) % 23).unwrap_or(0)) - 11.0) * 0.25)
        .collect();
    let query_bytes = encode_bf16(&query_host);
    let key_bytes = encode_bf16(&key_host);
    let value_bytes = encode_bf16(&value_host);
    let query_bits = decode_bf16_bits(&query_bytes);
    let key_bits = decode_bf16_bits(&key_bytes);
    let value_bits = decode_bf16_bits(&value_bytes);

    let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
    let key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
    let value = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let score_bytes = u64::try_from(QH * token_count * token_count * 2)?;
    let output_bytes = u64::try_from(token_count * QH * D * 2)?;
    let mut scores = context.allocate_device_buffer(score_bytes)?;
    let mut output = context.allocate_device_buffer(output_bytes)?;

    let expected_qk = cpu_qk(&query_bits, &key_bits, token_count);
    {
        let mut params = QkGqaParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            output: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            key_value_head_count: u64::try_from(KVH)?,
            head_size: u64::try_from(D)?,
        };
        qk_gqa(&mut params, &mut stream)?;
    }
    let actual_qk = decode_bf16_bits(&download(&context, &mut stream, &mut scores)?);
    assert_eq!(actual_qk, expected_qk, "raw QK BF16 checkpoint");

    let mut expected_probabilities = expected_qk;
    cpu_scale_mask(&mut expected_probabilities, token_count);
    {
        let mut params = ScaleCausalMaskInPlaceParams {
            scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            scale: SCALE,
        };
        scale_causal_mask_in_place(&mut params, &mut stream)?;
    }
    let actual_masked = decode_bf16_bits(&download(&context, &mut stream, &mut scores)?);
    assert_eq!(
        actual_masked, expected_probabilities,
        "scale/mask BF16 checkpoint"
    );
    for head in 0..QH {
        for query_token in 0..token_count {
            for key_token in query_token + 1..token_count {
                assert_eq!(
                    actual_masked[score_index(token_count, head, query_token, key_token)],
                    FUTURE_MASK_BITS,
                    "future mask bits at h={head}, q={query_token}, k={key_token}"
                );
            }
        }
    }

    cpu_softmax(&mut expected_probabilities, token_count);
    {
        let mut params = CausalSoftmaxInPlaceParams {
            scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
        };
        causal_softmax_in_place(&mut params, &mut stream)?;
    }
    let actual_probabilities = decode_bf16_bits(&download(&context, &mut stream, &mut scores)?);
    assert_bf16_ulps(
        &actual_probabilities,
        &expected_probabilities,
        1,
        "softmax probability",
    );
    for head in 0..QH {
        for query_token in 0..token_count {
            for key_token in query_token + 1..token_count {
                assert_eq!(
                    actual_probabilities[score_index(token_count, head, query_token, key_token,)],
                    0,
                    "future probability at h={head}, q={query_token}, k={key_token}"
                );
            }
        }
    }

    let expected_output = cpu_av(&actual_probabilities, &value_bits, token_count);
    {
        let mut params = AvGqaParams {
            probabilities: CudaBufferSpan::new(&scores, CudaDType::BF16, 0, score_bytes)?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
            token_count: u64::try_from(token_count)?,
            query_head_count: u64::try_from(QH)?,
            key_value_head_count: u64::try_from(KVH)?,
            head_size: u64::try_from(D)?,
        };
        av_gqa(&mut params, &mut stream)?;
    }
    let actual_output = decode_bf16_bits(&download(&context, &mut stream, &mut output)?);
    assert_eq!(actual_output, expected_output, "AV BF16 checkpoint");

    let before = context.allocation_stats()?;
    let baseline_output = actual_output;
    for _ in 0..16 {
        execute_attention(
            token_count,
            &query,
            &key,
            &value,
            &mut scores,
            &mut output,
            &mut stream,
        )?;
    }
    let after = context.allocation_stats()?;
    assert_eq!(before, after, "attention execution changed allocations");
    let repeated_output = decode_bf16_bits(&download(&context, &mut stream, &mut output)?);
    assert_eq!(
        repeated_output, baseline_output,
        "attention is not repeatable"
    );

    query.close()?;
    key.close()?;
    value.close()?;
    scores.close()?;
    output.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn causal_mask_matches_staged_bf16_reference_across_sequence_lengths() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;
    let baseline = context.allocation_stats()?;

    for &token_count in CAUSAL_SEQUENCE_LENGTHS {
        assert_causal_length_matches_cpu_reference(
            &context,
            &mut stream,
            &mut staging,
            token_count,
        )?;
        assert_eq!(
            context.allocation_stats()?,
            baseline,
            "S={token_count} leaked a CUDA allocation"
        );
    }

    staging.close()?;
    stream.close()?;
    close_context(context)
}
