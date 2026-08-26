#![allow(clippy::too_many_lines)]

use std::error::Error;

use rustinfer_cuda::{
    AttentionBackend, AttentionBackendAvailability, AttentionMask, AttentionPreference,
    AttentionReductionProfile, AvGqaParams, CausalSoftmaxInPlaceParams, CudaBufferSpan,
    CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaPinnedHostBuffer, CudaRuntime,
    CudaStream, PrefillAttentionParams, PrefillAttentionRequest, PreparedPrefillAttention,
    QkGqaParams, fixed37_av_gqa, fixed37_causal_softmax_in_place, fixed37_qk_gqa,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const D: usize = 64;
const SCALE: f32 = 0.125;

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    Ok((context, stream))
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
    if is_nan {
        0x7fff
    } else {
        let tie = (bits >> 16) & 1;
        u16::try_from(bits.wrapping_add(0x7fff + tie) >> 16).unwrap_or(0x7fff)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn round_bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_bits(value))
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32_to_bf16_bits(value).to_ne_bytes())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]])))
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

fn balanced_sum(mut partials: Vec<f32>) -> f32 {
    while partials.len() > 1 {
        let mut next = partials
            .chunks_exact(2)
            .map(|pair| pair[0] + pair[1])
            .collect::<Vec<_>>();
        if partials.len() % 2 != 0 {
            next.push(*partials.last().expect("non-empty fixed37 partials"));
        }
        partials = next;
    }
    partials[0]
}

fn balanced_max(mut partials: Vec<f32>) -> f32 {
    while partials.len() > 1 {
        let mut next = partials
            .chunks_exact(2)
            .map(|pair| pair[0].max(pair[1]))
            .collect::<Vec<_>>();
        if partials.len() % 2 != 0 {
            next.push(*partials.last().expect("non-empty fixed37 partials"));
        }
        partials = next;
    }
    partials[0]
}

fn fixed37_sum(values: &[f32]) -> f32 {
    balanced_sum(
        values
            .chunks(37)
            .map(|chunk| {
                chunk
                    .iter()
                    .copied()
                    .fold(0.0_f32, |sum, value| sum + value)
            })
            .collect(),
    )
}

fn fixed37_max(values: &[f32]) -> f32 {
    balanced_max(
        values
            .chunks(37)
            .map(|chunk| chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            .collect(),
    )
}

fn fixed37_dot(left: &[f32], right: &[f32]) -> f32 {
    let partials = left
        .chunks(37)
        .zip(right.chunks(37))
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .fold(0.0_f32, |sum, (&left, &right)| left.mul_add(right, sum))
        })
        .collect();
    balanced_sum(partials)
}

fn q_index(sequence: usize, token: usize, head: usize, depth: usize, heads: usize) -> usize {
    let _ = sequence;
    ((token * heads + head) * D) + depth
}

fn kv_index(sequence: usize, token: usize, head: usize, depth: usize, heads: usize) -> usize {
    let _ = sequence;
    ((token * heads + head) * D) + depth
}

fn visible(mask: AttentionMask, query: usize, key: usize) -> bool {
    if key > query {
        return false;
    }
    match mask {
        AttentionMask::Causal => true,
        AttentionMask::CausalLocal { window } => {
            window != 0 && u64::try_from(query - key).is_ok_and(|distance| distance < window)
        }
        _ => false,
    }
}

fn fixed37_attention_oracle(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
    mask: AttentionMask,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; sequence * query_heads * D];
    let group = query_heads / key_value_heads;
    let finite_min = f32::from_bits(0xff7f_0000);
    for query_token in 0..sequence {
        for query_head in 0..query_heads {
            let key_value_head = query_head / group;
            let query_base = q_index(sequence, query_token, query_head, 0, query_heads);
            let mut scores = Vec::with_capacity(sequence);
            for key_token in 0..sequence {
                let key_base = kv_index(sequence, key_token, key_value_head, 0, key_value_heads);
                let dot = fixed37_dot(
                    &query[query_base..query_base + D],
                    &key[key_base..key_base + D],
                );
                let raw = round_bf16(dot);
                let scaled = round_bf16(raw * SCALE);
                let mask_value = if visible(mask, query_token, key_token) {
                    0.0
                } else {
                    finite_min
                };
                scores.push(round_bf16(scaled + mask_value));
            }
            let maximum = fixed37_max(&scores);
            let exponentials = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = fixed37_sum(&exponentials);
            let probabilities = exponentials
                .iter()
                .map(|value| round_bf16(*value / denominator))
                .collect::<Vec<_>>();
            for depth in 0..D {
                let values = (0..sequence)
                    .map(|key_token| {
                        value[kv_index(sequence, key_token, key_value_head, depth, key_value_heads)]
                    })
                    .collect::<Vec<_>>();
                output[q_index(sequence, query_token, query_head, depth, query_heads)] =
                    round_bf16(fixed37_dot(&probabilities, &values));
            }
        }
    }
    output
}

fn deterministic_inputs(
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let query = (0..sequence * query_heads * D)
        .map(|index| (f32::from(u8::try_from((index * 11 + 3) % 31).unwrap_or(0)) - 15.0) * 0.03125)
        .collect();
    let key = (0..sequence * key_value_heads * D)
        .map(|index| (f32::from(u8::try_from((index * 7 + 5) % 29).unwrap_or(0)) - 14.0) * 0.03125)
        .collect();
    let value = (0..sequence * key_value_heads * D)
        .map(|index| (f32::from(u8::try_from((index * 5 + 1) % 23).unwrap_or(0)) - 11.0) * 0.0625)
        .collect();
    (query, key, value)
}

#[allow(clippy::too_many_arguments)]
fn execute_prefill(
    prepared: &PreparedPrefillAttention,
    query: &CudaDeviceBuffer,
    key: &CudaDeviceBuffer,
    value: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    stream: &mut CudaStream,
) -> TestResult {
    let output_bytes = output.byte_len();
    let workspace = match workspace {
        Some(workspace) => {
            let bytes = workspace.byte_len();
            Some(CudaBufferSpanMut::new(
                workspace,
                CudaDType::BF16,
                0,
                bytes,
            )?)
        }
        None => None,
    };
    let mut params = PrefillAttentionParams {
        query: CudaBufferSpan::new(query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(key, CudaDType::BF16, 0, key.byte_len())?,
        value: CudaBufferSpan::new(value, CudaDType::BF16, 0, value.byte_len())?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_bytes)?,
        workspace,
    };
    prepared.execute(&mut params, stream)?;
    Ok(())
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn fixed37_materialized_two_pass_local_and_cpu_oracle_agree() -> TestResult {
    const S: usize = 75;
    const QH: usize = 9;
    const KVH: usize = 3;
    let (context, mut stream) = first_context()?;
    let (query_host, key_host, value_host) = deterministic_inputs(S, QH, KVH);
    let query_bytes = encode_bf16(&query_host);
    let key_bytes = encode_bf16(&key_host);
    let value_bytes = encode_bf16(&value_host);
    let query_bf16 = decode_bf16(&query_bytes);
    let key_bf16 = decode_bf16(&key_bytes);
    let value_bf16 = decode_bf16(&value_bytes);
    let staging_len = u64::try_from(
        query_bytes
            .len()
            .max(key_bytes.len())
            .max(value_bytes.len()),
    )?;
    let mut staging = context.allocate_pinned_host_buffer(staging_len)?;
    let mut query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
    let mut key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
    let mut value = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let mut materialized_output =
        context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;
    let mut two_pass_output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;
    let mut local_output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;

    let causal_request = PrefillAttentionRequest::new(
        1,
        S as u64,
        QH as u64,
        KVH as u64,
        D as u64,
        SCALE,
        AttentionMask::Causal,
    );
    let materialized = PreparedPrefillAttention::select_with_reduction_profile(
        &context,
        causal_request,
        AttentionPreference::Reference,
        AttentionReductionProfile::FixedContiguous37BalancedV1,
        AttentionBackendAvailability::linked(),
    )?;
    let two_pass = PreparedPrefillAttention::select_with_reduction_profile(
        &context,
        causal_request,
        AttentionPreference::Optimized,
        AttentionReductionProfile::FixedContiguous37BalancedV1,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(
        materialized.backend(),
        AttentionBackend::Fixed37Materialized
    );
    assert_eq!(two_pass.backend(), AttentionBackend::Fixed37TwoPass);
    let mut workspace = context.allocate_device_buffer(materialized.workspace_bytes())?;

    let before = context.allocation_stats()?;
    for _ in 0..3 {
        execute_prefill(
            &materialized,
            &query,
            &key,
            &value,
            &mut materialized_output,
            Some(&mut workspace),
            &mut stream,
        )?;
        execute_prefill(
            &two_pass,
            &query,
            &key,
            &value,
            &mut two_pass_output,
            None,
            &mut stream,
        )?;
    }
    assert_eq!(context.allocation_stats()?, before);
    let materialized_bytes = download(&context, &mut stream, &mut materialized_output)?;
    let two_pass_bytes = download(&context, &mut stream, &mut two_pass_output)?;
    assert_eq!(
        materialized_bytes, two_pass_bytes,
        "fixed37 two-pass must byte-match fixed37 materialized causal attention"
    );

    let causal_expected = fixed37_attention_oracle(
        &query_bf16,
        &key_bf16,
        &value_bf16,
        S,
        QH,
        KVH,
        AttentionMask::Causal,
    );
    for (index, (&actual, &expected)) in decode_bf16(&two_pass_bytes)
        .iter()
        .zip(&causal_expected)
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 0.015625,
            "causal output[{index}] expected {expected}, got {actual}"
        );
    }

    let local_max = PreparedPrefillAttention::select_with_reduction_profile(
        &context,
        PrefillAttentionRequest::new(
            1,
            S as u64,
            QH as u64,
            KVH as u64,
            D as u64,
            SCALE,
            AttentionMask::CausalLocal { window: u64::MAX },
        ),
        AttentionPreference::Optimized,
        AttentionReductionProfile::FixedContiguous37BalancedV1,
        AttentionBackendAvailability::linked(),
    )?;
    execute_prefill(
        &local_max,
        &query,
        &key,
        &value,
        &mut local_output,
        None,
        &mut stream,
    )?;
    let local_max_bytes = download(&context, &mut stream, &mut local_output)?;
    assert_eq!(
        local_max_bytes, two_pass_bytes,
        "u64::MAX local window must be exactly causal without overflow"
    );

    let local37 = PreparedPrefillAttention::select_with_reduction_profile(
        &context,
        PrefillAttentionRequest::new(
            1,
            S as u64,
            QH as u64,
            KVH as u64,
            D as u64,
            SCALE,
            AttentionMask::CausalLocal { window: 37 },
        ),
        AttentionPreference::Optimized,
        AttentionReductionProfile::FixedContiguous37BalancedV1,
        AttentionBackendAvailability::linked(),
    )?;
    execute_prefill(
        &local37,
        &query,
        &key,
        &value,
        &mut local_output,
        None,
        &mut stream,
    )?;
    let local_actual = decode_bf16(&download(&context, &mut stream, &mut local_output)?);
    let local_expected = fixed37_attention_oracle(
        &query_bf16,
        &key_bf16,
        &value_bf16,
        S,
        QH,
        KVH,
        AttentionMask::CausalLocal { window: 37 },
    );
    for (index, (&actual, &expected)) in local_actual.iter().zip(&local_expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 0.015625,
            "local37 output[{index}] expected {expected}, got {actual}"
        );
    }

    local_output.close()?;
    two_pass_output.close()?;
    materialized_output.close()?;
    workspace.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    staging.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn fixed37_two_pass_short_sequences_reserve_both_partial_arrays() -> TestResult {
    let (context, mut stream) = first_context()?;
    let maximum_bytes = 37 * D * 2;
    let mut staging = context.allocate_pinned_host_buffer(maximum_bytes as u64)?;

    for (sequence, expected_shared_bytes) in [(1_usize, 20_u64), (36, 160), (37, 164)] {
        let zeros = vec![0.0_f32; sequence * D];
        let ones = vec![1.0_f32; sequence * D];
        let zero_bytes = encode_bf16(&zeros);
        let one_bytes = encode_bf16(&ones);
        let mut query = upload(&context, &mut stream, &mut staging, &zero_bytes)?;
        let mut key = upload(&context, &mut stream, &mut staging, &zero_bytes)?;
        let mut value = upload(&context, &mut stream, &mut staging, &one_bytes)?;
        let mut output = context.allocate_device_buffer(u64::try_from(zero_bytes.len())?)?;
        let prepared = PreparedPrefillAttention::select_with_reduction_profile(
            &context,
            PrefillAttentionRequest::new(
                1,
                sequence as u64,
                1,
                1,
                D as u64,
                SCALE,
                AttentionMask::Causal,
            ),
            AttentionPreference::Optimized,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            AttentionBackendAvailability::linked(),
        )?;
        assert_eq!(prepared.backend(), AttentionBackend::Fixed37TwoPass);
        assert_eq!(
            prepared.selection_trace().dynamic_shared_memory_bytes(),
            expected_shared_bytes
        );
        execute_prefill(
            &prepared,
            &query,
            &key,
            &value,
            &mut output,
            None,
            &mut stream,
        )?;
        let actual = download(&context, &mut stream, &mut output)?;
        for query_token in 0..sequence {
            let probability = round_bf16(1.0 / (query_token + 1) as f32);
            let mut contributions = vec![0.0_f32; sequence];
            contributions[..=query_token].fill(probability);
            let expected = f32_to_bf16_bits(fixed37_sum(&contributions));
            for depth in 0..D {
                let offset = (query_token * D + depth) * 2;
                assert_eq!(
                    u16::from_ne_bytes([actual[offset], actual[offset + 1]]),
                    expected,
                    "S={sequence} row={query_token} depth={depth}"
                );
            }
        }
        output.close()?;
        value.close()?;
        key.close()?;
        query.close()?;
    }

    staging.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn fixed37_qk_and_av_order_witnesses_survive_bf16_rounding() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut qk_left = vec![1.0_f32; D];
    qk_left[0] = 16_777_216.0;
    qk_left[D - 1] = -16_777_216.0;
    let qk_right = vec![1.0_f32; D];
    let fixed_qk = fixed37_dot(&qk_left, &qk_right);
    let flat_qk = qk_left
        .iter()
        .zip(&qk_right)
        .fold(0.0_f32, |sum, (&left, &right)| left.mul_add(right, sum));
    assert_ne!(f32_to_bf16_bits(fixed_qk), f32_to_bf16_bits(flat_qk));
    let mut staging = context.allocate_pinned_host_buffer((74 * 74 * 2) as u64)?;
    let mut query = upload(&context, &mut stream, &mut staging, &encode_bf16(&qk_left))?;
    let mut key = upload(&context, &mut stream, &mut staging, &encode_bf16(&qk_right))?;
    let mut scores = context.allocate_device_buffer(2)?;
    let mut qk_params = QkGqaParams {
        query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
        output: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, 2)?,
        token_count: 1,
        query_head_count: 1,
        key_value_head_count: 1,
        head_size: D as u64,
    };
    fixed37_qk_gqa(&mut qk_params, &mut stream)?;
    let qk_bytes = download(&context, &mut stream, &mut scores)?;
    let qk_bits = u16::from_ne_bytes([qk_bytes[0], qk_bytes[1]]);
    assert_eq!(qk_bits, f32_to_bf16_bits(fixed_qk));

    const S: usize = 74;
    let mut probabilities_host = vec![1.0_f32; S];
    probabilities_host[0] = 16_777_216.0;
    probabilities_host[S - 1] = -16_777_216.0;
    let values_host = vec![1.0_f32; S];
    let fixed_av = fixed37_dot(&probabilities_host, &values_host);
    let flat_av = probabilities_host
        .iter()
        .zip(&values_host)
        .fold(0.0_f32, |sum, (&left, &right)| left.mul_add(right, sum));
    assert_ne!(f32_to_bf16_bits(fixed_av), f32_to_bf16_bits(flat_av));
    let probabilities_matrix = probabilities_host.repeat(S);
    let mut probabilities = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&probabilities_matrix),
    )?;
    let mut values = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&values_host),
    )?;
    let mut output = context.allocate_device_buffer((S * 2) as u64)?;
    let mut av_params = AvGqaParams {
        probabilities: CudaBufferSpan::new(
            &probabilities,
            CudaDType::BF16,
            0,
            probabilities.byte_len(),
        )?,
        value: CudaBufferSpan::new(&values, CudaDType::BF16, 0, values.byte_len())?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, (S * 2) as u64)?,
        token_count: S as u64,
        query_head_count: 1,
        key_value_head_count: 1,
        head_size: 1,
    };
    fixed37_av_gqa(&mut av_params, &mut stream)?;
    let av_bytes = download(&context, &mut stream, &mut output)?;
    let av_bits = u16::from_ne_bytes([av_bytes[0], av_bytes[1]]);
    assert_eq!(av_bits, f32_to_bf16_bits(fixed_av));

    output.close()?;
    values.close()?;
    probabilities.close()?;
    scores.close()?;
    key.close()?;
    query.close()?;
    staging.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn fixed37_softmax_special_rows_are_complete_qnan_rows() -> TestResult {
    const S: usize = 4;
    const QH: usize = 1;
    let (context, mut stream) = first_context()?;
    let scores_host = [
        f32::NAN,
        0.0,
        0.0,
        0.0,
        f32::INFINITY,
        0.0,
        0.0,
        0.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        0.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let bytes = encode_bf16(&scores_host);
    let mut staging = context.allocate_pinned_host_buffer(u64::try_from(bytes.len())?)?;
    let mut scores = upload(&context, &mut stream, &mut staging, &bytes)?;
    let score_bytes = scores.byte_len();
    let mut params = CausalSoftmaxInPlaceParams {
        scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
        token_count: S as u64,
        query_head_count: QH as u64,
    };
    fixed37_causal_softmax_in_place(&mut params, &mut stream)?;
    let actual_bytes = download(&context, &mut stream, &mut scores)?;
    let actual_bits = actual_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    assert!(actual_bits[..12].iter().all(|&bits| bits == 0x7fff));
    assert_eq!(&actual_bits[12..], &[0x3f80, 0, 0, 0]);

    let probability_bytes = encode_bf16(&[1.0, 0.0, 1.0, 0.0]);
    let value_bytes = encode_bf16(&[1.0, f32::INFINITY]);
    let mut probabilities = upload(&context, &mut stream, &mut staging, &probability_bytes)?;
    let mut values = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let mut av_output = context.allocate_device_buffer(4)?;
    let mut av_params = AvGqaParams {
        probabilities: CudaBufferSpan::new(&probabilities, CudaDType::BF16, 0, 8)?,
        value: CudaBufferSpan::new(&values, CudaDType::BF16, 0, 4)?,
        output: CudaBufferSpanMut::new(&mut av_output, CudaDType::BF16, 0, 4)?,
        token_count: 2,
        query_head_count: 1,
        key_value_head_count: 1,
        head_size: 1,
    };
    fixed37_av_gqa(&mut av_params, &mut stream)?;
    let av_bytes = download(&context, &mut stream, &mut av_output)?;
    assert_eq!(u16::from_ne_bytes([av_bytes[0], av_bytes[1]]), 0x7fff);
    assert_eq!(u16::from_ne_bytes([av_bytes[2], av_bytes[3]]), 0x7fff);

    av_output.close()?;
    values.close()?;
    probabilities.close()?;
    scores.close()?;
    staging.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn fixed37_s8192_bound_and_softmax_order_witness() -> TestResult {
    const S: usize = 8192;
    const SOFTMAX_S: usize = 1024;
    let (context, mut stream) = first_context()?;
    let tensor_values = S * D;
    let zeros = vec![0.0_f32; tensor_values];
    let ones = vec![1.0_f32; tensor_values];
    let zero_bytes = encode_bf16(&zeros);
    let one_bytes = encode_bf16(&ones);
    let softmax_bytes = SOFTMAX_S * SOFTMAX_S * 2;
    let mut staging =
        context.allocate_pinned_host_buffer(u64::try_from(zero_bytes.len().max(softmax_bytes))?)?;
    let mut query = upload(&context, &mut stream, &mut staging, &zero_bytes)?;
    let mut key = upload(&context, &mut stream, &mut staging, &zero_bytes)?;
    let mut value = upload(&context, &mut stream, &mut staging, &one_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(zero_bytes.len())?)?;
    let prepared = PreparedPrefillAttention::select_with_reduction_profile(
        &context,
        PrefillAttentionRequest::new(1, S as u64, 1, 1, D as u64, SCALE, AttentionMask::Causal),
        AttentionPreference::Optimized,
        AttentionReductionProfile::FixedContiguous37BalancedV1,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), AttentionBackend::Fixed37TwoPass);
    assert_eq!(
        prepared.selection_trace().dynamic_shared_memory_bytes(),
        34_544
    );
    execute_prefill(
        &prepared,
        &query,
        &key,
        &value,
        &mut output,
        None,
        &mut stream,
    )?;
    let output_bytes = download(&context, &mut stream, &mut output)?;
    for query_token in [0_usize, 36, 37, S - 1] {
        let probability = round_bf16(1.0 / (query_token + 1) as f32);
        let mut contributions = vec![0.0_f32; S];
        contributions[..=query_token].fill(probability);
        let expected_bits = f32_to_bf16_bits(fixed37_sum(&contributions));
        for depth in [0_usize, 17, D - 1] {
            let offset = (query_token * D + depth) * 2;
            let actual_bits = u16::from_ne_bytes([output_bytes[offset], output_bytes[offset + 1]]);
            assert_eq!(
                actual_bits, expected_bits,
                "S8192 row {query_token} depth {depth}"
            );
        }
    }

    let mut score_row = vec![-12.0625_f32; SOFTMAX_S];
    score_row[0] = 0.0;
    let exponentials = score_row
        .iter()
        .map(|score| score.exp())
        .collect::<Vec<_>>();
    let fixed_denominator = fixed37_sum(&exponentials);
    let flat_denominator = exponentials
        .iter()
        .copied()
        .fold(0.0_f32, |sum, value| sum + value);
    let fixed_first = f32_to_bf16_bits(1.0 / fixed_denominator);
    let flat_first = f32_to_bf16_bits(1.0 / flat_denominator);
    assert_ne!(
        fixed_first, flat_first,
        "the S8192 softmax fixture must distinguish fixed37 from flat order"
    );
    let scores_host = score_row.repeat(SOFTMAX_S);
    let score_bytes = encode_bf16(&scores_host);
    let mut scores = upload(&context, &mut stream, &mut staging, &score_bytes)?;
    let scores_len = scores.byte_len();
    let mut params = CausalSoftmaxInPlaceParams {
        scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, scores_len)?,
        token_count: SOFTMAX_S as u64,
        query_head_count: 1,
    };
    fixed37_causal_softmax_in_place(&mut params, &mut stream)?;
    let actual = download(&context, &mut stream, &mut scores)?;
    assert_eq!(u16::from_ne_bytes([actual[0], actual[1]]), fixed_first);

    scores.close()?;
    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    staging.close()?;
    close_context(context)
}
