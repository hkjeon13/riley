#![allow(clippy::too_many_lines)]

use std::error::Error;
use std::time::Instant;

use rustinfer_cuda::{
    AttentionBackend, AttentionBackendAvailability, AttentionMask, AttentionPreference,
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaErrorKind,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, PrefillAttentionParams, PrefillAttentionRequest,
    PreparedPrefillAttention,
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

fn q_index(
    sequence: usize,
    query_heads: usize,
    batch: usize,
    token: usize,
    head: usize,
    depth: usize,
) -> usize {
    (((batch * sequence + token) * query_heads + head) * D) + depth
}

fn kv_index(
    sequence: usize,
    key_value_heads: usize,
    batch: usize,
    token: usize,
    head: usize,
    depth: usize,
) -> usize {
    (((batch * sequence + token) * key_value_heads + head) * D) + depth
}

fn key_is_visible(mask: AttentionMask, query: usize, key: usize) -> bool {
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

fn round_bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_bits(value))
}

fn update_online_normalizer(score: f32, maximum: &mut f32, denominator: &mut f32) {
    if score.is_nan() || (*maximum).is_nan() {
        *maximum = f32::NAN;
        *denominator = f32::NAN;
        return;
    }
    if score == f32::INFINITY {
        if *maximum == f32::INFINITY {
            *denominator += 1.0;
        } else {
            *maximum = f32::INFINITY;
            *denominator = 1.0;
        }
        return;
    }
    if *maximum == f32::INFINITY || score == f32::NEG_INFINITY {
        return;
    }

    let next_maximum = (*maximum).max(score);
    let alpha = if *denominator == 0.0 {
        0.0
    } else {
        (*maximum - next_maximum).exp()
    };
    let beta = (score - next_maximum).exp();
    *denominator = alpha.mul_add(*denominator, beta);
    *maximum = next_maximum;
}

fn staged_probability(score: f32, maximum: f32, denominator: f32) -> f32 {
    let probability = if score.is_nan() || maximum.is_nan() || denominator.is_nan() {
        f32::NAN
    } else if maximum == f32::INFINITY {
        if score == f32::INFINITY {
            denominator.recip()
        } else {
            0.0
        }
    } else if denominator > 0.0 {
        (score - maximum).exp() / denominator
    } else {
        0.0
    };
    round_bf16(probability)
}

#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch_size: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
    mask: AttentionMask,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch_size * sequence * query_heads * D];
    let group_size = query_heads / key_value_heads;
    for batch in 0..batch_size {
        for query_token in 0..sequence {
            for query_head in 0..query_heads {
                let key_value_head = query_head / group_size;
                let mut scores = Vec::with_capacity(query_token + 1);
                let mut maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0_f32;
                for key_token in 0..sequence {
                    if !key_is_visible(mask, query_token, key_token) {
                        continue;
                    }
                    let query_base =
                        q_index(sequence, query_heads, batch, query_token, query_head, 0);
                    let key_base = kv_index(
                        sequence,
                        key_value_heads,
                        batch,
                        key_token,
                        key_value_head,
                        0,
                    );
                    let mut lanes = [0.0_f32; 32];
                    for (lane, lane_sum) in lanes.iter_mut().enumerate() {
                        *lane_sum = query[query_base + lane].mul_add(
                            key[key_base + lane],
                            query[query_base + lane + 32] * key[key_base + lane + 32],
                        );
                    }
                    for offset in [16, 8, 4, 2, 1] {
                        let previous = lanes;
                        for (lane_sum, other) in
                            lanes[..32 - offset].iter_mut().zip(&previous[offset..])
                        {
                            *lane_sum += *other;
                        }
                    }
                    let score = round_bf16(round_bf16(lanes[0]) * SCALE);
                    update_online_normalizer(score, &mut maximum, &mut denominator);
                    scores.push((key_token, score));
                }
                if scores.is_empty() {
                    continue;
                }
                for depth in 0..D {
                    let mut accumulator = 0.0_f32;
                    for &(key_token, score) in &scores {
                        let probability = staged_probability(score, maximum, denominator);
                        if probability != 0.0 {
                            accumulator = probability.mul_add(
                                value[kv_index(
                                    sequence,
                                    key_value_heads,
                                    batch,
                                    key_token,
                                    key_value_head,
                                    depth,
                                )],
                                accumulator,
                            );
                        }
                    }
                    output[q_index(sequence, query_heads, batch, query_token, query_head, depth)] =
                        round_bf16(accumulator);
                }
            }
        }
    }
    output
}

fn deterministic_inputs(
    batch_size: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let query = (0..batch_size * sequence * query_heads * D)
        .map(|index| (f32::from(u8::try_from((index * 11 + 3) % 31).unwrap_or(0)) - 15.0) * 0.03125)
        .collect();
    let key = (0..batch_size * sequence * key_value_heads * D)
        .map(|index| (f32::from(u8::try_from((index * 7 + 5) % 29).unwrap_or(0)) - 14.0) * 0.03125)
        .collect();
    let value = (0..batch_size * sequence * key_value_heads * D)
        .map(|index| (f32::from(u8::try_from((index * 5 + 1) % 23).unwrap_or(0)) - 11.0) * 0.0625)
        .collect();
    (query, key, value)
}

#[allow(clippy::too_many_arguments)]
fn run_online_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    batch_size: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
    mask: AttentionMask,
) -> TestResult {
    let (query_host, key_host, value_host) =
        deterministic_inputs(batch_size, sequence, query_heads, key_value_heads);
    let query_bytes = encode_bf16(&query_host);
    let key_bytes = encode_bf16(&key_host);
    let value_bytes = encode_bf16(&value_host);
    let query_bf16 = decode_bf16(&query_bytes);
    let key_bf16 = decode_bf16(&key_bytes);
    let value_bf16 = decode_bf16(&value_bytes);
    let query = upload(context, stream, staging, &query_bytes)?;
    let key = upload(context, stream, staging, &key_bytes)?;
    let value = upload(context, stream, staging, &value_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;

    let request = PrefillAttentionRequest::new(
        u64::try_from(batch_size)?,
        u64::try_from(sequence)?,
        u64::try_from(query_heads)?,
        u64::try_from(key_value_heads)?,
        u64::try_from(D)?,
        SCALE,
        mask,
    );
    let prepared = PreparedPrefillAttention::select(
        context,
        request,
        AttentionPreference::Optimized,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), AttentionBackend::Online);
    assert_eq!(prepared.workspace_bytes(), 0);
    let before = context.allocation_stats()?;
    for _ in 0..3 {
        let output_bytes = output.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
            workspace: None,
        };
        prepared.execute(&mut params, stream)?;
    }
    assert_eq!(context.allocation_stats()?, before);

    let actual = decode_bf16(&download(context, stream, &mut output)?);
    let expected = cpu_attention(
        &query_bf16,
        &key_bf16,
        &value_bf16,
        batch_size,
        sequence,
        query_heads,
        key_value_heads,
        mask,
    );
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(actual.is_finite(), "output[{index}] is not finite");
        assert!(
            (actual - expected).abs() <= 0.03125,
            "output[{index}] expected {expected}, got {actual}"
        );
    }

    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_exact_causal_pair(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    batch_size: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
    query_bytes: &[u8],
    key_bytes: &[u8],
    value_bytes: &[u8],
) -> TestResult<Vec<u8>> {
    let query = upload(context, stream, staging, query_bytes)?;
    let key = upload(context, stream, staging, key_bytes)?;
    let value = upload(context, stream, staging, value_bytes)?;
    let output_bytes = u64::try_from(query_bytes.len())?;
    let mut reference_output = context.allocate_device_buffer(output_bytes)?;
    let mut online_output = context.allocate_device_buffer(output_bytes)?;
    let request = PrefillAttentionRequest::new(
        u64::try_from(batch_size)?,
        u64::try_from(sequence)?,
        u64::try_from(query_heads)?,
        u64::try_from(key_value_heads)?,
        u64::try_from(D)?,
        SCALE,
        AttentionMask::Causal,
    );
    let reference = PreparedPrefillAttention::select(
        context,
        request,
        AttentionPreference::Reference,
        AttentionBackendAvailability::linked(),
    )?;
    let online = PreparedPrefillAttention::select(
        context,
        request,
        AttentionPreference::Optimized,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(reference.backend(), AttentionBackend::MaterializedReference);
    assert_eq!(online.backend(), AttentionBackend::Online);
    assert!(!online.capability().uses_online_reduction());
    assert_eq!(online.workspace_bytes(), 0);
    let mut workspace = context.allocate_device_buffer(reference.workspace_bytes())?;

    let before = context.allocation_stats()?;
    {
        let workspace_bytes = workspace.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(
                &mut reference_output,
                CudaDType::BF16,
                0,
                output_bytes,
            )?,
            workspace: Some(CudaBufferSpanMut::new(
                &mut workspace,
                CudaDType::BF16,
                0,
                workspace_bytes,
            )?),
        };
        reference.execute(&mut params, stream)?;
    }
    {
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut online_output, CudaDType::BF16, 0, output_bytes)?,
            workspace: None,
        };
        online.execute(&mut params, stream)?;
    }
    assert_eq!(context.allocation_stats()?, before);

    let reference_bytes = download(context, stream, &mut reference_output)?;
    let online_bytes = download(context, stream, &mut online_output)?;
    assert_eq!(
        online_bytes, reference_bytes,
        "B={batch_size} S={sequence} QH={query_heads} KVH={key_value_heads} full-causal output differs from the materialized reference"
    );

    workspace.close()?;
    online_output.close()?;
    reference_output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    Ok(online_bytes)
}

fn zero_score_values(batch_size: usize, sequence: usize, key_value_heads: usize) -> Vec<f32> {
    let mut values: Vec<f32> = (0..batch_size * sequence * key_value_heads * D)
        .map(|index| (f32::from(u8::try_from((index * 13 + 7) % 31).unwrap_or(0)) - 15.0) * 0.03125)
        .collect();
    // For three equal scores, BF16(1/3) folded into these values produces a
    // different BF16 result than rounding their F32 average. Keep that witness
    // inside every long-shape fixture so the test proves probability staging,
    // rather than merely agreeing on inputs where both arithmetic paths alias.
    for (token, value) in [-0.125_f32, -5.375, 1.0625].into_iter().enumerate() {
        values[kv_index(sequence, key_value_heads, 0, token, 0, 0)] = value;
    }
    values
}

#[allow(clippy::cast_precision_loss)]
fn assert_zero_score_prefix_samples(
    actual: &[f32],
    value: &[f32],
    batch_size: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
) {
    let sampled_tokens = [0, 2, sequence / 2, sequence - 1];
    let sampled_heads = [0, query_heads / 2, query_heads - 1];
    let sampled_depths = [0, 17, D - 1];
    let group_size = query_heads / key_value_heads;
    let mut prefix = vec![0.0_f32; batch_size * key_value_heads * D];
    let mut witnessed_probability_staging = false;
    for token in 0..sequence {
        for batch in 0..batch_size {
            for key_value_head in 0..key_value_heads {
                for depth in 0..D {
                    let prefix_index = (batch * key_value_heads + key_value_head) * D + depth;
                    prefix[prefix_index] += value[kv_index(
                        sequence,
                        key_value_heads,
                        batch,
                        token,
                        key_value_head,
                        depth,
                    )];
                }
            }
        }
        if !sampled_tokens.contains(&token) {
            continue;
        }
        let denominator = (token + 1) as f32;
        for batch in 0..batch_size {
            for &query_head in &sampled_heads {
                let key_value_head = query_head / group_size;
                for &depth in &sampled_depths {
                    let prefix_index = (batch * key_value_heads + key_value_head) * D + depth;
                    let probability = round_bf16(denominator.recip());
                    let mut accumulator = 0.0_f32;
                    for key_token in 0..=token {
                        accumulator = probability.mul_add(
                            value[kv_index(
                                sequence,
                                key_value_heads,
                                batch,
                                key_token,
                                key_value_head,
                                depth,
                            )],
                            accumulator,
                        );
                    }
                    let expected = round_bf16(accumulator);
                    let unstaged_expected = round_bf16(prefix[prefix_index] / denominator);
                    witnessed_probability_staging |= expected != unstaged_expected;
                    let index = q_index(sequence, query_heads, batch, token, query_head, depth);
                    assert!(actual[index].is_finite(), "output[{index}] is not finite");
                    assert_eq!(
                        actual[index].to_bits(),
                        expected.to_bits(),
                        "B={batch_size} S={sequence} QH={query_heads} KVH={key_value_heads} output[{index}] must consume staged BF16 probabilities; expected {expected}, got {}",
                        actual[index],
                    );
                }
            }
        }
    }
    assert!(
        witnessed_probability_staging,
        "zero-score fixture must distinguish staged BF16 probabilities from normalize-after-AV"
    );
}

fn run_zero_score_analytic_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    batch_size: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
) -> TestResult {
    let query_elements = batch_size * sequence * query_heads * D;
    let key_value_elements = batch_size * sequence * key_value_heads * D;
    let query_bytes = encode_bf16(&vec![0.0; query_elements]);
    let key_bytes = encode_bf16(&vec![0.0; key_value_elements]);
    let value_bytes = encode_bf16(&zero_score_values(batch_size, sequence, key_value_heads));
    let value_bf16 = decode_bf16(&value_bytes);
    let query = upload(context, stream, staging, &query_bytes)?;
    let key = upload(context, stream, staging, &key_bytes)?;
    let value = upload(context, stream, staging, &value_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;
    let request = PrefillAttentionRequest::new(
        u64::try_from(batch_size)?,
        u64::try_from(sequence)?,
        u64::try_from(query_heads)?,
        u64::try_from(key_value_heads)?,
        u64::try_from(D)?,
        SCALE,
        AttentionMask::Causal,
    );
    let prepared = PreparedPrefillAttention::select(
        context,
        request,
        AttentionPreference::Optimized,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), AttentionBackend::Online);
    let trace = prepared.selection_trace();
    assert_eq!(trace.workspace_bytes(), 0);
    assert_eq!(trace.materialized_score_bytes(), 0);
    assert_eq!(trace.layout_copy_bytes(), 0);

    let before = context.allocation_stats()?;
    let output_bytes = output.byte_len();
    let mut params = PrefillAttentionParams {
        query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
        value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
        workspace: None,
    };
    prepared.execute(&mut params, stream)?;
    assert_eq!(context.allocation_stats()?, before);

    let actual = decode_bf16(&download(context, stream, &mut output)?);
    assert_zero_score_prefix_samples(
        &actual,
        &value_bf16,
        batch_size,
        sequence,
        query_heads,
        key_value_heads,
    );

    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_backend(
    context: &CudaContext,
    stream: &mut CudaStream,
    query: &CudaDeviceBuffer,
    key: &CudaDeviceBuffer,
    value: &CudaDeviceBuffer,
    batch_size: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
    preference: AttentionPreference,
    warmup: usize,
    iterations: usize,
) -> TestResult {
    let request = PrefillAttentionRequest::new(
        u64::try_from(batch_size)?,
        u64::try_from(sequence)?,
        u64::try_from(query_heads)?,
        u64::try_from(key_value_heads)?,
        u64::try_from(D)?,
        SCALE,
        AttentionMask::Causal,
    );
    let prepared = PreparedPrefillAttention::select(
        context,
        request,
        preference,
        AttentionBackendAvailability::linked(),
    )?;
    let output_bytes = u64::try_from(batch_size * sequence * query_heads * D * 2)?;
    let mut output = context.allocate_device_buffer(output_bytes)?;
    let mut workspace = if prepared.workspace_bytes() == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(prepared.workspace_bytes())?)
    };

    let (before, samples) = {
        let mut execute_once = || -> TestResult<f64> {
            let output_capacity = output.byte_len();
            let workspace_span = workspace
                .as_mut()
                .map(|buffer| {
                    let byte_len = buffer.byte_len();
                    CudaBufferSpanMut::new(buffer, CudaDType::BF16, 0, byte_len)
                })
                .transpose()?;
            let mut params = PrefillAttentionParams {
                query: CudaBufferSpan::new(query, CudaDType::BF16, 0, query.byte_len())?,
                key: CudaBufferSpan::new(key, CudaDType::BF16, 0, key.byte_len())?,
                value: CudaBufferSpan::new(value, CudaDType::BF16, 0, value.byte_len())?,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_capacity)?,
                workspace: workspace_span,
            };
            let started = Instant::now();
            prepared.execute(&mut params, stream)?;
            Ok(started.elapsed().as_secs_f64() * 1_000.0)
        };
        for _ in 0..warmup {
            let _ = execute_once()?;
        }
        let before = context.allocation_stats()?;
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            samples.push(execute_once()?);
        }
        (before, samples)
    };
    assert_eq!(context.allocation_stats()?, before);
    let trace = prepared.selection_trace();
    for (iteration, elapsed_ms) in samples.iter().copied().enumerate() {
        println!(
            "pr08-prefill-raw-sample backend={} B={} S={} QH={} KVH={} D={} iteration={} elapsed_ms={elapsed_ms:.6}",
            trace.implementation_id(),
            batch_size,
            sequence,
            query_heads,
            key_value_heads,
            D,
            iteration,
        );
    }
    let mut sorted_samples = samples.clone();
    sorted_samples.sort_by(f64::total_cmp);
    let median = sorted_samples[sorted_samples.len() / 2];
    let p95_index = (sorted_samples.len() * 95).div_ceil(100).saturating_sub(1);
    let p95 = sorted_samples[p95_index];
    println!(
        "pr08-prefill-summary backend={} version={} dependency={} compiled_architectures={} device_ordinal={} compute_capability={}.{} B={} S={} QH={} KVH={} D={} warmup={} iterations={} median_ms={median:.6} p95_ms={p95:.6} workspace_bytes={} materialized_score_bytes={} layout_copy_bytes={} device_live_bytes={} device_live_allocations={} pinned_host_live_bytes={} pinned_host_live_allocations={}",
        trace.implementation_id(),
        trace.implementation_version(),
        trace.native_dependency(),
        trace.compiled_architectures(),
        trace.device_ordinal(),
        trace.compute_capability().0,
        trace.compute_capability().1,
        batch_size,
        sequence,
        query_heads,
        key_value_heads,
        D,
        warmup,
        iterations,
        trace.workspace_bytes(),
        trace.materialized_score_bytes(),
        trace.layout_copy_bytes(),
        before.device_live_bytes(),
        before.device_live_allocations(),
        before.pinned_host_live_bytes(),
        before.pinned_host_live_allocations(),
    );

    if let Some(workspace) = workspace {
        workspace.close()?;
    }
    output.close()?;
    Ok(())
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn online_prefill_covers_batch_mha_gqa_and_tile_boundaries_without_allocating() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    for &(batch, sequence, query_heads, key_value_heads) in &[
        (1, 1, 4, 4),
        (2, 7, 6, 2),
        (1, 8, 4, 4),
        (2, 9, 6, 2),
        (1, 31, 6, 2),
        (1, 32, 4, 4),
        (1, 33, 6, 2),
    ] {
        let (query, key, value) =
            deterministic_inputs(batch, sequence, query_heads, key_value_heads);
        let output = run_exact_causal_pair(
            &context,
            &mut stream,
            &mut staging,
            batch,
            sequence,
            query_heads,
            key_value_heads,
            &encode_bf16(&query),
            &encode_bf16(&key),
            &encode_bf16(&value),
        )?;
        assert_eq!(output.len(), batch * sequence * query_heads * D * 2);
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn full_causal_exact_path_matches_reference_special_value_poisoning() -> TestResult {
    let batch_size = 1;
    let sequence = 2;
    let query_heads = 1;
    let key_value_heads = 1;
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;

    let query = vec![1.0_f32; batch_size * sequence * query_heads * D];
    let mut key = vec![0.0_f32; batch_size * sequence * key_value_heads * D];
    let value = vec![0.5_f32; batch_size * sequence * key_value_heads * D];
    key[kv_index(sequence, key_value_heads, 0, 1, 0, 0)] = f32::NAN;
    let future_nan = run_exact_causal_pair(
        &context,
        &mut stream,
        &mut staging,
        batch_size,
        sequence,
        query_heads,
        key_value_heads,
        &encode_bf16(&query),
        &encode_bf16(&key),
        &encode_bf16(&value),
    )?;
    assert!(
        decode_bf16(&future_nan[..D * 2])
            .into_iter()
            .all(f32::is_nan),
        "a future masked NaN score must poison the first reference row"
    );

    key.fill(0.0);
    key[kv_index(sequence, key_value_heads, 0, 1, 0, 0)] = f32::INFINITY;
    let future_infinity = run_exact_causal_pair(
        &context,
        &mut stream,
        &mut staging,
        batch_size,
        sequence,
        query_heads,
        key_value_heads,
        &encode_bf16(&query),
        &encode_bf16(&key),
        &encode_bf16(&value),
    )?;
    assert!(
        decode_bf16(&future_infinity[..D * 2])
            .into_iter()
            .all(f32::is_nan),
        "a future masked positive-infinity maximum must use reference NaN normalization"
    );

    key.fill(0.0);
    let mut infinite_value = value;
    for depth in 0..D {
        infinite_value[kv_index(sequence, key_value_heads, 0, 1, 0, depth)] = f32::INFINITY;
    }
    let masked_infinity = run_exact_causal_pair(
        &context,
        &mut stream,
        &mut staging,
        batch_size,
        sequence,
        query_heads,
        key_value_heads,
        &encode_bf16(&query),
        &encode_bf16(&key),
        &encode_bf16(&infinite_value),
    )?;
    assert!(
        decode_bf16(&masked_infinity[..D * 2])
            .into_iter()
            .all(f32::is_nan),
        "reference AV must evaluate zero-probability times infinite future values"
    );

    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn online_prefill_native_large_finite_score_gaps_are_stable() -> TestResult {
    let batch_size = 1;
    let sequence = 34;
    let query_heads = 9;
    let key_value_heads = 3;
    let mut query_host = vec![0.0_f32; batch_size * sequence * query_heads * D];
    let mut key_host = vec![0.0_f32; batch_size * sequence * key_value_heads * D];
    let mut value_host = vec![0.0_f32; batch_size * sequence * key_value_heads * D];
    for token in 0..sequence {
        let query_sign = if token & 1 == 0 { 16.0 } else { -16.0 };
        let key_sign = if token & 1 == 0 { -16.0 } else { 16.0 };
        for head in 0..query_heads {
            query_host[q_index(sequence, query_heads, 0, token, head, 0)] = query_sign;
        }
        for head in 0..key_value_heads {
            key_host[kv_index(sequence, key_value_heads, 0, token, head, 0)] = key_sign;
            for depth in 0..D {
                let code = u8::try_from((token * 17 + head * 11 + depth * 5) % 33).unwrap_or(0);
                value_host[kv_index(sequence, key_value_heads, 0, token, head, depth)] =
                    (f32::from(code) - 16.0) * 0.125;
            }
        }
    }

    let query_bytes = encode_bf16(&query_host);
    let key_bytes = encode_bf16(&key_host);
    let value_bytes = encode_bf16(&value_host);
    let query_bf16 = decode_bf16(&query_bytes);
    let key_bf16 = decode_bf16(&key_bytes);
    let value_bf16 = decode_bf16(&value_bytes);
    let expected = cpu_attention(
        &query_bf16,
        &key_bf16,
        &value_bf16,
        batch_size,
        sequence,
        query_heads,
        key_value_heads,
        AttentionMask::Causal,
    );

    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
    let key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
    let value = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;
    let request = PrefillAttentionRequest::new(
        u64::try_from(batch_size)?,
        u64::try_from(sequence)?,
        u64::try_from(query_heads)?,
        u64::try_from(key_value_heads)?,
        u64::try_from(D)?,
        SCALE,
        AttentionMask::Causal,
    );
    let prepared = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::Optimized,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), AttentionBackend::Online);
    let output_bytes = output.byte_len();
    let mut params = PrefillAttentionParams {
        query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
        value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
        workspace: None,
    };
    prepared.execute(&mut params, &mut stream)?;
    let actual = decode_bf16(&download(&context, &mut stream, &mut output)?);
    assert!(actual.iter().copied().all(f32::is_finite));

    let sampled_tokens = [0, sequence / 2, sequence - 1];
    for &token in &sampled_tokens {
        if token != 0 {
            let query_scalar = query_bf16[q_index(sequence, query_heads, 0, token, 0, 0)];
            let mut minimum = f32::INFINITY;
            let mut maximum = f32::NEG_INFINITY;
            for key_token in 0..=token {
                let key_scalar = key_bf16[kv_index(sequence, key_value_heads, 0, key_token, 0, 0)];
                let score = query_scalar * key_scalar * SCALE;
                minimum = minimum.min(score);
                maximum = maximum.max(score);
            }
            assert!(((maximum - minimum) - 64.0).abs() <= f32::EPSILON);
        }
        for &head in &[0, query_heads / 2, query_heads - 1] {
            for &depth in &[0, 17, D - 1] {
                let index = q_index(sequence, query_heads, 0, token, head, depth);
                assert!(actual[index].is_finite(), "output[{index}] is not finite");
                assert!(
                    (actual[index] - expected[index]).abs() <= 0.03125,
                    "large-gap BSHD output[{index}] at token={token} head={head} depth={depth} expected {}, got {}",
                    expected[index],
                    actual[index]
                );
            }
        }
    }

    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    staging.close()?;
    drop(prepared);
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn prepared_prefill_rejects_a_different_context_owner_before_launch() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0);
    let device = runtime.device(0)?;
    let selected_context = device.create_context()?;
    let execution_context = device.create_context()?;
    let mut stream = execution_context.create_stream()?;
    let request =
        PrefillAttentionRequest::new(1, 1, 1, 1, u64::try_from(D)?, SCALE, AttentionMask::Causal);
    let prepared = PreparedPrefillAttention::select(
        &selected_context,
        request,
        AttentionPreference::Optimized,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(
        prepared.selection_trace().compute_capability(),
        selected_context.compute_capability()
    );

    let tensor_bytes = u64::try_from(D * 2)?;
    let query = execution_context.allocate_device_buffer(tensor_bytes)?;
    let key = execution_context.allocate_device_buffer(tensor_bytes)?;
    let value = execution_context.allocate_device_buffer(tensor_bytes)?;
    let mut output = execution_context.allocate_device_buffer(tensor_bytes)?;
    let mut params = PrefillAttentionParams {
        query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, tensor_bytes)?,
        key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, tensor_bytes)?,
        value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, tensor_bytes)?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, tensor_bytes)?,
        workspace: None,
    };
    let error = prepared
        .execute(&mut params, &mut stream)
        .expect_err("a prepared plan must reject another context owner before native launch");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);

    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    stream.close()?;
    close_context(execution_context)?;
    drop(prepared);
    close_context(selected_context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn target_long_prefill_shapes_match_staged_probability_prefix_oracle() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    for &(batch, sequence, query_heads, key_value_heads) in &[
        // Pairwise coverage of every required B/S value at target GQA D64.
        (4, 128, 9, 3),
        (2, 1_024, 9, 3),
        (1, 4_096, 9, 3),
        // Target-width MHA in addition to the 9/3 GQA matrix.
        (1, 128, 9, 9),
    ] {
        run_zero_score_analytic_case(
            &context,
            &mut stream,
            &mut staging,
            batch,
            sequence,
            query_heads,
            key_value_heads,
        )?;
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn reference_and_online_match_target_gqa_at_s128() -> TestResult {
    let batch_size = 1;
    let sequence = 128;
    let query_heads = 9;
    let key_value_heads = 3;
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let (query_host, key_host, value_host) =
        deterministic_inputs(batch_size, sequence, query_heads, key_value_heads);
    let query_bytes = encode_bf16(&query_host);
    let key_bytes = encode_bf16(&key_host);
    let value_bytes = encode_bf16(&value_host);
    let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
    let key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
    let value = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let mut reference_output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;
    let mut online_output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;
    let request = PrefillAttentionRequest::new(
        u64::try_from(batch_size)?,
        u64::try_from(sequence)?,
        u64::try_from(query_heads)?,
        u64::try_from(key_value_heads)?,
        u64::try_from(D)?,
        SCALE,
        AttentionMask::Causal,
    );
    let reference = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::Reference,
        AttentionBackendAvailability::linked(),
    )?;
    let online = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::Optimized,
        AttentionBackendAvailability::linked(),
    )?;
    let mut workspace = context.allocate_device_buffer(reference.workspace_bytes())?;
    let before = context.allocation_stats()?;
    {
        let output_bytes = reference_output.byte_len();
        let workspace_bytes = workspace.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(
                &mut reference_output,
                CudaDType::BF16,
                0,
                output_bytes,
            )?,
            workspace: Some(CudaBufferSpanMut::new(
                &mut workspace,
                CudaDType::BF16,
                0,
                workspace_bytes,
            )?),
        };
        reference.execute(&mut params, &mut stream)?;
    }
    {
        let output_bytes = online_output.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut online_output, CudaDType::BF16, 0, output_bytes)?,
            workspace: None,
        };
        online.execute(&mut params, &mut stream)?;
    }
    assert_eq!(context.allocation_stats()?, before);

    let reference_bytes = download(&context, &mut stream, &mut reference_output)?;
    let online_bytes = download(&context, &mut stream, &mut online_output)?;
    println!(
        "pr16-prefill-exact-parity B={batch_size} S={sequence} QH={query_heads} KVH={key_value_heads} D={D} bytes={} byte_exact=true",
        reference_bytes.len(),
    );
    assert_eq!(
        online_bytes, reference_bytes,
        "full-causal no-HBM output must be byte-exact with the materialized reference"
    );

    workspace.close()?;
    online_output.close()?;
    reference_output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    staging.close()?;
    drop(online);
    drop(reference);
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn online_local_mask_covers_fully_masked_and_window_boundaries() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    for window in [0, 1, 3, 9] {
        run_online_case(
            &context,
            &mut stream,
            &mut staging,
            2,
            9,
            6,
            2,
            AttentionMask::CausalLocal { window },
        )?;
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn materialized_reference_reuses_batch_workspace_and_invalid_workspace_fails_early() -> TestResult {
    let batch_size = 2;
    let sequence = 7;
    let query_heads = 6;
    let key_value_heads = 2;
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let (query_host, key_host, value_host) =
        deterministic_inputs(batch_size, sequence, query_heads, key_value_heads);
    let query_bytes = encode_bf16(&query_host);
    let key_bytes = encode_bf16(&key_host);
    let value_bytes = encode_bf16(&value_host);
    let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
    let key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
    let value = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;

    let request = PrefillAttentionRequest::new(
        u64::try_from(batch_size)?,
        u64::try_from(sequence)?,
        u64::try_from(query_heads)?,
        u64::try_from(key_value_heads)?,
        u64::try_from(D)?,
        SCALE,
        AttentionMask::Causal,
    );
    let prepared = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::Reference,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), AttentionBackend::MaterializedReference);
    assert_eq!(
        prepared.workspace_bytes(),
        u64::try_from(query_heads * sequence * sequence * 2)?
    );
    let mut workspace = context.allocate_device_buffer(prepared.workspace_bytes())?;

    let online = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::Optimized,
        AttentionBackendAvailability::linked(),
    )?;
    let output_bytes = output.byte_len();
    let workspace_bytes = workspace.byte_len();
    let mut unexpected_workspace = PrefillAttentionParams {
        query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
        value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
        workspace: Some(CudaBufferSpanMut::new(
            &mut workspace,
            CudaDType::BF16,
            0,
            workspace_bytes,
        )?),
    };
    let error = online
        .execute(&mut unexpected_workspace, &mut stream)
        .expect_err("online workspace must fail before launch");
    assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);

    let mut missing_workspace = PrefillAttentionParams {
        query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
        value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
        workspace: None,
    };
    assert!(
        prepared
            .execute(&mut missing_workspace, &mut stream)
            .is_err()
    );

    let before = context.allocation_stats()?;
    for _ in 0..2 {
        let output_bytes = output.byte_len();
        let workspace_bytes = workspace.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
            workspace: Some(CudaBufferSpanMut::new(
                &mut workspace,
                CudaDType::BF16,
                0,
                workspace_bytes,
            )?),
        };
        prepared.execute(&mut params, &mut stream)?;
    }
    assert_eq!(context.allocation_stats()?, before);
    assert!(
        decode_bf16(&download(&context, &mut stream, &mut output)?)
            .into_iter()
            .all(f32::is_finite)
    );

    workspace.close()?;
    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    staging.close()?;
    drop(online);
    drop(prepared);
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote-only raw latency benchmark; may allocate a 288 MiB reference workspace"]
fn prefill_reference_and_online_raw_latency() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    for &sequence in &[128, 1_024, 4_096] {
        let batch_size = 1;
        let query_heads = 9;
        let key_value_heads = 3;
        let query_bytes = encode_bf16(&vec![0.0; batch_size * sequence * query_heads * D]);
        let key_bytes = encode_bf16(&vec![0.0; batch_size * sequence * key_value_heads * D]);
        let value_bytes = encode_bf16(&zero_score_values(batch_size, sequence, key_value_heads));
        let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
        let key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
        let value = upload(&context, &mut stream, &mut staging, &value_bytes)?;
        let (warmup, iterations) = match sequence {
            128 => (3, 10),
            1_024 => (2, 5),
            4_096 => (1, 2),
            _ => unreachable!(),
        };
        for preference in [
            AttentionPreference::Reference,
            AttentionPreference::Optimized,
        ] {
            benchmark_backend(
                &context,
                &mut stream,
                &query,
                &key,
                &value,
                batch_size,
                sequence,
                query_heads,
                key_value_heads,
                preference,
                warmup,
                iterations,
            )?;
        }
        value.close()?;
        key.close()?;
        query.close()?;
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}
