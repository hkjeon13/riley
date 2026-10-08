#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, DecodeAttentionBackend,
    DecodeAttentionBackendAvailability, DecodeAttentionParams, DecodeAttentionPreference,
    DecodeAttentionRequest, DecodePartialReductionOrder, DecodePartialState,
    DecodePartialStateReduceParams, KvCacheAppendParams, PreparedDecodeAttention,
    decode_partial_states_reduce, kv_cache_append,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const D: usize = 64;
const SCALE: f32 = 0.125;
const ONLINE_REFERENCE_ABS_TOLERANCE: f32 = 0.0625;

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

fn encode_f32(values: &[f32]) -> Vec<u8> {
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

fn source_index(token: usize, head: usize, depth: usize, heads: usize) -> usize {
    (token * heads + head) * D + depth
}

fn cache_index(head: usize, token: usize, depth: usize, maximum: usize) -> usize {
    (head * maximum + token) * D + depth
}

fn cpu_decode(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    logical: usize,
    maximum: usize,
    query_heads: usize,
    key_value_heads: usize,
) -> Vec<f32> {
    let group_size = query_heads / key_value_heads;
    let mut output = vec![0.0_f32; query_heads * D];
    for query_head in 0..query_heads {
        let key_value_head = query_head / group_size;
        let mut scores = vec![0.0_f32; logical];
        for (token, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0_f32;
            for depth in 0..D {
                dot = query[query_head * D + depth].mul_add(
                    key_cache[cache_index(key_value_head, token, depth, maximum)],
                    dot,
                );
            }
            let staged_dot = bf16_to_f32(f32_to_bf16_bits(dot));
            *score = bf16_to_f32(f32_to_bf16_bits(staged_dot * SCALE));
        }
        let maximum_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = scores
            .iter()
            .map(|score| (*score - maximum_score).exp())
            .sum();
        for score in &mut scores {
            *score = bf16_to_f32(f32_to_bf16_bits(
                (*score - maximum_score).exp() / denominator,
            ));
        }
        for depth in 0..D {
            let mut sum = 0.0_f32;
            for (token, &probability) in scores.iter().enumerate() {
                sum = probability.mul_add(
                    value_cache[cache_index(key_value_head, token, depth, maximum)],
                    sum,
                );
            }
            output[query_head * D + depth] = bf16_to_f32(f32_to_bf16_bits(sum));
        }
    }
    output
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn kv_cache_scatter_is_bit_exact_and_preserves_unwritten_positions() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let key_value_heads = 3;
    let maximum = 40;
    let source_tokens = 33;
    let destination_start = 3;
    let source_elements = source_tokens * key_value_heads * D;
    let cache_elements = key_value_heads * maximum * D;
    let key_values: Vec<f32> = (0..source_elements)
        .map(|index| (f32::from(u8::try_from(index % 101).unwrap_or(0)) - 50.0) / 32.0)
        .collect();
    let value_values: Vec<f32> = (0..source_elements)
        .map(|index| (f32::from(u8::try_from((index * 7) % 97).unwrap_or(0)) - 48.0) / 64.0)
        .collect();
    let sentinel = -7.5_f32;
    let key_source_bytes = encode_bf16(&key_values);
    let value_source_bytes = encode_bf16(&value_values);
    let cache_bytes = encode_bf16(&vec![sentinel; cache_elements]);
    let key_source = upload(&context, &mut stream, &mut staging, &key_source_bytes)?;
    let value_source = upload(&context, &mut stream, &mut staging, &value_source_bytes)?;
    let mut key_cache = upload(&context, &mut stream, &mut staging, &cache_bytes)?;
    let mut value_cache = upload(&context, &mut stream, &mut staging, &cache_bytes)?;

    let key_cache_bytes = key_cache.byte_len();
    let value_cache_bytes = value_cache.byte_len();
    let mut params = KvCacheAppendParams {
        key_source: CudaBufferSpan::new(&key_source, CudaDType::BF16, 0, key_source.byte_len())?,
        value_source: CudaBufferSpan::new(
            &value_source,
            CudaDType::BF16,
            0,
            value_source.byte_len(),
        )?,
        key_cache: CudaBufferSpanMut::new(&mut key_cache, CudaDType::BF16, 0, key_cache_bytes)?,
        value_cache: CudaBufferSpanMut::new(
            &mut value_cache,
            CudaDType::BF16,
            0,
            value_cache_bytes,
        )?,
        source_token_count: u64::try_from(source_tokens)?,
        destination_token_start: u64::try_from(destination_start)?,
        maximum_token_count: u64::try_from(maximum)?,
        key_value_head_count: u64::try_from(key_value_heads)?,
        head_size: u64::try_from(D)?,
    };
    kv_cache_append(&mut params, &mut stream)?;

    let key_actual = decode_bf16(&download(&context, &mut stream, &mut key_cache)?);
    let value_actual = decode_bf16(&download(&context, &mut stream, &mut value_cache)?);
    let key_expected = decode_bf16(&key_source_bytes);
    let value_expected = decode_bf16(&value_source_bytes);
    let sentinel = bf16_to_f32(f32_to_bf16_bits(sentinel));
    for head in 0..key_value_heads {
        for token in 0..maximum {
            for depth in 0..D {
                let destination = cache_index(head, token, depth, maximum);
                if (destination_start..destination_start + source_tokens).contains(&token) {
                    let source =
                        source_index(token - destination_start, head, depth, key_value_heads);
                    assert_eq!(
                        key_actual[destination].to_bits(),
                        key_expected[source].to_bits()
                    );
                    assert_eq!(
                        value_actual[destination].to_bits(),
                        value_expected[source].to_bits()
                    );
                } else {
                    assert_eq!(key_actual[destination].to_bits(), sentinel.to_bits());
                    assert_eq!(value_actual[destination].to_bits(), sentinel.to_bits());
                }
            }
        }
    }

    value_cache.close()?;
    key_cache.close()?;
    value_source.close()?;
    key_source.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn materialized_and_chunked_decode_cover_gqa_boundaries_without_allocating() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let mut multi_range_output = None;
    let mut one_range_output = None;
    for &(logical, maximum, query_heads, key_value_heads, chunk) in &[
        (1, 1, 9, 9, 1),
        (2, 7, 6, 2, 1),
        (31, 33, 9, 3, 7),
        (32, 33, 9, 3, 13),
        (33, 40, 9, 3, 7),
        (33, 40, 9, 3, 64),
        (129, 129, 9, 3, 128),
    ] {
        let query_host: Vec<f32> = (0..query_heads * D)
            .map(|index| (f32::from(u8::try_from((index * 11) % 29).unwrap_or(0)) - 14.0) / 32.0)
            .collect();
        let cache_elements = key_value_heads * maximum * D;
        let key_host: Vec<f32> = (0..cache_elements)
            .map(|index| (f32::from(u8::try_from((index * 5 + 3) % 31).unwrap_or(0)) - 15.0) / 32.0)
            .collect();
        let value_host: Vec<f32> = (0..cache_elements)
            .map(|index| {
                (f32::from(u8::try_from((index * 13 + 7) % 37).unwrap_or(0)) - 18.0) / 32.0
            })
            .collect();
        let query_bytes = encode_bf16(&query_host);
        let key_bytes = encode_bf16(&key_host);
        let value_bytes = encode_bf16(&value_host);
        let query_values = decode_bf16(&query_bytes);
        let key_values = decode_bf16(&key_bytes);
        let value_values = decode_bf16(&value_bytes);
        let expected = cpu_decode(
            &query_values,
            &key_values,
            &value_values,
            logical,
            maximum,
            query_heads,
            key_value_heads,
        );
        let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
        let key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
        let value = upload(&context, &mut stream, &mut staging, &value_bytes)?;

        let request = DecodeAttentionRequest::new(
            u64::try_from(maximum)?,
            u64::try_from(query_heads)?,
            u64::try_from(key_value_heads)?,
            u64::try_from(D)?,
            SCALE,
        )
        .with_tokens_per_partition(u64::try_from(chunk)?);
        let reference = PreparedDecodeAttention::select(
            &context,
            request,
            DecodeAttentionPreference::Reference,
            DecodeAttentionBackendAvailability::linked(),
        )?;
        let online = PreparedDecodeAttention::select(
            &context,
            request,
            DecodeAttentionPreference::Optimized,
            DecodeAttentionBackendAvailability::linked(),
        )?;
        assert_eq!(
            reference.backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        assert_eq!(online.backend(), DecodeAttentionBackend::ChunkedOnline);
        let reviewed_hybrid = query_heads == 9 && key_value_heads == 3;
        assert_eq!(
            online.selection_trace().short_materialized_token_limit(),
            reviewed_hybrid.then_some(32)
        );
        let output_bytes = u64::try_from(query_bytes.len())?;
        let mut reference_output = context.allocate_device_buffer(output_bytes)?;
        let mut online_output = context.allocate_device_buffer(output_bytes)?;
        let mut reference_workspace =
            context.allocate_device_buffer(reference.workspace_bytes())?;
        let online_sentinel = -1234.5_f32;
        let online_workspace_elements = usize::try_from(online.workspace_bytes() / 4)?;
        let mut online_workspace = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_f32(&vec![online_sentinel; online_workspace_elements]),
        )?;
        let before = context.allocation_stats()?;

        for _ in 0..3 {
            let reference_output_len = reference_output.byte_len();
            let reference_workspace_len = reference_workspace.byte_len();
            let mut params = DecodeAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_cache: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
                value_cache: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
                output: CudaBufferSpanMut::new(
                    &mut reference_output,
                    CudaDType::BF16,
                    0,
                    reference_output_len,
                )?,
                workspace: CudaBufferSpanMut::new(
                    &mut reference_workspace,
                    CudaDType::BF16,
                    0,
                    reference_workspace_len,
                )?,
            };
            reference.execute(u64::try_from(logical)?, &mut params, &mut stream)?;

            let online_output_len = online_output.byte_len();
            let online_workspace_len = online_workspace.byte_len();
            let mut params = DecodeAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_cache: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
                value_cache: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
                output: CudaBufferSpanMut::new(
                    &mut online_output,
                    CudaDType::BF16,
                    0,
                    online_output_len,
                )?,
                workspace: CudaBufferSpanMut::new(
                    &mut online_workspace,
                    CudaDType::F32,
                    0,
                    online_workspace_len,
                )?,
            };
            online.execute(u64::try_from(logical)?, &mut params, &mut stream)?;
        }
        assert_eq!(context.allocation_stats()?, before);

        let reference_values =
            decode_bf16(&download(&context, &mut stream, &mut reference_output)?);
        let online_values = decode_bf16(&download(&context, &mut stream, &mut online_output)?);
        for (index, ((&reference, &online), &expected)) in reference_values
            .iter()
            .zip(&online_values)
            .zip(&expected)
            .enumerate()
        {
            assert!(
                (reference - expected).abs() <= 0.03125,
                "reference[{index}] expected {expected}, got {reference}"
            );
            assert!(
                (online - reference).abs() <= ONLINE_REFERENCE_ABS_TOLERANCE,
                "online[{index}] reference {reference}, got {online}"
            );
        }
        if (logical, maximum, query_heads, key_value_heads) == (33, 40, 9, 3) {
            match chunk {
                7 => multi_range_output = Some(online_values.clone()),
                64 => one_range_output = Some(online_values.clone()),
                _ => {}
            }
        }
        let active_partitions = logical / chunk + usize::from(logical % chunk != 0);
        let active_state_elements = active_partitions * query_heads * (D + 2);
        let online_workspace_values =
            decode_f32(&download(&context, &mut stream, &mut online_workspace)?);
        if reviewed_hybrid && logical == 33 {
            for partition in 0..active_partitions {
                for query_head in 0..query_heads {
                    let denominator = online_workspace_values
                        [(partition * query_heads + query_head) * (D + 2) + 1];
                    assert!(
                        denominator.is_finite() && denominator > 0.0,
                        "T=33 must retain the online partial-state layout; partition={partition} query_head={query_head} denominator={denominator}"
                    );
                }
            }
        }
        if reviewed_hybrid && logical <= 32 {
            let prefix_elements =
                usize::try_from(online.selection_trace().materialized_score_bytes() / 4)?;
            assert!(
                online_workspace_values[prefix_elements..]
                    .iter()
                    .all(|value| value.to_bits() == online_sentinel.to_bits()),
                "short hybrid modified bytes after its 576-byte score prefix"
            );
        }
        assert!(
            online_workspace_values[active_state_elements..]
                .iter()
                .all(|value| value.to_bits() == online_sentinel.to_bits()),
            "online decode modified partial-state capacity tail"
        );

        online_workspace.close()?;
        reference_workspace.close()?;
        online_output.close()?;
        reference_output.close()?;
        value.close()?;
        key.close()?;
        query.close()?;
    }
    let multi_range_output = multi_range_output.ok_or("missing chunk-7 decode output")?;
    let one_range_output = one_range_output.ok_or("missing chunk-64 decode output")?;
    let mut maximum_difference = 0.0_f32;
    for (index, (&multi_range, &one_range)) in
        multi_range_output.iter().zip(&one_range_output).enumerate()
    {
        let difference = (multi_range - one_range).abs();
        maximum_difference = maximum_difference.max(difference);
        assert!(
            difference <= ONLINE_REFERENCE_ABS_TOLERANCE,
            "multi-range output[{index}] {multi_range} differs from one-range {one_range}"
        );
    }
    println!(
        "pr09-decode-range-parity schema_version=1 logical_length=33 \
multi_range_partition_tokens=7 one_range_partition_tokens=64 max_abs={maximum_difference:.9} \
tolerance={ONLINE_REFERENCE_ABS_TOLERANCE:.9} status=passed"
    );
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn materialized_and_chunked_decode_agree_on_infinite_score_rules() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let logical = 3;
    let request = DecodeAttentionRequest::new(logical, 1, 1, u64::try_from(D)?, SCALE)
        .with_tokens_per_partition(1);
    let reference = PreparedDecodeAttention::select(
        &context,
        request,
        DecodeAttentionPreference::Reference,
        DecodeAttentionBackendAvailability::linked(),
    )?;
    let online = PreparedDecodeAttention::select(
        &context,
        request,
        DecodeAttentionPreference::Optimized,
        DecodeAttentionBackendAvailability::linked(),
    )?;

    for positive_infinity_ties in [false, true] {
        let query = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&vec![f32::INFINITY; D]),
        )?;
        let mut key_values = vec![-1.0_f32; usize::try_from(logical)? * D];
        if positive_infinity_ties {
            key_values[..2 * D].fill(1.0);
        }
        let mut value_values = vec![100.0_f32; usize::try_from(logical)? * D];
        value_values[..D].fill(2.0);
        value_values[D..2 * D].fill(6.0);
        let key = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&key_values),
        )?;
        let value = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&value_values),
        )?;
        let mut reference_output = context.allocate_device_buffer(u64::try_from(D * 2)?)?;
        let mut online_output = context.allocate_device_buffer(u64::try_from(D * 2)?)?;
        let mut reference_workspace =
            context.allocate_device_buffer(reference.workspace_bytes())?;
        let mut online_workspace = context.allocate_device_buffer(online.workspace_bytes())?;

        let reference_output_len = reference_output.byte_len();
        let reference_workspace_len = reference_workspace.byte_len();
        reference.execute(
            logical,
            &mut DecodeAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_cache: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
                value_cache: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
                output: CudaBufferSpanMut::new(
                    &mut reference_output,
                    CudaDType::BF16,
                    0,
                    reference_output_len,
                )?,
                workspace: CudaBufferSpanMut::new(
                    &mut reference_workspace,
                    CudaDType::BF16,
                    0,
                    reference_workspace_len,
                )?,
            },
            &mut stream,
        )?;
        let online_output_len = online_output.byte_len();
        let online_workspace_len = online_workspace.byte_len();
        online.execute(
            logical,
            &mut DecodeAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_cache: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
                value_cache: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
                output: CudaBufferSpanMut::new(
                    &mut online_output,
                    CudaDType::BF16,
                    0,
                    online_output_len,
                )?,
                workspace: CudaBufferSpanMut::new(
                    &mut online_workspace,
                    CudaDType::F32,
                    0,
                    online_workspace_len,
                )?,
            },
            &mut stream,
        )?;

        let expected = if positive_infinity_ties { 4.0 } else { 0.0 };
        let reference_values =
            decode_bf16(&download(&context, &mut stream, &mut reference_output)?);
        let online_values = decode_bf16(&download(&context, &mut stream, &mut online_output)?);
        assert!(
            reference_values
                .iter()
                .all(|&value| (value - expected).abs() <= f32::EPSILON)
        );
        assert!(
            online_values
                .iter()
                .all(|&value| (value - expected).abs() <= f32::EPSILON)
        );

        online_workspace.close()?;
        reference_workspace.close()?;
        online_output.close()?;
        reference_output.close()?;
        value.close()?;
        key.close()?;
        query.close()?;
    }

    drop(online);
    drop(reference);
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn standalone_reducer_handles_empty_and_merge_order() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    let query_heads = 2;
    let head_size = 3;
    let capacity = 3;
    let stride = head_size + 2;
    let mut packed = vec![0.0_f32; capacity * query_heads * stride];
    let mut expected = vec![0.0_f32; query_heads * head_size];
    for head in 0..query_heads {
        let mut merged = DecodePartialState::new(head_size)?;
        for partition in 0..capacity {
            let mut state = DecodePartialState::new(head_size)?;
            if partition != 1 {
                let score = f32::from(u8::try_from(partition * 3 + head).unwrap_or(0)) - 2.0;
                let value: Vec<f32> = (0..head_size)
                    .map(|depth| f32::from(u8::try_from(partition + head + depth).unwrap_or(0)))
                    .collect();
                state.accumulate(score, &value)?;
            }
            merged.merge(&state)?;
            let offset = (partition * query_heads + head) * stride;
            state.write_packed(&mut packed[offset..offset + stride])?;
        }
        merged.finalize(&mut expected[head * head_size..(head + 1) * head_size])?;
    }
    let states = upload(&context, &mut stream, &mut staging, &encode_f32(&packed))?;
    for order in [
        DecodePartialReductionOrder::LogicalAscending,
        DecodePartialReductionOrder::LogicalDescending,
    ] {
        let mut output = context.allocate_device_buffer(u64::try_from(expected.len() * 2)?)?;
        let output_len = output.byte_len();
        let mut params = DecodePartialStateReduceParams {
            partial_states: CudaBufferSpan::new(&states, CudaDType::F32, 0, states.byte_len())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            partial_state_count: u64::try_from(capacity)?,
            partial_state_capacity: u64::try_from(capacity)?,
            query_head_count: u64::try_from(query_heads)?,
            head_size: u64::try_from(head_size)?,
            order,
        };
        decode_partial_states_reduce(&mut params, &mut stream)?;
        let actual = decode_bf16(&download(&context, &mut stream, &mut output)?);
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 0.03125);
        }
        output.close()?;
    }

    let mut empty_output = context.allocate_device_buffer(u64::try_from(expected.len() * 2)?)?;
    let empty_output_len = empty_output.byte_len();
    let mut params = DecodePartialStateReduceParams {
        partial_states: CudaBufferSpan::new(&states, CudaDType::F32, 0, states.byte_len())?,
        output: CudaBufferSpanMut::new(&mut empty_output, CudaDType::BF16, 0, empty_output_len)?,
        partial_state_count: 0,
        partial_state_capacity: u64::try_from(capacity)?,
        query_head_count: u64::try_from(query_heads)?,
        head_size: u64::try_from(head_size)?,
        order: DecodePartialReductionOrder::LogicalAscending,
    };
    decode_partial_states_reduce(&mut params, &mut stream)?;
    assert!(
        decode_bf16(&download(&context, &mut stream, &mut empty_output)?)
            .iter()
            .all(|&value| value == 0.0)
    );

    empty_output.close()?;
    states.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}
