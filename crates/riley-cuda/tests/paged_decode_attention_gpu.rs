#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, DecodeAttentionBackendAvailability,
    DecodeAttentionParams, DecodeAttentionPreference, DecodeAttentionRequest,
    DecodePartialReductionOrder, DecodePartialStateReduceParams, PAGED_KV_BLOCK_SIZE,
    PagedDecodeAttentionParams, PagedDecodeAttentionRequest, PagedKvBlockTableHostV1,
    PagedKvBlockTableV1, PagedKvCacheAppendParams, PreparedDecodeAttention,
    PreparedPagedDecodeAttention, decode_partial_states_reduce, paged_kv_cache_append,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const D: usize = 64;
const SCALE: f32 = 0.125;
const REFERENCE_CPU_ABS_TOLERANCE: f32 = 0.03125;
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

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn encode_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
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

fn contiguous_index(head: usize, token: usize, depth: usize, logical: usize) -> usize {
    (head * logical + token) * D + depth
}

fn paged_index(
    physical_block: usize,
    head: usize,
    token_in_block: usize,
    depth: usize,
    heads: usize,
) -> usize {
    (((physical_block * heads + head) * 16 + token_in_block) * D) + depth
}

fn cpu_decode(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    logical: usize,
    query_heads: usize,
    key_value_heads: usize,
) -> Vec<f32> {
    let group_size = query_heads / key_value_heads;
    let mut output = vec![0.0_f32; query_heads * D];
    for query_head in 0..query_heads {
        let kv_head = query_head / group_size;
        let mut scores = vec![0.0_f32; logical];
        for (token, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0_f32;
            for depth in 0..D {
                dot = query[query_head * D + depth]
                    .mul_add(key[contiguous_index(kv_head, token, depth, logical)], dot);
            }
            let staged_dot = bf16_to_f32(f32_to_bf16_bits(dot));
            *score = bf16_to_f32(f32_to_bf16_bits(staged_dot * SCALE));
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
        for score in &mut scores {
            *score = bf16_to_f32(f32_to_bf16_bits((*score - maximum).exp() / denominator));
        }
        for depth in 0..D {
            let mut sum = 0.0_f32;
            for (token, &probability) in scores.iter().enumerate() {
                sum = probability
                    .mul_add(value[contiguous_index(kv_head, token, depth, logical)], sum);
            }
            output[query_head * D + depth] = bf16_to_f32(f32_to_bf16_bits(sum));
        }
    }
    output
}

fn device_table<'a>(
    host: PagedKvBlockTableHostV1<'a>,
    ids: &'a CudaDeviceBuffer,
    valid: &'a CudaDeviceBuffer,
) -> TestResult<PagedKvBlockTableV1<'a>> {
    Ok(PagedKvBlockTableV1::new(
        host,
        CudaBufferSpan::new(ids, CudaDType::U32, 0, ids.byte_len())?,
        CudaBufferSpan::new(valid, CudaDType::U16, 0, valid.byte_len())?,
    )?)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn paged_scatter_reference_and_online_match_contiguous_across_boundaries() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let query_heads = 9;
    let key_value_heads = 3;

    for logical in [1_usize, 15, 16, 17, 31, 32, 33, 128, 129] {
        let logical_blocks = logical.div_ceil(16);
        let physical_blocks = logical_blocks + 3;
        let block_ids: Vec<u32> = (0..logical_blocks)
            .rev()
            .map(|index| u32::try_from(index + 2))
            .collect::<Result<_, _>>()?;
        let mut valid_tokens = vec![16_u16; logical_blocks];
        *valid_tokens.last_mut().ok_or("missing final block")? =
            u16::try_from((logical - 1) % 16 + 1)?;
        let host_table = PagedKvBlockTableHostV1::new(
            &block_ids,
            &valid_tokens,
            u64::try_from(logical)?,
            u64::try_from(physical_blocks)?,
        )?;

        let query_values: Vec<f32> = (0..query_heads * D)
            .map(|index| (f32::from(u8::try_from((index * 11) % 29).unwrap_or(0)) - 14.0) / 32.0)
            .collect();
        let source_elements = logical * key_value_heads * D;
        let key_source_values: Vec<f32> = (0..source_elements)
            .map(|index| (f32::from(u8::try_from((index * 5 + 3) % 31).unwrap_or(0)) - 15.0) / 32.0)
            .collect();
        let value_source_values: Vec<f32> = (0..source_elements)
            .map(|index| {
                (f32::from(u8::try_from((index * 13 + 7) % 37).unwrap_or(0)) - 18.0) / 32.0
            })
            .collect();
        let query_bytes = encode_bf16(&query_values);
        let key_source_bytes = encode_bf16(&key_source_values);
        let value_source_bytes = encode_bf16(&value_source_values);
        let key_source_exact = decode_bf16(&key_source_bytes);
        let value_source_exact = decode_bf16(&value_source_bytes);

        let mut contiguous_key = vec![0.0_f32; source_elements];
        let mut contiguous_value = vec![0.0_f32; source_elements];
        for token in 0..logical {
            for head in 0..key_value_heads {
                for depth in 0..D {
                    let source = source_index(token, head, depth, key_value_heads);
                    let destination = contiguous_index(head, token, depth, logical);
                    contiguous_key[destination] = key_source_exact[source];
                    contiguous_value[destination] = value_source_exact[source];
                }
            }
        }
        let expected = cpu_decode(
            &decode_bf16(&query_bytes),
            &contiguous_key,
            &contiguous_value,
            logical,
            query_heads,
            key_value_heads,
        );

        let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
        let key_source = upload(&context, &mut stream, &mut staging, &key_source_bytes)?;
        let value_source = upload(&context, &mut stream, &mut staging, &value_source_bytes)?;
        let ids = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
        let valid = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u16(&valid_tokens),
        )?;
        let sentinel = -77.0_f32;
        let pool_elements = physical_blocks * key_value_heads * 16 * D;
        let pool_bytes = encode_bf16(&vec![sentinel; pool_elements]);
        let mut key_pool = upload(&context, &mut stream, &mut staging, &pool_bytes)?;
        let mut value_pool = upload(&context, &mut stream, &mut staging, &pool_bytes)?;

        let key_pool_len = key_pool.byte_len();
        let value_pool_len = value_pool.byte_len();
        paged_kv_cache_append(
            &mut PagedKvCacheAppendParams {
                key_source: CudaBufferSpan::new(
                    &key_source,
                    CudaDType::BF16,
                    0,
                    key_source.byte_len(),
                )?,
                value_source: CudaBufferSpan::new(
                    &value_source,
                    CudaDType::BF16,
                    0,
                    value_source.byte_len(),
                )?,
                key_pool: CudaBufferSpanMut::new(&mut key_pool, CudaDType::BF16, 0, key_pool_len)?,
                value_pool: CudaBufferSpanMut::new(
                    &mut value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool_len,
                )?,
                block_table: device_table(host_table, &ids, &valid)?,
                source_token_count: u64::try_from(logical)?,
                destination_token_start: 0,
                key_value_head_count: u64::try_from(key_value_heads)?,
                head_size: u64::try_from(D)?,
            },
            &mut stream,
        )?;

        let key_pool_values = decode_bf16(&download(&context, &mut stream, &mut key_pool)?);
        let value_pool_values = decode_bf16(&download(&context, &mut stream, &mut value_pool)?);
        let sentinel = bf16_to_f32(f32_to_bf16_bits(sentinel));
        for physical in 0..physical_blocks {
            for head in 0..key_value_heads {
                for offset in 0..16 {
                    for depth in 0..D {
                        let destination =
                            paged_index(physical, head, offset, depth, key_value_heads);
                        let logical_token = block_ids
                            .iter()
                            .position(|&id| usize::try_from(id).ok() == Some(physical))
                            .map(|block| block * 16 + offset);
                        if let Some(token) = logical_token.filter(|&token| token < logical) {
                            let source = source_index(token, head, depth, key_value_heads);
                            assert_eq!(
                                key_pool_values[destination].to_bits(),
                                key_source_exact[source].to_bits()
                            );
                            assert_eq!(
                                value_pool_values[destination].to_bits(),
                                value_source_exact[source].to_bits()
                            );
                        } else {
                            assert_eq!(key_pool_values[destination].to_bits(), sentinel.to_bits());
                            assert_eq!(
                                value_pool_values[destination].to_bits(),
                                sentinel.to_bits()
                            );
                        }
                    }
                }
            }
        }

        let paged_request = PagedDecodeAttentionRequest::new(
            129,
            u64::try_from(physical_blocks)?,
            u64::try_from(query_heads)?,
            u64::try_from(key_value_heads)?,
            u64::try_from(D)?,
            SCALE,
        );
        let paged_reference = PreparedPagedDecodeAttention::select(
            &context,
            paged_request,
            DecodeAttentionPreference::Reference,
            DecodeAttentionBackendAvailability::linked(),
        )?;
        let paged_online = PreparedPagedDecodeAttention::select(
            &context,
            paged_request,
            DecodeAttentionPreference::Optimized,
            DecodeAttentionBackendAvailability::linked(),
        )?;
        assert_eq!(
            paged_online
                .selection_trace()
                .short_materialized_token_limit(),
            Some(32)
        );
        let contiguous_request = DecodeAttentionRequest::new(
            u64::try_from(logical)?,
            u64::try_from(query_heads)?,
            u64::try_from(key_value_heads)?,
            u64::try_from(D)?,
            SCALE,
        );
        let contiguous_reference = PreparedDecodeAttention::select(
            &context,
            contiguous_request,
            DecodeAttentionPreference::Reference,
            DecodeAttentionBackendAvailability::linked(),
        )?;
        let contiguous_key_device = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&contiguous_key),
        )?;
        let contiguous_value_device = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&contiguous_value),
        )?;
        let output_bytes = u64::try_from(query_heads * D * 2)?;
        let mut contiguous_output = context.allocate_device_buffer(output_bytes)?;
        let mut paged_reference_output = context.allocate_device_buffer(output_bytes)?;
        let mut paged_online_output = context.allocate_device_buffer(output_bytes)?;
        let mut paged_descending_output = context.allocate_device_buffer(output_bytes)?;
        let mut contiguous_workspace =
            context.allocate_device_buffer(contiguous_reference.workspace_bytes())?;
        let mut paged_reference_workspace =
            context.allocate_device_buffer(paged_reference.workspace_bytes())?;
        let online_sentinel = -1234.5_f32;
        let mut paged_online_workspace = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_f32(&vec![
                online_sentinel;
                usize::try_from(paged_online.workspace_bytes() / 4)?
            ]),
        )?;
        let before = context.allocation_stats()?;

        let contiguous_output_len = contiguous_output.byte_len();
        let contiguous_workspace_len = contiguous_workspace.byte_len();
        contiguous_reference.execute(
            u64::try_from(logical)?,
            &mut DecodeAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_cache: CudaBufferSpan::new(
                    &contiguous_key_device,
                    CudaDType::BF16,
                    0,
                    contiguous_key_device.byte_len(),
                )?,
                value_cache: CudaBufferSpan::new(
                    &contiguous_value_device,
                    CudaDType::BF16,
                    0,
                    contiguous_value_device.byte_len(),
                )?,
                output: CudaBufferSpanMut::new(
                    &mut contiguous_output,
                    CudaDType::BF16,
                    0,
                    contiguous_output_len,
                )?,
                workspace: CudaBufferSpanMut::new(
                    &mut contiguous_workspace,
                    CudaDType::BF16,
                    0,
                    contiguous_workspace_len,
                )?,
            },
            &mut stream,
        )?;

        for _ in 0..3 {
            let reference_output_len = paged_reference_output.byte_len();
            let reference_workspace_len = paged_reference_workspace.byte_len();
            paged_reference.execute(
                &mut PagedDecodeAttentionParams {
                    query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                    key_pool: CudaBufferSpan::new(
                        &key_pool,
                        CudaDType::BF16,
                        0,
                        key_pool.byte_len(),
                    )?,
                    value_pool: CudaBufferSpan::new(
                        &value_pool,
                        CudaDType::BF16,
                        0,
                        value_pool.byte_len(),
                    )?,
                    workspace: CudaBufferSpanMut::new(
                        &mut paged_reference_workspace,
                        CudaDType::BF16,
                        0,
                        reference_workspace_len,
                    )?,
                    output: CudaBufferSpanMut::new(
                        &mut paged_reference_output,
                        CudaDType::BF16,
                        0,
                        reference_output_len,
                    )?,
                    block_table: device_table(host_table, &ids, &valid)?,
                },
                &mut stream,
            )?;

            let online_output_len = paged_online_output.byte_len();
            let online_workspace_len = paged_online_workspace.byte_len();
            paged_online.execute(
                &mut PagedDecodeAttentionParams {
                    query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                    key_pool: CudaBufferSpan::new(
                        &key_pool,
                        CudaDType::BF16,
                        0,
                        key_pool.byte_len(),
                    )?,
                    value_pool: CudaBufferSpan::new(
                        &value_pool,
                        CudaDType::BF16,
                        0,
                        value_pool.byte_len(),
                    )?,
                    workspace: CudaBufferSpanMut::new(
                        &mut paged_online_workspace,
                        CudaDType::F32,
                        0,
                        online_workspace_len,
                    )?,
                    output: CudaBufferSpanMut::new(
                        &mut paged_online_output,
                        CudaDType::BF16,
                        0,
                        online_output_len,
                    )?,
                    block_table: device_table(host_table, &ids, &valid)?,
                },
                &mut stream,
            )?;
        }
        if logical > 32 {
            let descending_output_len = paged_descending_output.byte_len();
            let mut descending_params = DecodePartialStateReduceParams {
                partial_states: CudaBufferSpan::new(
                    &paged_online_workspace,
                    CudaDType::F32,
                    0,
                    paged_online.workspace_bytes(),
                )?,
                output: CudaBufferSpanMut::new(
                    &mut paged_descending_output,
                    CudaDType::BF16,
                    0,
                    descending_output_len,
                )?,
                partial_state_count: u64::try_from(logical_blocks)?,
                partial_state_capacity: paged_online.partial_state_capacity(),
                query_head_count: u64::try_from(query_heads)?,
                head_size: u64::try_from(D)?,
                order: DecodePartialReductionOrder::LogicalDescending,
            };
            decode_partial_states_reduce(&mut descending_params, &mut stream)?;
        }
        assert_eq!(context.allocation_stats()?, before);

        let contiguous_actual =
            decode_bf16(&download(&context, &mut stream, &mut contiguous_output)?);
        let reference_actual = decode_bf16(&download(
            &context,
            &mut stream,
            &mut paged_reference_output,
        )?);
        let online_actual =
            decode_bf16(&download(&context, &mut stream, &mut paged_online_output)?);
        let descending_actual = if logical > 32 {
            decode_bf16(&download(
                &context,
                &mut stream,
                &mut paged_descending_output,
            )?)
        } else {
            online_actual.clone()
        };
        for (index, ((((&contiguous, &reference), &online), &descending), &expected)) in
            contiguous_actual
                .iter()
                .zip(&reference_actual)
                .zip(&online_actual)
                .zip(&descending_actual)
                .zip(&expected)
                .enumerate()
        {
            assert_eq!(
                reference.to_bits(),
                contiguous.to_bits(),
                "paged reference[{index}] differs from contiguous reference"
            );
            assert!(
                (reference - expected).abs() <= REFERENCE_CPU_ABS_TOLERANCE,
                "reference[{index}] expected {expected}, got {reference}"
            );
            assert!(
                (online - reference).abs() <= ONLINE_REFERENCE_ABS_TOLERANCE,
                "paged online[{index}] reference {reference}, got {online}"
            );
            assert!(
                (descending - reference).abs() <= ONLINE_REFERENCE_ABS_TOLERANCE,
                "descending paged reduction[{index}] reference {reference}, got {descending}"
            );
        }
        let active_state_elements = logical_blocks * query_heads * (D + 2);
        let online_workspace = decode_f32(&download(
            &context,
            &mut stream,
            &mut paged_online_workspace,
        )?);
        if logical == 33 {
            for logical_block in 0..logical_blocks {
                for query_head in 0..query_heads {
                    let denominator =
                        online_workspace[(logical_block * query_heads + query_head) * (D + 2) + 1];
                    assert!(
                        denominator.is_finite() && denominator > 0.0,
                        "T=33 must retain the paged online state layout; logical_block={logical_block} query_head={query_head} denominator={denominator}"
                    );
                }
            }
        }
        if logical <= 32 {
            let prefix_elements =
                usize::try_from(paged_online.selection_trace().materialized_score_bytes() / 4)?;
            assert!(
                online_workspace[prefix_elements..]
                    .iter()
                    .all(|value| value.to_bits() == online_sentinel.to_bits()),
                "short paged hybrid modified bytes after its 576-byte score prefix"
            );
        }
        assert!(
            online_workspace[active_state_elements..]
                .iter()
                .all(|value| value.to_bits() == online_sentinel.to_bits()),
            "paged online modified the preallocated state-capacity tail"
        );
        println!(
            "pr10-paged-decode schema_version=1 logical_length={logical} block_count={logical_blocks} physical_blocks={physical_blocks} shuffled_ids=true contiguous_reference_exact=true descending_reduce_matches=true status=passed"
        );

        paged_online_workspace.close()?;
        paged_reference_workspace.close()?;
        contiguous_workspace.close()?;
        paged_online_output.close()?;
        paged_descending_output.close()?;
        paged_reference_output.close()?;
        contiguous_output.close()?;
        contiguous_value_device.close()?;
        contiguous_key_device.close()?;
        value_pool.close()?;
        key_pool.close()?;
        valid.close()?;
        ids.close()?;
        value_source.close()?;
        key_source.close()?;
        query.close()?;
    }

    assert_eq!(PAGED_KV_BLOCK_SIZE, 16);
    staging.close()?;
    stream.close()?;
    close_context(context)
}
