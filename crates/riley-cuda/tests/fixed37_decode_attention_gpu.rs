#![allow(clippy::too_many_arguments, clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    AttentionReductionProfile, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaErrorKind, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    DecodeAttentionBackend, DecodeAttentionBackendAvailability, DecodeAttentionNoWorkspaceParams,
    DecodeAttentionParams, DecodeAttentionPreference, DecodeAttentionRequest,
    PagedDecodeAttentionNoWorkspaceParams, PagedDecodeAttentionParams, PagedDecodeAttentionRequest,
    PagedKvBlockTableHostV1, PagedKvBlockTableV1, PreparedDecodeAttention,
    PreparedPagedDecodeAttention,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const FIXED37: AttentionReductionProfile = AttentionReductionProfile::FixedContiguous37BalancedV1;

#[derive(Debug, Eq, PartialEq)]
struct ExecutionBits {
    output: Vec<u16>,
    probabilities: Vec<u16>,
}

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

fn decode_bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
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

fn download_bits(
    context: &CudaContext,
    stream: &mut CudaStream,
    buffer: &mut CudaDeviceBuffer,
) -> TestResult<Vec<u16>> {
    let mut staging = context.allocate_pinned_host_buffer(buffer.byte_len())?;
    buffer
        .copy_to_pinned_async(0, &mut staging, 0, buffer.byte_len(), stream)?
        .synchronize()?;
    let bits = decode_bf16_bits(&staging.to_vec()?);
    staging.close()?;
    Ok(bits)
}

fn balanced_sum(mut values: Vec<f32>) -> f32 {
    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut pairs = values.chunks_exact(2);
        for pair in &mut pairs {
            next.push(pair[0] + pair[1]);
        }
        if let Some(&odd) = pairs.remainder().first() {
            next.push(odd);
        }
        values = next;
    }
    values[0]
}

fn fixed37_sum(values: &[f32]) -> f32 {
    let partials = values
        .chunks(37)
        .map(|chunk| chunk.iter().fold(0.0_f32, |sum, &value| sum + value))
        .collect();
    balanced_sum(partials)
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

fn cpu_fixed37_decode(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    token_count: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_size: usize,
    scale: f32,
) -> Vec<u16> {
    let query: Vec<f32> = query.iter().copied().map(round_bf16).collect();
    let key: Vec<f32> = key.iter().copied().map(round_bf16).collect();
    let value: Vec<f32> = value.iter().copied().map(round_bf16).collect();
    let group_size = query_head_count / key_value_head_count;
    let mut output = vec![0_u16; query_head_count * head_size];
    for query_head in 0..query_head_count {
        let key_value_head = query_head / group_size;
        let query_row = &query[query_head * head_size..(query_head + 1) * head_size];
        let mut scores = Vec::with_capacity(token_count);
        for token in 0..token_count {
            let key_base = (key_value_head * token_count + token) * head_size;
            let raw = round_bf16(fixed37_dot(query_row, &key[key_base..key_base + head_size]));
            scores.push(round_bf16(raw * scale));
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = scores
            .iter()
            .map(|&score| (score - maximum).exp())
            .collect();
        let denominator = fixed37_sum(&exponentials);
        let probabilities: Vec<f32> = exponentials
            .iter()
            .map(|&numerator| round_bf16(numerator / denominator))
            .collect();
        for depth in 0..head_size {
            let values: Vec<f32> = (0..token_count)
                .map(|token| {
                    let index = (key_value_head * token_count + token) * head_size + depth;
                    value[index]
                })
                .collect();
            output[query_head * head_size + depth] =
                f32_to_bf16_bits(fixed37_dot(&probabilities, &values));
        }
    }
    output
}

fn run_contiguous(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    token_count: usize,
    maximum_token_count: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_size: usize,
    scale: f32,
    workspace_sentinel: Option<f32>,
    preference: DecodeAttentionPreference,
) -> TestResult<ExecutionBits> {
    let baseline = context.allocation_stats()?;
    let query = upload(context, stream, staging, &encode_bf16(query))?;
    let key = upload(context, stream, staging, &encode_bf16(key))?;
    let value = upload(context, stream, staging, &encode_bf16(value))?;
    let prepared = PreparedDecodeAttention::select_with_reduction_profile(
        context,
        DecodeAttentionRequest::new(
            u64::try_from(maximum_token_count)?,
            u64::try_from(query_head_count)?,
            u64::try_from(key_value_head_count)?,
            u64::try_from(head_size)?,
            scale,
        ),
        preference,
        FIXED37,
        DecodeAttentionBackendAvailability::linked(),
    )?;
    if preference == DecodeAttentionPreference::Optimized {
        assert_eq!(prepared.backend(), DecodeAttentionBackend::Fixed37TwoPass);
        assert_eq!(prepared.workspace_bytes(), 0);
    }
    let mut output =
        context.allocate_device_buffer(u64::try_from(query_head_count * head_size * 2)?)?;
    let mut workspace = if prepared.workspace_bytes() == 0 {
        assert!(workspace_sentinel.is_none());
        None
    } else if let Some(sentinel) = workspace_sentinel {
        Some(upload(
            context,
            stream,
            staging,
            &encode_bf16(&vec![
                sentinel;
                usize::try_from(prepared.workspace_bytes() / 2)?
            ]),
        )?)
    } else {
        Some(context.allocate_device_buffer(prepared.workspace_bytes())?)
    };
    let active = context.allocation_stats()?;
    let mut observed = None;
    for _ in 0..3 {
        let output_len = output.byte_len();
        if prepared.backend() == DecodeAttentionBackend::Fixed37TwoPass {
            prepared.execute_without_workspace(
                u64::try_from(token_count)?,
                &mut DecodeAttentionNoWorkspaceParams {
                    query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                    key_cache: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
                    value_cache: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                },
                stream,
            )?;
        } else {
            let workspace = workspace.as_mut().ok_or("missing materialized workspace")?;
            let workspace_len = workspace.byte_len();
            prepared.execute(
                u64::try_from(token_count)?,
                &mut DecodeAttentionParams {
                    query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                    key_cache: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
                    value_cache: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                    workspace: CudaBufferSpanMut::new(
                        workspace,
                        CudaDType::BF16,
                        0,
                        workspace_len,
                    )?,
                },
                stream,
            )?;
        }
        assert_eq!(context.allocation_stats()?, active);
        let current = ExecutionBits {
            output: download_bits(context, stream, &mut output)?,
            probabilities: if let Some(workspace) = workspace.as_mut() {
                download_bits(context, stream, workspace)?
            } else {
                Vec::new()
            },
        };
        if let Some(previous) = &observed {
            assert_eq!(current, *previous, "fixed37 contiguous repeatability");
        }
        observed = Some(current);
    }
    if let Some(workspace) = workspace {
        workspace.close()?;
    }
    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    assert_eq!(context.allocation_stats()?, baseline);
    observed.ok_or_else(|| "missing contiguous execution".into())
}

fn paged_index(
    physical_block: usize,
    head: usize,
    token_in_block: usize,
    depth: usize,
    key_value_head_count: usize,
    head_size: usize,
) -> usize {
    (((physical_block * key_value_head_count + head) * 16 + token_in_block) * head_size) + depth
}

fn run_paged(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    token_count: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_size: usize,
    scale: f32,
    preference: DecodeAttentionPreference,
) -> TestResult<ExecutionBits> {
    let baseline = context.allocation_stats()?;
    let logical_block_count = token_count.div_ceil(16);
    let physical_block_count = logical_block_count + 3;
    let block_ids: Vec<u32> = (0..logical_block_count)
        .map(|logical| u32::try_from(physical_block_count - 1 - logical))
        .collect::<Result<_, _>>()?;
    let mut valid_tokens = vec![16_u16; logical_block_count];
    *valid_tokens.last_mut().ok_or("missing final paged block")? =
        u16::try_from((token_count - 1) % 16 + 1)?;
    let host_table = PagedKvBlockTableHostV1::new(
        &block_ids,
        &valid_tokens,
        u64::try_from(token_count)?,
        u64::try_from(physical_block_count)?,
    )?;
    let pool_elements = physical_block_count * key_value_head_count * 16 * head_size;
    let mut key_pool = vec![f32::NAN; pool_elements];
    let mut value_pool = vec![f32::NAN; pool_elements];
    for token in 0..token_count {
        let logical_block = token / 16;
        let physical_block = usize::try_from(block_ids[logical_block])?;
        for head in 0..key_value_head_count {
            for depth in 0..head_size {
                let contiguous = (head * token_count + token) * head_size + depth;
                let paged = paged_index(
                    physical_block,
                    head,
                    token % 16,
                    depth,
                    key_value_head_count,
                    head_size,
                );
                key_pool[paged] = key[contiguous];
                value_pool[paged] = value[contiguous];
            }
        }
    }
    let query = upload(context, stream, staging, &encode_bf16(query))?;
    let key_pool = upload(context, stream, staging, &encode_bf16(&key_pool))?;
    let value_pool = upload(context, stream, staging, &encode_bf16(&value_pool))?;
    let device_ids = upload(context, stream, staging, &encode_u32(&block_ids))?;
    let device_valid = upload(context, stream, staging, &encode_u16(&valid_tokens))?;
    let prepared = PreparedPagedDecodeAttention::select_with_reduction_profile(
        context,
        PagedDecodeAttentionRequest::new(
            u64::try_from(token_count)?,
            u64::try_from(physical_block_count)?,
            u64::try_from(query_head_count)?,
            u64::try_from(key_value_head_count)?,
            u64::try_from(head_size)?,
            scale,
        ),
        preference,
        FIXED37,
        DecodeAttentionBackendAvailability::linked(),
    )?;
    if preference == DecodeAttentionPreference::Optimized {
        assert_eq!(prepared.backend(), DecodeAttentionBackend::Fixed37TwoPass);
        assert_eq!(prepared.workspace_bytes(), 0);
    }
    let mut output =
        context.allocate_device_buffer(u64::try_from(query_head_count * head_size * 2)?)?;
    let mut workspace = if prepared.workspace_bytes() == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(prepared.workspace_bytes())?)
    };
    let active = context.allocation_stats()?;
    let mut observed = None;
    for _ in 0..3 {
        let output_len = output.byte_len();
        if prepared.backend() == DecodeAttentionBackend::Fixed37TwoPass {
            prepared.execute_without_workspace(
                &mut PagedDecodeAttentionNoWorkspaceParams {
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
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                    block_table: PagedKvBlockTableV1::new(
                        host_table,
                        CudaBufferSpan::new(&device_ids, CudaDType::U32, 0, device_ids.byte_len())?,
                        CudaBufferSpan::new(
                            &device_valid,
                            CudaDType::U16,
                            0,
                            device_valid.byte_len(),
                        )?,
                    )?,
                },
                stream,
            )?;
        } else {
            let workspace = workspace.as_mut().ok_or("missing materialized workspace")?;
            let workspace_len = workspace.byte_len();
            prepared.execute(
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
                        workspace,
                        CudaDType::BF16,
                        0,
                        workspace_len,
                    )?,
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                    block_table: PagedKvBlockTableV1::new(
                        host_table,
                        CudaBufferSpan::new(&device_ids, CudaDType::U32, 0, device_ids.byte_len())?,
                        CudaBufferSpan::new(
                            &device_valid,
                            CudaDType::U16,
                            0,
                            device_valid.byte_len(),
                        )?,
                    )?,
                },
                stream,
            )?;
        }
        assert_eq!(context.allocation_stats()?, active);
        let current = ExecutionBits {
            output: download_bits(context, stream, &mut output)?,
            probabilities: if let Some(workspace) = workspace.as_mut() {
                download_bits(context, stream, workspace)?
            } else {
                Vec::new()
            },
        };
        if let Some(previous) = &observed {
            assert_eq!(current, *previous, "fixed37 paged repeatability");
        }
        observed = Some(current);
    }
    if let Some(workspace) = workspace {
        workspace.close()?;
    }
    output.close()?;
    device_valid.close()?;
    device_ids.close()?;
    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    assert_eq!(context.allocation_stats()?, baseline);
    observed.ok_or_else(|| "missing paged execution".into())
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn two_pass_contiguous_and_shuffled_paged_are_exact_at_boundaries_and_t8192() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(4 << 20)?;
    let query_head_count = 2;
    let key_value_head_count = 1;
    let head_size = 64;
    for token_count in [1, 36, 37, 38, 8192] {
        let query: Vec<f32> = (0..query_head_count * head_size)
            .map(|index| {
                (f32::from(u8::try_from((index * 11 + 3) % 29).unwrap_or(0)) - 14.0) / 32.0
            })
            .collect();
        let cache_elements = key_value_head_count * token_count * head_size;
        let key: Vec<f32> = (0..cache_elements)
            .map(|index| (f32::from(u8::try_from((index * 5 + 7) % 31).unwrap_or(0)) - 15.0) / 32.0)
            .collect();
        let value: Vec<f32> = (0..cache_elements)
            .map(|index| {
                (f32::from(u8::try_from((index * 13 + 9) % 37).unwrap_or(0)) - 18.0) / 32.0
            })
            .collect();
        let expected = cpu_fixed37_decode(
            &query,
            &key,
            &value,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            0.125,
        );
        let contiguous = run_contiguous(
            &context,
            &mut stream,
            &mut staging,
            &query,
            &key,
            &value,
            token_count,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            0.125,
            None,
            DecodeAttentionPreference::Optimized,
        )?;
        let paged = run_paged(
            &context,
            &mut stream,
            &mut staging,
            &query,
            &key,
            &value,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            0.125,
            DecodeAttentionPreference::Optimized,
        )?;
        let materialized = run_contiguous(
            &context,
            &mut stream,
            &mut staging,
            &query,
            &key,
            &value,
            token_count,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            0.125,
            None,
            DecodeAttentionPreference::Reference,
        )?;
        assert!(contiguous.probabilities.is_empty());
        assert_eq!(contiguous, paged, "page16 must be address translation only");
        assert_eq!(
            contiguous.output, materialized.output,
            "two-pass must byte-match fixed37 materialized at T={token_count}"
        );
        for (index, (&actual, &expected)) in contiguous.output.iter().zip(&expected).enumerate() {
            let difference = (bf16_to_f32(actual) - bf16_to_f32(expected)).abs();
            assert!(
                difference <= 0.03125,
                "two-pass oracle mismatch at T={token_count}, index={index}: {difference}"
            );
        }
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn contiguous_and_shuffled_paged_match_fixed37_oracle_at_chunk_and_page_boundaries() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let query_head_count = 9;
    let key_value_head_count = 3;
    for (token_count, head_size) in [(36, 37), (38, 74), (75, 73)] {
        let query: Vec<f32> = (0..query_head_count * head_size)
            .map(|index| {
                (f32::from(u8::try_from((index * 11 + 3) % 29).unwrap_or(0)) - 14.0) / 32.0
            })
            .collect();
        let cache_elements = key_value_head_count * token_count * head_size;
        let key: Vec<f32> = (0..cache_elements)
            .map(|index| (f32::from(u8::try_from((index * 5 + 7) % 31).unwrap_or(0)) - 15.0) / 32.0)
            .collect();
        let value: Vec<f32> = (0..cache_elements)
            .map(|index| {
                (f32::from(u8::try_from((index * 13 + 9) % 37).unwrap_or(0)) - 18.0) / 32.0
            })
            .collect();
        let expected = cpu_fixed37_decode(
            &query,
            &key,
            &value,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            0.125,
        );
        let contiguous = run_contiguous(
            &context,
            &mut stream,
            &mut staging,
            &query,
            &key,
            &value,
            token_count,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            0.125,
            None,
            DecodeAttentionPreference::Reference,
        )?;
        let paged = run_paged(
            &context,
            &mut stream,
            &mut staging,
            &query,
            &key,
            &value,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            0.125,
            DecodeAttentionPreference::Reference,
        )?;
        assert_eq!(contiguous, paged, "page16 must be address translation only");
        for (index, (&actual, &expected)) in contiguous.output.iter().zip(&expected).enumerate() {
            let difference = (bf16_to_f32(actual) - bf16_to_f32(expected)).abs();
            assert!(
                difference <= 0.03125,
                "fixed37 oracle mismatch at T={token_count}, D={head_size}, index={index}: {difference}"
            );
        }
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn qk_and_av_order_witnesses_pin_staged_bf16_fixed37_results() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;

    let query = vec![1.0_f32; 74];
    let mut key = vec![0.0_f32; 2 * 74];
    key[0] = 16_777_216.0;
    key[1..73].fill(1.0);
    key[73] = -16_777_216.0;
    let mut value = vec![0.0_f32; 2 * 74];
    value[..74].fill(1.0);
    let qk = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        2,
        2,
        1,
        1,
        74,
        1.0,
        None,
        DecodeAttentionPreference::Reference,
    )?;
    let qk_paged = run_paged(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        2,
        1,
        1,
        74,
        1.0,
        DecodeAttentionPreference::Reference,
    )?;
    assert_eq!(qk_paged, qk, "paged QK must keep the logical D reduction");
    assert_eq!(qk.output, vec![f32_to_bf16_bits(1.0); 74]);
    let flat_dot = query
        .iter()
        .zip(&key[..74])
        .fold(0.0_f32, |sum, (&left, &right)| left.mul_add(right, sum));
    assert_eq!(flat_dot.to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        fixed37_dot(&query, &key[..74]).to_bits(),
        36.0_f32.to_bits()
    );
    assert_ne!(qk.output[0], f32_to_bf16_bits(0.5));

    let token_count = 74;
    let query = [1.0_f32];
    let key = vec![0.0_f32; token_count];
    let mut value = vec![1.0_f32; token_count];
    value[0] = 16_777_216.0;
    value[token_count - 1] = -16_777_216.0;
    let av = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        token_count,
        token_count,
        1,
        1,
        1,
        1.0,
        None,
        DecodeAttentionPreference::Reference,
    )?;
    let av_paged = run_paged(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        token_count,
        1,
        1,
        1,
        1.0,
        DecodeAttentionPreference::Reference,
    )?;
    assert_eq!(av_paged, av, "paged AV must keep token-zero chunk anchors");
    let probability = round_bf16(1.0 / 74.0);
    let probabilities = vec![probability; token_count];
    let values: Vec<f32> = value.iter().copied().map(round_bf16).collect();
    let fixed_bits = f32_to_bf16_bits(fixed37_dot(&probabilities, &values));
    let flat_bits = f32_to_bf16_bits(
        probabilities
            .iter()
            .zip(&values)
            .fold(0.0_f32, |sum, (&probability, &value)| {
                probability.mul_add(value, sum)
            }),
    );
    assert_ne!(
        fixed_bits, flat_bits,
        "AV witness must distinguish reduction order"
    );
    assert_eq!(av.output, vec![fixed_bits]);

    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn two_pass_d64_order_witnesses_and_special_values_are_pinned() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let head_size = 64;

    let query = vec![1.0_f32; head_size];
    let mut key = vec![0.0_f32; 2 * head_size];
    key[0] = 16_777_216.0;
    key[1..head_size - 1].fill(1.0);
    key[head_size - 1] = -16_777_216.0;
    let mut value = vec![0.0_f32; 2 * head_size];
    value[..head_size].fill(1.0);
    let flat = query
        .iter()
        .zip(&key[..head_size])
        .fold(0.0_f32, |sum, (&left, &right)| left.mul_add(right, sum));
    let fixed = fixed37_dot(&query, &key[..head_size]);
    assert_eq!(flat.to_bits(), 0.0_f32.to_bits());
    assert_ne!(fixed.to_bits(), flat.to_bits());
    let qk = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        2,
        2,
        1,
        1,
        head_size,
        1.0,
        None,
        DecodeAttentionPreference::Optimized,
    )?;
    let qk_paged = run_paged(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        2,
        1,
        1,
        head_size,
        1.0,
        DecodeAttentionPreference::Optimized,
    )?;
    let qk_materialized = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        2,
        2,
        1,
        1,
        head_size,
        1.0,
        None,
        DecodeAttentionPreference::Reference,
    )?;
    assert_eq!(qk, qk_paged);
    assert_eq!(qk.output, qk_materialized.output);
    assert_ne!(qk.output[0], f32_to_bf16_bits(0.5));

    let token_count = 74;
    let query = vec![0.0_f32; head_size];
    let key = vec![0.0_f32; token_count * head_size];
    let mut token_values = vec![1.0_f32; token_count];
    token_values[0] = 16_777_216.0;
    token_values[token_count - 1] = -16_777_216.0;
    let value: Vec<f32> = token_values
        .iter()
        .flat_map(|&token_value| std::iter::repeat_n(token_value, head_size))
        .collect();
    let probability = round_bf16(1.0 / f32::from(u16::try_from(token_count)?));
    let fixed_av = f32_to_bf16_bits(fixed37_dot(&vec![probability; token_count], &token_values));
    let flat_av = f32_to_bf16_bits(
        token_values
            .iter()
            .fold(0.0_f32, |sum, &item| probability.mul_add(item, sum)),
    );
    assert_ne!(fixed_av, flat_av);
    let av = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        token_count,
        token_count,
        1,
        1,
        head_size,
        1.0,
        None,
        DecodeAttentionPreference::Optimized,
    )?;
    let av_paged = run_paged(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        token_count,
        1,
        1,
        head_size,
        1.0,
        DecodeAttentionPreference::Optimized,
    )?;
    let av_materialized = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        token_count,
        token_count,
        1,
        1,
        head_size,
        1.0,
        None,
        DecodeAttentionPreference::Reference,
    )?;
    assert_eq!(av, av_paged);
    assert_eq!(av.output, av_materialized.output);
    assert_eq!(av.output, vec![fixed_av; head_size]);

    for (query_scalar, key_scalar) in [
        (f32::NAN, 1.0_f32),
        (f32::INFINITY, 1.0),
        (f32::INFINITY, -1.0),
    ] {
        let mut query = vec![0.0_f32; head_size];
        query[0] = query_scalar;
        let mut key = vec![0.0_f32; head_size];
        key[0] = key_scalar;
        let special = run_contiguous(
            &context,
            &mut stream,
            &mut staging,
            &query,
            &key,
            &vec![2.0; head_size],
            1,
            1,
            1,
            1,
            head_size,
            1.0,
            None,
            DecodeAttentionPreference::Optimized,
        )?;
        assert_eq!(special.output, vec![0x7fff; head_size]);
    }

    let mut query = vec![0.0_f32; head_size];
    query[0] = 1.0;
    let mut key = vec![0.0_f32; 2 * head_size];
    key[head_size] = f32::NEG_INFINITY;
    let mut value = vec![2.0_f32; 2 * head_size];
    value[head_size..].fill(f32::INFINITY);
    let zero_times_infinity = run_paged(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        2,
        1,
        1,
        head_size,
        1.0,
        DecodeAttentionPreference::Optimized,
    )?;
    assert_eq!(zero_times_infinity.output, vec![0x7fff; head_size]);

    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn contiguous_materialized_workspace_preserves_capacity_tail() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let token_count = 38;
    let maximum_token_count = 75;
    let query_head_count = 9;
    let key_value_head_count = 3;
    let head_size = 37;
    let sentinel = -7.5_f32;
    let query = vec![1.0_f32; query_head_count * head_size];
    let key = vec![0.0_f32; key_value_head_count * maximum_token_count * head_size];
    let mut value = vec![0.0_f32; key_value_head_count * maximum_token_count * head_size];
    for head in 0..key_value_head_count {
        for token in 0..token_count {
            let begin = (head * maximum_token_count + token) * head_size;
            value[begin..begin + head_size].fill(1.0);
        }
    }
    let execution = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &query,
        &key,
        &value,
        token_count,
        maximum_token_count,
        query_head_count,
        key_value_head_count,
        head_size,
        1.0,
        Some(sentinel),
        DecodeAttentionPreference::Reference,
    )?;
    assert_eq!(
        execution.output,
        vec![f32_to_bf16_bits(1.0); query_head_count * head_size]
    );
    let active_elements = query_head_count * token_count;
    let sentinel_bits = f32_to_bf16_bits(sentinel);
    assert!(
        execution.probabilities[..active_elements]
            .iter()
            .all(|&bits| bits != sentinel_bits)
    );
    assert!(
        execution.probabilities[active_elements..]
            .iter()
            .all(|&bits| bits == sentinel_bits),
        "materialized decode must not touch QH*(M-T) workspace capacity tail"
    );

    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn special_rows_and_zero_probability_times_infinity_are_canonical_qnan() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    for (query, key) in [
        (f32::NAN, 1.0_f32),
        (f32::INFINITY, 1.0),
        (f32::INFINITY, -1.0),
    ] {
        let execution = run_contiguous(
            &context,
            &mut stream,
            &mut staging,
            &[query],
            &[key],
            &[2.0],
            1,
            1,
            1,
            1,
            1,
            1.0,
            None,
            DecodeAttentionPreference::Reference,
        )?;
        assert_eq!(execution.probabilities, vec![0x7fff]);
        assert_eq!(execution.output, vec![0x7fff]);
    }

    let zero_times_infinity = run_contiguous(
        &context,
        &mut stream,
        &mut staging,
        &[1.0],
        &[0.0, f32::NEG_INFINITY],
        &[2.0, f32::INFINITY],
        2,
        2,
        1,
        1,
        1,
        1.0,
        None,
        DecodeAttentionPreference::Reference,
    )?;
    assert_eq!(
        zero_times_infinity.probabilities,
        vec![f32_to_bf16_bits(1.0), 0]
    );
    assert_eq!(zero_times_infinity.output, vec![0x7fff]);

    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn paged_corrupt_device_metadata_returns_qnan_without_out_of_bounds_access() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    let baseline = context.allocation_stats()?;
    let host_ids = [0_u32];
    let host_valid = [1_u16];
    let host_table = PagedKvBlockTableHostV1::new(&host_ids, &host_valid, 1, 1)?;
    let query = upload(&context, &mut stream, &mut staging, &encode_bf16(&[1.0]))?;
    let key_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[1.0; 16]),
    )?;
    let value_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[2.0; 16]),
    )?;
    let prepared = PreparedPagedDecodeAttention::select_with_reduction_profile(
        &context,
        PagedDecodeAttentionRequest::new(1, 1, 1, 1, 1, 1.0),
        DecodeAttentionPreference::Reference,
        FIXED37,
        DecodeAttentionBackendAvailability::linked(),
    )?;

    for (actual_ids, actual_valid) in [([1_u32], [1_u16]), ([0], [0])] {
        let device_ids = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u32(&actual_ids),
        )?;
        let device_valid = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u16(&actual_valid),
        )?;
        let mut workspace = context.allocate_device_buffer(prepared.workspace_bytes())?;
        let mut output = context.allocate_device_buffer(2)?;
        let workspace_len = workspace.byte_len();
        let output_len = output.byte_len();
        prepared.execute(
            &mut PagedDecodeAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
                value_pool: CudaBufferSpan::new(
                    &value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool.byte_len(),
                )?,
                workspace: CudaBufferSpanMut::new(
                    &mut workspace,
                    CudaDType::BF16,
                    0,
                    workspace_len,
                )?,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                block_table: PagedKvBlockTableV1::new(
                    host_table,
                    CudaBufferSpan::new(&device_ids, CudaDType::U32, 0, device_ids.byte_len())?,
                    CudaBufferSpan::new(&device_valid, CudaDType::U16, 0, device_valid.byte_len())?,
                )?,
            },
            &mut stream,
        )?;
        assert_eq!(
            download_bits(&context, &mut stream, &mut workspace)?,
            vec![0x7fff]
        );
        assert_eq!(
            download_bits(&context, &mut stream, &mut output)?,
            vec![0x7fff]
        );
        output.close()?;
        workspace.close()?;
        device_valid.close()?;
        device_ids.close()?;
    }

    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    assert_eq!(context.allocation_stats()?, baseline);
    drop(prepared);
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires remote CUDA GPU"]
fn two_pass_paged_corrupt_metadata_is_full_qnan_without_workspace() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 16)?;
    let baseline = context.allocation_stats()?;
    let host_ids = [0_u32];
    let host_valid = [1_u16];
    let host_table = PagedKvBlockTableHostV1::new(&host_ids, &host_valid, 1, 1)?;
    let query = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[1.0; 64]),
    )?;
    let key_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[1.0; 16 * 64]),
    )?;
    let value_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[2.0; 16 * 64]),
    )?;
    let prepared = PreparedPagedDecodeAttention::select_with_reduction_profile(
        &context,
        PagedDecodeAttentionRequest::new(1, 1, 1, 1, 64, 1.0),
        DecodeAttentionPreference::Optimized,
        FIXED37,
        DecodeAttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), DecodeAttentionBackend::Fixed37TwoPass);
    assert_eq!(prepared.workspace_bytes(), 0);

    let valid_device_ids = upload(&context, &mut stream, &mut staging, &encode_u32(&host_ids))?;
    let valid_device_tokens = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&host_valid),
    )?;
    let mut compatibility_workspace =
        upload(&context, &mut stream, &mut staging, &encode_bf16(&[9.0]))?;
    let mut compatibility_output = context.allocate_device_buffer(64 * 2)?;
    let compatibility_workspace_len = compatibility_workspace.byte_len();
    let compatibility_output_len = compatibility_output.byte_len();
    let compatibility_active = context.allocation_stats()?;
    prepared.execute(
        &mut PagedDecodeAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            workspace: CudaBufferSpanMut::new(
                &mut compatibility_workspace,
                CudaDType::BF16,
                0,
                compatibility_workspace_len,
            )?,
            output: CudaBufferSpanMut::new(
                &mut compatibility_output,
                CudaDType::BF16,
                0,
                compatibility_output_len,
            )?,
            block_table: PagedKvBlockTableV1::new(
                host_table,
                CudaBufferSpan::new(
                    &valid_device_ids,
                    CudaDType::U32,
                    0,
                    valid_device_ids.byte_len(),
                )?,
                CudaBufferSpan::new(
                    &valid_device_tokens,
                    CudaDType::U16,
                    0,
                    valid_device_tokens.byte_len(),
                )?,
            )?,
        },
        &mut stream,
    )?;
    assert_eq!(context.allocation_stats()?, compatibility_active);
    assert_eq!(
        download_bits(&context, &mut stream, &mut compatibility_workspace)?,
        vec![f32_to_bf16_bits(9.0)],
        "legacy two-pass execute must not touch its compatibility workspace"
    );
    assert_eq!(
        download_bits(&context, &mut stream, &mut compatibility_output)?,
        vec![f32_to_bf16_bits(2.0); 64]
    );

    let materialized = PreparedPagedDecodeAttention::select_with_reduction_profile(
        &context,
        PagedDecodeAttentionRequest::new(1, 1, 1, 1, 64, 1.0),
        DecodeAttentionPreference::Reference,
        FIXED37,
        DecodeAttentionBackendAvailability::linked(),
    )?;
    let error = materialized
        .execute_without_workspace(
            &mut PagedDecodeAttentionNoWorkspaceParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
                value_pool: CudaBufferSpan::new(
                    &value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool.byte_len(),
                )?,
                output: CudaBufferSpanMut::new(
                    &mut compatibility_output,
                    CudaDType::BF16,
                    0,
                    compatibility_output_len,
                )?,
                block_table: PagedKvBlockTableV1::new(
                    host_table,
                    CudaBufferSpan::new(
                        &valid_device_ids,
                        CudaDType::U32,
                        0,
                        valid_device_ids.byte_len(),
                    )?,
                    CudaBufferSpan::new(
                        &valid_device_tokens,
                        CudaDType::U16,
                        0,
                        valid_device_tokens.byte_len(),
                    )?,
                )?,
            },
            &mut stream,
        )
        .expect_err("no-workspace execution must reject a materialized plan before launch");
    assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
    drop(materialized);
    compatibility_output.close()?;
    compatibility_workspace.close()?;
    valid_device_tokens.close()?;
    valid_device_ids.close()?;

    for (actual_ids, actual_valid) in [([1_u32], [1_u16]), ([0], [0])] {
        let device_ids = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u32(&actual_ids),
        )?;
        let device_valid = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u16(&actual_valid),
        )?;
        let mut output = context.allocate_device_buffer(64 * 2)?;
        let output_len = output.byte_len();
        let active = context.allocation_stats()?;
        for _ in 0..3 {
            prepared.execute_without_workspace(
                &mut PagedDecodeAttentionNoWorkspaceParams {
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
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                    block_table: PagedKvBlockTableV1::new(
                        host_table,
                        CudaBufferSpan::new(&device_ids, CudaDType::U32, 0, device_ids.byte_len())?,
                        CudaBufferSpan::new(
                            &device_valid,
                            CudaDType::U16,
                            0,
                            device_valid.byte_len(),
                        )?,
                    )?,
                },
                &mut stream,
            )?;
            assert_eq!(context.allocation_stats()?, active);
            assert_eq!(
                download_bits(&context, &mut stream, &mut output)?,
                vec![0x7fff; 64]
            );
        }
        output.close()?;
        device_valid.close()?;
        device_ids.close()?;
    }

    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    assert_eq!(context.allocation_stats()?, baseline);
    drop(prepared);
    staging.close()?;
    stream.close()?;
    close_context(context)
}
