#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaError,
    CudaErrorKind, CudaEvent, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    DecodeAttentionBackend, DecodeAttentionBackendAvailability, DecodeAttentionPreference,
    DecodePartialReductionOrder, DecodePartialStateReduceParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout, PAGED_KV_BLOCK_SIZE,
    PackedBatchHostV1, PackedBatchV1, PagedDecodeAttentionParams, PagedDecodeAttentionRequest,
    PagedKvBlockTableHostV1, PagedKvBlockTableV1, PagedKvCacheAppendParams,
    PreparedPagedDecodeAttention, decode_partial_states_reduce,
    native_bf16_ragged_paged_split_gqa_d128_two_stage, paged_kv_cache_append,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_SIZE: usize = 128;
const PAGE_SIZE: usize = 16;
// IEEE-754 f32 representation of 1 / sqrt(128), pinned without a non-const
// `sqrt` call so this CUDA integration target builds on the workspace MSRV.
const SCALE: f32 = 0.088_388_346;
const MAX_CONTEXT: usize = 16_384;
const PHYSICAL_BLOCKS: usize = MAX_CONTEXT.div_ceil(PAGE_SIZE) + 3;
const STAGING_BYTES: u64 = 16 * 1024 * 1024;
const REFERENCE_CPU_ABS_TOLERANCE: f32 = 0.03125;
const NATIVE_REFERENCE_ABS_TOLERANCE: f32 = 0.0625;
const POOL_SENTINEL: f32 = -77.0;
const STATE_SENTINEL: f32 = -1_234.5;
const OUTPUT_SENTINEL: f32 = -7.25;
const INTERNAL_WARMUPS: usize = 8;
const PAIRED_ROUNDS: usize = 24;

#[derive(Clone, Copy, Debug)]
enum FixtureStyle {
    Patterned,
    Cancellation,
}

impl FixtureStyle {
    const fn label(self) -> &'static str {
        match self {
            Self::Patterned => "patterned",
            Self::Cancellation => "high-cancellation",
        }
    }
}

struct PagedFixture {
    logical_token_count: usize,
    physical_block_count: usize,
    block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
    query: CudaDeviceBuffer,
    key_pool: CudaDeviceBuffer,
    value_pool: CudaDeviceBuffer,
    ids: CudaDeviceBuffer,
    valid: CudaDeviceBuffer,
}

impl PagedFixture {
    fn device_table(&self) -> TestResult<PagedKvBlockTableV1<'_>> {
        let host = PagedKvBlockTableHostV1::new(
            &self.block_ids,
            &self.valid_tokens,
            u64::try_from(self.logical_token_count)?,
            u64::try_from(self.physical_block_count)?,
        )?;
        Ok(PagedKvBlockTableV1::new(
            host,
            CudaBufferSpan::new(&self.ids, CudaDType::U32, 0, self.ids.byte_len())?,
            CudaBufferSpan::new(&self.valid, CudaDType::U16, 0, self.valid.byte_len())?,
        )?)
    }

    fn close(self) -> TestResult {
        self.valid.close()?;
        self.ids.close()?;
        self.value_pool.close()?;
        self.key_pool.close()?;
        self.query.close()?;
        Ok(())
    }
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

fn source_index(token: usize, head: usize, depth: usize) -> usize {
    (token * KEY_VALUE_HEADS + head) * HEAD_SIZE + depth
}

fn centered_pattern(
    count: usize,
    multiplier: usize,
    addend: usize,
    modulus: usize,
    center: f32,
    divisor: f32,
) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let value = u8::try_from((index * multiplier + addend) % modulus).unwrap_or(0);
            (f32::from(value) - center) / divisor
        })
        .collect()
}

fn fixture_values(logical: usize, style: FixtureStyle) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    match style {
        FixtureStyle::Patterned => (
            centered_pattern(QUERY_HEADS * HEAD_SIZE, 11, 0, 29, 14.0, 32.0),
            centered_pattern(logical * KEY_VALUE_HEADS * HEAD_SIZE, 5, 3, 31, 15.0, 32.0),
            centered_pattern(logical * KEY_VALUE_HEADS * HEAD_SIZE, 13, 7, 37, 18.0, 32.0),
        ),
        FixtureStyle::Cancellation => {
            let mut query = Vec::with_capacity(QUERY_HEADS * HEAD_SIZE);
            for _head in 0..QUERY_HEADS {
                for depth in 0..HEAD_SIZE {
                    query.push(if depth % 2 == 0 { 0.25 } else { -0.25 });
                }
            }
            let mut key = Vec::with_capacity(logical * KEY_VALUE_HEADS * HEAD_SIZE);
            let mut value = Vec::with_capacity(logical * KEY_VALUE_HEADS * HEAD_SIZE);
            for token in 0..logical {
                for head in 0..KEY_VALUE_HEADS {
                    for depth in 0..HEAD_SIZE {
                        let perturbation = match (token + head) % 5 {
                            0 => -0.007_812_5,
                            1 => -0.003_906_25,
                            2 => 0.0,
                            3 => 0.003_906_25,
                            _ => 0.007_812_5,
                        };
                        key.push(if depth == 0 {
                            0.25 + perturbation
                        } else {
                            0.25
                        });
                        value.push(if (token + head + depth) % 2 == 0 {
                            0.5
                        } else {
                            -0.5
                        });
                    }
                }
            }
            (query, key, value)
        }
    }
}

fn cpu_decode(query: &[f32], key: &[f32], value: &[f32], logical: usize) -> Vec<f32> {
    let group_size = QUERY_HEADS / KEY_VALUE_HEADS;
    let mut output = vec![0.0_f32; QUERY_HEADS * HEAD_SIZE];
    for query_head in 0..QUERY_HEADS {
        let key_value_head = query_head / group_size;
        let mut scores = vec![0.0_f32; logical];
        for (token, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0_f32;
            for depth in 0..HEAD_SIZE {
                dot = query[query_head * HEAD_SIZE + depth]
                    .mul_add(key[source_index(token, key_value_head, depth)], dot);
            }
            let staged_dot = bf16_to_f32(f32_to_bf16_bits(dot));
            *score = bf16_to_f32(f32_to_bf16_bits(staged_dot * SCALE));
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
        for score in &mut scores {
            *score = bf16_to_f32(f32_to_bf16_bits((*score - maximum).exp() / denominator));
        }
        for depth in 0..HEAD_SIZE {
            let mut sum = 0.0_f32;
            for (token, &probability) in scores.iter().enumerate() {
                sum = probability.mul_add(value[source_index(token, key_value_head, depth)], sum);
            }
            output[query_head * HEAD_SIZE + depth] = bf16_to_f32(f32_to_bf16_bits(sum));
        }
    }
    output
}

fn setup_fixture(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    logical: usize,
    physical_block_count: usize,
    style: FixtureStyle,
) -> TestResult<(PagedFixture, Vec<f32>)> {
    let logical_blocks = logical.div_ceil(PAGE_SIZE);
    assert!(
        logical_blocks + 3 <= physical_block_count,
        "fixture physical pool must leave unused blocks"
    );
    let block_ids: Vec<u32> = (0..logical_blocks)
        .rev()
        .map(|index| u32::try_from(index + 3))
        .collect::<Result<_, _>>()?;
    let mut valid_tokens = vec![u16::try_from(PAGE_SIZE)?; logical_blocks];
    *valid_tokens.last_mut().ok_or("missing final block")? =
        u16::try_from((logical - 1) % PAGE_SIZE + 1)?;

    let (query_values, key_source_values, value_source_values) = fixture_values(logical, style);
    let query_bytes = encode_bf16(&query_values);
    let key_source_bytes = encode_bf16(&key_source_values);
    let value_source_bytes = encode_bf16(&value_source_values);
    let expected = cpu_decode(
        &decode_bf16(&query_bytes),
        &decode_bf16(&key_source_bytes),
        &decode_bf16(&value_source_bytes),
        logical,
    );

    let query = upload(context, stream, staging, &query_bytes)?;
    let key_source = upload(context, stream, staging, &key_source_bytes)?;
    let value_source = upload(context, stream, staging, &value_source_bytes)?;
    let ids = upload(context, stream, staging, &encode_u32(&block_ids))?;
    let valid = upload(context, stream, staging, &encode_u16(&valid_tokens))?;
    let pool_elements = physical_block_count * KEY_VALUE_HEADS * PAGE_SIZE * HEAD_SIZE;
    let pool_bytes = encode_bf16(&vec![POOL_SENTINEL; pool_elements]);
    let mut key_pool = upload(context, stream, staging, &pool_bytes)?;
    let mut value_pool = upload(context, stream, staging, &pool_bytes)?;
    let host_table = PagedKvBlockTableHostV1::new(
        &block_ids,
        &valid_tokens,
        u64::try_from(logical)?,
        u64::try_from(physical_block_count)?,
    )?;
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
            block_table: PagedKvBlockTableV1::new(
                host_table,
                CudaBufferSpan::new(&ids, CudaDType::U32, 0, ids.byte_len())?,
                CudaBufferSpan::new(&valid, CudaDType::U16, 0, valid.byte_len())?,
            )?,
            source_token_count: u64::try_from(logical)?,
            destination_token_start: 0,
            key_value_head_count: u64::try_from(KEY_VALUE_HEADS)?,
            head_size: u64::try_from(HEAD_SIZE)?,
        },
        stream,
    )?;
    stream.synchronize()?;
    key_source.close()?;
    value_source.close()?;
    Ok((
        PagedFixture {
            logical_token_count: logical,
            physical_block_count,
            block_ids,
            valid_tokens,
            query,
            key_pool,
            value_pool,
            ids,
            valid,
        },
        expected,
    ))
}

fn execute_prepared(
    prepared: &PreparedPagedDecodeAttention,
    fixture: &PagedFixture,
    workspace: &mut CudaDeviceBuffer,
    workspace_bytes: u64,
    output: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let output_len = output.byte_len();
    prepared.execute(
        &mut PagedDecodeAttentionParams {
            query: CudaBufferSpan::new(
                &fixture.query,
                CudaDType::BF16,
                0,
                fixture.query.byte_len(),
            )?,
            key_pool: CudaBufferSpan::new(
                &fixture.key_pool,
                CudaDType::BF16,
                0,
                fixture.key_pool.byte_len(),
            )?,
            value_pool: CudaBufferSpan::new(
                &fixture.value_pool,
                CudaDType::BF16,
                0,
                fixture.value_pool.byte_len(),
            )?,
            workspace: CudaBufferSpanMut::new(
                workspace,
                prepared.workspace_dtype(),
                0,
                workspace_bytes,
            )?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_len)?,
            block_table: fixture.device_table()?,
        },
        stream,
    )?;
    Ok(())
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length differs");
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (observed - reference).abs() <= tolerance,
            "{label}[{index}] expected {reference}, got {observed}, tolerance={tolerance}"
        );
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let upper = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[upper - 1] + samples[upper]) / 2.0
    } else {
        samples[upper]
    }
}

fn percentile(mut samples: Vec<f64>, numerator: usize, denominator: usize) -> f64 {
    samples.sort_by(f64::total_cmp);
    let rank = samples
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator);
    samples[rank.saturating_sub(1)]
}

fn timed_execute(
    prepared: &PreparedPagedDecodeAttention,
    fixture: &PagedFixture,
    workspace: &mut CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
    start: &mut CudaEvent,
    end: &mut CudaEvent,
) -> TestResult<f64> {
    start.record(stream)?;
    execute_prepared(
        prepared,
        fixture,
        workspace,
        prepared.workspace_bytes(),
        output,
        stream,
    )?;
    end.record(stream)?;
    end.synchronize()?;
    let elapsed = f64::from(start.elapsed_ms(end)?);
    assert!(
        elapsed > 0.0 && elapsed.is_finite(),
        "invalid CUDA event time {elapsed}"
    );
    Ok(elapsed)
}

#[derive(Clone, Copy, Debug)]
struct NativeV2WorkspaceLayout {
    state_prefix_bytes: u64,
    transition_step_bytes: u64,
    normalizer_bytes: u64,
}

impl NativeV2WorkspaceLayout {
    fn for_prepared(native_v2: &PreparedPagedDecodeAttention) -> TestResult<Self> {
        let state_prefix_bytes = native_v2.selection_trace().partial_state_bytes();
        let transition_step_bytes = native_v2
            .partial_state_capacity()
            .checked_mul(u64::try_from(QUERY_HEADS)?)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or("V2 transition-step workspace size overflowed")?;
        let normalizer_bytes = u64::try_from(QUERY_HEADS)?
            .checked_mul(4)
            .ok_or("V2 normalizer workspace size overflowed")?;
        let total = state_prefix_bytes
            .checked_add(transition_step_bytes)
            .and_then(|bytes| bytes.checked_add(normalizer_bytes))
            .ok_or("V2 workspace size overflowed")?;
        assert_eq!(
            native_v2.workspace_bytes(),
            total,
            "V2 workspace must be state prefix + transition steps + normalizers"
        );
        assert_eq!(state_prefix_bytes % 4, 0);
        assert_eq!(transition_step_bytes % 4, 0);
        assert_eq!(normalizer_bytes % 4, 0);
        Ok(Self {
            state_prefix_bytes,
            transition_step_bytes,
            normalizer_bytes,
        })
    }
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn native_bf16_paged_split_gqa_d128_matches_reference_and_cpu_across_pages() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(STAGING_BYTES)?;
    assert_eq!(PAGED_KV_BLOCK_SIZE, u64::try_from(PAGE_SIZE)?);
    let request = PagedDecodeAttentionRequest::new(
        u64::try_from(MAX_CONTEXT)?,
        u64::try_from(PHYSICAL_BLOCKS)?,
        u64::try_from(QUERY_HEADS)?,
        u64::try_from(KEY_VALUE_HEADS)?,
        u64::try_from(HEAD_SIZE)?,
        SCALE,
    );
    let availability = DecodeAttentionBackendAvailability::linked();
    let reference = PreparedPagedDecodeAttention::select(
        &context,
        request,
        DecodeAttentionPreference::Reference,
        availability,
    )?;
    let native_v1 = PreparedPagedDecodeAttention::select_native_bf16_paged_split_gqa_d128(
        &context,
        request,
        availability,
    )?;
    assert_eq!(
        native_v1.backend(),
        DecodeAttentionBackend::NativeBf16PagedSplitGqaD128
    );
    let native_v2 =
        PreparedPagedDecodeAttention::select_native_bf16_paged_split_gqa_d128_two_stage(
            &context,
            request,
            availability,
        )?;
    assert_eq!(
        native_v2.backend(),
        DecodeAttentionBackend::NativeBf16PagedSplitGqaD128TwoStage
    );
    assert_eq!(native_v2.workspace_dtype(), CudaDType::F32);
    assert_eq!(
        native_v2.partial_state_capacity(),
        u64::try_from(MAX_CONTEXT / PAGE_SIZE)?
    );
    assert_eq!(
        native_v1.partial_state_capacity(),
        native_v2.partial_state_capacity(),
        "V1 and V2 must share the fixed partial-state capacity"
    );
    let native_v2_layout = NativeV2WorkspaceLayout::for_prepared(&native_v2)?;
    assert_eq!(
        native_v1.workspace_bytes(),
        native_v2_layout.state_prefix_bytes,
        "the V2 state prefix must remain independently reducible by V1"
    );
    assert!(native_v2.capability().supports_partial_state_merge());
    assert!(!native_v2.capability().materializes_scores());
    assert!(!native_v2.selection_trace().supports_graph_capture());
    assert_eq!(native_v2.selection_trace().implementation_version(), "2");

    for (logical, style) in [
        (1_usize, FixtureStyle::Patterned),
        (15, FixtureStyle::Patterned),
        (16, FixtureStyle::Patterned),
        (17, FixtureStyle::Patterned),
        (33, FixtureStyle::Patterned),
        (129, FixtureStyle::Cancellation),
        (2_048, FixtureStyle::Patterned),
        (8_192, FixtureStyle::Patterned),
        (16_384, FixtureStyle::Patterned),
    ] {
        let (fixture, expected_cpu) = setup_fixture(
            &context,
            &mut stream,
            &mut staging,
            logical,
            PHYSICAL_BLOCKS,
            style,
        )?;
        let output_bytes = u64::try_from(QUERY_HEADS * HEAD_SIZE * 2)?;
        let mut reference_workspace =
            context.allocate_device_buffer(reference.workspace_bytes())?;
        let mut native_v1_workspace = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_f32(&vec![
                STATE_SENTINEL;
                usize::try_from(native_v1.workspace_bytes() / 4)?
            ]),
        )?;
        let mut native_v2_workspace = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_f32(&vec![
                STATE_SENTINEL;
                usize::try_from(native_v2.workspace_bytes() / 4)?
            ]),
        )?;
        let mut reference_output = context.allocate_device_buffer(output_bytes)?;
        let mut native_v1_output = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&vec![OUTPUT_SENTINEL; QUERY_HEADS * HEAD_SIZE]),
        )?;
        let mut native_v2_output = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&vec![OUTPUT_SENTINEL; QUERY_HEADS * HEAD_SIZE]),
        )?;
        let mut descending_output = context.allocate_device_buffer(output_bytes)?;
        let before_repeated_execution = context.allocation_stats()?;
        let mut first_native_v2_output = None;
        for attempt in 0..3 {
            execute_prepared(
                &reference,
                &fixture,
                &mut reference_workspace,
                reference.workspace_bytes(),
                &mut reference_output,
                &mut stream,
            )?;
            execute_prepared(
                &native_v1,
                &fixture,
                &mut native_v1_workspace,
                native_v1.workspace_bytes(),
                &mut native_v1_output,
                &mut stream,
            )?;
            execute_prepared(
                &native_v2,
                &fixture,
                &mut native_v2_workspace,
                native_v2.workspace_bytes(),
                &mut native_v2_output,
                &mut stream,
            )?;
            stream.synchronize()?;
            let observed = download(&context, &mut stream, &mut native_v2_output)?;
            if let Some(first) = &first_native_v2_output {
                assert_eq!(
                    &observed, first,
                    "V2 output changed on deterministic repetition {attempt} at T={logical}"
                );
            } else {
                first_native_v2_output = Some(observed);
            }
            assert_eq!(
                before_repeated_execution,
                context.allocation_stats()?,
                "prepared V2 execution allocated at T={logical}"
            );
        }

        let logical_blocks = logical.div_ceil(PAGE_SIZE);
        let descending_output_len = descending_output.byte_len();
        decode_partial_states_reduce(
            &mut DecodePartialStateReduceParams {
                partial_states: CudaBufferSpan::new(
                    &native_v2_workspace,
                    CudaDType::F32,
                    0,
                    native_v2_layout.state_prefix_bytes,
                )?,
                output: CudaBufferSpanMut::new(
                    &mut descending_output,
                    CudaDType::BF16,
                    0,
                    descending_output_len,
                )?,
                partial_state_count: u64::try_from(logical_blocks)?,
                partial_state_capacity: native_v2.partial_state_capacity(),
                query_head_count: u64::try_from(QUERY_HEADS)?,
                head_size: u64::try_from(HEAD_SIZE)?,
                order: DecodePartialReductionOrder::LogicalDescending,
            },
            &mut stream,
        )?;
        stream.synchronize()?;
        assert_eq!(
            before_repeated_execution,
            context.allocation_stats()?,
            "V1 generic descending D128 partial-state reduction allocated at T={logical}"
        );

        let reference_actual =
            decode_bf16(&download(&context, &mut stream, &mut reference_output)?);
        let native_v1_output_bytes = download(&context, &mut stream, &mut native_v1_output)?;
        let native_v2_output_bytes = first_native_v2_output
            .take()
            .ok_or("V2 deterministic output was not captured")?;
        assert_eq!(
            native_v2_output_bytes, native_v1_output_bytes,
            "V2 must replay the V1 generic ascending reducer exactly at T={logical}"
        );
        let native_v1_actual = decode_bf16(&native_v1_output_bytes);
        let native_v2_actual = decode_bf16(&native_v2_output_bytes);
        let descending_actual =
            decode_bf16(&download(&context, &mut stream, &mut descending_output)?);
        assert_close(
            &reference_actual,
            &expected_cpu,
            REFERENCE_CPU_ABS_TOLERANCE,
            &format!("materialized reference ({}) T={logical}", style.label()),
        );
        assert_close(
            &native_v1_actual,
            &reference_actual,
            NATIVE_REFERENCE_ABS_TOLERANCE,
            &format!("V1 generic D128 control ({}) T={logical}", style.label()),
        );
        assert_close(
            &native_v2_actual,
            &reference_actual,
            NATIVE_REFERENCE_ABS_TOLERANCE,
            &format!("V2 two-stage D128 ({}) T={logical}", style.label()),
        );
        assert_close(
            &descending_actual,
            &reference_actual,
            NATIVE_REFERENCE_ABS_TOLERANCE,
            &format!(
                "V1 generic descending reducer ({}) T={logical}",
                style.label()
            ),
        );

        let workspace_values =
            decode_f32(&download(&context, &mut stream, &mut native_v2_workspace)?);
        let state_prefix_elements = usize::try_from(native_v2_layout.state_prefix_bytes / 4)?;
        let transition_step_elements = usize::try_from(native_v2_layout.transition_step_bytes / 4)?;
        let normalizer_elements = usize::try_from(native_v2_layout.normalizer_bytes / 4)?;
        assert_eq!(
            workspace_values.len(),
            state_prefix_elements + transition_step_elements + normalizer_elements,
            "V2 workspace must have no unaccounted regions"
        );
        let (state_values, native_v2_scratch) = workspace_values.split_at(state_prefix_elements);
        let (transition_steps, normalizers) = native_v2_scratch.split_at(transition_step_elements);
        assert_eq!(normalizers.len(), normalizer_elements);
        let state_stride = HEAD_SIZE + 2;
        for logical_block in 0..logical_blocks {
            for query_head in 0..QUERY_HEADS {
                let state_base = (logical_block * QUERY_HEADS + query_head) * state_stride;
                assert!(
                    state_values[state_base].is_finite(),
                    "T={logical} state maximum is not finite for block={logical_block}, head={query_head}"
                );
                assert!(
                    state_values[state_base + 1].is_finite() && state_values[state_base + 1] > 0.0,
                    "T={logical} state denominator is invalid for block={logical_block}, head={query_head}"
                );
            }
        }
        let active_state_elements = logical_blocks * QUERY_HEADS * state_stride;
        assert!(
            state_values[active_state_elements..]
                .iter()
                .all(|value| value.to_bits() == STATE_SENTINEL.to_bits()),
            "V2 modified the preallocated partial-state tail at T={logical}"
        );
        let active_transition_elements = logical_blocks * QUERY_HEADS * 2;
        assert!(
            transition_steps[..active_transition_elements]
                .iter()
                .all(|value| value.is_finite()
                    && *value >= 0.0
                    && value.to_bits() != STATE_SENTINEL.to_bits()),
            "V2 transition scratch was not populated with finite scales at T={logical}"
        );
        assert!(
            transition_steps[active_transition_elements..]
                .iter()
                .all(|value| value.to_bits() == STATE_SENTINEL.to_bits()),
            "V2 modified the preallocated transition-scratch tail at T={logical}"
        );
        assert!(
            normalizers.iter().all(|value| {
                value.is_finite() && *value > 0.0 && value.to_bits() != STATE_SENTINEL.to_bits()
            }),
            "V2 normalizer scratch was not populated at T={logical}"
        );

        if logical == 33 {
            let mut failed_output = upload(
                &context,
                &mut stream,
                &mut staging,
                &encode_bf16(&vec![OUTPUT_SENTINEL; QUERY_HEADS * HEAD_SIZE]),
            )?;
            let error = execute_prepared(
                &native_v2,
                &fixture,
                &mut native_v2_workspace,
                native_v2.workspace_bytes() - 4,
                &mut failed_output,
                &mut stream,
            )
            .expect_err("undersized full V2 D128 workspace must fail before launch");
            let error = error
                .downcast_ref::<CudaError>()
                .ok_or("undersized workspace error did not preserve CudaError")?;
            assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
            assert_eq!(
                download(&context, &mut stream, &mut failed_output)?,
                encode_bf16(&vec![OUTPUT_SENTINEL; QUERY_HEADS * HEAD_SIZE]),
                "failed V2 D128 execution modified the output"
            );
            failed_output.close()?;
        }

        println!(
            "n03a-paged-d128-correctness schema_version=2 logical_tokens={logical} fixture={} shuffled_page_ids=true partial_last_page={} native_v2_backend=true v1_generic_control=true reference_cpu=true descending_reduce=true repeats=3 state_prefix_bytes={} transition_step_bytes={} normalizer_bytes={} native_v2_workspace_bytes={} zero_allocations=true python_free=true status=passed",
            style.label(),
            logical % PAGE_SIZE != 0,
            native_v2_layout.state_prefix_bytes,
            native_v2_layout.transition_step_bytes,
            native_v2_layout.normalizer_bytes,
            native_v2.workspace_bytes(),
        );
        descending_output.close()?;
        native_v2_output.close()?;
        native_v1_output.close()?;
        reference_output.close()?;
        native_v2_workspace.close()?;
        native_v1_workspace.close()?;
        reference_workspace.close()?;
        fixture.close()?;
    }

    drop(native_v2);
    drop(native_v1);
    drop(reference);
    staging.close()?;
    stream.close()?;
    close_context(context)
}

fn run_paired_operator_control_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    logical: usize,
) -> TestResult {
    let physical_blocks = logical.div_ceil(PAGE_SIZE) + 3;
    let request = PagedDecodeAttentionRequest::new(
        u64::try_from(logical)?,
        u64::try_from(physical_blocks)?,
        u64::try_from(QUERY_HEADS)?,
        u64::try_from(KEY_VALUE_HEADS)?,
        u64::try_from(HEAD_SIZE)?,
        SCALE,
    );
    let availability = DecodeAttentionBackendAvailability::linked();
    let reference = PreparedPagedDecodeAttention::select(
        context,
        request,
        DecodeAttentionPreference::Reference,
        availability,
    )?;
    let native_v2 =
        PreparedPagedDecodeAttention::select_native_bf16_paged_split_gqa_d128_two_stage(
            context,
            request,
            availability,
        )?;
    assert_eq!(
        native_v2.backend(),
        DecodeAttentionBackend::NativeBf16PagedSplitGqaD128TwoStage
    );
    let native_v2_layout = NativeV2WorkspaceLayout::for_prepared(&native_v2)?;
    let (fixture, _) = setup_fixture(
        context,
        stream,
        staging,
        logical,
        physical_blocks,
        FixtureStyle::Patterned,
    )?;
    let output_bytes = u64::try_from(QUERY_HEADS * HEAD_SIZE * 2)?;
    let mut reference_workspace = context.allocate_device_buffer(reference.workspace_bytes())?;
    let mut native_v2_workspace = upload(
        context,
        stream,
        staging,
        &encode_f32(&vec![
            STATE_SENTINEL;
            usize::try_from(native_v2.workspace_bytes() / 4)?
        ]),
    )?;
    let mut reference_output = context.allocate_device_buffer(output_bytes)?;
    let mut native_v2_output = context.allocate_device_buffer(output_bytes)?;
    for _ in 0..INTERNAL_WARMUPS {
        execute_prepared(
            &reference,
            &fixture,
            &mut reference_workspace,
            reference.workspace_bytes(),
            &mut reference_output,
            stream,
        )?;
        execute_prepared(
            &native_v2,
            &fixture,
            &mut native_v2_workspace,
            native_v2.workspace_bytes(),
            &mut native_v2_output,
            stream,
        )?;
    }
    stream.synchronize()?;
    let mut start = context.create_event()?;
    let mut end = context.create_event()?;
    let allocation_baseline = context.allocation_stats()?;
    let mut reference_rounds = Vec::with_capacity(PAIRED_ROUNDS);
    let mut native_rounds = Vec::with_capacity(PAIRED_ROUNDS);
    for _ in 0..PAIRED_ROUNDS {
        let reference_first = timed_execute(
            &reference,
            &fixture,
            &mut reference_workspace,
            &mut reference_output,
            stream,
            &mut start,
            &mut end,
        )?;
        let native_first = timed_execute(
            &native_v2,
            &fixture,
            &mut native_v2_workspace,
            &mut native_v2_output,
            stream,
            &mut start,
            &mut end,
        )?;
        let native_second = timed_execute(
            &native_v2,
            &fixture,
            &mut native_v2_workspace,
            &mut native_v2_output,
            stream,
            &mut start,
            &mut end,
        )?;
        let reference_second = timed_execute(
            &reference,
            &fixture,
            &mut reference_workspace,
            &mut reference_output,
            stream,
            &mut start,
            &mut end,
        )?;
        reference_rounds.push((reference_first + reference_second) / 2.0);
        native_rounds.push((native_first + native_second) / 2.0);
    }
    assert_eq!(
        allocation_baseline,
        context.allocation_stats()?,
        "ABBA paired V2 native control allocated at T={logical}"
    );
    let reference_actual = decode_bf16(&download(context, stream, &mut reference_output)?);
    let native_actual = decode_bf16(&download(context, stream, &mut native_v2_output)?);
    assert_close(
        &native_actual,
        &reference_actual,
        NATIVE_REFERENCE_ABS_TOLERANCE,
        &format!("paired operator control T={logical}"),
    );
    let mut reference_median_rounds = reference_rounds.clone();
    let mut native_median_rounds = native_rounds.clone();
    let reference_median_ms = median(&mut reference_median_rounds);
    let native_median_ms = median(&mut native_median_rounds);
    let reference_p95_ms = percentile(reference_rounds, 95, 100);
    let native_p95_ms = percentile(native_rounds, 95, 100);
    let paired_speedup_ratio = reference_median_ms / native_median_ms;
    let paired_delta_ms = reference_median_ms - native_median_ms;
    assert!(
        paired_speedup_ratio.is_finite() && paired_delta_ms.is_finite(),
        "paired operator metrics must be finite"
    );
    println!(
        "riley-cuda-n03a-paged-gqa schema_version=2 case=b1-c{logical}-qh16-kvh2-d128 logical_tokens={logical} batch=1 query_heads=16 key_value_heads=2 head_size=128 page_size=16 fixture=patterned shuffled_page_ids=true partial_last_page=false timing_scope=prepared_paged_decode_execute_cuda_event internal_warmups_per_backend={INTERNAL_WARMUPS} paired_rounds={PAIRED_ROUNDS} paired_order=ABBA native_median_ms={native_median_ms:.6} native_p95_ms={native_p95_ms:.6} reference_median_ms={reference_median_ms:.6} reference_p95_ms={reference_p95_ms:.6} paired_speedup_ratio={paired_speedup_ratio:.6} paired_delta_ms={paired_delta_ms:.6} native_workspace_bytes={} native_v2_state_prefix_bytes={} native_v2_transition_step_bytes={} native_v2_normalizer_bytes={} reference_workspace_bytes={} implementation_id={} implementation_version={} graph_capture_supported=false operator_parity=passed allocation_delta=0 python_free=true full_model_serving=false vllm_comparison=false status=passed",
        native_v2.workspace_bytes(),
        native_v2_layout.state_prefix_bytes,
        native_v2_layout.transition_step_bytes,
        native_v2_layout.normalizer_bytes,
        reference.workspace_bytes(),
        native_v2.capability().implementation_id(),
        native_v2.selection_trace().implementation_version(),
    );
    start.close()?;
    end.close()?;
    native_v2_output.close()?;
    reference_output.close()?;
    native_v2_workspace.close()?;
    reference_workspace.close()?;
    fixture.close()?;
    Ok(())
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn n03a_paged_d128_paired_cuda_event_control() -> TestResult {
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(STAGING_BYTES)?;
    for logical in [2_048_usize, 16_384] {
        run_paired_operator_control_case(&context, &mut stream, &mut staging, logical)?;
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn native_bf16_ragged_paged_d128_two_stage_v3_c2048_b8_matches_v1_prefix_reducer_per_row()
-> TestResult {
    const ACTIVE_ROWS: usize = 8;
    const OUTPUT_ROWS: usize = 9;
    const PARTIAL_CAPACITY: usize = 2_048;
    const PAGES_PER_SEQUENCE: usize = 512;
    const PHYSICAL_BLOCKS: usize = ACTIVE_ROWS * PAGES_PER_SEQUENCE;

    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(STAGING_BYTES)?;
    // B8 has uneven active prefixes from one through 128 pages, while each
    // backing CSR sequence has 512 pages. Thus aggregate blocks (4096) exceed
    // P=2048, but every row fits its independent workspace slice. The exact
    // producer launch extent is L=128, avoiding the old all-P grid.
    let row_positions = [0_u32, 16, 47, 63, 255, 511, 1_023, 2_047];
    let row_sequence_slots = (0..ACTIVE_ROWS)
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let sequence_block_offsets = (0..=ACTIVE_ROWS)
        .map(|sequence| u32::try_from(sequence * PAGES_PER_SEQUENCE))
        .collect::<Result<Vec<_>, _>>()?;
    let block_ids = (0..PHYSICAL_BLOCKS)
        .rev()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let valid_tokens = vec![u16::try_from(PAGE_SIZE)?; PHYSICAL_BLOCKS];
    let launch_partial_state_count = row_positions
        .iter()
        .map(|&position| usize::try_from(u64::from(position) / PAGE_SIZE as u64 + 1))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .expect("B8 fixture has active rows");
    assert_eq!(launch_partial_state_count, 128);
    assert_eq!(PARTIAL_CAPACITY / launch_partial_state_count, 16);
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions,
        u64::try_from(PHYSICAL_BLOCKS)?,
    )?;
    assert!(
        host.block_count() > u64::try_from(PARTIAL_CAPACITY)?,
        "fixture must exercise aggregate CSR blocks greater than per-row P"
    );

    let query_values =
        centered_pattern(ACTIVE_ROWS * QUERY_HEADS * HEAD_SIZE, 11, 7, 29, 14.0, 32.0);
    let key_values = centered_pattern(
        PHYSICAL_BLOCKS * KEY_VALUE_HEADS * PAGE_SIZE * HEAD_SIZE,
        5,
        3,
        31,
        15.0,
        32.0,
    );
    let value_values = centered_pattern(
        PHYSICAL_BLOCKS * KEY_VALUE_HEADS * PAGE_SIZE * HEAD_SIZE,
        13,
        9,
        37,
        18.0,
        32.0,
    );
    let query = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&query_values),
    )?;
    let key_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&key_values),
    )?;
    let value_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&value_values),
    )?;
    let offsets = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let ids = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
    let valid = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let positions = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;
    let batch = PackedBatchV1::new(
        host,
        CudaBufferSpan::new(&offsets, CudaDType::U32, 0, offsets.byte_len())?,
        CudaBufferSpan::new(&ids, CudaDType::U32, 0, ids.byte_len())?,
        CudaBufferSpan::new(&valid, CudaDType::U16, 0, valid.byte_len())?,
        CudaBufferSpan::new(&row_slots, CudaDType::U32, 0, row_slots.byte_len())?,
        CudaBufferSpan::new(&positions, CudaDType::U32, 0, positions.byte_len())?,
    )?;

    let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::new(
        u64::try_from(OUTPUT_ROWS)?,
        u64::try_from(PARTIAL_CAPACITY)?,
    )?;
    let mut partial_states = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_f32(&vec![
            STATE_SENTINEL;
            usize::try_from(layout.partial_state_bytes() / 4)?
        ]),
    )?;
    let mut reduction_steps = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_f32(&vec![
            STATE_SENTINEL;
            usize::try_from(layout.reduction_step_bytes() / 4)?
        ]),
    )?;
    let mut reduction_normalizers = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_f32(&vec![
            STATE_SENTINEL;
            usize::try_from(
                layout.reduction_normalizer_bytes() / 4
            )?
        ]),
    )?;
    let output_elements = OUTPUT_ROWS * QUERY_HEADS * HEAD_SIZE;
    let mut output = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&vec![OUTPUT_SENTINEL; output_elements]),
    )?;
    let partial_state_len = partial_states.byte_len();
    let reduction_step_len = reduction_steps.byte_len();
    let normalizer_len = reduction_normalizers.byte_len();
    let output_len = output.byte_len();
    native_bf16_ragged_paged_split_gqa_d128_two_stage(
        &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            partial_states: CudaBufferSpanMut::new(
                &mut partial_states,
                CudaDType::F32,
                0,
                partial_state_len,
            )?,
            reduction_steps: CudaBufferSpanMut::new(
                &mut reduction_steps,
                CudaDType::F32,
                0,
                reduction_step_len,
            )?,
            reduction_normalizers: CudaBufferSpanMut::new(
                &mut reduction_normalizers,
                CudaDType::F32,
                0,
                normalizer_len,
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch,
            query_head_count: u64::try_from(QUERY_HEADS)?,
            key_value_head_count: u64::try_from(KEY_VALUE_HEADS)?,
            head_size: u64::try_from(HEAD_SIZE)?,
            output_row_count: u64::try_from(OUTPUT_ROWS)?,
            partial_state_capacity: u64::try_from(PARTIAL_CAPACITY)?,
            scale: SCALE,
            reduction_order: DecodePartialReductionOrder::LogicalAscending,
        },
        &mut stream,
    )?;
    stream.synchronize()?;

    let row_state_bytes = layout.partial_state_bytes() / u64::try_from(OUTPUT_ROWS)?;
    let mut generic_outputs = Vec::with_capacity(ACTIVE_ROWS);
    for row in 0..ACTIVE_ROWS {
        let mut generic = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(&vec![OUTPUT_SENTINEL; QUERY_HEADS * HEAD_SIZE]),
        )?;
        let generic_len = generic.byte_len();
        let partial_state_count = u64::from(row_positions[row]) / u64::try_from(PAGE_SIZE)? + 1;
        decode_partial_states_reduce(
            &mut DecodePartialStateReduceParams {
                partial_states: CudaBufferSpan::new(
                    &partial_states,
                    CudaDType::F32,
                    u64::try_from(row)? * row_state_bytes,
                    row_state_bytes,
                )?,
                output: CudaBufferSpanMut::new(&mut generic, CudaDType::BF16, 0, generic_len)?,
                partial_state_count,
                partial_state_capacity: u64::try_from(PARTIAL_CAPACITY)?,
                query_head_count: u64::try_from(QUERY_HEADS)?,
                head_size: u64::try_from(HEAD_SIZE)?,
                order: DecodePartialReductionOrder::LogicalAscending,
            },
            &mut stream,
        )?;
        generic_outputs.push(generic);
    }
    stream.synchronize()?;
    let v3_output = download(&context, &mut stream, &mut output)?;
    let active_output_bytes = ACTIVE_ROWS * QUERY_HEADS * HEAD_SIZE * 2;
    let mut generic_joined = Vec::with_capacity(active_output_bytes);
    for generic in &mut generic_outputs {
        generic_joined.extend(download(&context, &mut stream, generic)?);
    }
    assert_eq!(
        &v3_output[..active_output_bytes],
        generic_joined.as_slice(),
        "V3 ragged output must exactly replay the V1 generic ascending reducer for every row"
    );
    assert_eq!(
        &v3_output[active_output_bytes..],
        encode_bf16(&vec![0.0; QUERY_HEADS * HEAD_SIZE]).as_slice(),
        "inactive prepared output rows must be BF16 zero"
    );

    // The V1 reduction compares the shared producer state exactly. Exercise a
    // separate materialized paged reference for each ragged row as well: its
    // block table clips the backing 512-page sequence to that row's inclusive
    // prefix, while preserving the shared shuffled physical K/V pool.
    let availability = DecodeAttentionBackendAvailability::linked();
    let row_query_bytes = u64::try_from(QUERY_HEADS * HEAD_SIZE * 2)?;
    let mut materialized_reference = Vec::with_capacity(active_output_bytes / 2);
    for row in 0..ACTIVE_ROWS {
        let logical_token_count = u64::from(row_positions[row]) + 1;
        let partial_state_count = logical_token_count.div_ceil(u64::try_from(PAGE_SIZE)?);
        let block_begin = usize::try_from(sequence_block_offsets[row])?;
        let block_end = block_begin + usize::try_from(partial_state_count)?;
        assert!(
            block_end <= usize::try_from(sequence_block_offsets[row + 1])?,
            "row {row} materialized table exceeds its CSR range"
        );
        let mut reference_valid_tokens = vec![u16::try_from(PAGE_SIZE)?; block_end - block_begin];
        let final_valid_tokens =
            u16::try_from((logical_token_count - 1) % u64::try_from(PAGE_SIZE)? + 1)?;
        *reference_valid_tokens
            .last_mut()
            .expect("active row has at least one logical page") = final_valid_tokens;
        let reference_table_host = PagedKvBlockTableHostV1::new(
            &block_ids[block_begin..block_end],
            &reference_valid_tokens,
            logical_token_count,
            u64::try_from(PHYSICAL_BLOCKS)?,
        )?;
        let reference = PreparedPagedDecodeAttention::select(
            &context,
            PagedDecodeAttentionRequest::new(
                logical_token_count,
                u64::try_from(PHYSICAL_BLOCKS)?,
                u64::try_from(QUERY_HEADS)?,
                u64::try_from(KEY_VALUE_HEADS)?,
                u64::try_from(HEAD_SIZE)?,
                SCALE,
            ),
            DecodeAttentionPreference::Reference,
            availability,
        )?;
        assert_eq!(
            reference.backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        let mut reference_workspace =
            context.allocate_device_buffer(reference.workspace_bytes())?;
        let mut reference_output = context.allocate_device_buffer(row_query_bytes)?;
        let reference_valid = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u16(&reference_valid_tokens),
        )?;
        let reference_workspace_len = reference_workspace.byte_len();
        let reference_output_len = reference_output.byte_len();
        let reference_valid_len = reference_valid.byte_len();
        reference.execute(
            &mut PagedDecodeAttentionParams {
                query: CudaBufferSpan::new(
                    &query,
                    CudaDType::BF16,
                    u64::try_from(row)? * row_query_bytes,
                    row_query_bytes,
                )?,
                key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
                value_pool: CudaBufferSpan::new(
                    &value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool.byte_len(),
                )?,
                workspace: CudaBufferSpanMut::new(
                    &mut reference_workspace,
                    reference.workspace_dtype(),
                    0,
                    reference_workspace_len,
                )?,
                output: CudaBufferSpanMut::new(
                    &mut reference_output,
                    CudaDType::BF16,
                    0,
                    reference_output_len,
                )?,
                block_table: PagedKvBlockTableV1::new(
                    reference_table_host,
                    CudaBufferSpan::new(
                        &ids,
                        CudaDType::U32,
                        u64::try_from(block_begin)? * 4,
                        u64::try_from(block_end - block_begin)? * 4,
                    )?,
                    CudaBufferSpan::new(&reference_valid, CudaDType::U16, 0, reference_valid_len)?,
                )?,
            },
            &mut stream,
        )?;
        stream.synchronize()?;
        materialized_reference.extend(decode_bf16(&download(
            &context,
            &mut stream,
            &mut reference_output,
        )?));
        reference_valid.close()?;
        reference_output.close()?;
        reference_workspace.close()?;
        drop(reference);
    }
    assert_close(
        &decode_bf16(&v3_output[..active_output_bytes]),
        &materialized_reference,
        NATIVE_REFERENCE_ABS_TOLERANCE,
        "V3 ragged B8 C2048 materialized-reference control",
    );

    let states = decode_f32(&download(&context, &mut stream, &mut partial_states)?);
    let steps = decode_f32(&download(&context, &mut stream, &mut reduction_steps)?);
    let normalizers = decode_f32(&download(
        &context,
        &mut stream,
        &mut reduction_normalizers,
    )?);
    let state_stride = HEAD_SIZE + 2;
    let row_state_elements = PARTIAL_CAPACITY * QUERY_HEADS * state_stride;
    let row_step_elements = PARTIAL_CAPACITY * QUERY_HEADS * 2;
    for row in 0..ACTIVE_ROWS {
        let active_pages =
            usize::try_from(u64::from(row_positions[row]) / u64::try_from(PAGE_SIZE)? + 1)?;
        for partition in 0..active_pages {
            for head in 0..QUERY_HEADS {
                let state_base =
                    row * row_state_elements + (partition * QUERY_HEADS + head) * state_stride;
                assert!(states[state_base].is_finite());
                assert!(states[state_base + 1].is_finite() && states[state_base + 1] > 0.0);
                let step_base = row * row_step_elements + (partition * QUERY_HEADS + head) * 2;
                assert!(steps[step_base].is_finite() && steps[step_base + 1].is_finite());
            }
        }
        let state_tail_start = row * row_state_elements + active_pages * QUERY_HEADS * state_stride;
        let state_tail_end = (row + 1) * row_state_elements;
        assert!(
            states[state_tail_start..state_tail_end]
                .iter()
                .all(|value| value.to_bits() == STATE_SENTINEL.to_bits())
        );
        let step_tail_start = row * row_step_elements + active_pages * QUERY_HEADS * 2;
        let step_tail_end = (row + 1) * row_step_elements;
        assert!(
            steps[step_tail_start..step_tail_end]
                .iter()
                .all(|value| value.to_bits() == STATE_SENTINEL.to_bits())
        );
        assert!(
            normalizers[row * QUERY_HEADS..(row + 1) * QUERY_HEADS]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
    }
    assert!(
        states[ACTIVE_ROWS * row_state_elements..]
            .iter()
            .all(|value| value.to_bits() == STATE_SENTINEL.to_bits())
    );
    assert!(
        steps[ACTIVE_ROWS * row_step_elements..]
            .iter()
            .all(|value| value.to_bits() == STATE_SENTINEL.to_bits())
    );
    assert!(
        normalizers[ACTIVE_ROWS * QUERY_HEADS..]
            .iter()
            .all(|value| value.to_bits() == STATE_SENTINEL.to_bits())
    );

    let mut untouched_output = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&vec![OUTPUT_SENTINEL; output_elements]),
    )?;
    let untouched_len = untouched_output.byte_len();
    let error = native_bf16_ragged_paged_split_gqa_d128_two_stage(
        &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            partial_states: CudaBufferSpanMut::new(
                &mut partial_states,
                CudaDType::F32,
                0,
                partial_state_len - 4,
            )?,
            reduction_steps: CudaBufferSpanMut::new(
                &mut reduction_steps,
                CudaDType::F32,
                0,
                reduction_step_len,
            )?,
            reduction_normalizers: CudaBufferSpanMut::new(
                &mut reduction_normalizers,
                CudaDType::F32,
                0,
                normalizer_len,
            )?,
            output: CudaBufferSpanMut::new(
                &mut untouched_output,
                CudaDType::BF16,
                0,
                untouched_len,
            )?,
            batch,
            query_head_count: u64::try_from(QUERY_HEADS)?,
            key_value_head_count: u64::try_from(KEY_VALUE_HEADS)?,
            head_size: u64::try_from(HEAD_SIZE)?,
            output_row_count: u64::try_from(OUTPUT_ROWS)?,
            partial_state_capacity: u64::try_from(PARTIAL_CAPACITY)?,
            scale: SCALE,
            reduction_order: DecodePartialReductionOrder::LogicalAscending,
        },
        &mut stream,
    )
    .expect_err("undersized V3 ragged workspace must fail before launch");
    assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
    assert_eq!(
        download(&context, &mut stream, &mut untouched_output)?,
        encode_bf16(&vec![OUTPUT_SENTINEL; output_elements]),
        "undersized V3 ragged workspace modified output"
    );
    println!(
        "n03b-ragged-paged-d128-correctness schema_version=1 implementation_id=native-bf16-ragged-paged-split-gqa-d128-two-stage-v3 active_rows={ACTIVE_ROWS} output_rows={OUTPUT_ROWS} per_row_page_capacity={PARTIAL_CAPACITY} aggregate_csr_blocks={} v1_generic_ascending_exact=true inactive_output_zero=true state_bytes={} step_bytes={} normalizer_bytes={} workspace_total_bytes={} graph_capture_supported=false python_free=true status=passed",
        host.block_count(),
        layout.partial_state_bytes(),
        layout.reduction_step_bytes(),
        layout.reduction_normalizer_bytes(),
        layout.total_bytes(),
    );
    untouched_output.close()?;
    for generic in generic_outputs {
        generic.close()?;
    }
    output.close()?;
    reduction_normalizers.close()?;
    reduction_steps.close()?;
    partial_states.close()?;
    positions.close()?;
    row_slots.close()?;
    valid.close()?;
    ids.close()?;
    offsets.close()?;
    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}
