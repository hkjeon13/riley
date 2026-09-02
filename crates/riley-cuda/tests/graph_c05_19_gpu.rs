#![allow(clippy::too_many_arguments, clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaErrorKind,
    CudaGraphCaptureMode, CudaRuntime, CudaStream, PackedBatchHostV1, PackedBatchV1,
    RaggedPagedAttentionParams, grouped_ragged_paged_attention,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SEQUENCE_COUNT: u64 = 2;
const BLOCK_COUNT: u64 = 3;
const ACTIVE_ROW_COUNT: u64 = 5;
const PHYSICAL_BLOCK_COUNT: u64 = 6;
const QUERY_HEAD_COUNT: u64 = 9;
const KEY_VALUE_HEAD_COUNT: u64 = 3;
const HEAD_SIZE: u64 = 64;
const OUTPUT_ROW_COUNT: u64 = 8;
const BLOCK_SIZE: u64 = 16;
const BF16_BYTES: u64 = 2;
const SCALE: f32 = 0.125;

struct GroupedAttentionFixture {
    offsets: Vec<u32>,
    block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
    row_slots: Vec<u32>,
    row_positions: Vec<u32>,
    query: Vec<u8>,
    key_pool: Vec<u8>,
    value_pool: Vec<u8>,
    output_sentinel: Vec<u8>,
}

impl GroupedAttentionFixture {
    fn new() -> Self {
        // This is the production grouped-GQA geometry. Sequence 0 crosses a
        // page boundary at 15/16 through shuffled physical blocks, while the
        // padded output tail exercises M > T zero-fill semantics.
        let offsets = vec![0_u32, 2, 3];
        let block_ids = vec![4_u32, 1, 3];
        let valid_tokens = vec![16_u16, 2, 5];
        let row_slots = vec![0_u32, 1, 0, 1, 0];
        let row_positions = vec![17_u32, 4, 0, 2, 16];
        assert_eq!(u64::try_from(block_ids.len()).unwrap(), BLOCK_COUNT);
        assert_eq!(u64::try_from(row_slots.len()).unwrap(), ACTIVE_ROW_COUNT);

        let query_elements = ACTIVE_ROW_COUNT
            .checked_mul(QUERY_HEAD_COUNT)
            .and_then(|value| value.checked_mul(HEAD_SIZE))
            .expect("C05-19 query element count fits u64");
        let pool_elements = PHYSICAL_BLOCK_COUNT
            .checked_mul(KEY_VALUE_HEAD_COUNT)
            .and_then(|value| value.checked_mul(BLOCK_SIZE))
            .and_then(|value| value.checked_mul(HEAD_SIZE))
            .expect("C05-19 pool element count fits u64");
        let output_elements = OUTPUT_ROW_COUNT
            .checked_mul(QUERY_HEAD_COUNT)
            .and_then(|value| value.checked_mul(HEAD_SIZE))
            .expect("C05-19 output element count fits u64");
        Self {
            offsets,
            block_ids,
            valid_tokens,
            row_slots,
            row_positions,
            query: finite_bf16_pattern(query_elements, 7),
            key_pool: finite_bf16_pattern(pool_elements, 19),
            value_pool: finite_bf16_pattern(pool_elements, 37),
            output_sentinel: vec![0xa5; usize::try_from(output_elements * BF16_BYTES).unwrap()],
        }
    }

    fn host(&self) -> TestResult<PackedBatchHostV1<'_>> {
        Ok(PackedBatchHostV1::new(
            &self.offsets,
            &self.block_ids,
            &self.valid_tokens,
            &self.row_slots,
            &self.row_positions,
            PHYSICAL_BLOCK_COUNT,
        )?)
    }
}

fn finite_bf16_pattern(elements: u64, seed: usize) -> Vec<u8> {
    (0..usize::try_from(elements).expect("C05-19 fixture element count fits usize"))
        .flat_map(|index| {
            // Values in [roughly 0.125, 0.25) are finite normal BF16 values.
            let word = 0x3e00_u16 + u16::try_from((index * seed) % 0x100).unwrap();
            word.to_ne_bytes()
        })
        .collect()
}

fn u32_words_to_ne_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn u16_words_to_ne_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn upload(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    bytes: &[u8],
) -> TestResult<CudaDeviceBuffer> {
    let mut buffer = context.allocate_device_buffer(u64::try_from(bytes.len())?)?;
    buffer.upload_from_slice(0, bytes, staging, stream)?;
    Ok(buffer)
}

fn download(
    buffer: &mut CudaDeviceBuffer,
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> TestResult<Vec<u8>> {
    let mut bytes = vec![0_u8; usize::try_from(buffer.byte_len())?];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}

fn bind_batch<'a>(
    host: PackedBatchHostV1<'a>,
    offsets: &'a CudaDeviceBuffer,
    block_ids: &'a CudaDeviceBuffer,
    valid_tokens: &'a CudaDeviceBuffer,
    row_slots: &'a CudaDeviceBuffer,
    row_positions: &'a CudaDeviceBuffer,
) -> TestResult<PackedBatchV1<'a>> {
    Ok(PackedBatchV1::new(
        host,
        CudaBufferSpan::new(offsets, CudaDType::U32, 0, offsets.byte_len())?,
        CudaBufferSpan::new(block_ids, CudaDType::U32, 0, block_ids.byte_len())?,
        CudaBufferSpan::new(valid_tokens, CudaDType::U16, 0, valid_tokens.byte_len())?,
        CudaBufferSpan::new(row_slots, CudaDType::U32, 0, row_slots.byte_len())?,
        CudaBufferSpan::new(row_positions, CudaDType::U32, 0, row_positions.byte_len())?,
    )?)
}

fn assert_invalid_state<T>(result: riley_cuda::CudaResult<T>, operation: &str) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("{operation} unexpectedly succeeded a second time"));
    assert_eq!(
        error.kind(),
        CudaErrorKind::InvalidState,
        "{operation} must reject before recording another CUDA graph node"
    );
}

fn assert_bf16_nan(word_bytes: &[u8]) {
    let word = u16::from_ne_bytes([word_bytes[0], word_bytes[1]]);
    assert_eq!(
        word & 0x7f80,
        0x7f80,
        "BF16 output must have an all-one exponent"
    );
    assert_ne!(
        word & 0x007f,
        0,
        "BF16 output must be NaN rather than infinity"
    );
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_grouped_ragged_attention_graph_replays_byte_exact_against_eager() -> TestResult {
    const REPLAYS: usize = 64;
    let fixture = GroupedAttentionFixture::new();
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let staging_bytes = u64::try_from(
        fixture
            .query
            .len()
            .max(fixture.key_pool.len())
            .max(fixture.value_pool.len())
            .max(fixture.output_sentinel.len()),
    )?;
    let mut staging = context.allocate_pinned_host_buffer(staging_bytes)?;

    let offsets_bytes = u32_words_to_ne_bytes(&fixture.offsets);
    let block_ids_bytes = u32_words_to_ne_bytes(&fixture.block_ids);
    let valid_tokens_bytes = u16_words_to_ne_bytes(&fixture.valid_tokens);
    let row_slots_bytes = u32_words_to_ne_bytes(&fixture.row_slots);
    let row_positions_bytes = u32_words_to_ne_bytes(&fixture.row_positions);

    let eager_query = upload(&context, &mut eager_stream, &mut staging, &fixture.query)?;
    let eager_key_pool = upload(&context, &mut eager_stream, &mut staging, &fixture.key_pool)?;
    let eager_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool,
    )?;
    let eager_offsets = upload(&context, &mut eager_stream, &mut staging, &offsets_bytes)?;
    let eager_block_ids = upload(&context, &mut eager_stream, &mut staging, &block_ids_bytes)?;
    let eager_valid_tokens = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &valid_tokens_bytes,
    )?;
    let eager_row_slots = upload(&context, &mut eager_stream, &mut staging, &row_slots_bytes)?;
    let eager_row_positions = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &row_positions_bytes,
    )?;
    let mut eager_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;

    let graph_query = upload(&context, &mut eager_stream, &mut staging, &fixture.query)?;
    let graph_key_pool = upload(&context, &mut eager_stream, &mut staging, &fixture.key_pool)?;
    let graph_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool,
    )?;
    let graph_offsets = upload(&context, &mut eager_stream, &mut staging, &offsets_bytes)?;
    let graph_block_ids = upload(&context, &mut eager_stream, &mut staging, &block_ids_bytes)?;
    let graph_valid_tokens = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &valid_tokens_bytes,
    )?;
    let graph_row_slots = upload(&context, &mut eager_stream, &mut staging, &row_slots_bytes)?;
    let graph_row_positions = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &row_positions_bytes,
    )?;
    let graph_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;

    let eager_output_len = eager_output.byte_len();
    grouped_ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&eager_query, CudaDType::BF16, 0, eager_query.byte_len())?,
            key_pool: CudaBufferSpan::new(
                &eager_key_pool,
                CudaDType::BF16,
                0,
                eager_key_pool.byte_len(),
            )?,
            value_pool: CudaBufferSpan::new(
                &eager_value_pool,
                CudaDType::BF16,
                0,
                eager_value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(
                &mut eager_output,
                CudaDType::BF16,
                0,
                eager_output_len,
            )?,
            batch: bind_batch(
                fixture.host()?,
                &eager_offsets,
                &eager_block_ids,
                &eager_valid_tokens,
                &eager_row_slots,
                &eager_row_positions,
            )?,
            query_head_count: QUERY_HEAD_COUNT,
            key_value_head_count: KEY_VALUE_HEAD_COUNT,
            head_size: HEAD_SIZE,
            output_row_count: OUTPUT_ROW_COUNT,
            scale: SCALE,
        },
        &mut eager_stream,
    )?;
    eager_stream.synchronize()?;

    let admission_offsets = fixture.offsets.clone();
    let admission_block_ids = fixture.block_ids.clone();
    let admission_valid_tokens = fixture.valid_tokens.clone();
    let admission_row_slots = fixture.row_slots.clone();
    let admission_row_positions = fixture.row_positions.clone();
    let allocation_with_resources = context.allocation_stats()?;
    let mut capture = {
        let admission = PackedBatchHostV1::new(
            &admission_offsets,
            &admission_block_ids,
            &admission_valid_tokens,
            &admission_row_slots,
            &admission_row_positions,
            PHYSICAL_BLOCK_COUNT,
        )?;
        capture_stream.begin_owned_graph_grouped_ragged_paged_attention_bf16_capture(
            graph_query,
            graph_key_pool,
            graph_value_pool,
            graph_output,
            graph_offsets,
            graph_block_ids,
            graph_valid_tokens,
            graph_row_slots,
            graph_row_positions,
            admission,
            QUERY_HEAD_COUNT,
            KEY_VALUE_HEAD_COUNT,
            OUTPUT_ROW_COUNT,
            SCALE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
    };
    drop(admission_offsets);
    drop(admission_block_ids);
    drop(admission_valid_tokens);
    drop(admission_row_slots);
    drop(admission_row_positions);
    capture.enqueue_grouped_ragged_paged_attention_bf16()?;
    assert_invalid_state(
        capture.enqueue_grouped_ragged_paged_attention_bf16(),
        "second C05-19 grouped ragged-attention graph enqueue",
    );
    let mut exec = capture.end()?.instantiate()?;
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(context.allocation_stats()?, allocation_with_resources);

    let resources = exec.close()?;
    let (
        capture_stream,
        mut graph_query,
        mut graph_key_pool,
        mut graph_value_pool,
        mut graph_output,
        mut graph_offsets,
        mut graph_block_ids,
        mut graph_valid_tokens,
        mut graph_row_slots,
        mut graph_row_positions,
    ) = resources.into_parts();
    let graph_output_bytes = download(&mut graph_output, &mut staging, &mut transfer_stream)?;
    assert_eq!(
        graph_output_bytes,
        download(&mut eager_output, &mut staging, &mut transfer_stream)?,
        "C05-19 graph output must equal eager grouped BF16 bytes exactly",
    );
    let active_bytes =
        usize::try_from(ACTIVE_ROW_COUNT * QUERY_HEAD_COUNT * HEAD_SIZE * BF16_BYTES)?;
    assert!(
        graph_output_bytes[active_bytes..]
            .iter()
            .all(|byte| *byte == 0),
        "every padded [T,M) grouped-attention output BF16 byte must be zero",
    );
    for (name, buffer, expected) in [
        ("query", &mut graph_query, fixture.query.as_slice()),
        ("key pool", &mut graph_key_pool, fixture.key_pool.as_slice()),
        (
            "value pool",
            &mut graph_value_pool,
            fixture.value_pool.as_slice(),
        ),
        ("offsets", &mut graph_offsets, offsets_bytes.as_slice()),
        (
            "block ids",
            &mut graph_block_ids,
            block_ids_bytes.as_slice(),
        ),
        (
            "valid tokens",
            &mut graph_valid_tokens,
            valid_tokens_bytes.as_slice(),
        ),
        (
            "row slots",
            &mut graph_row_slots,
            row_slots_bytes.as_slice(),
        ),
        (
            "row positions",
            &mut graph_row_positions,
            row_positions_bytes.as_slice(),
        ),
    ] {
        assert_eq!(
            download(buffer, &mut staging, &mut transfer_stream)?,
            expected,
            "C05-19 graph must not mutate fixed {name} bytes",
        );
    }

    graph_row_positions.close()?;
    graph_row_slots.close()?;
    graph_valid_tokens.close()?;
    graph_block_ids.close()?;
    graph_offsets.close()?;
    graph_output.close()?;
    graph_value_pool.close()?;
    graph_key_pool.close()?;
    graph_query.close()?;
    eager_output.close()?;
    eager_row_positions.close()?;
    eager_row_slots.close()?;
    eager_valid_tokens.close()?;
    eager_block_ids.close()?;
    eager_offsets.close()?;
    eager_value_pool.close()?;
    eager_key_pool.close()?;
    eager_query.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-19-grouped-ragged-attention-valid-replays={REPLAYS} status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_grouped_ragged_attention_graph_generic_group_one_matches_eager() -> TestResult {
    // A group size of one deliberately bypasses the shared-KV CTA branch and
    // covers the graph's generic grouped-head launch topology.
    const GENERIC_QUERY_HEAD_COUNT: u64 = 2;
    const GENERIC_KEY_VALUE_HEAD_COUNT: u64 = 2;
    assert_eq!(GENERIC_QUERY_HEAD_COUNT / GENERIC_KEY_VALUE_HEAD_COUNT, 1);
    let fixture = GroupedAttentionFixture::new();
    let query = finite_bf16_pattern(ACTIVE_ROW_COUNT * GENERIC_QUERY_HEAD_COUNT * HEAD_SIZE, 11);
    let key_pool = finite_bf16_pattern(
        PHYSICAL_BLOCK_COUNT * GENERIC_KEY_VALUE_HEAD_COUNT * BLOCK_SIZE * HEAD_SIZE,
        23,
    );
    let value_pool = finite_bf16_pattern(
        PHYSICAL_BLOCK_COUNT * GENERIC_KEY_VALUE_HEAD_COUNT * BLOCK_SIZE * HEAD_SIZE,
        41,
    );
    let output_sentinel =
        vec![
            0x5a;
            usize::try_from(OUTPUT_ROW_COUNT * GENERIC_QUERY_HEAD_COUNT * HEAD_SIZE * BF16_BYTES,)?
        ];
    let offsets_bytes = u32_words_to_ne_bytes(&fixture.offsets);
    let block_ids_bytes = u32_words_to_ne_bytes(&fixture.block_ids);
    let valid_tokens_bytes = u16_words_to_ne_bytes(&fixture.valid_tokens);
    let row_slots_bytes = u32_words_to_ne_bytes(&fixture.row_slots);
    let row_positions_bytes = u32_words_to_ne_bytes(&fixture.row_positions);

    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(u64::try_from(
        query
            .len()
            .max(key_pool.len())
            .max(value_pool.len())
            .max(output_sentinel.len()),
    )?)?;

    let eager_query = upload(&context, &mut eager_stream, &mut staging, &query)?;
    let eager_key_pool = upload(&context, &mut eager_stream, &mut staging, &key_pool)?;
    let eager_value_pool = upload(&context, &mut eager_stream, &mut staging, &value_pool)?;
    let eager_offsets = upload(&context, &mut eager_stream, &mut staging, &offsets_bytes)?;
    let eager_block_ids = upload(&context, &mut eager_stream, &mut staging, &block_ids_bytes)?;
    let eager_valid_tokens = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &valid_tokens_bytes,
    )?;
    let eager_row_slots = upload(&context, &mut eager_stream, &mut staging, &row_slots_bytes)?;
    let eager_row_positions = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &row_positions_bytes,
    )?;
    let mut eager_output = upload(&context, &mut eager_stream, &mut staging, &output_sentinel)?;
    let graph_query = upload(&context, &mut eager_stream, &mut staging, &query)?;
    let graph_key_pool = upload(&context, &mut eager_stream, &mut staging, &key_pool)?;
    let graph_value_pool = upload(&context, &mut eager_stream, &mut staging, &value_pool)?;
    let graph_offsets = upload(&context, &mut eager_stream, &mut staging, &offsets_bytes)?;
    let graph_block_ids = upload(&context, &mut eager_stream, &mut staging, &block_ids_bytes)?;
    let graph_valid_tokens = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &valid_tokens_bytes,
    )?;
    let graph_row_slots = upload(&context, &mut eager_stream, &mut staging, &row_slots_bytes)?;
    let graph_row_positions = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &row_positions_bytes,
    )?;
    let graph_output = upload(&context, &mut eager_stream, &mut staging, &output_sentinel)?;

    let eager_output_len = eager_output.byte_len();
    grouped_ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&eager_query, CudaDType::BF16, 0, eager_query.byte_len())?,
            key_pool: CudaBufferSpan::new(
                &eager_key_pool,
                CudaDType::BF16,
                0,
                eager_key_pool.byte_len(),
            )?,
            value_pool: CudaBufferSpan::new(
                &eager_value_pool,
                CudaDType::BF16,
                0,
                eager_value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(
                &mut eager_output,
                CudaDType::BF16,
                0,
                eager_output_len,
            )?,
            batch: bind_batch(
                fixture.host()?,
                &eager_offsets,
                &eager_block_ids,
                &eager_valid_tokens,
                &eager_row_slots,
                &eager_row_positions,
            )?,
            query_head_count: GENERIC_QUERY_HEAD_COUNT,
            key_value_head_count: GENERIC_KEY_VALUE_HEAD_COUNT,
            head_size: HEAD_SIZE,
            output_row_count: OUTPUT_ROW_COUNT,
            scale: SCALE,
        },
        &mut eager_stream,
    )?;
    eager_stream.synchronize()?;

    let mut capture = capture_stream
        .begin_owned_graph_grouped_ragged_paged_attention_bf16_capture(
            graph_query,
            graph_key_pool,
            graph_value_pool,
            graph_output,
            graph_offsets,
            graph_block_ids,
            graph_valid_tokens,
            graph_row_slots,
            graph_row_positions,
            fixture.host()?,
            GENERIC_QUERY_HEAD_COUNT,
            GENERIC_KEY_VALUE_HEAD_COUNT,
            OUTPUT_ROW_COUNT,
            SCALE,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
    capture.enqueue_grouped_ragged_paged_attention_bf16()?;
    let mut exec = capture.end()?.instantiate()?;
    exec.launch()?.finish()?;
    let resources = exec.close()?;
    let (
        capture_stream,
        graph_query,
        graph_key_pool,
        graph_value_pool,
        mut graph_output,
        graph_offsets,
        graph_block_ids,
        graph_valid_tokens,
        graph_row_slots,
        graph_row_positions,
    ) = resources.into_parts();
    let graph_output_bytes = download(&mut graph_output, &mut staging, &mut transfer_stream)?;
    assert_eq!(
        graph_output_bytes,
        download(&mut eager_output, &mut staging, &mut transfer_stream)?,
        "generic group-one graph output must exactly match eager grouped attention",
    );
    let active_bytes =
        usize::try_from(ACTIVE_ROW_COUNT * GENERIC_QUERY_HEAD_COUNT * HEAD_SIZE * BF16_BYTES)?;
    assert!(
        graph_output_bytes[active_bytes..]
            .iter()
            .all(|byte| *byte == 0)
    );

    graph_row_positions.close()?;
    graph_row_slots.close()?;
    graph_valid_tokens.close()?;
    graph_block_ids.close()?;
    graph_offsets.close()?;
    graph_output.close()?;
    graph_value_pool.close()?;
    graph_key_pool.close()?;
    graph_query.close()?;
    eager_output.close()?;
    eager_row_positions.close()?;
    eager_row_slots.close()?;
    eager_valid_tokens.close()?;
    eager_block_ids.close()?;
    eager_offsets.close()?;
    eager_value_pool.close()?;
    eager_key_pool.close()?;
    eager_query.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-19-grouped-ragged-attention-generic-group-one status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_grouped_ragged_attention_graph_preserves_bounds_invalid_raw_row_nan() -> TestResult {
    let fixture = GroupedAttentionFixture::new();
    // Host admission remains valid; one fixed device row slot becomes invalid
    // to exercise the eager kernel's in-kernel non-finite output path.
    let raw_row_slots = vec![0_u32, SEQUENCE_COUNT as u32, 0, 1, 0];
    let raw_row_slots_bytes = u32_words_to_ne_bytes(&raw_row_slots);
    let offsets_bytes = u32_words_to_ne_bytes(&fixture.offsets);
    let block_ids_bytes = u32_words_to_ne_bytes(&fixture.block_ids);
    let valid_tokens_bytes = u16_words_to_ne_bytes(&fixture.valid_tokens);
    let row_positions_bytes = u32_words_to_ne_bytes(&fixture.row_positions);

    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let mut staging =
        context.allocate_pinned_host_buffer(u64::try_from(fixture.key_pool.len())?)?;

    let eager_query = upload(&context, &mut eager_stream, &mut staging, &fixture.query)?;
    let eager_key_pool = upload(&context, &mut eager_stream, &mut staging, &fixture.key_pool)?;
    let eager_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool,
    )?;
    let eager_offsets = upload(&context, &mut eager_stream, &mut staging, &offsets_bytes)?;
    let eager_block_ids = upload(&context, &mut eager_stream, &mut staging, &block_ids_bytes)?;
    let eager_valid_tokens = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &valid_tokens_bytes,
    )?;
    let eager_row_slots = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &raw_row_slots_bytes,
    )?;
    let eager_row_positions = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &row_positions_bytes,
    )?;
    let mut eager_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;
    let graph_query = upload(&context, &mut eager_stream, &mut staging, &fixture.query)?;
    let graph_key_pool = upload(&context, &mut eager_stream, &mut staging, &fixture.key_pool)?;
    let graph_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool,
    )?;
    let graph_offsets = upload(&context, &mut eager_stream, &mut staging, &offsets_bytes)?;
    let graph_block_ids = upload(&context, &mut eager_stream, &mut staging, &block_ids_bytes)?;
    let graph_valid_tokens = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &valid_tokens_bytes,
    )?;
    let graph_row_slots = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &raw_row_slots_bytes,
    )?;
    let graph_row_positions = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &row_positions_bytes,
    )?;
    let graph_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;

    let eager_output_len = eager_output.byte_len();
    grouped_ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&eager_query, CudaDType::BF16, 0, eager_query.byte_len())?,
            key_pool: CudaBufferSpan::new(
                &eager_key_pool,
                CudaDType::BF16,
                0,
                eager_key_pool.byte_len(),
            )?,
            value_pool: CudaBufferSpan::new(
                &eager_value_pool,
                CudaDType::BF16,
                0,
                eager_value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(
                &mut eager_output,
                CudaDType::BF16,
                0,
                eager_output_len,
            )?,
            batch: bind_batch(
                fixture.host()?,
                &eager_offsets,
                &eager_block_ids,
                &eager_valid_tokens,
                &eager_row_slots,
                &eager_row_positions,
            )?,
            query_head_count: QUERY_HEAD_COUNT,
            key_value_head_count: KEY_VALUE_HEAD_COUNT,
            head_size: HEAD_SIZE,
            output_row_count: OUTPUT_ROW_COUNT,
            scale: SCALE,
        },
        &mut eager_stream,
    )?;
    eager_stream.synchronize()?;

    let mut capture = capture_stream
        .begin_owned_graph_grouped_ragged_paged_attention_bf16_capture(
            graph_query,
            graph_key_pool,
            graph_value_pool,
            graph_output,
            graph_offsets,
            graph_block_ids,
            graph_valid_tokens,
            graph_row_slots,
            graph_row_positions,
            fixture.host()?,
            QUERY_HEAD_COUNT,
            KEY_VALUE_HEAD_COUNT,
            OUTPUT_ROW_COUNT,
            SCALE,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
    capture.enqueue_grouped_ragged_paged_attention_bf16()?;
    let mut exec = capture.end()?.instantiate()?;
    exec.launch()?.finish()?;
    let resources = exec.close()?;
    let (
        capture_stream,
        _graph_query,
        _graph_key_pool,
        _graph_value_pool,
        mut graph_output,
        _graph_offsets,
        _graph_block_ids,
        _graph_valid_tokens,
        _graph_row_slots,
        _graph_row_positions,
    ) = resources.into_parts();
    let graph_output_bytes = download(&mut graph_output, &mut staging, &mut transfer_stream)?;
    let eager_output_bytes = download(&mut eager_output, &mut staging, &mut transfer_stream)?;
    assert_eq!(
        graph_output_bytes, eager_output_bytes,
        "bounds-invalid raw metadata output must exactly match eager grouped attention",
    );
    let invalid_row_start = usize::try_from(QUERY_HEAD_COUNT * HEAD_SIZE * BF16_BYTES)?;
    let invalid_row_end = invalid_row_start * 2;
    for word in graph_output_bytes[invalid_row_start..invalid_row_end].chunks_exact(2) {
        assert_bf16_nan(word);
    }

    graph_output.close()?;
    let (
        _stream,
        graph_query,
        graph_key_pool,
        graph_value_pool,
        graph_offsets,
        graph_block_ids,
        graph_valid_tokens,
        graph_row_slots,
        graph_row_positions,
    ) = (
        capture_stream,
        _graph_query,
        _graph_key_pool,
        _graph_value_pool,
        _graph_offsets,
        _graph_block_ids,
        _graph_valid_tokens,
        _graph_row_slots,
        _graph_row_positions,
    );
    graph_row_positions.close()?;
    graph_row_slots.close()?;
    graph_valid_tokens.close()?;
    graph_block_ids.close()?;
    graph_offsets.close()?;
    graph_value_pool.close()?;
    graph_key_pool.close()?;
    graph_query.close()?;
    eager_output.close()?;
    eager_row_positions.close()?;
    eager_row_slots.close()?;
    eager_valid_tokens.close()?;
    eager_block_ids.close()?;
    eager_offsets.close()?;
    eager_value_pool.close()?;
    eager_key_pool.close()?;
    eager_query.close()?;
    staging.close()?;
    _stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-19-grouped-ragged-attention-bounds-invalid-row status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_grouped_ragged_attention_graph_preflight_and_abort_recover_every_resource() -> TestResult {
    let fixture = GroupedAttentionFixture::new();
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let query_bytes = u64::try_from(fixture.query.len())?;
    let pool_bytes = u64::try_from(fixture.key_pool.len())?;
    let output_bytes = u64::try_from(fixture.output_sentinel.len())?;
    let offsets_bytes =
        u64::try_from(fixture.offsets.len())? * u64::try_from(std::mem::size_of::<u32>())?;
    let block_ids_bytes =
        u64::try_from(fixture.block_ids.len())? * u64::try_from(std::mem::size_of::<u32>())?;
    let valid_tokens_bytes =
        u64::try_from(fixture.valid_tokens.len())? * u64::try_from(std::mem::size_of::<u16>())?;
    let rows_bytes =
        u64::try_from(fixture.row_slots.len())? * u64::try_from(std::mem::size_of::<u32>())?;

    let query = context.allocate_device_buffer(query_bytes)?;
    let key_pool = context.allocate_device_buffer(pool_bytes)?;
    let value_pool = context.allocate_device_buffer(pool_bytes)?;
    let short_output = context.allocate_device_buffer(output_bytes - BF16_BYTES)?;
    let offsets = context.allocate_device_buffer(offsets_bytes)?;
    let block_ids = context.allocate_device_buffer(block_ids_bytes)?;
    let valid_tokens = context.allocate_device_buffer(valid_tokens_bytes)?;
    let row_slots = context.allocate_device_buffer(rows_bytes)?;
    let row_positions = context.allocate_device_buffer(rows_bytes)?;
    let allocation_with_resources = context.allocation_stats()?;

    let error = match stream.begin_owned_graph_grouped_ragged_paged_attention_bf16_capture(
        query,
        key_pool,
        value_pool,
        short_output,
        offsets,
        block_ids,
        valid_tokens,
        row_slots,
        row_positions,
        fixture.host()?,
        QUERY_HEAD_COUNT,
        KEY_VALUE_HEAD_COUNT,
        OUTPUT_ROW_COUNT,
        SCALE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("C05-19 short output preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("Rust preflight must recover every untouched C05-19 resource");
    let (
        stream,
        query,
        key_pool,
        value_pool,
        short_output,
        offsets,
        block_ids,
        valid_tokens,
        row_slots,
        row_positions,
    ) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    short_output.close()?;
    let output = context.allocate_device_buffer(output_bytes)?;
    let resources = stream
        .begin_owned_graph_grouped_ragged_paged_attention_bf16_capture(
            query,
            key_pool,
            value_pool,
            output,
            offsets,
            block_ids,
            valid_tokens,
            row_slots,
            row_positions,
            fixture.host()?,
            QUERY_HEAD_COUNT,
            KEY_VALUE_HEAD_COUNT,
            OUTPUT_ROW_COUNT,
            SCALE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    resources.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-19-grouped-ragged-attention-preflight-abort-recovery status=passed");
    Ok(())
}
