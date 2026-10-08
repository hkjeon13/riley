#![allow(clippy::too_many_arguments, clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaErrorKind,
    CudaGraphCaptureMode, CudaRuntime, CudaStream, PackedBatchHostV1, PackedBatchV1,
    RaggedPagedKvCacheWriteParams, ragged_paged_kv_cache_write,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SEQUENCE_COUNT: u64 = 2;
const BLOCK_COUNT: u64 = 4;
const ACTIVE_ROW_COUNT: u64 = 3;
const PHYSICAL_BLOCK_COUNT: u64 = 4;
const KEY_VALUE_HEAD_COUNT: u64 = 1;
const HEAD_SIZE: u64 = 2;
const BLOCK_SIZE: u64 = 16;
const BF16_BYTES: u64 = 2;

struct RaggedFixture {
    offsets: Vec<u32>,
    block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
    row_slots: Vec<u32>,
    row_positions: Vec<u32>,
    key_source: Vec<u8>,
    value_source: Vec<u8>,
    key_pool_sentinel: Vec<u8>,
    value_pool_sentinel: Vec<u8>,
}

impl RaggedFixture {
    fn new() -> Self {
        // Sequence 0 uses physical blocks 2 then 0, and sequence 1 uses 3
        // then 1. The three active rows exercise token 0, the 15/16 page
        // transition, and a final valid token at position 17.
        let offsets = vec![0_u32, 2, 4];
        let block_ids = vec![2_u32, 0, 3, 1];
        let valid_tokens = vec![16_u16, 1, 16, 2];
        let row_slots = vec![0_u32, 0, 1];
        let row_positions = vec![0_u32, 16, 17];
        assert_eq!(
            block_ids.len(),
            usize::try_from(BLOCK_COUNT).expect("C05-18 fixture block count fits usize")
        );
        let key_source = bf16_words_to_ne_bytes(&[
            0x3f80, 0x4000, // row 0
            0xc040, 0x4040, // row 1 / sequence 0 page 2
            0x7fc1, 0x8000, // row 2 / sequence 1 final block
        ]);
        let value_source = bf16_words_to_ne_bytes(&[
            0xbf80, 0x3f00, // row 0
            0x4080, 0xc000, // row 1
            0xffc1, 0x0000, // row 2
        ]);
        let pool_byte_len = PHYSICAL_BLOCK_COUNT
            .checked_mul(KEY_VALUE_HEAD_COUNT)
            .and_then(|value| value.checked_mul(BLOCK_SIZE))
            .and_then(|value| value.checked_mul(HEAD_SIZE))
            .and_then(|value| value.checked_mul(BF16_BYTES))
            .expect("C05-18 fixture pool byte length fits u64");
        Self {
            offsets,
            block_ids,
            valid_tokens,
            row_slots,
            row_positions,
            key_source,
            value_source,
            key_pool_sentinel: vec![0xa5; usize::try_from(pool_byte_len).unwrap()],
            value_pool_sentinel: vec![0x5a; usize::try_from(pool_byte_len).unwrap()],
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

fn bf16_words_to_ne_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
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
        "{operation} must reject before issuing another CUDA capture node"
    );
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_ragged_paged_kv_write_graph_replays_byte_exact_against_eager() -> TestResult {
    const REPLAYS: usize = 64;
    let fixture = RaggedFixture::new();
    assert_eq!(fixture.key_source.len(), fixture.value_source.len());
    assert_eq!(u64::try_from(fixture.row_slots.len())?, ACTIVE_ROW_COUNT);

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
    let staging_byte_len = u64::try_from(
        fixture
            .key_pool_sentinel
            .len()
            .max(fixture.value_pool_sentinel.len())
            .max(fixture.key_source.len()),
    )?;
    let mut staging = context.allocate_pinned_host_buffer(staging_byte_len)?;

    let offsets_bytes = u32_words_to_ne_bytes(&fixture.offsets);
    let block_ids_bytes = u32_words_to_ne_bytes(&fixture.block_ids);
    let valid_tokens_bytes = u16_words_to_ne_bytes(&fixture.valid_tokens);
    let row_slots_bytes = u32_words_to_ne_bytes(&fixture.row_slots);
    let row_positions_bytes = u32_words_to_ne_bytes(&fixture.row_positions);

    let eager_key_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_source,
    )?;
    let eager_value_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_source,
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
    let mut eager_key_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_pool_sentinel,
    )?;
    let mut eager_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool_sentinel,
    )?;

    let graph_key_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_source,
    )?;
    let graph_value_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_source,
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
    let graph_key_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_pool_sentinel,
    )?;
    let graph_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool_sentinel,
    )?;

    let eager_key_pool_len = eager_key_pool.byte_len();
    let eager_value_pool_len = eager_value_pool.byte_len();
    ragged_paged_kv_cache_write(
        &mut RaggedPagedKvCacheWriteParams {
            key_source: CudaBufferSpan::new(
                &eager_key_source,
                CudaDType::BF16,
                0,
                eager_key_source.byte_len(),
            )?,
            value_source: CudaBufferSpan::new(
                &eager_value_source,
                CudaDType::BF16,
                0,
                eager_value_source.byte_len(),
            )?,
            key_pool: CudaBufferSpanMut::new(
                &mut eager_key_pool,
                CudaDType::BF16,
                0,
                eager_key_pool_len,
            )?,
            value_pool: CudaBufferSpanMut::new(
                &mut eager_value_pool,
                CudaDType::BF16,
                0,
                eager_value_pool_len,
            )?,
            batch: bind_batch(
                fixture.host()?,
                &eager_offsets,
                &eager_block_ids,
                &eager_valid_tokens,
                &eager_row_slots,
                &eager_row_positions,
            )?,
            key_value_head_count: KEY_VALUE_HEAD_COUNT,
            head_size: HEAD_SIZE,
        },
        &mut eager_stream,
    )?;
    eager_stream.synchronize()?;

    // Make a separate, temporary admission witness. The raw device metadata
    // that the graph reads was uploaded above and the owner must not retain
    // these host arrays after capture begin returns.
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
        capture_stream.begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture(
            graph_key_source,
            graph_value_source,
            graph_key_pool,
            graph_value_pool,
            graph_offsets,
            graph_block_ids,
            graph_valid_tokens,
            graph_row_slots,
            graph_row_positions,
            admission,
            KEY_VALUE_HEAD_COUNT,
            HEAD_SIZE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
    };
    drop(admission_offsets);
    drop(admission_block_ids);
    drop(admission_valid_tokens);
    drop(admission_row_slots);
    drop(admission_row_positions);
    capture.enqueue_ragged_paged_kv_cache_write_bf16()?;
    assert_invalid_state(
        capture.enqueue_ragged_paged_kv_cache_write_bf16(),
        "second C05-18 ragged paged-K/V graph enqueue",
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
        mut graph_key_source,
        mut graph_value_source,
        mut graph_key_pool,
        mut graph_value_pool,
        mut graph_offsets,
        mut graph_block_ids,
        mut graph_valid_tokens,
        mut graph_row_slots,
        mut graph_row_positions,
    ) = resources.into_parts();
    assert_eq!(
        download(&mut graph_key_pool, &mut staging, &mut transfer_stream)?,
        download(&mut eager_key_pool, &mut staging, &mut transfer_stream)?,
        "C05-18 graph key pool must match eager BF16 storage bytes exactly",
    );
    assert_eq!(
        download(&mut graph_value_pool, &mut staging, &mut transfer_stream)?,
        download(&mut eager_value_pool, &mut staging, &mut transfer_stream)?,
        "C05-18 graph value pool must match eager BF16 storage bytes exactly",
    );
    assert_eq!(
        download(&mut graph_key_source, &mut staging, &mut transfer_stream)?,
        fixture.key_source,
        "fixed graph must not mutate key source bytes",
    );
    assert_eq!(
        download(&mut graph_value_source, &mut staging, &mut transfer_stream)?,
        fixture.value_source,
        "fixed graph must not mutate value source bytes",
    );
    for (name, buffer, expected) in [
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
            "fixed graph must not mutate packed {name} metadata bytes",
        );
    }

    graph_row_positions.close()?;
    graph_row_slots.close()?;
    graph_valid_tokens.close()?;
    graph_block_ids.close()?;
    graph_offsets.close()?;
    graph_value_pool.close()?;
    graph_key_pool.close()?;
    graph_value_source.close()?;
    graph_key_source.close()?;
    eager_value_pool.close()?;
    eager_key_pool.close()?;
    eager_row_positions.close()?;
    eager_row_slots.close()?;
    eager_valid_tokens.close()?;
    eager_block_ids.close()?;
    eager_offsets.close()?;
    eager_value_source.close()?;
    eager_key_source.close()?;
    staging.close()?;
    capture_stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-18-ragged-paged-kv-write-valid-replays={REPLAYS} status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_ragged_paged_kv_write_graph_preserves_bounds_invalid_raw_row_noop() -> TestResult {
    let fixture = RaggedFixture::new();
    // Host admission stays valid. Only the pre-uploaded fixed device row
    // metadata becomes invalid, exercising the same bounds guard as eager.
    let raw_row_slots = vec![0_u32, SEQUENCE_COUNT as u32, 1];
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
        context.allocate_pinned_host_buffer(u64::try_from(fixture.key_pool_sentinel.len())?)?;

    let eager_key_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_source,
    )?;
    let eager_value_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_source,
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
    let mut eager_key_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_pool_sentinel,
    )?;
    let mut eager_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool_sentinel,
    )?;
    let graph_key_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_source,
    )?;
    let graph_value_source = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_source,
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
    let graph_key_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.key_pool_sentinel,
    )?;
    let graph_value_pool = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.value_pool_sentinel,
    )?;

    let eager_key_pool_len = eager_key_pool.byte_len();
    let eager_value_pool_len = eager_value_pool.byte_len();
    ragged_paged_kv_cache_write(
        &mut RaggedPagedKvCacheWriteParams {
            key_source: CudaBufferSpan::new(
                &eager_key_source,
                CudaDType::BF16,
                0,
                eager_key_source.byte_len(),
            )?,
            value_source: CudaBufferSpan::new(
                &eager_value_source,
                CudaDType::BF16,
                0,
                eager_value_source.byte_len(),
            )?,
            key_pool: CudaBufferSpanMut::new(
                &mut eager_key_pool,
                CudaDType::BF16,
                0,
                eager_key_pool_len,
            )?,
            value_pool: CudaBufferSpanMut::new(
                &mut eager_value_pool,
                CudaDType::BF16,
                0,
                eager_value_pool_len,
            )?,
            batch: bind_batch(
                fixture.host()?,
                &eager_offsets,
                &eager_block_ids,
                &eager_valid_tokens,
                &eager_row_slots,
                &eager_row_positions,
            )?,
            key_value_head_count: KEY_VALUE_HEAD_COUNT,
            head_size: HEAD_SIZE,
        },
        &mut eager_stream,
    )?;
    eager_stream.synchronize()?;

    let admission_offsets = fixture.offsets.clone();
    let admission_block_ids = fixture.block_ids.clone();
    let admission_valid_tokens = fixture.valid_tokens.clone();
    let admission_row_slots = fixture.row_slots.clone();
    let admission_row_positions = fixture.row_positions.clone();
    let mut capture = {
        let admission = PackedBatchHostV1::new(
            &admission_offsets,
            &admission_block_ids,
            &admission_valid_tokens,
            &admission_row_slots,
            &admission_row_positions,
            PHYSICAL_BLOCK_COUNT,
        )?;
        capture_stream.begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture(
            graph_key_source,
            graph_value_source,
            graph_key_pool,
            graph_value_pool,
            graph_offsets,
            graph_block_ids,
            graph_valid_tokens,
            graph_row_slots,
            graph_row_positions,
            admission,
            KEY_VALUE_HEAD_COUNT,
            HEAD_SIZE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
    };
    drop(admission_offsets);
    drop(admission_block_ids);
    drop(admission_valid_tokens);
    drop(admission_row_slots);
    drop(admission_row_positions);
    capture.enqueue_ragged_paged_kv_cache_write_bf16()?;
    let mut exec = capture.end()?.instantiate()?;
    exec.launch()?.finish()?;
    let resources = exec.close()?;
    let (
        capture_stream,
        _graph_key_source,
        _graph_value_source,
        mut graph_key_pool,
        mut graph_value_pool,
        _graph_offsets,
        _graph_block_ids,
        _graph_valid_tokens,
        _graph_row_slots,
        _graph_row_positions,
    ) = resources.into_parts();
    let graph_key_pool_bytes = download(&mut graph_key_pool, &mut staging, &mut transfer_stream)?;
    let graph_value_pool_bytes =
        download(&mut graph_value_pool, &mut staging, &mut transfer_stream)?;
    assert_eq!(
        graph_key_pool_bytes,
        download(&mut eager_key_pool, &mut staging, &mut transfer_stream)?,
        "bounds-invalid raw device row must retain eager key-pool bytes",
    );
    assert_eq!(
        graph_value_pool_bytes,
        download(&mut eager_value_pool, &mut staging, &mut transfer_stream)?,
        "bounds-invalid raw device row must retain eager value-pool bytes",
    );
    // Row 1 would have addressed physical block 0, token 0, two BF16 lanes.
    // Its invalid raw sequence slot must leave exactly that K/V destination at
    // its respective initial sentinel bytes.
    assert_eq!(&graph_key_pool_bytes[..4], &fixture.key_pool_sentinel[..4]);
    assert_eq!(
        &graph_value_pool_bytes[..4],
        &fixture.value_pool_sentinel[..4]
    );

    graph_value_pool.close()?;
    graph_key_pool.close()?;
    // The remaining graph buffers were intentionally not read, but must still
    // be closed after their known graph lease release.
    let (
        _stream,
        graph_key_source,
        graph_value_source,
        graph_offsets,
        graph_block_ids,
        graph_valid_tokens,
        graph_row_slots,
        graph_row_positions,
    ) = (
        capture_stream,
        _graph_key_source,
        _graph_value_source,
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
    graph_value_source.close()?;
    graph_key_source.close()?;
    eager_value_pool.close()?;
    eager_key_pool.close()?;
    eager_row_positions.close()?;
    eager_row_slots.close()?;
    eager_valid_tokens.close()?;
    eager_block_ids.close()?;
    eager_offsets.close()?;
    eager_value_source.close()?;
    eager_key_source.close()?;
    staging.close()?;
    _stream.close()?;
    eager_stream.close()?;
    transfer_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-18-ragged-paged-kv-write-bounds-invalid-row status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_ragged_paged_kv_write_graph_preflight_and_abort_recover_every_resource() -> TestResult {
    let fixture = RaggedFixture::new();
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let stream = context.create_stream()?;
    let source_bytes = u64::try_from(fixture.key_source.len())?;
    let pool_bytes = u64::try_from(fixture.key_pool_sentinel.len())?;
    let offsets_bytes =
        u64::try_from(fixture.offsets.len())? * u64::try_from(std::mem::size_of::<u32>())?;
    let block_ids_bytes =
        u64::try_from(fixture.block_ids.len())? * u64::try_from(std::mem::size_of::<u32>())?;
    let valid_tokens_bytes =
        u64::try_from(fixture.valid_tokens.len())? * u64::try_from(std::mem::size_of::<u16>())?;
    let rows_bytes =
        u64::try_from(fixture.row_slots.len())? * u64::try_from(std::mem::size_of::<u32>())?;

    let key_source = context.allocate_device_buffer(source_bytes)?;
    let value_source = context.allocate_device_buffer(source_bytes)?;
    let key_pool = context.allocate_device_buffer(pool_bytes)?;
    let short_value_pool = context.allocate_device_buffer(pool_bytes - BF16_BYTES)?;
    let offsets = context.allocate_device_buffer(offsets_bytes)?;
    let block_ids = context.allocate_device_buffer(block_ids_bytes)?;
    let valid_tokens = context.allocate_device_buffer(valid_tokens_bytes)?;
    let row_slots = context.allocate_device_buffer(rows_bytes)?;
    let row_positions = context.allocate_device_buffer(rows_bytes)?;
    let allocation_with_resources = context.allocation_stats()?;

    let error = match stream.begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture(
        key_source,
        value_source,
        key_pool,
        short_value_pool,
        offsets,
        block_ids,
        valid_tokens,
        row_slots,
        row_positions,
        fixture.host()?,
        KEY_VALUE_HEAD_COUNT,
        HEAD_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("C05-18 short value-pool preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    let resources = error
        .into_resources()
        .expect("Rust preflight must recover every untouched C05-18 resource");
    let (
        stream,
        key_source,
        value_source,
        key_pool,
        short_value_pool,
        offsets,
        block_ids,
        valid_tokens,
        row_slots,
        row_positions,
    ) = resources.into_parts();
    assert_eq!(context.allocation_stats()?, allocation_with_resources);
    short_value_pool.close()?;
    let value_pool = context.allocate_device_buffer(pool_bytes)?;

    let resources = stream
        .begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture(
            key_source,
            value_source,
            key_pool,
            value_pool,
            offsets,
            block_ids,
            valid_tokens,
            row_slots,
            row_positions,
            fixture.host()?,
            KEY_VALUE_HEAD_COUNT,
            HEAD_SIZE,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    resources.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-18-ragged-paged-kv-write-preflight-abort-recovery status=passed");
    Ok(())
}
