#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    Bf16EmbeddingStatusD2HStatus, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaErrorKind, CudaGraphCaptureMode, CudaRuntime, CudaStream, EmbeddingError,
    EmbeddingParams, embedding,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const TOKEN_COUNT: u64 = 4;
const VOCABULARY_SIZE: u64 = 5;
const HIDDEN_SIZE: u64 = 3;
const BF16_BYTES: u64 = 2;
const U32_BYTES: u64 = 4;
const REPORT_BYTES: u64 = 32;

struct EmbeddingFixture {
    table: Vec<u8>,
    token_ids: Vec<u32>,
    output_sentinel: Vec<u8>,
}

impl EmbeddingFixture {
    fn valid() -> Self {
        // Preserve ordinary BF16 values, signed zero, and NaN payloads so the
        // graph must remain a storage move rather than an FP32 round trip.
        let table = bf16_words_to_ne_bytes(&[
            0x3f80, 0x7fc1, 0x8000, // vocabulary row 0
            0x4000, 0xc040, 0x0001, // row 1
            0x7fff, 0x3e00, 0xbf80, // row 2
            0x4040, 0x7f81, 0x3f00, // row 3
            0xc000, 0x3f40, 0xffc1, // row 4
        ]);
        let output_bytes = TOKEN_COUNT
            .checked_mul(HIDDEN_SIZE)
            .and_then(|value| value.checked_mul(BF16_BYTES))
            .expect("C05-20 output byte count fits u64");
        Self {
            table,
            token_ids: vec![4, 0, 3, 1],
            output_sentinel: vec![0xa5; usize::try_from(output_bytes).unwrap()],
        }
    }

    fn invalid() -> Self {
        let mut fixture = Self::valid();
        // No host mirror is accepted by C05-20. These are raw fixed device
        // bytes at replay, and index 1 must win over the later OOB index 2.
        fixture.token_ids = vec![1, 9, 7, 2];
        fixture
    }
}

fn bf16_words_to_ne_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn u32_words_to_ne_bytes(words: &[u32]) -> Vec<u8> {
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

fn assert_invalid_state<T>(result: riley_cuda::CudaResult<T>, operation: &str) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("{operation} unexpectedly succeeded a second time"));
    assert_eq!(
        error.kind(),
        CudaErrorKind::InvalidState,
        "{operation} must reject before recording another graph node"
    );
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_embedding_status_d2h_graph_replays_byte_exact_against_eager() -> TestResult {
    const REPLAYS: usize = 64;
    let fixture = EmbeddingFixture::valid();
    let token_bytes = u32_words_to_ne_bytes(&fixture.token_ids);
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
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let eager_table = upload(&context, &mut eager_stream, &mut staging, &fixture.table)?;
    let eager_token_ids = upload(&context, &mut eager_stream, &mut staging, &token_bytes)?;
    let mut eager_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;
    let mut eager_error_scratch = context.allocate_device_buffer(REPORT_BYTES)?;
    {
        let table_len = eager_table.byte_len();
        let token_ids_len = eager_token_ids.byte_len();
        let output_len = eager_output.byte_len();
        let error_scratch_len = eager_error_scratch.byte_len();
        embedding(
            &mut EmbeddingParams {
                table: CudaBufferSpan::new(&eager_table, CudaDType::BF16, 0, table_len)?,
                token_ids: CudaBufferSpan::new(&eager_token_ids, CudaDType::U32, 0, token_ids_len)?,
                output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, output_len)?,
                error_scratch: CudaBufferSpanMut::new(
                    &mut eager_error_scratch,
                    CudaDType::U8,
                    0,
                    error_scratch_len,
                )?,
                token_count: TOKEN_COUNT,
                vocabulary_size: VOCABULARY_SIZE,
                hidden_size: HIDDEN_SIZE,
            },
            &mut eager_stream,
        )?;
    }
    let eager_output_bytes = download(&mut eager_output, &mut staging, &mut transfer_stream)?;

    let graph_table = upload(&context, &mut eager_stream, &mut staging, &fixture.table)?;
    let graph_token_ids = upload(&context, &mut eager_stream, &mut staging, &token_bytes)?;
    let graph_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;
    let graph_error_scratch = context.allocate_device_buffer(REPORT_BYTES)?;
    let graph_pinned_report = context.allocate_pinned_host_buffer(REPORT_BYTES)?;

    let mut capture = capture_stream.begin_owned_graph_bf16_embedding_status_d2h_capture(
        graph_table,
        graph_token_ids,
        graph_output,
        graph_error_scratch,
        graph_pinned_report,
        TOKEN_COUNT,
        VOCABULARY_SIZE,
        HIDDEN_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_bf16_embedding_status_d2h()?;
    assert_invalid_state(
        capture.enqueue_bf16_embedding_status_d2h(),
        "C05-20 embedding graph second enqueue",
    );
    let mut exec = capture.end()?.instantiate()?;
    for _ in 0..REPLAYS {
        let mut completion = exec.launch()?.finish()?;
        assert_eq!(
            completion.read_status()?,
            Bf16EmbeddingStatusD2HStatus::Success,
            "valid graph replay must publish the eager successful embedding status"
        );
    }

    let resources = exec.close()?;
    let (
        graph_stream,
        mut graph_table,
        mut graph_token_ids,
        mut graph_output,
        graph_error_scratch,
        graph_pinned_report,
    ) = resources.into_parts();
    assert_eq!(
        download(&mut graph_output, &mut staging, &mut transfer_stream)?,
        eager_output_bytes,
        "C05-20 graph BF16 gather must match eager bytes exactly"
    );
    assert_eq!(
        download(&mut graph_table, &mut staging, &mut transfer_stream)?,
        fixture.table,
        "graph embedding must not mutate the fixed table"
    );
    assert_eq!(
        download(&mut graph_token_ids, &mut staging, &mut transfer_stream)?,
        token_bytes,
        "graph embedding must not mutate fixed token IDs"
    );

    graph_pinned_report.close()?;
    graph_error_scratch.close()?;
    graph_output.close()?;
    graph_token_ids.close()?;
    graph_table.close()?;
    graph_stream.close()?;
    eager_error_scratch.close()?;
    eager_output.close()?;
    eager_token_ids.close()?;
    eager_table.close()?;
    staging.close()?;
    transfer_stream.close()?;
    eager_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_embedding_status_d2h_graph_preserves_raw_oob_no_write_parity() -> TestResult {
    let fixture = EmbeddingFixture::invalid();
    let token_bytes = u32_words_to_ne_bytes(&fixture.token_ids);
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
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let eager_table = upload(&context, &mut eager_stream, &mut staging, &fixture.table)?;
    let eager_token_ids = upload(&context, &mut eager_stream, &mut staging, &token_bytes)?;
    let mut eager_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;
    let mut eager_error_scratch = context.allocate_device_buffer(REPORT_BYTES)?;
    let eager_error = {
        let table_len = eager_table.byte_len();
        let token_ids_len = eager_token_ids.byte_len();
        let output_len = eager_output.byte_len();
        let error_scratch_len = eager_error_scratch.byte_len();
        embedding(
            &mut EmbeddingParams {
                table: CudaBufferSpan::new(&eager_table, CudaDType::BF16, 0, table_len)?,
                token_ids: CudaBufferSpan::new(&eager_token_ids, CudaDType::U32, 0, token_ids_len)?,
                output: CudaBufferSpanMut::new(&mut eager_output, CudaDType::BF16, 0, output_len)?,
                error_scratch: CudaBufferSpanMut::new(
                    &mut eager_error_scratch,
                    CudaDType::U8,
                    0,
                    error_scratch_len,
                )?,
                token_count: TOKEN_COUNT,
                vocabulary_size: VOCABULARY_SIZE,
                hidden_size: HIDDEN_SIZE,
            },
            &mut eager_stream,
        )
        .expect_err("raw OOB device IDs must retain eager structured failure")
    };
    match eager_error {
        EmbeddingError::TokenOutOfRange {
            token_position,
            token_id,
            source,
        } => {
            assert_eq!(token_position, 1);
            assert_eq!(token_id, 9);
            assert_eq!(source.kind(), CudaErrorKind::OutOfRange);
        }
        EmbeddingError::Cuda(error) => panic!("expected eager token OOB, got {error}"),
    }
    assert_eq!(
        download(&mut eager_output, &mut staging, &mut transfer_stream)?,
        fixture.output_sentinel,
        "eager OOB embedding must leave its complete output untouched"
    );

    let graph_table = upload(&context, &mut eager_stream, &mut staging, &fixture.table)?;
    let graph_token_ids = upload(&context, &mut eager_stream, &mut staging, &token_bytes)?;
    let graph_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &fixture.output_sentinel,
    )?;
    let graph_error_scratch = context.allocate_device_buffer(REPORT_BYTES)?;
    let graph_pinned_report = context.allocate_pinned_host_buffer(REPORT_BYTES)?;
    let mut capture = capture_stream.begin_owned_graph_bf16_embedding_status_d2h_capture(
        graph_table,
        graph_token_ids,
        graph_output,
        graph_error_scratch,
        graph_pinned_report,
        TOKEN_COUNT,
        VOCABULARY_SIZE,
        HIDDEN_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_bf16_embedding_status_d2h()?;
    let mut exec = capture.end()?.instantiate()?;
    let mut completion = exec.launch()?.finish()?;
    assert_eq!(
        completion.read_status()?,
        Bf16EmbeddingStatusD2HStatus::TokenOutOfRange {
            token_position: 1,
            token_id: 9,
        },
        "C05-20 must expose the earliest raw OOB device token as a reusable status"
    );
    drop(completion);

    let resources = exec.close()?;
    let (
        graph_stream,
        mut graph_table,
        mut graph_token_ids,
        mut graph_output,
        graph_error_scratch,
        graph_pinned_report,
    ) = resources.into_parts();
    assert_eq!(
        download(&mut graph_output, &mut staging, &mut transfer_stream)?,
        fixture.output_sentinel,
        "graph OOB embedding must suppress every output write like eager"
    );
    assert_eq!(
        download(&mut graph_table, &mut staging, &mut transfer_stream)?,
        fixture.table,
        "graph OOB embedding must not mutate the fixed table"
    );
    assert_eq!(
        download(&mut graph_token_ids, &mut staging, &mut transfer_stream)?,
        token_bytes,
        "graph OOB embedding must not mutate fixed token IDs"
    );

    graph_pinned_report.close()?;
    graph_error_scratch.close()?;
    graph_output.close()?;
    graph_token_ids.close()?;
    graph_table.close()?;
    graph_stream.close()?;
    eager_error_scratch.close()?;
    eager_output.close()?;
    eager_token_ids.close()?;
    eager_table.close()?;
    staging.close()?;
    transfer_stream.close()?;
    eager_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}

#[derive(Clone, Copy)]
enum ShortResource {
    Output,
    ErrorScratch,
    PinnedReport,
}

fn rejected_short_resource_recovers(context: &CudaContext, short: ShortResource) -> TestResult {
    let table_bytes = VOCABULARY_SIZE * HIDDEN_SIZE * BF16_BYTES;
    let token_bytes = TOKEN_COUNT * U32_BYTES;
    let output_bytes = TOKEN_COUNT * HIDDEN_SIZE * BF16_BYTES;
    let stream = context.create_stream()?;
    let table = context.allocate_device_buffer(table_bytes)?;
    let token_ids = context.allocate_device_buffer(token_bytes)?;
    let output = context.allocate_device_buffer(if matches!(short, ShortResource::Output) {
        output_bytes - 1
    } else {
        output_bytes
    })?;
    let error_scratch =
        context.allocate_device_buffer(if matches!(short, ShortResource::ErrorScratch) {
            REPORT_BYTES - 1
        } else {
            REPORT_BYTES
        })?;
    let pinned_report =
        context.allocate_pinned_host_buffer(if matches!(short, ShortResource::PinnedReport) {
            REPORT_BYTES - 1
        } else {
            REPORT_BYTES
        })?;
    let error = match stream.begin_owned_graph_bf16_embedding_status_d2h_capture(
        table,
        token_ids,
        output,
        error_scratch,
        pinned_report,
        TOKEN_COUNT,
        VOCABULARY_SIZE,
        HIDDEN_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("short C05-20 fixed allocation must fail before native capture"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("pure Rust preflight must recover every untouched fixed resource")
        .close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_bf16_embedding_status_d2h_graph_recovers_preflight_and_abort_resources() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    for short in [
        ShortResource::Output,
        ShortResource::ErrorScratch,
        ShortResource::PinnedReport,
    ] {
        rejected_short_resource_recovers(&context, short)?;
        assert_eq!(
            context.allocation_stats()?,
            allocation_baseline,
            "rejected C05-20 preflight must return to the allocation baseline"
        );
    }

    let table_bytes = VOCABULARY_SIZE * HIDDEN_SIZE * BF16_BYTES;
    let token_bytes = TOKEN_COUNT * U32_BYTES;
    let output_bytes = TOKEN_COUNT * HIDDEN_SIZE * BF16_BYTES;
    let stream = context.create_stream()?;
    let capture = stream.begin_owned_graph_bf16_embedding_status_d2h_capture(
        context.allocate_device_buffer(table_bytes)?,
        context.allocate_device_buffer(token_bytes)?,
        context.allocate_device_buffer(output_bytes)?,
        context.allocate_device_buffer(REPORT_BYTES)?,
        context.allocate_pinned_host_buffer(REPORT_BYTES)?,
        TOKEN_COUNT,
        VOCABULARY_SIZE,
        HIDDEN_SIZE,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.abort()?.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}
