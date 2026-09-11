#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaErrorKind,
    CudaGemmConfig, CudaGraphCaptureMode, CudaPinnedHostBuffer, CudaPreparedGemm, CudaRuntime,
    CudaStream, GemmParams,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

// This is the canonical decode projection shape used by the roadmap.  It is
// large enough to exercise the actual cuBLASLt path without turning this
// lifecycle/parity test into a throughput benchmark.
const M: u64 = 1;
const N: u64 = 576;
const K: u64 = 576;
const MAX_WORKSPACE_BYTES: u64 = 8 * 1024 * 1024;
const REPLAYS: usize = 64;

fn finite_bf16_pattern(element_count: u64, seed: u64) -> TestResult<Vec<u8>> {
    let byte_len = element_count
        .checked_mul(u64::try_from(std::mem::size_of::<u16>())?)
        .ok_or("C05-21 BF16 fixture byte count overflow")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(byte_len)?)?;
    for index in 0..element_count {
        // Keep every input finite and small.  The varied storage words make a
        // byte-for-byte eager comparison stronger than a single repeated value.
        let word = 0x3e00_u16 + u16::try_from(index.wrapping_mul(31).wrapping_add(seed) % 0x0100)?;
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    Ok(bytes)
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
    buffer: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> TestResult<Vec<u8>> {
    let mut bytes = vec![0_u8; usize::try_from(buffer.byte_len())?];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}

fn execute_eager(
    plan: &mut CudaPreparedGemm,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let config = plan.config();
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    assert_eq!(workspace.byte_len(), workspace_bytes);
    let workspace_span = if workspace_bytes == 0 {
        None
    } else {
        Some(CudaBufferSpanMut::new(
            workspace,
            CudaDType::U8,
            0,
            workspace_bytes,
        )?)
    };
    let mut params = GemmParams {
        input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
        weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
        workspace: workspace_span,
    };
    plan.execute(&mut params, stream)?;
    Ok(())
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
fn owned_canonical_gemm_bf16_graph_cold_capture_replays_byte_exact_against_eager() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let input_host = finite_bf16_pattern(M * K, 7)?;
    let weight_host = finite_bf16_pattern(N * K, 19)?;
    let output_sentinel = vec![0xa5_u8; usize::try_from(config.output_bytes())?];

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let mut eager_plan = context.prepare_gemm(config)?;
    let eager_workspace_bytes = eager_plan.algorithm_metadata().workspace_bytes();
    let eager_input = upload(&context, &mut eager_stream, &mut staging, &input_host)?;
    let eager_weight = upload(&context, &mut eager_stream, &mut staging, &weight_host)?;
    let mut eager_output = upload(&context, &mut eager_stream, &mut staging, &output_sentinel)?;
    let mut eager_workspace = context.allocate_device_buffer(eager_workspace_bytes)?;
    execute_eager(
        &mut eager_plan,
        &eager_input,
        &eager_weight,
        &mut eager_output,
        &mut eager_workspace,
        &mut eager_stream,
    )?;
    let eager_output_bytes = download(&mut eager_output, &mut staging, &mut transfer_stream)?;
    assert_ne!(
        eager_output_bytes, output_sentinel,
        "the eager baseline must execute rather than preserve the output sentinel"
    );

    // This plan is deliberately never executed eagerly.  Metadata inspection
    // and allocation setup are cold preparation; its first cuBLASLt matmul is
    // the capture-only graph node below.
    let graph_plan = context.prepare_gemm(config)?;
    let graph_workspace_bytes = graph_plan.algorithm_metadata().workspace_bytes();
    let graph_input = upload(&context, &mut eager_stream, &mut staging, &input_host)?;
    let graph_weight = upload(&context, &mut eager_stream, &mut staging, &weight_host)?;
    let graph_output = upload(&context, &mut eager_stream, &mut staging, &output_sentinel)?;
    // A distinct zero-byte allocation is still required when cuBLASLt chooses
    // no workspace; it must never be substituted with an output alias.
    let graph_workspace = context.allocate_device_buffer(graph_workspace_bytes)?;

    let mut capture = capture_stream.begin_owned_graph_canonical_gemm_bf16_capture(
        graph_plan,
        graph_input,
        graph_weight,
        graph_output,
        graph_workspace,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_canonical_gemm_bf16()?;
    assert_invalid_state(
        capture.enqueue_canonical_gemm_bf16(),
        "C05-21 canonical GEMM graph second enqueue",
    );
    let mut exec = capture.end()?.instantiate()?;
    let allocations_before_replay = context.allocation_stats()?;
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(
        context.allocation_stats()?,
        allocations_before_replay,
        "C05-21 graph replay must not change caller allocation accounting"
    );

    let resources = exec.close()?;
    let (
        graph_stream,
        graph_plan,
        mut graph_input,
        mut graph_weight,
        mut graph_output,
        graph_workspace,
    ) = resources.into_parts();
    let graph_output_bytes = download(&mut graph_output, &mut staging, &mut transfer_stream)?;
    assert_eq!(
        graph_output_bytes, eager_output_bytes,
        "C05-21 cold cuBLASLt graph replay must match eager BF16 output bytes exactly"
    );
    assert_ne!(
        graph_output_bytes, output_sentinel,
        "C05-21 graph must execute rather than preserve the output sentinel"
    );
    assert_eq!(
        download(&mut graph_input, &mut staging, &mut transfer_stream)?,
        input_host,
        "C05-21 graph must not mutate its fixed BF16 input allocation"
    );
    assert_eq!(
        download(&mut graph_weight, &mut staging, &mut transfer_stream)?,
        weight_host,
        "C05-21 graph must not mutate its fixed BF16 weight allocation"
    );

    graph_plan.close()?;
    graph_workspace.close()?;
    graph_output.close()?;
    graph_weight.close()?;
    graph_input.close()?;
    graph_stream.close()?;
    eager_plan.close()?;
    eager_workspace.close()?;
    eager_output.close()?;
    eager_weight.close()?;
    eager_input.close()?;
    staging.close()?;
    transfer_stream.close()?;
    eager_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}

#[derive(Clone, Copy)]
enum WrongExactAllocation {
    Input,
    Weight,
    Output,
    Workspace,
}

fn wrong_exact_byte_len(required: u64) -> u64 {
    // A zero-byte workspace has no smaller length.  One byte is still a
    // deterministic wrong-exact-size preflight case.
    if required == 0 { 1 } else { required - 1 }
}

fn rejected_wrong_exact_resource_recovers(
    context: &CudaContext,
    wrong: WrongExactAllocation,
) -> TestResult {
    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let stream = context.create_stream()?;
    let input =
        context.allocate_device_buffer(if matches!(wrong, WrongExactAllocation::Input) {
            wrong_exact_byte_len(config.input_bytes())
        } else {
            config.input_bytes()
        })?;
    let weight =
        context.allocate_device_buffer(if matches!(wrong, WrongExactAllocation::Weight) {
            wrong_exact_byte_len(config.weight_bytes())
        } else {
            config.weight_bytes()
        })?;
    let output =
        context.allocate_device_buffer(if matches!(wrong, WrongExactAllocation::Output) {
            wrong_exact_byte_len(config.output_bytes())
        } else {
            config.output_bytes()
        })?;
    let workspace =
        context.allocate_device_buffer(if matches!(wrong, WrongExactAllocation::Workspace) {
            wrong_exact_byte_len(workspace_bytes)
        } else {
            workspace_bytes
        })?;

    let error = match stream.begin_owned_graph_canonical_gemm_bf16_capture(
        plan,
        input,
        weight,
        output,
        workspace,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("wrong C05-21 fixed allocation must fail before native capture"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("pure Rust C05-21 preflight must recover every untouched resource")
        .close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_canonical_gemm_bf16_graph_recovers_preflight_and_abort_resources() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    for wrong in [
        WrongExactAllocation::Input,
        WrongExactAllocation::Weight,
        WrongExactAllocation::Output,
        WrongExactAllocation::Workspace,
    ] {
        rejected_wrong_exact_resource_recovers(&context, wrong)?;
        assert_eq!(
            context.allocation_stats()?,
            allocation_baseline,
            "rejected C05-21 preflight must return to the allocation baseline"
        );
    }

    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let stream = context.create_stream()?;
    let capture = stream.begin_owned_graph_canonical_gemm_bf16_capture(
        plan,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(config.weight_bytes())?,
        context.allocate_device_buffer(config.output_bytes())?,
        context.allocate_device_buffer(workspace_bytes)?,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.abort()?.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}
