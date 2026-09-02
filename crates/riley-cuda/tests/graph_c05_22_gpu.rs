#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaErrorKind,
    CudaGemmConfig, CudaGraphCaptureMode, CudaPinnedHostBuffer, CudaPreparedGemm, CudaRuntime,
    CudaStream, GemmParams, RmsNormParams, rms_norm,
};

#[cfg(feature = "cuda-test-fault-injection")]
use riley_cuda::{CudaErrorDomain, CudaErrorStage, CudaMemoryFault};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

// This is the canonical decode projection geometry. It keeps the two-node
// graph on the same reviewed RMSNorm and cuBLASLt path as C05-21.
const M: u64 = 1;
const N: u64 = 576;
const K: u64 = 576;
const BF16_BYTES: u64 = 2;
const EPSILON: f32 = 1.0e-5;
const MAX_WORKSPACE_BYTES: u64 = 8 * 1024 * 1024;
const REPLAYS: usize = 64;

fn finite_bf16_pattern(element_count: u64, seed: u64) -> TestResult<Vec<u8>> {
    let byte_len = element_count
        .checked_mul(BF16_BYTES)
        .ok_or("C05-22 BF16 fixture byte count overflow")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(byte_len)?)?;
    for index in 0..element_count {
        // Keep every source finite and non-uniform. This catches an accidental
        // host round trip or a graph that does not read its retained sources.
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

fn execute_eager_rms_norm(
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let input_bytes = input.byte_len();
    let weight_bytes = weight.byte_len();
    let output_bytes = output.byte_len();
    let mut params = RmsNormParams {
        input: CudaBufferSpan::new(input, CudaDType::BF16, 0, input_bytes)?,
        weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, weight_bytes)?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_bytes)?,
        row_count: M,
        hidden_size: K,
        epsilon: EPSILON,
    };
    rms_norm(&mut params, stream)?;
    Ok(())
}

fn execute_eager_gemm(
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
fn owned_canonical_rms_norm_gemm_bf16_graph_cold_capture_replays_byte_exact_against_eager()
-> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let rms_norm_weight_bytes = K
        .checked_mul(BF16_BYTES)
        .ok_or("C05-22 RMSNorm weight byte count overflow")?;
    let rms_norm_input_host = finite_bf16_pattern(M * K, 7)?;
    let rms_norm_weight_host = finite_bf16_pattern(K, 19)?;
    let gemm_weight_host = finite_bf16_pattern(N * K, 37)?;
    let rms_norm_output_sentinel = vec![0xa5_u8; usize::try_from(config.input_bytes())?];
    let gemm_output_sentinel = vec![0x5a_u8; usize::try_from(config.output_bytes())?];
    assert_eq!(
        u64::try_from(rms_norm_input_host.len())?,
        config.input_bytes()
    );
    assert_eq!(
        u64::try_from(rms_norm_weight_host.len())?,
        rms_norm_weight_bytes
    );
    assert_eq!(
        u64::try_from(gemm_weight_host.len())?,
        config.weight_bytes()
    );

    let mut eager_stream = context.create_stream()?;
    let capture_stream = context.create_stream()?;
    let mut transfer_stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;

    let eager_rms_norm_input = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &rms_norm_input_host,
    )?;
    let eager_rms_norm_weight = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &rms_norm_weight_host,
    )?;
    let mut eager_rms_norm_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &rms_norm_output_sentinel,
    )?;
    let eager_gemm_weight = upload(&context, &mut eager_stream, &mut staging, &gemm_weight_host)?;
    let mut eager_gemm_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &gemm_output_sentinel,
    )?;
    let mut eager_plan = context.prepare_gemm(config)?;
    let mut eager_workspace =
        context.allocate_device_buffer(eager_plan.algorithm_metadata().workspace_bytes())?;
    execute_eager_rms_norm(
        &eager_rms_norm_input,
        &eager_rms_norm_weight,
        &mut eager_rms_norm_output,
        &mut eager_stream,
    )?;
    execute_eager_gemm(
        &mut eager_plan,
        &eager_rms_norm_output,
        &eager_gemm_weight,
        &mut eager_gemm_output,
        &mut eager_workspace,
        &mut eager_stream,
    )?;
    let eager_output_bytes = download(&mut eager_gemm_output, &mut staging, &mut transfer_stream)?;
    assert_ne!(
        eager_output_bytes, gemm_output_sentinel,
        "the eager RMSNorm -> GEMM baseline must execute rather than preserve the output sentinel"
    );

    // This plan is deliberately cold: its first RMSNorm and cuBLASLt work is
    // the exact two-node graph capture below, not an eager warm-up.
    let graph_plan = context.prepare_gemm(config)?;
    let graph_workspace =
        context.allocate_device_buffer(graph_plan.algorithm_metadata().workspace_bytes())?;
    let graph_rms_norm_input = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &rms_norm_input_host,
    )?;
    let graph_rms_norm_weight = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &rms_norm_weight_host,
    )?;
    let graph_rms_norm_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &rms_norm_output_sentinel,
    )?;
    let graph_gemm_weight = upload(&context, &mut eager_stream, &mut staging, &gemm_weight_host)?;
    let graph_gemm_output = upload(
        &context,
        &mut eager_stream,
        &mut staging,
        &gemm_output_sentinel,
    )?;

    let mut capture = capture_stream.begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
        graph_plan,
        graph_rms_norm_input,
        graph_rms_norm_weight,
        graph_rms_norm_output,
        graph_gemm_weight,
        graph_gemm_output,
        graph_workspace,
        M,
        K,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_canonical_rms_norm_gemm_bf16()?;
    assert_invalid_state(
        capture.enqueue_canonical_rms_norm_gemm_bf16(),
        "C05-22 canonical RMSNorm -> GEMM graph second enqueue",
    );
    let mut exec = capture.end()?.instantiate()?;
    let allocations_before_replay = context.allocation_stats()?;
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }
    assert_eq!(
        context.allocation_stats()?,
        allocations_before_replay,
        "C05-22 graph replay must not change caller allocation accounting"
    );

    let resources = exec.close()?;
    let (
        graph_stream,
        graph_plan,
        mut graph_rms_norm_input,
        mut graph_rms_norm_weight,
        mut graph_rms_norm_output,
        mut graph_gemm_weight,
        mut graph_gemm_output,
        graph_workspace,
    ) = resources.into_parts();
    let eager_intermediate_bytes = download(
        &mut eager_rms_norm_output,
        &mut staging,
        &mut transfer_stream,
    )?;
    let graph_intermediate_bytes = download(
        &mut graph_rms_norm_output,
        &mut staging,
        &mut transfer_stream,
    )?;
    assert_eq!(
        graph_intermediate_bytes, eager_intermediate_bytes,
        "C05-22 cold graph RMSNorm intermediate must match eager BF16 bytes exactly"
    );
    assert_ne!(
        graph_intermediate_bytes, rms_norm_output_sentinel,
        "C05-22 graph must execute RMSNorm rather than preserve its intermediate sentinel"
    );
    let graph_output_bytes = download(&mut graph_gemm_output, &mut staging, &mut transfer_stream)?;
    assert_eq!(
        graph_output_bytes, eager_output_bytes,
        "C05-22 cold RMSNorm -> cuBLASLt graph replay must match eager BF16 output bytes exactly"
    );
    assert_ne!(
        graph_output_bytes, gemm_output_sentinel,
        "C05-22 graph must execute rather than preserve the output sentinel"
    );
    for (name, buffer, expected) in [
        (
            "RMSNorm input",
            &mut graph_rms_norm_input,
            rms_norm_input_host.as_slice(),
        ),
        (
            "RMSNorm weight",
            &mut graph_rms_norm_weight,
            rms_norm_weight_host.as_slice(),
        ),
        (
            "GEMM weight",
            &mut graph_gemm_weight,
            gemm_weight_host.as_slice(),
        ),
    ] {
        assert_eq!(
            download(buffer, &mut staging, &mut transfer_stream)?,
            expected,
            "C05-22 graph must not mutate its fixed {name} bytes"
        );
    }

    graph_plan.close()?;
    graph_workspace.close()?;
    graph_gemm_output.close()?;
    graph_gemm_weight.close()?;
    graph_rms_norm_output.close()?;
    graph_rms_norm_weight.close()?;
    graph_rms_norm_input.close()?;
    graph_stream.close()?;
    eager_plan.close()?;
    eager_workspace.close()?;
    eager_gemm_output.close()?;
    eager_gemm_weight.close()?;
    eager_rms_norm_output.close()?;
    eager_rms_norm_weight.close()?;
    eager_rms_norm_input.close()?;
    staging.close()?;
    transfer_stream.close()?;
    eager_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}

#[derive(Clone, Copy)]
enum WrongExactAllocation {
    RmsNormInput,
    RmsNormWeight,
    RmsNormOutput,
    GemmWeight,
    GemmOutput,
    GemmWorkspace,
}

fn wrong_exact_byte_len(required: u64, too_long: bool) -> u64 {
    if too_long {
        required
            .checked_add(1)
            .expect("reviewed C05-22 test allocation lengths cannot overflow")
    } else {
        // A zero-byte workspace has no smaller length. One byte remains a
        // deterministic wrong-exact-size preflight case.
        if required == 0 { 1 } else { required - 1 }
    }
}

fn rejected_wrong_exact_resource_recovers(
    context: &CudaContext,
    wrong: WrongExactAllocation,
    too_long: bool,
) -> TestResult {
    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let rms_norm_weight_bytes = K
        .checked_mul(BF16_BYTES)
        .ok_or("C05-22 RMSNorm weight byte count overflow")?;
    let stream = context.create_stream()?;
    let rms_norm_input =
        context.allocate_device_buffer(if matches!(wrong, WrongExactAllocation::RmsNormInput) {
            wrong_exact_byte_len(config.input_bytes(), too_long)
        } else {
            config.input_bytes()
        })?;
    let rms_norm_weight = context.allocate_device_buffer(
        if matches!(wrong, WrongExactAllocation::RmsNormWeight) {
            wrong_exact_byte_len(rms_norm_weight_bytes, too_long)
        } else {
            rms_norm_weight_bytes
        },
    )?;
    let rms_norm_output = context.allocate_device_buffer(
        if matches!(wrong, WrongExactAllocation::RmsNormOutput) {
            wrong_exact_byte_len(config.input_bytes(), too_long)
        } else {
            config.input_bytes()
        },
    )?;
    let gemm_weight =
        context.allocate_device_buffer(if matches!(wrong, WrongExactAllocation::GemmWeight) {
            wrong_exact_byte_len(config.weight_bytes(), too_long)
        } else {
            config.weight_bytes()
        })?;
    let gemm_output =
        context.allocate_device_buffer(if matches!(wrong, WrongExactAllocation::GemmOutput) {
            wrong_exact_byte_len(config.output_bytes(), too_long)
        } else {
            config.output_bytes()
        })?;
    let gemm_workspace = context.allocate_device_buffer(
        if matches!(wrong, WrongExactAllocation::GemmWorkspace) {
            wrong_exact_byte_len(workspace_bytes, too_long)
        } else {
            workspace_bytes
        },
    )?;

    let error = match stream.begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
        plan,
        rms_norm_input,
        rms_norm_weight,
        rms_norm_output,
        gemm_weight,
        gemm_output,
        gemm_workspace,
        M,
        K,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("wrong C05-22 fixed allocation must fail before native capture"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::OutOfRange);
    error
        .into_resources()
        .expect("pure Rust C05-22 preflight must recover every untouched resource")
        .close()?;
    Ok(())
}

fn rejected_geometry_recovers(
    context: &CudaContext,
    row_count: u64,
    hidden_size: u64,
) -> TestResult {
    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let rms_norm_weight_bytes = K
        .checked_mul(BF16_BYTES)
        .ok_or("C05-22 RMSNorm weight byte count overflow")?;
    let stream = context.create_stream()?;
    let error = match stream.begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
        plan,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(rms_norm_weight_bytes)?,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(config.weight_bytes())?,
        context.allocate_device_buffer(config.output_bytes())?,
        context.allocate_device_buffer(workspace_bytes)?,
        row_count,
        hidden_size,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("wrong C05-22 geometry must fail before native capture"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::InvalidArgument);
    error
        .into_resources()
        .expect("C05-22 geometry preflight must return every untouched resource")
        .close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn owned_canonical_rms_norm_gemm_bf16_graph_recovers_exact_preflight_geometry_and_abort()
-> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());

    for too_long in [false, true] {
        for wrong in [
            WrongExactAllocation::RmsNormInput,
            WrongExactAllocation::RmsNormWeight,
            WrongExactAllocation::RmsNormOutput,
            WrongExactAllocation::GemmWeight,
            WrongExactAllocation::GemmOutput,
            WrongExactAllocation::GemmWorkspace,
        ] {
            rejected_wrong_exact_resource_recovers(&context, wrong, too_long)?;
            assert_eq!(
                context.allocation_stats()?,
                allocation_baseline,
                "rejected C05-22 exact-size preflight must return to the allocation baseline"
            );
        }
    }
    for (row_count, hidden_size) in [(M + 1, K), (M, K + 1), (2, K / 2)] {
        rejected_geometry_recovers(&context, row_count, hidden_size)?;
        assert_eq!(
            context.allocation_stats()?,
            allocation_baseline,
            "rejected C05-22 geometry preflight must return to the allocation baseline"
        );
    }

    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let rms_norm_weight_bytes = K
        .checked_mul(BF16_BYTES)
        .ok_or("C05-22 RMSNorm weight byte count overflow")?;
    let stream = context.create_stream()?;
    stream
        .begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
            plan,
            context.allocate_device_buffer(config.input_bytes())?,
            context.allocate_device_buffer(rms_norm_weight_bytes)?,
            context.allocate_device_buffer(config.input_bytes())?,
            context.allocate_device_buffer(config.weight_bytes())?,
            context.allocate_device_buffer(config.output_bytes())?,
            context.allocate_device_buffer(workspace_bytes)?,
            M,
            K,
            EPSILON,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?
        .close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);

    // Graph close after a successful capture is a distinct native release
    // branch from executable close. No launch occurs here; the test proves
    // that its captured graph owner returns all six retained resources.
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let stream = context.create_stream()?;
    let mut capture = stream.begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
        plan,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(rms_norm_weight_bytes)?,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(config.weight_bytes())?,
        context.allocate_device_buffer(config.output_bytes())?,
        context.allocate_device_buffer(workspace_bytes)?,
        M,
        K,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    capture.enqueue_canonical_rms_norm_gemm_bf16()?;
    capture.end()?.close()?.close()?;
    assert_eq!(
        context.allocation_stats()?,
        allocation_baseline,
        "C05-22 captured-graph close must return every retained resource"
    );
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn owned_canonical_rms_norm_gemm_bf16_graph_rejects_foreign_context_and_recovers() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let foreign_context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    let foreign_allocation_baseline = foreign_context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    assert!(foreign_allocation_baseline.is_zero());

    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let rms_norm_weight_bytes = K
        .checked_mul(BF16_BYTES)
        .ok_or("C05-22 RMSNorm weight byte count overflow")?;
    let stream = context.create_stream()?;
    let error = match stream.begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
        plan,
        foreign_context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(rms_norm_weight_bytes)?,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(config.weight_bytes())?,
        context.allocate_device_buffer(config.output_bytes())?,
        context.allocate_device_buffer(workspace_bytes)?,
        M,
        K,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    ) {
        Ok(_) => panic!("foreign C05-22 RMSNorm input must fail before native capture"),
        Err(error) => error,
    };
    assert_eq!(error.error().kind(), CudaErrorKind::InvalidState);
    error
        .into_resources()
        .expect("foreign-context C05-22 preflight must return every untouched resource")
        .close()?;

    assert_eq!(context.allocation_stats()?, allocation_baseline);
    assert_eq!(
        foreign_context.allocation_stats()?,
        foreign_allocation_baseline,
        "foreign preflight recovery must release its moved foreign allocation"
    );
    close_context(foreign_context)?;
    close_context(context)
}

#[cfg(feature = "cuda-test-fault-injection")]
#[test]
#[ignore = "remote GPU"]
fn owned_canonical_rms_norm_gemm_bf16_graph_test_fault_is_abort_only_and_recovers_six_resources()
-> TestResult {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    context.reset_memory_fault_injection()?;

    let config = CudaGemmConfig::new(M, N, K, MAX_WORKSPACE_BYTES)?;
    let plan = context.prepare_gemm(config)?;
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let rms_norm_weight_bytes = K
        .checked_mul(BF16_BYTES)
        .ok_or("C05-22 RMSNorm weight byte count overflow")?;
    let stream = context.create_stream()?;
    let mut capture = stream.begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
        plan,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(rms_norm_weight_bytes)?,
        context.allocate_device_buffer(config.input_bytes())?,
        context.allocate_device_buffer(config.weight_bytes())?,
        context.allocate_device_buffer(config.output_bytes())?,
        context.allocate_device_buffer(workspace_bytes)?,
        M,
        K,
        EPSILON,
        CudaGraphCaptureMode::ThreadLocal,
    )?;
    context.arm_memory_fault(CudaMemoryFault::C05_22GemmSubmissionNotSupported)?;
    let error = capture
        .enqueue_canonical_rms_norm_gemm_bf16()
        .expect_err("the test seam must reject the second C05-22 node");
    assert_eq!(error.kind(), CudaErrorKind::NotSupported);
    assert_eq!(error.domain(), CudaErrorDomain::CuBlasLt);
    assert_eq!(error.stage(), CudaErrorStage::Launch);
    assert_ne!(error.native_code(), 0);
    assert_invalid_state(
        capture.enqueue_canonical_rms_norm_gemm_bf16(),
        "C05-22 capture after test-injected second-node failure",
    );
    let injected = context.memory_fault_stats()?;
    assert_eq!(injected.faults_fired(), 1);
    assert_eq!(injected.armed_fault(), None);

    let (
        stream,
        plan,
        rms_norm_input,
        rms_norm_weight,
        rms_norm_output,
        gemm_weight,
        gemm_output,
        gemm_workspace,
    ) = capture.abort()?.into_parts();
    plan.close()?;
    gemm_workspace.close()?;
    gemm_output.close()?;
    gemm_weight.close()?;
    rms_norm_output.close()?;
    rms_norm_weight.close()?;
    rms_norm_input.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}
