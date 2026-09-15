//! Remote-only paired CUDA-event control for the N02B fused-BIAS candidate.
//!
//! This is an operator receipt. It compares one strict prepared GEMM plus the
//! standalone row-bias kernel against a prepared cuBLASLt BIAS epilogue under
//! the same stream and workspace cap. It does not select either mode from a
//! serving request and must not be interpreted as a vLLM or serving result.
#![allow(clippy::cast_precision_loss)]

use std::error::Error;

use riley_cuda::{
    BiasGemmParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaEvent, CudaGemmAlgorithmMetadata, CudaGemmConfig, CudaGemmReductionPolicy,
    CudaPinnedHostBuffer, CudaPreparedBiasEpilogueGemm, CudaPreparedGemm, CudaRuntime, CudaStream,
    GemmParams, RowBiasAddInPlaceParams, row_bias_add_in_place,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const MAX_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
const UPLOAD_STAGING_BYTES: u64 = 16 * 1024 * 1024;
const INTERNAL_WARMUPS: usize = 8;
const PAIRED_ROUNDS: usize = 24;

#[derive(Clone, Copy, Debug)]
struct ProjectionCase {
    label: &'static str,
    m: u64,
    n: u64,
    k: u64,
    seed: u64,
}

const QWEN2_5_3B_QKV_CASES: &[ProjectionCase] = &[
    ProjectionCase {
        label: "q-proj-decode-m1",
        m: 1,
        n: 2_048,
        k: 2_048,
        seed: 1,
    },
    ProjectionCase {
        label: "q-proj-batch-m8",
        m: 8,
        n: 2_048,
        k: 2_048,
        seed: 2,
    },
    ProjectionCase {
        label: "q-proj-batch-m32",
        m: 32,
        n: 2_048,
        k: 2_048,
        seed: 3,
    },
    ProjectionCase {
        label: "q-proj-prefill-m2048",
        m: 2_048,
        n: 2_048,
        k: 2_048,
        seed: 4,
    },
    ProjectionCase {
        label: "k-proj-decode-m1",
        m: 1,
        n: 256,
        k: 2_048,
        seed: 5,
    },
    ProjectionCase {
        label: "k-proj-batch-m8",
        m: 8,
        n: 256,
        k: 2_048,
        seed: 6,
    },
    ProjectionCase {
        label: "k-proj-batch-m32",
        m: 32,
        n: 256,
        k: 2_048,
        seed: 7,
    },
    ProjectionCase {
        label: "k-proj-prefill-m2048",
        m: 2_048,
        n: 256,
        k: 2_048,
        seed: 8,
    },
    ProjectionCase {
        label: "v-proj-decode-m1",
        m: 1,
        n: 256,
        k: 2_048,
        seed: 9,
    },
    ProjectionCase {
        label: "v-proj-batch-m8",
        m: 8,
        n: 256,
        k: 2_048,
        seed: 10,
    },
    ProjectionCase {
        label: "v-proj-batch-m32",
        m: 32,
        n: 256,
        k: 2_048,
        seed: 11,
    },
    ProjectionCase {
        label: "v-proj-prefill-m2048",
        m: 2_048,
        n: 256,
        k: 2_048,
        seed: 12,
    },
];

fn first_device() -> TestResult<(CudaRuntime, riley_cuda::CudaDevice)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    Ok((runtime, device))
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let is_nan = bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0;
    let rounded = if is_nan {
        0x7fff
    } else {
        let tie = (bits >> 16) & 1;
        bits.wrapping_add(0x7fff + tie) >> 16
    };
    u16::try_from(rounded).unwrap_or(0x7fff)
}

fn patterned_bf16_bytes(element_count: u64, seed: u64) -> TestResult<Vec<u8>> {
    let element_count = usize::try_from(element_count)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(
        element_count
            .checked_mul(2)
            .ok_or("BF16 host byte length overflow")?,
    )?;
    for index in 0..element_count {
        let index = u64::try_from(index)?;
        let bucket = index.wrapping_mul(17).wrapping_add(seed.wrapping_mul(29)) % 31;
        let centered = i8::try_from(bucket)? - 15;
        let value = f32::from(centered) / 64.0;
        bytes.extend_from_slice(&f32_to_bf16_bits(value).to_ne_bytes());
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

fn workspace_span<'a>(
    workspace: Option<&'a mut CudaDeviceBuffer>,
    bytes: u64,
) -> TestResult<Option<CudaBufferSpanMut<'a>>> {
    workspace
        .map(|buffer| CudaBufferSpanMut::new(buffer, CudaDType::U8, 0, bytes))
        .transpose()
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn execute_strict_batched(
    plan: &mut CudaPreparedGemm,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    bias: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    stream: &mut CudaStream,
) -> TestResult {
    let mut batch = stream.begin_command_batch()?;
    {
        let mut commands = batch.commands();
        {
            let mut params = GemmParams {
                input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
                weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
                output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
                workspace: workspace_span(workspace, metadata.workspace_bytes())?,
            };
            plan.execute(&mut params, &mut commands)?;
        }
        let mut params = RowBiasAddInPlaceParams {
            matrix: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
            bias: CudaBufferSpan::new(bias, CudaDType::BF16, 0, config.bias_bytes())?,
            row_count: config.m(),
            column_count: config.n(),
        };
        row_bias_add_in_place(&mut params, &mut commands)?;
    }
    batch.finish()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_fused_batched(
    plan: &mut CudaPreparedBiasEpilogueGemm,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    bias: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    stream: &mut CudaStream,
) -> TestResult {
    let mut batch = stream.begin_command_batch()?;
    {
        let mut commands = batch.commands();
        let mut params = BiasGemmParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
            bias: CudaBufferSpan::new(bias, CudaDType::BF16, 0, config.bias_bytes())?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
            workspace: workspace_span(workspace, metadata.workspace_bytes())?,
        };
        plan.execute(&mut params, &mut commands)?;
    }
    batch.finish()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn timed_strict(
    plan: &mut CudaPreparedGemm,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    bias: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    stream: &mut CudaStream,
    start: &mut CudaEvent,
    end: &mut CudaEvent,
) -> TestResult<f64> {
    start.record(stream)?;
    execute_strict_batched(
        plan, config, metadata, input, weight, bias, output, workspace, stream,
    )?;
    end.record(stream)?;
    end.synchronize()?;
    let elapsed = f64::from(start.elapsed_ms(end)?);
    if !elapsed.is_finite() || elapsed <= 0.0 {
        return Err(format!("invalid strict CUDA event time {elapsed}").into());
    }
    Ok(elapsed)
}

#[allow(clippy::too_many_arguments)]
fn timed_fused(
    plan: &mut CudaPreparedBiasEpilogueGemm,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    bias: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    stream: &mut CudaStream,
    start: &mut CudaEvent,
    end: &mut CudaEvent,
) -> TestResult<f64> {
    start.record(stream)?;
    execute_fused_batched(
        plan, config, metadata, input, weight, bias, output, workspace, stream,
    )?;
    end.record(stream)?;
    end.synchronize()?;
    let elapsed = f64::from(start.elapsed_ms(end)?);
    if !elapsed.is_finite() || elapsed <= 0.0 {
        return Err(format!("invalid fused CUDA event time {elapsed}").into());
    }
    Ok(elapsed)
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
        .checked_mul(numerator)
        .expect("percentile rank fits usize")
        .div_ceil(denominator);
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

fn assert_metadata(
    label: &str,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    compute_capability: (u32, u32),
) -> TestResult {
    if metadata.backend_id() != CudaGemmAlgorithmMetadata::CUBLASLT_BACKEND_ID
        || !metadata.deterministic()
        || metadata.dimensions() != (config.m(), config.n(), config.k())
        || metadata.compute_capability() != compute_capability
        || metadata.workspace_bytes() > config.max_workspace_bytes()
        || metadata.split_k() > 1
        || metadata.reduction_scheme() != 0
    {
        return Err(format!("{label} selected an ineligible strict-no-split algorithm").into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    compute_capability: (u32, u32),
    case: ProjectionCase,
) -> TestResult {
    let config = CudaGemmConfig::new(case.m, case.n, case.k, MAX_WORKSPACE_BYTES)?
        .with_reduction_policy(CudaGemmReductionPolicy::StrictNoSplitV1);
    let input_host = patterned_bf16_bytes(case.m * case.k, case.seed)?;
    let weight_host = patterned_bf16_bytes(case.n * case.k, case.seed.wrapping_add(101))?;
    let bias_host = patterned_bf16_bytes(case.n, case.seed.wrapping_add(211))?;
    let input = upload(context, stream, staging, &input_host)?;
    let weight = upload(context, stream, staging, &weight_host)?;
    let bias = upload(context, stream, staging, &bias_host)?;
    let mut strict_output = context.allocate_device_buffer(config.output_bytes())?;
    let mut fused_output = context.allocate_device_buffer(config.output_bytes())?;
    let mut strict_plan = context.prepare_gemm(config)?;
    let mut fused_plan = context.prepare_bias_epilogue_gemm(
        config,
        CudaBufferSpan::new(&bias, CudaDType::BF16, 0, config.bias_bytes())?,
    )?;
    let strict_metadata = strict_plan.algorithm_metadata();
    let fused_metadata = fused_plan.algorithm_metadata();
    assert_metadata("strict", config, strict_metadata, compute_capability)?;
    assert_metadata("fused", config, fused_metadata, compute_capability)?;
    let mut strict_workspace = if strict_metadata.workspace_bytes() == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(strict_metadata.workspace_bytes())?)
    };
    let mut fused_workspace = if fused_metadata.workspace_bytes() == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(fused_metadata.workspace_bytes())?)
    };

    for _ in 0..INTERNAL_WARMUPS {
        execute_strict_batched(
            &mut strict_plan,
            config,
            strict_metadata,
            &input,
            &weight,
            &bias,
            &mut strict_output,
            strict_workspace.as_mut(),
            stream,
        )?;
        execute_fused_batched(
            &mut fused_plan,
            config,
            fused_metadata,
            &input,
            &weight,
            &bias,
            &mut fused_output,
            fused_workspace.as_mut(),
            stream,
        )?;
    }
    let strict_reference = download(context, stream, &mut strict_output)?;
    let fused_reference = download(context, stream, &mut fused_output)?;
    let mut start = context.create_event()?;
    let mut end = context.create_event()?;
    let allocation_baseline = context.allocation_stats()?;
    let mut strict_rounds = Vec::with_capacity(PAIRED_ROUNDS);
    let mut fused_rounds = Vec::with_capacity(PAIRED_ROUNDS);
    for round in 0..PAIRED_ROUNDS {
        let strict_first = timed_strict(
            &mut strict_plan,
            config,
            strict_metadata,
            &input,
            &weight,
            &bias,
            &mut strict_output,
            strict_workspace.as_mut(),
            stream,
            &mut start,
            &mut end,
        )?;
        let fused_first = timed_fused(
            &mut fused_plan,
            config,
            fused_metadata,
            &input,
            &weight,
            &bias,
            &mut fused_output,
            fused_workspace.as_mut(),
            stream,
            &mut start,
            &mut end,
        )?;
        let fused_second = timed_fused(
            &mut fused_plan,
            config,
            fused_metadata,
            &input,
            &weight,
            &bias,
            &mut fused_output,
            fused_workspace.as_mut(),
            stream,
            &mut start,
            &mut end,
        )?;
        let strict_second = timed_strict(
            &mut strict_plan,
            config,
            strict_metadata,
            &input,
            &weight,
            &bias,
            &mut strict_output,
            strict_workspace.as_mut(),
            stream,
            &mut start,
            &mut end,
        )?;
        let strict_ms = (strict_first + strict_second) / 2.0;
        let fused_ms = (fused_first + fused_second) / 2.0;
        println!(
            "n02b-bias-epilogue-ab-sample case={} round={} strict_ms={strict_ms:.9} fused_ms={fused_ms:.9} order=ABBA",
            case.label,
            round + 1,
        );
        strict_rounds.push(strict_ms);
        fused_rounds.push(fused_ms);
    }
    if context.allocation_stats()? != allocation_baseline {
        return Err(format!("{} ABBA control changed allocation accounting", case.label).into());
    }
    let strict_repeated = download(context, stream, &mut strict_output)?;
    let fused_repeated = download(context, stream, &mut fused_output)?;
    if strict_repeated != strict_reference {
        return Err(format!("{} strict output changed across ABBA repeats", case.label).into());
    }
    if fused_repeated != fused_reference {
        return Err(format!("{} fused output changed across ABBA repeats", case.label).into());
    }
    let mut strict_for_median = strict_rounds.clone();
    let mut fused_for_median = fused_rounds.clone();
    let strict_median_ms = median(&mut strict_for_median);
    let fused_median_ms = median(&mut fused_for_median);
    let strict_p95_ms = percentile(strict_rounds, 95, 100);
    let fused_p95_ms = percentile(fused_rounds, 95, 100);
    let speedup_ratio = strict_median_ms / fused_median_ms;
    let delta_ms = strict_median_ms - fused_median_ms;
    if !speedup_ratio.is_finite() || !delta_ms.is_finite() {
        return Err(format!("{} ABBA metrics are non-finite", case.label).into());
    }
    println!(
        "n02b-bias-epilogue-ab-summary schema_version=1 case={} m={} n={} k={} timing_scope=command_batch_cuda_event internal_warmups_per_backend={} paired_rounds={} paired_order=ABBA strict_median_ms={strict_median_ms:.9} strict_p95_ms={strict_p95_ms:.9} fused_median_ms={fused_median_ms:.9} fused_p95_ms={fused_p95_ms:.9} median_speedup_ratio={speedup_ratio:.9} median_delta_ms={delta_ms:.9} strict_algorithm_id={} strict_tile_id={} strict_stages_id={} strict_workspace_bytes={} fused_algorithm_id={} fused_tile_id={} fused_stages_id={} fused_workspace_bytes={} strict_output_repeated=true fused_output_repeated=true allocation_delta=0 strict_reduction_policy={} fused_bias_contract=hf-module-qualified-v1 cuda_graph=false python_free=true full_model_serving=false vllm_comparison=false status=passed",
        case.label,
        case.m,
        case.n,
        case.k,
        INTERNAL_WARMUPS,
        PAIRED_ROUNDS,
        strict_metadata.algorithm_id(),
        strict_metadata.tile_id(),
        strict_metadata.stages_id(),
        strict_metadata.workspace_bytes(),
        fused_metadata.algorithm_id(),
        fused_metadata.tile_id(),
        fused_metadata.stages_id(),
        fused_metadata.workspace_bytes(),
        CudaGemmReductionPolicy::StrictNoSplitV1.id(),
    );

    start.close()?;
    end.close()?;
    strict_plan.close()?;
    fused_plan.close()?;
    input.close()?;
    weight.close()?;
    bias.close()?;
    strict_output.close()?;
    fused_output.close()?;
    if let Some(workspace) = strict_workspace {
        workspace.close()?;
    }
    if let Some(workspace) = fused_workspace {
        workspace.close()?;
    }
    Ok(())
}

#[test]
#[ignore = "remote-only N02B cuBLASLt BIAS versus strict Qwen operator ABBA control"]
fn qwen2_5_3b_fused_bias_epilogue_paired_cuda_event_control() -> TestResult {
    let (_runtime, device) = first_device()?;
    let compute_capability = device.properties().compute_capability();
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
    for &case in QWEN2_5_3B_QKV_CASES {
        run_case(
            &context,
            &mut stream,
            &mut staging,
            compute_capability,
            case,
        )?;
    }
    staging.close()?;
    context.synchronize()?;
    if !context.allocation_stats()?.is_zero() {
        return Err("N02B operator control left a CUDA allocation after close".into());
    }
    stream.close()?;
    context.close()?;
    println!(
        "n02b-bias-epilogue-ab-complete cases={} paired_rounds={} timing_scope=command_batch_cuda_event full_model_serving=false vllm_comparison=false performance_claim_eligible=false",
        QWEN2_5_3B_QKV_CASES.len(),
        PAIRED_ROUNDS,
    );
    Ok(())
}
