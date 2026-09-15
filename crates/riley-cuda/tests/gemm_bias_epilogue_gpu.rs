//! Remote-only qualification for the separate cuBLASLt BF16 BIAS epilogue.
//!
//! This target exercises the N02B native owner directly. It deliberately does
//! not select the plan from a serving forward or make a throughput claim. A
//! later Qwen artifact gate decides whether this distinct rounding contract is
//! eligible for any serving integration.
#![allow(clippy::cast_precision_loss)]

use std::error::Error;

use riley_cuda::{
    BiasGemmParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice,
    CudaDeviceBuffer, CudaGemmAlgorithmMetadata, CudaGemmConfig, CudaPinnedHostBuffer, CudaRuntime,
    CudaStream,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const MAX_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
const UPLOAD_STAGING_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct ProjectionCase {
    label: &'static str,
    m: u64,
    n: u64,
    k: u64,
    seed: u64,
}

// These are the three layer-zero Qwen2.5-3B projection geometries. K and V
// intentionally run as distinct cases despite their equal dimensions so the
// receipt identifies both deployment sites.
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

fn first_device() -> TestResult<(CudaRuntime, CudaDevice)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    Ok((runtime, device))
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

fn bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| f32_to_bf16_bits(value).to_ne_bytes())
        .collect()
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
        let bucket = index.wrapping_mul(17).wrapping_add(seed.wrapping_mul(29)) % 17;
        let centered = i8::try_from(bucket)? - 8;
        bytes.extend_from_slice(&f32_to_bf16_bits(f32::from(centered) / 32.0).to_ne_bytes());
    }
    Ok(bytes)
}

fn constant_bf16_bytes(element_count: u64, value: f32) -> TestResult<Vec<u8>> {
    Ok(bf16_bytes(std::iter::repeat_n(
        value,
        usize::try_from(element_count)?,
    )))
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

fn execute(
    plan: &mut riley_cuda::CudaPreparedBiasEpilogueGemm,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    bias: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    stream: &mut CudaStream,
) -> TestResult {
    let mut params = BiasGemmParams {
        input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
        weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
        bias: CudaBufferSpan::new(bias, CudaDType::BF16, 0, config.bias_bytes())?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
        workspace: workspace_span(workspace, metadata.workspace_bytes())?,
    };
    plan.execute(&mut params, stream)?;
    Ok(())
}

fn assert_metadata(
    case: ProjectionCase,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    compute_capability: (u32, u32),
) {
    assert_eq!(
        metadata.backend_id(),
        CudaGemmAlgorithmMetadata::CUBLASLT_BACKEND_ID,
        "{} backend",
        case.label
    );
    assert!(metadata.deterministic(), "{} deterministic", case.label);
    assert_eq!(
        metadata.dimensions(),
        (config.m(), config.n(), config.k()),
        "{} dimensions",
        case.label
    );
    assert_eq!(
        metadata.compute_capability(),
        compute_capability,
        "{} compute capability",
        case.label
    );
    assert!(
        metadata.runtime_version() > 0,
        "{} CUDA runtime",
        case.label
    );
    assert!(metadata.cublaslt_version() > 0, "{} cuBLASLt", case.label);
    assert!(
        metadata.workspace_bytes() <= config.max_workspace_bytes(),
        "{} workspace",
        case.label
    );
    assert!(
        metadata.split_k() <= 1 && metadata.reduction_scheme() == 0,
        "{} selected non-strict topology ({}, {})",
        case.label,
        metadata.split_k(),
        metadata.reduction_scheme(),
    );
}

fn run_qwen_projection_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    compute_capability: (u32, u32),
    case: ProjectionCase,
) -> TestResult {
    let config = CudaGemmConfig::new(case.m, case.n, case.k, MAX_WORKSPACE_BYTES)?;
    let input_bytes = patterned_bf16_bytes(case.m * case.k, case.seed)?;
    let weight_bytes = patterned_bf16_bytes(case.n * case.k, case.seed.wrapping_add(101))?;
    let preparation_bias_bytes = constant_bf16_bytes(case.n, -1.0)?;
    let alternate_bias_bytes = constant_bf16_bytes(case.n, 1.0)?;
    let input = upload(context, stream, staging, &input_bytes)?;
    let weight = upload(context, stream, staging, &weight_bytes)?;
    let preparation_bias = upload(context, stream, staging, &preparation_bias_bytes)?;
    let alternate_bias = upload(context, stream, staging, &alternate_bias_bytes)?;
    let mut output = context.allocate_device_buffer(config.output_bytes())?;
    let mut plan = context.prepare_bias_epilogue_gemm(
        config,
        CudaBufferSpan::new(&preparation_bias, CudaDType::BF16, 0, config.bias_bytes())?,
    )?;
    let metadata = plan.algorithm_metadata();
    assert_metadata(case, config, metadata, compute_capability);
    let mut workspace = if metadata.workspace_bytes() == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(metadata.workspace_bytes())?)
    };
    let allocations_before = context.allocation_stats()?;

    execute(
        &mut plan,
        config,
        metadata,
        &input,
        &weight,
        &preparation_bias,
        &mut output,
        workspace.as_mut(),
        stream,
    )?;
    let first = download(context, stream, &mut output)?;

    execute(
        &mut plan,
        config,
        metadata,
        &input,
        &weight,
        &alternate_bias,
        &mut output,
        workspace.as_mut(),
        stream,
    )?;
    let alternate = download(context, stream, &mut output)?;
    assert_ne!(
        first, alternate,
        "{} execution reused the cold preparation bias instead of the supplied per-execution bias",
        case.label
    );

    execute(
        &mut plan,
        config,
        metadata,
        &input,
        &weight,
        &alternate_bias,
        &mut output,
        workspace.as_mut(),
        stream,
    )?;
    let repeated = download(context, stream, &mut output)?;
    assert_eq!(
        alternate, repeated,
        "{} repeated fused-bias execution changed output bytes",
        case.label
    );
    assert_eq!(
        allocations_before,
        context.allocation_stats()?,
        "{} fused-bias execution changed allocation accounting",
        case.label
    );
    println!(
        "riley-cuda-bias-epilogue case={} m={} n={} k={} workspace_bytes={} algorithm_id={} tile_id={} stages_id={} split_k={} reduction_scheme={} cc={}.{} runtime_version={} cublaslt_version={} per_execute_bias_pointer=true deterministic=true allocation_free_repetition=true performance_claim_eligible=false",
        case.label,
        case.m,
        case.n,
        case.k,
        metadata.workspace_bytes(),
        metadata.algorithm_id(),
        metadata.tile_id(),
        metadata.stages_id(),
        metadata.split_k(),
        metadata.reduction_scheme(),
        metadata.compute_capability().0,
        metadata.compute_capability().1,
        metadata.runtime_version(),
        metadata.cublaslt_version(),
    );

    plan.close()?;
    input.close()?;
    weight.close()?;
    preparation_bias.close()?;
    alternate_bias.close()?;
    output.close()?;
    if let Some(workspace) = workspace {
        workspace.close()?;
    }
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn fused_bias_epilogue_matches_a_small_row_major_bf16_contract() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
    let config = CudaGemmConfig::new(2, 4, 3, MAX_WORKSPACE_BYTES)?;
    let input = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )?;
    let weight = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes([
            1.0, 0.0, 0.0, // output column 0
            0.0, 1.0, 0.0, // output column 1
            0.0, 0.0, 1.0, // output column 2
            1.0, 1.0, 1.0, // output column 3
        ]),
    )?;
    let preparation_bias = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes([1.0, 2.0, 4.0, 8.0]),
    )?;
    let alternate_bias = upload(
        &context,
        &mut stream,
        &mut staging,
        &bf16_bytes([2.0, 4.0, 8.0, 16.0]),
    )?;
    let mut output = context.allocate_device_buffer(config.output_bytes())?;
    let mut plan = context.prepare_bias_epilogue_gemm(
        config,
        CudaBufferSpan::new(&preparation_bias, CudaDType::BF16, 0, config.bias_bytes())?,
    )?;
    let metadata = plan.algorithm_metadata();
    let mut workspace = if metadata.workspace_bytes() == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(metadata.workspace_bytes())?)
    };

    execute(
        &mut plan,
        config,
        metadata,
        &input,
        &weight,
        &preparation_bias,
        &mut output,
        workspace.as_mut(),
        &mut stream,
    )?;
    assert_eq!(
        download(&context, &mut stream, &mut output)?,
        bf16_bytes([2.0, 4.0, 7.0, 14.0, 5.0, 7.0, 10.0, 23.0]),
        "fused BIAS did not broadcast row-major B[N] across both output rows"
    );

    execute(
        &mut plan,
        config,
        metadata,
        &input,
        &weight,
        &alternate_bias,
        &mut output,
        workspace.as_mut(),
        &mut stream,
    )?;
    assert_eq!(
        download(&context, &mut stream, &mut output)?,
        bf16_bytes([3.0, 6.0, 11.0, 22.0, 6.0, 9.0, 14.0, 31.0]),
        "fused BIAS did not update to the supplied per-execution vector"
    );

    plan.close()?;
    input.close()?;
    weight.close()?;
    preparation_bias.close()?;
    alternate_bias.close()?;
    output.close()?;
    if let Some(workspace) = workspace {
        workspace.close()?;
    }
    staging.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn fused_bias_epilogue_qualifies_qwen2_5_3b_qkv_geometry() -> TestResult {
    let (_runtime, device) = first_device()?;
    let compute_capability = device.properties().compute_capability();
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
    for &case in QWEN2_5_3B_QKV_CASES {
        run_qwen_projection_case(
            &context,
            &mut stream,
            &mut staging,
            compute_capability,
            case,
        )?;
    }
    staging.close()?;
    println!(
        "riley-cuda-bias-epilogue qwen2_5_3b_qkv_geometry_complete cases={} performance_claim_eligible=false",
        QWEN2_5_3B_QKV_CASES.len()
    );
    Ok(())
}
