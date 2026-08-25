#![allow(clippy::too_many_lines)]

use std::collections::BTreeSet;
use std::error::Error;
use std::time::Instant;

use rustinfer_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice, CudaDeviceBuffer,
    CudaGemmAlgorithmMetadata, CudaGemmConfig, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    GemmParams,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const STANDARD_MAX_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
const QWEN_MAX_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
const UPLOAD_STAGING_BYTES: u64 = 1024 * 1024;
const WARMUP_ITERATIONS: usize = 2;
const MEASURED_ITERATIONS: usize = 11;
const LARGE_CASE_SAMPLES: u64 = 97;

#[derive(Clone, Copy, Debug)]
struct GemmCase {
    label: &'static str,
    m: u64,
    n: u64,
    k: u64,
}

const CASES: &[GemmCase] = &[
    GemmCase {
        label: "odd-3x5x7",
        m: 3,
        n: 5,
        k: 7,
    },
    GemmCase {
        label: "odd-7x11x13",
        m: 7,
        n: 11,
        k: 13,
    },
    GemmCase {
        label: "q-o-m1",
        m: 1,
        n: 576,
        k: 576,
    },
    GemmCase {
        label: "q-o-m17",
        m: 17,
        n: 576,
        k: 576,
    },
    GemmCase {
        label: "q-o-m128",
        m: 128,
        n: 576,
        k: 576,
    },
    GemmCase {
        label: "k-v-m1",
        m: 1,
        n: 192,
        k: 576,
    },
    GemmCase {
        label: "k-v-m17",
        m: 17,
        n: 192,
        k: 576,
    },
    GemmCase {
        label: "k-v-m128",
        m: 128,
        n: 192,
        k: 576,
    },
    GemmCase {
        label: "gate-up-m1",
        m: 1,
        n: 1_536,
        k: 576,
    },
    GemmCase {
        label: "gate-up-m17",
        m: 17,
        n: 1_536,
        k: 576,
    },
    GemmCase {
        label: "gate-up-m128",
        m: 128,
        n: 1_536,
        k: 576,
    },
    GemmCase {
        label: "down-m1",
        m: 1,
        n: 576,
        k: 1_536,
    },
    GemmCase {
        label: "down-m17",
        m: 17,
        n: 576,
        k: 1_536,
    },
    GemmCase {
        label: "down-m128",
        m: 128,
        n: 576,
        k: 1_536,
    },
    GemmCase {
        label: "lm-head-m1",
        m: 1,
        n: 49_152,
        k: 576,
    },
    GemmCase {
        label: "lm-head-m7",
        m: 7,
        n: 49_152,
        k: 576,
    },
];

const QWEN_CASES: &[GemmCase] = &[
    GemmCase {
        label: "qwen-down-decode-m1",
        m: 1,
        n: 896,
        k: 4_864,
    },
    GemmCase {
        label: "qwen-down-prefill-m30",
        m: 30,
        n: 896,
        k: 4_864,
    },
    GemmCase {
        label: "qwen-down-prefill-m40",
        m: 40,
        n: 896,
        k: 4_864,
    },
    GemmCase {
        label: "qwen-down-prefill-m46",
        m: 46,
        n: 896,
        k: 4_864,
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
    let byte_len = u64::try_from(bytes.len())?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;
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
    let output = staging.to_vec()?;
    staging.close()?;
    Ok(output)
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

fn decode_bf16_bits(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn pattern_value(index: u64, seed: u64) -> f32 {
    let bucket = (index.wrapping_mul(17).wrapping_add(seed.wrapping_mul(29))) % 9;
    let centered = i8::try_from(bucket).expect("modulo nine fits i8") - 4;
    f32::from(centered) / 16.0
}

fn patterned_bf16_bytes(element_count: u64, seed: u64) -> TestResult<Vec<u8>> {
    let byte_len = element_count
        .checked_mul(2)
        .ok_or("BF16 host byte length overflow")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(byte_len)?)?;
    for index in 0..element_count {
        bytes.extend_from_slice(&f32_to_bf16_bits(pattern_value(index, seed)).to_ne_bytes());
    }
    Ok(bytes)
}

fn sample_coordinates(case: GemmCase) -> Vec<(u64, u64)> {
    let output_elements = case.m.checked_mul(case.n).expect("test shape fits u64");
    if output_elements <= 4_096 {
        return (0..case.m)
            .flat_map(|row| (0..case.n).map(move |column| (row, column)))
            .collect();
    }

    let mut coordinates = BTreeSet::new();
    coordinates.insert((0, 0));
    coordinates.insert((0, case.n - 1));
    coordinates.insert((case.m - 1, 0));
    coordinates.insert((case.m - 1, case.n - 1));
    coordinates.insert((case.m / 2, case.n / 2));
    for sample in 0..LARGE_CASE_SAMPLES {
        let row = sample.wrapping_mul(37).wrapping_add(11) % case.m;
        let column = sample
            .wrapping_mul(sample.wrapping_add(3))
            .wrapping_mul(104_729)
            .wrapping_add(97)
            % case.n;
        coordinates.insert((row, column));
    }
    coordinates.into_iter().collect()
}

fn reference_output(
    case: GemmCase,
    row: u64,
    column: u64,
    input_seed: u64,
    weight_seed: u64,
) -> f32 {
    let mut accumulator = 0.0_f32;
    for reduction in 0..case.k {
        let input_index = row * case.k + reduction;
        let weight_index = column * case.k + reduction;
        accumulator +=
            pattern_value(input_index, input_seed) * pattern_value(weight_index, weight_seed);
    }
    accumulator
}

fn output_bf16(bytes: &[u8], case: GemmCase, row: u64, column: u64) -> f32 {
    let element_index = row * case.n + column;
    let byte_index = usize::try_from(element_index * 2).expect("test output index fits usize");
    decode_bf16_bits(u16::from_ne_bytes([
        bytes[byte_index],
        bytes[byte_index + 1],
    ]))
}

fn bf16_rne_tolerance(expected: f32) -> f32 {
    let magnitude_bits = f32_to_bf16_bits(expected.abs());
    let next_bits = magnitude_bits
        .checked_add(1)
        .expect("test reference stays below maximum BF16");
    let one_ulp = decode_bf16_bits(next_bits) - decode_bf16_bits(magnitude_bits);
    one_ulp.max(1.0 / 128.0)
}

fn assert_reference_samples(bytes: &[u8], case: GemmCase, input_seed: u64, weight_seed: u64) {
    assert_eq!(
        u64::try_from(bytes.len()).expect("host output length fits u64"),
        case.m * case.n * 2
    );
    for (row, column) in sample_coordinates(case) {
        let reference_f32 = reference_output(case, row, column, input_seed, weight_seed);
        let expected_rne = decode_bf16_bits(f32_to_bf16_bits(reference_f32));
        let actual = output_bf16(bytes, case, row, column);
        let tolerance = bf16_rne_tolerance(expected_rne);
        assert!(
            (actual - expected_rne).abs() <= tolerance,
            "{} output[{row},{column}]: F32 reference {reference_f32}, BF16 RNE {expected_rne}, actual {actual}, tolerance {tolerance}",
            case.label
        );
    }
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    assert!(!sorted.is_empty());
    let rank = sorted
        .len()
        .checked_mul(numerator)
        .expect("benchmark sample count fits usize")
        .div_ceil(denominator);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[allow(clippy::cast_precision_loss)]
fn effective_tflops(case: GemmCase, latency_ms: f64) -> f64 {
    let floating_point_operations = 2.0 * case.m as f64 * case.n as f64 * case.k as f64;
    floating_point_operations / (latency_ms * 1.0e9)
}

fn run_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    upload_staging: &mut CudaPinnedHostBuffer,
    expected_compute_capability: (u32, u32),
    case: GemmCase,
    case_index: u64,
    max_workspace_bytes: u64,
) -> TestResult {
    let config = CudaGemmConfig::new(case.m, case.n, case.k, max_workspace_bytes)?;
    assert_eq!(config.input_dtype(), CudaDType::BF16);
    assert_eq!(config.weight_dtype(), CudaDType::BF16);
    assert_eq!(config.accumulator_dtype(), CudaDType::F32);
    assert_eq!(config.output_dtype(), CudaDType::BF16);
    assert!(config.deterministic());

    let mut plan = context.prepare_gemm(config)?;
    let metadata = plan.algorithm_metadata();
    assert_eq!(
        metadata.backend_id(),
        CudaGemmAlgorithmMetadata::CUBLASLT_BACKEND_ID
    );
    assert!(metadata.deterministic());
    assert_eq!(metadata.dimensions(), (case.m, case.n, case.k));
    assert_eq!(metadata.compute_capability(), expected_compute_capability);
    assert!(metadata.runtime_version() > 0);
    assert!(metadata.cublaslt_version() > 0);
    assert!(metadata.workspace_bytes() <= config.max_workspace_bytes());
    assert!(metadata.split_k() <= 1);
    assert_eq!(metadata.reduction_scheme(), 0);

    let input_seed = case_index.wrapping_mul(2).wrapping_add(1);
    let weight_seed = case_index.wrapping_mul(2).wrapping_add(2);
    let input_host = patterned_bf16_bytes(case.m * case.k, input_seed)?;
    let input = upload(context, stream, upload_staging, &input_host)?;
    drop(input_host);
    let weight_host = patterned_bf16_bytes(case.n * case.k, weight_seed)?;
    let weight = upload(context, stream, upload_staging, &weight_host)?;
    drop(weight_host);
    let mut output = context.allocate_device_buffer(config.output_bytes())?;
    let mut workspace = if metadata.workspace_bytes() == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(metadata.workspace_bytes())?)
    };

    {
        let workspace_span = workspace
            .as_mut()
            .map(|buffer| {
                CudaBufferSpanMut::new(buffer, CudaDType::U8, 0, metadata.workspace_bytes())
            })
            .transpose()?;
        let mut params = GemmParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, config.output_bytes())?,
            workspace: workspace_span,
        };
        plan.execute(&mut params, stream)?;
    }
    let first_output = download(context, stream, &mut output)?;
    assert_reference_samples(&first_output, case, input_seed, weight_seed);

    let before_repeated_execution = context.allocation_stats()?;
    let mut elapsed_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    {
        let workspace_span = workspace
            .as_mut()
            .map(|buffer| {
                CudaBufferSpanMut::new(buffer, CudaDType::U8, 0, metadata.workspace_bytes())
            })
            .transpose()?;
        let mut params = GemmParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, config.output_bytes())?,
            workspace: workspace_span,
        };
        for _ in 0..WARMUP_ITERATIONS {
            plan.execute(&mut params, stream)?;
        }
        for _ in 0..MEASURED_ITERATIONS {
            let started = Instant::now();
            plan.execute(&mut params, stream)?;
            elapsed_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
    }
    assert_eq!(
        before_repeated_execution,
        context.allocation_stats()?,
        "{} repeated prepared execution changed allocation accounting",
        case.label
    );
    let repeated_output = download(context, stream, &mut output)?;
    assert_eq!(
        repeated_output, first_output,
        "{} deterministic repeated execution changed output bytes",
        case.label
    );

    elapsed_ms.sort_by(f64::total_cmp);
    let median_ms = percentile(&elapsed_ms, 50, 100);
    let p95_ms = percentile(&elapsed_ms, 95, 100);
    let median_tflops = effective_tflops(case, median_ms);
    println!(
        "rustinfer-cuda-gemm case={} m={} n={} k={} latency_scope=ffi_execute_sync latency_median_ms={median_ms:.6} latency_p95_ms={p95_ms:.6} effective_median_tflops={median_tflops:.6} temporary_bytes={} implementation_id=cublaslt:algo={}:tile={}:stages={}:split_k={}:reduction={}:swizzle={}:custom={} numerical_flags=0x{:x} cc={}.{} runtime_version={} cublaslt_version={} explicit_stream=true python_free=true",
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
        metadata.cta_swizzling(),
        metadata.custom_option(),
        metadata.numerical_implementation_flags(),
        metadata.compute_capability().0,
        metadata.compute_capability().1,
        metadata.runtime_version(),
        metadata.cublaslt_version(),
    );

    plan.close()?;
    input.close()?;
    weight.close()?;
    output.close()?;
    if let Some(workspace) = workspace {
        workspace.close()?;
    }
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn deterministic_bf16_gemm_matches_f32_reference_for_odd_smollm2_and_qwen_shapes() -> TestResult {
    let (_runtime, device) = first_device()?;
    let expected_compute_capability = device.properties().compute_capability();
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut upload_staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;

    for (case_index, &case) in CASES.iter().enumerate() {
        run_case(
            &context,
            &mut stream,
            &mut upload_staging,
            expected_compute_capability,
            case,
            u64::try_from(case_index)?,
            STANDARD_MAX_WORKSPACE_BYTES,
        )?;
    }
    for (case_index, &case) in QWEN_CASES.iter().enumerate() {
        run_case(
            &context,
            &mut stream,
            &mut upload_staging,
            expected_compute_capability,
            case,
            u64::try_from(
                CASES
                    .len()
                    .checked_add(case_index)
                    .ok_or("case index overflow")?,
            )?,
            QWEN_MAX_WORKSPACE_BYTES,
        )?;
    }

    upload_staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.synchronize()?;
    context.close()?;
    Ok(())
}
