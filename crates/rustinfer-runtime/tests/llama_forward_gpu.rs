//! Remote-only end-to-end PR07 CUDA and pinned-golden validation.

#![cfg(feature = "cuda")]
#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

#[cfg(not(target_endian = "little"))]
compile_error!("the pinned safetensors golden gate requires a little-endian target");

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rustinfer_cuda::{
    AttentionSelectionTrace, CudaAllocationStats, CudaContext, CudaErrorStage, CudaRuntime,
    CudaStream,
};
use rustinfer_model::{LoadLimits, LoadedModel};
use rustinfer_runtime::llama::{
    LlamaForwardError, LlamaPlanError, LlamaTracePoint, PreparedLlamaAllocationReport,
    PreparedLlamaForward, PreparedLlamaForwardConfig, PreparedLlamaTrace,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SEQUENCE_LENGTH: usize = 7;
const SEQUENCE_LENGTH_U64: u64 = 7;
const COMMON_PREFIX_LENGTH: usize = 4;
const HIDDEN_SIZE: usize = 576;
const INTERMEDIATE_SIZE: usize = 1_536;
const QUERY_HEADS: u64 = 9;
const KEY_VALUE_WIDTH: u64 = 192;
const HEAD_SIZE: u64 = 64;
const VOCABULARY_SIZE: usize = 49_152;
const PINNED_TOKENS_A: [u32; SEQUENCE_LENGTH] = [504, 2_365, 6_354, 16_438, 11_139, 253, 1_890];
const PINNED_TOKENS_B: [u32; SEQUENCE_LENGTH] = [504, 2_365, 6_354, 16_438, 42, 43, 44];
const MODEL_BENCHMARK_SEQUENCE_LENGTH: usize = 128;
const MODEL_BENCHMARK_WARMUP_ITERATIONS: usize = 3;
const MODEL_BENCHMARK_MEASURED_ITERATIONS: usize = 10;

// Immutable PR01 E0 v2 full-corpus final-logits thresholds. PR08 reuses only
// these predeclared three metric bounds for its pinned traces instead of
// adjusting a threshold after observing one differential run; these checks do
// not replace or reactivate the full 31-case PR01 gate.
const PR01_E0_V2_FINAL_LOGITS_TOLERANCE: NumericTolerance = NumericTolerance {
    cosine_min: 0.997_903_530_549_539_3,
    max_abs_max: 5.852_936_458_587_647,
    mean_abs_max: 1.151_280_319_263_363,
};

#[derive(Clone, Copy, Debug)]
struct NumericTolerance {
    cosine_min: f64,
    max_abs_max: f64,
    mean_abs_max: f64,
}

#[derive(Clone, Copy, Debug)]
struct NumericMetrics {
    cosine: f64,
    max_abs: f64,
    mean_abs: f64,
}

struct LlamaPrefillBenchmarkOutcome {
    selection: AttentionSelectionTrace,
    report: PreparedLlamaAllocationReport,
    last_logits: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct LlamaPrefillBenchmarkSpec<'a> {
    backend_label: &'static str,
    expected_implementation_id: &'static str,
    sequence_length: usize,
    token_ids: &'a [u32],
    config: PreparedLlamaForwardConfig,
}

fn expected_shape(point: LlamaTracePoint) -> &'static [u64] {
    const HIDDEN: &[u64] = &[SEQUENCE_LENGTH_U64, 576];
    const KEY_VALUE: &[u64] = &[SEQUENCE_LENGTH_U64, KEY_VALUE_WIDTH];
    const INTERMEDIATE: &[u64] = &[SEQUENCE_LENGTH_U64, 1_536];
    const PROBABILITIES: &[u64] = &[QUERY_HEADS, SEQUENCE_LENGTH_U64, SEQUENCE_LENGTH_U64];
    const CONTEXT: &[u64] = &[SEQUENCE_LENGTH_U64, QUERY_HEADS, HEAD_SIZE];
    const LAST_LOGITS: &[u64] = &[49_152];
    match point {
        LlamaTracePoint::Layer0KeyProjection | LlamaTracePoint::Layer0ValueProjection => KEY_VALUE,
        LlamaTracePoint::Layer0AttentionProbabilities => PROBABILITIES,
        LlamaTracePoint::Layer0AttentionContext => CONTEXT,
        LlamaTracePoint::Layer0GateProjection
        | LlamaTracePoint::Layer0UpProjection
        | LlamaTracePoint::Layer0Gated => INTERMEDIATE,
        LlamaTracePoint::LastLogits => LAST_LOGITS,
        LlamaTracePoint::Embedding
        | LlamaTracePoint::Layer0InputNorm
        | LlamaTracePoint::Layer0QueryProjection
        | LlamaTracePoint::Layer0AfterAttentionResidual
        | LlamaTracePoint::Layer0PostAttentionNorm
        | LlamaTracePoint::Layer0DownProjection
        | LlamaTracePoint::Layer0Output
        | LlamaTracePoint::Layer14Output
        | LlamaTracePoint::FinalNormInput
        | LlamaTracePoint::FinalNormOutput => HIDDEN,
    }
}

fn tolerance(point: LlamaTracePoint) -> Option<NumericTolerance> {
    let strict_early = NumericTolerance {
        cosine_min: 0.999,
        max_abs_max: 0.5,
        mean_abs_max: 0.02,
    };
    let cumulative_hidden = NumericTolerance {
        cosine_min: 0.998,
        max_abs_max: 3.0,
        mean_abs_max: 0.35,
    };
    match point {
        // Embeddings are checked byte-for-byte by the caller. The unnormalized
        // final residual stream is diagnostic-only; FinalNormOutput below is
        // the gated final hidden state and actual LM-head input.
        LlamaTracePoint::Embedding | LlamaTracePoint::FinalNormInput => None,
        LlamaTracePoint::Layer0AttentionProbabilities => Some(NumericTolerance {
            cosine_min: 0.999,
            max_abs_max: 0.02,
            mean_abs_max: 0.002,
        }),
        // Predeclared first-layer threshold from the immutable PR01 E0 v2 matrix.
        LlamaTracePoint::Layer0Output => Some(NumericTolerance {
            cosine_min: 0.999_983_706_829_855,
            max_abs_max: 0.388_427_257_537_841_8,
            mean_abs_max: 0.008_509_292_567_237_658,
        }),
        LlamaTracePoint::Layer14Output | LlamaTracePoint::FinalNormOutput => {
            Some(cumulative_hidden)
        }
        // Predeclared final-logits threshold from the immutable PR01 E0 v2 matrix.
        LlamaTracePoint::LastLogits => Some(PR01_E0_V2_FINAL_LOGITS_TOLERANCE),
        _ => Some(strict_early),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().fold(
        String::with_capacity(digest.len() * 2),
        |mut output, byte| {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
            output
        },
    )
}

fn json_array_u64(value: &Value) -> TestResult<Vec<u64>> {
    let array = value.as_array().ok_or("expected a JSON array")?;
    let mut output = Vec::with_capacity(array.len());
    for item in array {
        output.push(item.as_u64().ok_or("expected an unsigned integer")?);
    }
    Ok(output)
}

fn parse_golden_trace(manifest_path: &Path) -> TestResult<Vec<Vec<u8>>> {
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    assert_eq!(manifest["schema_version"], "1.0.0");
    assert_eq!(manifest["artifact_kind"], "pr07-hf-bf16-forward-trace");
    assert_eq!(
        manifest["trace_id"],
        "smollm2-pr07-bf16-reference-forward-v1"
    );
    assert_eq!(
        manifest["contract"]["input_token_ids"],
        serde_json::json!(PINNED_TOKENS_A)
    );
    assert_eq!(
        manifest["contract"]["execution"]["sequence_length"].as_u64(),
        Some(SEQUENCE_LENGTH_U64)
    );
    assert_eq!(
        manifest["contract"]["execution"]["use_cache"].as_bool(),
        Some(false)
    );
    assert_eq!(manifest["sidecar"]["format"], "safetensors");
    assert_eq!(
        manifest["sidecar"]["tensor_count"].as_u64(),
        Some(u64::try_from(LlamaTracePoint::ALL.len())?)
    );

    let sidecar_name = manifest["sidecar"]["path"]
        .as_str()
        .ok_or("golden manifest sidecar.path is not a string")?;
    let sidecar_component = Path::new(sidecar_name);
    assert_eq!(
        sidecar_component.file_name(),
        Some(sidecar_component.as_os_str())
    );
    assert_eq!(sidecar_component.components().count(), 1);
    let sidecar_path = manifest_path
        .parent()
        .ok_or("golden manifest has no parent")?
        .join(sidecar_component);
    let sidecar = fs::read(sidecar_path)?;
    let digest = sha256_hex(&sidecar);
    assert_eq!(
        manifest["sidecar"]["sha256"].as_str(),
        Some(digest.as_str()),
        "golden sidecar SHA-256 differs from its manifest"
    );

    let header_length_bytes: [u8; 8] = sidecar
        .get(..8)
        .ok_or("golden sidecar lacks its header prefix")?
        .try_into()?;
    let header_length = usize::try_from(u64::from_le_bytes(header_length_bytes))?;
    let data_start = 8_usize
        .checked_add(header_length)
        .ok_or("golden sidecar header offset overflow")?;
    let header: Value = serde_json::from_slice(
        sidecar
            .get(8..data_start)
            .ok_or("golden sidecar header is truncated")?,
    )?;
    let header_object = header
        .as_object()
        .ok_or("golden sidecar header is not an object")?;
    assert_eq!(header_object.len(), LlamaTracePoint::ALL.len());

    let mut tensors = Vec::with_capacity(LlamaTracePoint::ALL.len());
    let mut relative_ranges = Vec::with_capacity(LlamaTracePoint::ALL.len());
    for point in LlamaTracePoint::ALL {
        let reference = &manifest["tensors"][point.name()];
        assert_eq!(reference["dtype"], "bfloat16", "{} dtype", point.name());
        let shape = json_array_u64(&reference["shape"])?;
        assert_eq!(shape, expected_shape(point), "{} shape", point.name());
        let expected_key = format!("trace/{}", point.name().replace('.', "/"));
        assert_eq!(reference["key"], expected_key, "{} key", point.name());

        let metadata = header_object
            .get(&expected_key)
            .ok_or("golden sidecar tensor key is missing")?;
        assert_eq!(metadata["dtype"], "BF16", "{} sidecar dtype", point.name());
        assert_eq!(
            json_array_u64(&metadata["shape"])?,
            expected_shape(point),
            "{} sidecar shape",
            point.name()
        );
        let offsets = json_array_u64(&metadata["data_offsets"])?;
        assert_eq!(offsets.len(), 2, "{} data offsets", point.name());
        assert!(
            offsets[0] <= offsets[1],
            "{} descending range",
            point.name()
        );
        relative_ranges.push((offsets[0], offsets[1]));
        let start = data_start
            .checked_add(usize::try_from(offsets[0])?)
            .ok_or("golden tensor start overflow")?;
        let end = data_start
            .checked_add(usize::try_from(offsets[1])?)
            .ok_or("golden tensor end overflow")?;
        let bytes = sidecar
            .get(start..end)
            .ok_or("golden tensor range exceeds sidecar")?
            .to_vec();
        let element_count = expected_shape(point)
            .iter()
            .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
            .ok_or("golden element-count overflow")?;
        assert_eq!(
            u64::try_from(bytes.len())?,
            element_count
                .checked_mul(2)
                .ok_or("golden byte length overflow")?,
            "{} byte length",
            point.name()
        );
        tensors.push(bytes);
    }
    relative_ranges.sort_unstable();
    let mut covered = 0_u64;
    for (start, end) in relative_ranges {
        assert_eq!(start, covered, "golden sidecar tensor ranges have a gap");
        covered = end;
    }
    assert_eq!(
        usize::try_from(covered)?,
        sidecar.len() - data_start,
        "golden sidecar has unbound data bytes"
    );
    Ok(tensors)
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .chunks_exact(2)
        .map(|scalar| {
            let bits = u16::from_le_bytes([scalar[0], scalar[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect()
}

fn numeric_metrics(actual: &[u8], expected: &[u8]) -> NumericMetrics {
    assert_eq!(actual.len(), expected.len());
    let actual = decode_bf16(actual);
    let expected = decode_bf16(expected);
    let mut dot = 0.0_f64;
    let mut actual_norm = 0.0_f64;
    let mut expected_norm = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(&expected) {
        assert!(actual.is_finite() && expected.is_finite());
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        dot = actual.mul_add(expected, dot);
        actual_norm = actual.mul_add(actual, actual_norm);
        expected_norm = expected.mul_add(expected, expected_norm);
    }
    let cosine = if actual_norm == 0.0 || expected_norm == 0.0 {
        if actual == expected { 1.0 } else { 0.0 }
    } else {
        dot / (actual_norm.sqrt() * expected_norm.sqrt())
    };
    NumericMetrics {
        cosine,
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
    }
}

fn top_k(bytes: &[u8], count: usize) -> Vec<usize> {
    let values = decode_bf16(bytes);
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(count);
    indices
}

fn assert_trace_matches_golden(trace: &PreparedLlamaTrace, golden: &[Vec<u8>]) {
    assert_eq!(trace.tensor_count(), LlamaTracePoint::ALL.len());
    assert_eq!(
        trace.captured_count(),
        u32::try_from(LlamaTracePoint::ALL.len()).expect("trace point count fits u32")
    );
    assert_eq!(golden.len(), LlamaTracePoint::ALL.len());
    let mut first_numeric_divergence = None;
    for (index, point) in LlamaTracePoint::ALL.into_iter().enumerate() {
        let actual = trace
            .tensor(point)
            .unwrap_or_else(|| panic!("{} was not captured", point.name()));
        let expected = &golden[index];
        assert_eq!(
            trace.tensor_byte_len(point),
            expected.len(),
            "{} prepared byte length",
            point.name()
        );
        if point == LlamaTracePoint::Embedding {
            assert_eq!(actual, expected, "{} must be byte-exact", point.name());
        } else {
            let metrics = numeric_metrics(actual, expected);
            println!(
                "pr07-trace point={} cosine={:.12} max_abs={:.9} mean_abs={:.9}",
                point.name(),
                metrics.cosine,
                metrics.max_abs,
                metrics.mean_abs
            );
            if let Some(tolerance) = tolerance(point) {
                let passes = metrics.cosine >= tolerance.cosine_min
                    && metrics.max_abs <= tolerance.max_abs_max
                    && metrics.mean_abs <= tolerance.mean_abs_max;
                if !passes && first_numeric_divergence.is_none() {
                    first_numeric_divergence = Some((point, metrics, tolerance));
                }
            } else {
                assert_eq!(point, LlamaTracePoint::FinalNormInput);
            }
        }
    }

    let actual_logits = trace
        .tensor(LlamaTracePoint::LastLogits)
        .expect("last logits were captured");
    let expected_logits = golden.last().expect("golden trace contains last logits");
    if let Some((point, metrics, tolerance)) = first_numeric_divergence {
        panic!(
            "first divergent PR07 checkpoint {}: metrics={metrics:?}, tolerance={tolerance:?}",
            point.name()
        );
    }
    let actual_top = top_k(actual_logits, 10);
    let expected_top = top_k(expected_logits, 10);
    assert_eq!(actual_top[0], expected_top[0], "golden greedy next token");
    assert_eq!(
        actual_top.into_iter().collect::<BTreeSet<_>>(),
        expected_top.into_iter().collect::<BTreeSet<_>>(),
        "golden top-10 token set"
    );
}

fn assert_report_matches_context(
    report: PreparedLlamaAllocationReport,
    stats: CudaAllocationStats,
) {
    assert_eq!(report.total_device_bytes(), stats.device_live_bytes());
    assert_eq!(
        report.device_allocation_count(),
        stats.device_live_allocations()
    );
    assert_eq!(report.pinned_host_bytes(), stats.pinned_host_live_bytes());
    assert_eq!(
        report.pinned_host_allocation_count(),
        stats.pinned_host_live_allocations()
    );
    assert_eq!(
        report.total_device_bytes(),
        report.weight_bytes() + report.graph_bytes() + report.gemm_workspace_bytes()
    );
}

fn assert_bf16_logits_are_finite(bytes: &[u8]) {
    assert_eq!(bytes.len() % 2, 0, "BF16 output must contain whole scalars");
    for (index, scalar) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_ne_bytes([scalar[0], scalar[1]]);
        assert_ne!(
            bits & 0x7f80,
            0x7f80,
            "logit {index} is NaN or infinity (BF16 bits 0x{bits:04x})"
        );
    }
}

fn benchmark_tokens(sequence_length: usize) -> Vec<u32> {
    (0..sequence_length)
        .map(|position| PINNED_TOKENS_A[position % PINNED_TOKENS_A.len()])
        .collect()
}

fn percentile_nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    assert!(!sorted.is_empty());
    assert!((1..=100).contains(&percentile));
    let rank = percentile
        .checked_mul(sorted.len())
        .expect("benchmark percentile rank fits usize")
        .div_ceil(100);
    sorted[rank - 1]
}

fn run_llama_prefill_benchmark(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    spec: LlamaPrefillBenchmarkSpec<'_>,
) -> TestResult<LlamaPrefillBenchmarkOutcome> {
    let LlamaPrefillBenchmarkSpec {
        backend_label,
        expected_implementation_id,
        sequence_length,
        token_ids,
        config,
    } = spec;
    let mut forward =
        PreparedLlamaForward::prepare(model, context, stream, sequence_length, config)?;
    let selection = forward.attention_selection();
    let report = forward.allocation_report();
    assert_eq!(selection.implementation_id(), expected_implementation_id);
    assert_report_matches_context(report, context.allocation_stats()?);
    assert_eq!(token_ids.len(), sequence_length);

    println!(
        "pr08-llama-prefill-benchmark-metadata schema_version=1 \
metric=prefill_execute_proxy_ttft backend={backend_label} sequence_length={sequence_length} \
token_pattern=repeat_pr07_pinned_tokens_a implementation_id={} implementation_version={} \
native_dependency={} selection_reason={:?} score_materialization={:?} \
materialized_score_bytes={} attention_workspace_bytes={} layout_copy_bytes={} graph_bytes={} \
gemm_workspace_bytes={} total_device_bytes={} device_allocation_count={} warmup_iterations={} \
measured_iterations={} timing_boundary=execute_plus_stream_synchronize \
decode_sampling=incomplete_excluded",
        selection.implementation_id(),
        selection.implementation_version(),
        selection.native_dependency(),
        selection.reason(),
        selection.score_materialization(),
        selection.materialized_score_bytes(),
        selection.workspace_bytes(),
        selection.layout_copy_bytes(),
        report.graph_bytes(),
        report.gemm_workspace_bytes(),
        report.total_device_bytes(),
        report.device_allocation_count(),
        MODEL_BENCHMARK_WARMUP_ITERATIONS,
        MODEL_BENCHMARK_MEASURED_ITERATIONS,
    );

    forward.upload_tokens(token_ids, stream)?;
    for _ in 0..MODEL_BENCHMARK_WARMUP_ITERATIONS {
        forward.execute(stream)?;
        stream.synchronize()?;
    }

    let stable_allocations = context.allocation_stats()?;
    let mut latencies_ns = Vec::with_capacity(MODEL_BENCHMARK_MEASURED_ITERATIONS);
    for iteration in 0..MODEL_BENCHMARK_MEASURED_ITERATIONS {
        let started = Instant::now();
        forward.execute(stream)?;
        stream.synchronize()?;
        let latency_ns = u64::try_from(started.elapsed().as_nanos())?;
        latencies_ns.push(latency_ns);
        println!(
            "pr08-llama-prefill-benchmark-raw schema_version=1 \
metric=prefill_execute_proxy_ttft backend={backend_label} sequence_length={sequence_length} \
iteration={iteration} latency_ns={latency_ns} timing_boundary=execute_plus_stream_synchronize \
decode_sampling=incomplete_excluded"
        );
    }
    assert_eq!(
        context.allocation_stats()?,
        stable_allocations,
        "hot benchmark iterations must not change CUDA allocation accounting"
    );

    let mut sorted_latencies_ns = latencies_ns;
    sorted_latencies_ns.sort_unstable();
    let median_ns = percentile_nearest_rank(&sorted_latencies_ns, 50);
    let p95_ns = percentile_nearest_rank(&sorted_latencies_ns, 95);
    println!(
        "pr08-llama-prefill-benchmark-summary schema_version=1 \
metric=prefill_execute_proxy_ttft backend={backend_label} sequence_length={sequence_length} \
samples={} median_ns={median_ns} p95_ns={p95_ns} \
timing_boundary=execute_plus_stream_synchronize decode_sampling=incomplete_excluded",
        sorted_latencies_ns.len(),
    );

    let row_bytes = forward
        .plan()
        .dimensions()
        .vocabulary_size()
        .checked_mul(2)
        .ok_or("last-logit byte length overflow")?;
    let mut last_logits = vec![0_u8; row_bytes];
    forward.download_last_logits(&mut last_logits, stream)?;
    assert_bf16_logits_are_finite(&last_logits);
    forward.close()?;
    assert!(
        context.allocation_stats()?.is_zero(),
        "sequential benchmark owners must release all CUDA allocations"
    );
    Ok(LlamaPrefillBenchmarkOutcome {
        selection,
        report,
        last_logits,
    })
}

#[test]
#[ignore = "requires the pinned checkpoint/golden and a CUDA GPU on server-4096"]
fn pinned_smollm2_fixed_sequence_forward_matches_golden_and_is_causal() -> TestResult {
    let checkpoint = std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory");
    let golden_manifest = std::env::var_os("RUSTINFER_PR07_GOLDEN_MANIFEST")
        .map(PathBuf::from)
        .expect("RUSTINFER_PR07_GOLDEN_MANIFEST must name the remote golden manifest");
    let golden = parse_golden_trace(&golden_manifest)?;
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;

    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());

    let invalid_config = PreparedLlamaForwardConfig::new(0, 4_096, 0, u64::MAX);
    let error = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        SEQUENCE_LENGTH,
        invalid_config,
    )
    .expect_err("zero upload staging must fail before allocating CUDA buffers");
    assert!(matches!(
        error,
        LlamaForwardError::InvalidConfiguration {
            field: "upload_staging_bytes",
            ..
        }
    ));
    assert!(context.allocation_stats()?.is_zero());

    let error = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        0,
        PreparedLlamaForwardConfig::default(),
    )
    .expect_err("zero sequence length must fail before uploading weights");
    assert!(matches!(
        error,
        LlamaForwardError::Plan(LlamaPlanError::InvalidSequenceLength { requested: 0, .. })
    ));
    assert!(context.allocation_stats()?.is_zero());

    let defaults = PreparedLlamaForwardConfig::default();
    let zero_attention_budget = PreparedLlamaForwardConfig::new(
        defaults.upload_staging_bytes(),
        defaults.io_staging_bytes(),
        defaults.gemm_workspace_cap_bytes(),
        0,
    )
    .with_reference_attention();
    let error = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        SEQUENCE_LENGTH,
        zero_attention_budget,
    )
    .expect_err("attention budget failure must release the uploaded checkpoint");
    assert!(matches!(
        error,
        LlamaForwardError::AttentionBudgetExceeded {
            maximum_bytes: 0,
            ..
        }
    ));
    assert!(
        context.allocation_stats()?.is_zero(),
        "failed cold preparation must release all partial CUDA resources"
    );

    let mut forward = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        SEQUENCE_LENGTH,
        PreparedLlamaForwardConfig::default().with_reference_attention(),
    )?;
    assert_report_matches_context(forward.allocation_report(), context.allocation_stats()?);
    assert_eq!(forward.plan().layers().len(), 30);
    assert_eq!(forward.plan().dimensions().hidden_size(), HIDDEN_SIZE);
    assert_eq!(
        forward.plan().dimensions().intermediate_size(),
        INTERMEDIATE_SIZE
    );
    assert_eq!(
        forward.plan().dimensions().vocabulary_size(),
        VOCABULARY_SIZE
    );
    assert!(!forward.tokens_ready());
    assert!(!forward.output_ready());
    assert!(!forward.is_poisoned());

    let error = forward
        .execute(&mut stream)
        .expect_err("execution before token upload must fail closed");
    assert!(matches!(error, LlamaForwardError::TokensNotUploaded));
    let error = forward
        .download_last_logits(&mut [], &mut stream)
        .expect_err("download before execution must fail closed");
    assert!(matches!(error, LlamaForwardError::OutputNotReady));
    let error = forward
        .upload_tokens(&PINNED_TOKENS_A[..SEQUENCE_LENGTH - 1], &mut stream)
        .expect_err("wrong token count must fail before copying");
    match error {
        LlamaForwardError::InvalidTokenCount {
            expected, actual, ..
        } => {
            assert_eq!(expected, SEQUENCE_LENGTH);
            assert_eq!(actual, SEQUENCE_LENGTH - 1);
        }
        other => panic!("expected token-count validation error, got {other}"),
    }
    let mut out_of_range = PINNED_TOKENS_A;
    out_of_range[SEQUENCE_LENGTH - 1] = u32::MAX;
    let error = forward
        .upload_tokens(&out_of_range, &mut stream)
        .expect_err("out-of-vocabulary token must fail before copying");
    match error {
        LlamaForwardError::TokenOutOfRange {
            position, token_id, ..
        } => {
            assert_eq!(position, SEQUENCE_LENGTH - 1);
            assert_eq!(token_id, u32::MAX);
        }
        other => panic!("expected token-range validation error, got {other}"),
    }
    assert!(!forward.tokens_ready());
    assert!(!forward.is_poisoned());
    assert_report_matches_context(forward.allocation_report(), context.allocation_stats()?);

    let mut trace = forward.prepare_trace()?;
    assert_eq!(trace.tensor_count(), LlamaTracePoint::ALL.len());
    assert_eq!(trace.captured_count(), 0);
    forward.upload_tokens(&PINNED_TOKENS_A, &mut stream)?;
    forward.execute_traced(&mut stream, &mut trace)?;
    assert!(forward.tokens_ready());
    assert!(forward.output_ready());
    assert_trace_matches_golden(&trace, &golden);
    assert_report_matches_context(forward.allocation_report(), context.allocation_stats()?);

    let logits_bytes = usize::try_from(forward.plan().workspace_spec().logits_bytes())?;
    let vocabulary_size = forward.plan().dimensions().vocabulary_size();
    let row_bytes = vocabulary_size.checked_mul(2).ok_or("logit row overflow")?;
    assert_eq!(logits_bytes, SEQUENCE_LENGTH * row_bytes);

    let mut first_logits = vec![0_u8; logits_bytes];
    let mut first_last_logits = vec![0_u8; row_bytes];
    forward.download_logits(&mut first_logits, &mut stream)?;
    forward.download_last_logits(&mut first_last_logits, &mut stream)?;
    assert_bf16_logits_are_finite(&first_logits);
    assert_eq!(
        first_last_logits,
        first_logits[first_logits.len() - row_bytes..],
        "last-logit download must equal the final full-logit row"
    );
    assert_eq!(
        trace
            .tensor(LlamaTracePoint::LastLogits)
            .expect("trace last logits"),
        first_last_logits,
        "trace and public last-logit downloads must agree"
    );

    let foreign_context = device.create_context()?;
    let mut foreign_stream = foreign_context.create_stream()?;
    let mut rejected_download = vec![0_u8; row_bytes];
    let error = forward
        .download_last_logits(&mut rejected_download, &mut foreign_stream)
        .expect_err("a foreign-context stream must fail validation before copying");
    match error {
        LlamaForwardError::Cuda { source, .. } => {
            assert_eq!(source.stage(), CudaErrorStage::Validation);
        }
        other => panic!("expected CUDA validation error, got {other}"),
    }
    assert!(forward.output_ready());
    assert!(!forward.is_poisoned());
    let mut retried_download = vec![0_u8; row_bytes];
    forward.download_last_logits(&mut retried_download, &mut stream)?;
    assert_eq!(retried_download, first_last_logits);
    foreign_stream.close()?;
    assert!(foreign_context.allocation_stats()?.is_zero());
    foreign_context.close()?;

    forward.execute(&mut stream)?;
    let mut repeated_logits = vec![0_u8; logits_bytes];
    let mut repeated_last_logits = vec![0_u8; row_bytes];
    forward.download_logits(&mut repeated_logits, &mut stream)?;
    forward.download_last_logits(&mut repeated_last_logits, &mut stream)?;
    assert_eq!(repeated_logits, first_logits);
    assert_eq!(repeated_last_logits, first_last_logits);

    let repeat_allocations = context.allocation_stats()?;
    for _ in 0..100 {
        forward.execute(&mut stream)?;
    }
    assert_eq!(
        repeat_allocations,
        context.allocation_stats()?,
        "100 hot executions must not change CUDA allocation accounting"
    );
    let mut hundredth_logits = vec![0_u8; logits_bytes];
    forward.download_logits(&mut hundredth_logits, &mut stream)?;
    assert_eq!(hundredth_logits, first_logits);

    forward.forward(&PINNED_TOKENS_B, &mut stream)?;
    let mut divergent_suffix_logits = vec![0_u8; logits_bytes];
    forward.download_logits(&mut divergent_suffix_logits, &mut stream)?;
    assert_bf16_logits_are_finite(&divergent_suffix_logits);
    let prefix_bytes = COMMON_PREFIX_LENGTH
        .checked_mul(row_bytes)
        .ok_or("common-prefix byte length overflow")?;
    assert_eq!(
        &divergent_suffix_logits[..prefix_bytes],
        &first_logits[..prefix_bytes],
        "a divergent future suffix must not change common-prefix logits"
    );
    assert_ne!(
        &divergent_suffix_logits[prefix_bytes..],
        &first_logits[prefix_bytes..],
        "the two pinned suffixes must exercise distinct forward inputs"
    );
    assert_eq!(repeat_allocations, context.allocation_stats()?);

    forward.close()?;
    assert!(
        context.allocation_stats()?.is_zero(),
        "explicit forward close must release weights, graph buffers, and pinned staging"
    );
    stream.close()?;
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "requires the pinned checkpoint/golden and a CUDA GPU on server-4096"]
fn pinned_smollm2_online_prefill_matches_reference_without_score_storage() -> TestResult {
    let checkpoint = std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory");
    let golden_manifest = std::env::var_os("RUSTINFER_PR07_GOLDEN_MANIFEST")
        .map(PathBuf::from)
        .expect("RUSTINFER_PR07_GOLDEN_MANIFEST must name the remote golden manifest");
    let golden = parse_golden_trace(&golden_manifest)?;
    let golden_last_logits = golden.last().ok_or("golden trace has no last logits")?;
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;

    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());

    let mut reference = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        SEQUENCE_LENGTH,
        PreparedLlamaForwardConfig::default().with_reference_attention(),
    )?;
    let reference_selection = reference.attention_selection();
    let reference_report = reference.allocation_report();
    assert_eq!(
        reference_selection.implementation_id(),
        "rustinfer.cuda.materialized-gqa-prefill.bf16"
    );
    assert_eq!(
        reference_selection.workspace_bytes(),
        reference.plan().workspace_spec().attention_buffer_bytes()
    );
    assert_eq!(
        reference_selection.materialized_score_bytes(),
        reference.plan().workspace_spec().attention_buffer_bytes()
    );
    assert_report_matches_context(reference_report, context.allocation_stats()?);

    reference.forward(&PINNED_TOKENS_A, &mut stream)?;
    let row_bytes = VOCABULARY_SIZE
        .checked_mul(2)
        .ok_or("last-logit byte length overflow")?;
    let logits_bytes = SEQUENCE_LENGTH
        .checked_mul(row_bytes)
        .ok_or("full-logit byte length overflow")?;
    assert_eq!(
        usize::try_from(reference.plan().workspace_spec().logits_bytes())?,
        logits_bytes
    );
    let mut reference_logits = vec![0_u8; logits_bytes];
    reference.download_logits(&mut reference_logits, &mut stream)?;
    let mut reference_last_logits = vec![0_u8; row_bytes];
    reference.download_last_logits(&mut reference_last_logits, &mut stream)?;
    let reference_greedy = top_k(&reference_last_logits, 1)[0];
    reference.close()?;
    assert!(context.allocation_stats()?.is_zero());

    let defaults = PreparedLlamaForwardConfig::default();
    let zero_workspace_optimized = PreparedLlamaForwardConfig::new(
        defaults.upload_staging_bytes(),
        defaults.io_staging_bytes(),
        defaults.gemm_workspace_cap_bytes(),
        0,
    )
    .with_optimized_attention();
    let mut optimized = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        SEQUENCE_LENGTH,
        zero_workspace_optimized,
    )?;
    let optimized_selection = optimized.attention_selection();
    let optimized_report = optimized.allocation_report();
    assert_eq!(
        optimized_selection.implementation_id(),
        "rustinfer.cuda.online-gqa-prefill.bf16.d64"
    );
    assert_eq!(optimized_selection.workspace_bytes(), 0);
    assert_eq!(optimized_selection.materialized_score_bytes(), 0);
    assert_ne!(
        optimized_selection.implementation_id(),
        reference_selection.implementation_id()
    );
    assert_eq!(
        optimized_report.graph_bytes(),
        optimized
            .plan()
            .workspace_spec()
            .non_attention_planned_bytes()
    );
    assert_eq!(
        optimized_report.graph_bytes() + reference_selection.workspace_bytes(),
        reference_report.graph_bytes()
    );
    assert_eq!(
        optimized_report.gemm_workspace_bytes(),
        reference_report.gemm_workspace_bytes()
    );
    assert_eq!(
        optimized_report.total_device_bytes() + reference_selection.workspace_bytes(),
        reference_report.total_device_bytes()
    );
    assert_eq!(
        optimized_report.device_allocation_count() + 1,
        reference_report.device_allocation_count()
    );
    assert!(matches!(
        optimized.prepare_trace(),
        Err(LlamaForwardError::TraceRequiresReferenceAttention)
    ));
    assert_report_matches_context(optimized_report, context.allocation_stats()?);

    optimized.forward(&PINNED_TOKENS_A, &mut stream)?;
    let mut optimized_prefix_source = vec![0_u8; logits_bytes];
    optimized.download_logits(&mut optimized_prefix_source, &mut stream)?;
    let mut optimized_last_logits = vec![0_u8; row_bytes];
    optimized.download_last_logits(&mut optimized_last_logits, &mut stream)?;
    assert_bf16_logits_are_finite(&optimized_last_logits);
    assert_eq!(
        &optimized_prefix_source[..row_bytes],
        &reference_logits[..row_bytes],
        "the first causal row must remain byte-exact across attention backends"
    );

    let reference_metrics = numeric_metrics(&optimized_last_logits, &reference_last_logits);
    let golden_metrics = numeric_metrics(&optimized_last_logits, golden_last_logits);
    eprintln!(
        "pr08-online-vs-reference cosine={:.12} max_abs={:.9} mean_abs={:.9}",
        reference_metrics.cosine, reference_metrics.max_abs, reference_metrics.mean_abs
    );
    eprintln!(
        "pr08-online-vs-golden cosine={:.12} max_abs={:.9} mean_abs={:.9}",
        golden_metrics.cosine, golden_metrics.max_abs, golden_metrics.mean_abs
    );
    let optimized_greedy = top_k(&optimized_last_logits, 1)[0];
    let optimized_top_ten = top_k(&optimized_last_logits, 10);
    let golden_top_ten = top_k(golden_last_logits, 10);
    let optimized_top_ten_csv = optimized_top_ten
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "pr08-online-semantic schema_version=1 optimized_top1={optimized_greedy} top10={optimized_top_ten_csv} row0_byte_exact=true"
    );
    assert_eq!(optimized_greedy, reference_greedy);
    assert_eq!(optimized_greedy, top_k(golden_last_logits, 1)[0]);
    assert_eq!(
        optimized_top_ten.into_iter().collect::<BTreeSet<_>>(),
        golden_top_ten.into_iter().collect::<BTreeSet<_>>(),
        "online and golden last logits must preserve the top-10 token set"
    );
    assert!(
        reference_metrics.cosine >= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.cosine_min,
        "online/reference cosine {reference_metrics:?}"
    );
    assert!(
        reference_metrics.max_abs <= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.max_abs_max,
        "online/reference max abs {reference_metrics:?}"
    );
    assert!(
        reference_metrics.mean_abs <= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.mean_abs_max,
        "online/reference mean abs {reference_metrics:?}"
    );
    assert!(
        golden_metrics.cosine >= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.cosine_min,
        "online/golden cosine {golden_metrics:?}"
    );
    assert!(
        golden_metrics.max_abs <= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.max_abs_max,
        "online/golden max abs {golden_metrics:?}"
    );
    assert!(
        golden_metrics.mean_abs <= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.mean_abs_max,
        "online/golden mean abs {golden_metrics:?}"
    );

    let stable_allocations = context.allocation_stats()?;
    let mut repeated_logits = vec![0_u8; logits_bytes];
    for iteration in 0..100 {
        optimized.execute(&mut stream)?;
        optimized.download_logits(&mut repeated_logits, &mut stream)?;
        assert_eq!(
            repeated_logits,
            optimized_prefix_source,
            "online hot execution {} must remain byte-deterministic",
            iteration + 1
        );
    }
    assert_eq!(
        context.allocation_stats()?,
        stable_allocations,
        "100 online hot executions must not allocate score or workspace buffers"
    );
    eprintln!(
        "pr08-online-determinism schema_version=1 executions=100 logits_sha256={} byte_exact=true",
        sha256_hex(&repeated_logits)
    );

    optimized.forward(&PINNED_TOKENS_B, &mut stream)?;
    let mut optimized_divergent_suffix = vec![0_u8; logits_bytes];
    optimized.download_logits(&mut optimized_divergent_suffix, &mut stream)?;
    let prefix_bytes = COMMON_PREFIX_LENGTH
        .checked_mul(row_bytes)
        .ok_or("common-prefix byte length overflow")?;
    assert_eq!(
        &optimized_divergent_suffix[..prefix_bytes],
        &optimized_prefix_source[..prefix_bytes],
        "online attention must not let a future suffix change prefix logits"
    );
    assert_ne!(
        &optimized_divergent_suffix[prefix_bytes..],
        &optimized_prefix_source[prefix_bytes..],
        "the optimized causal regression inputs must have distinct suffix logits"
    );
    assert_eq!(context.allocation_stats()?, stable_allocations);

    optimized.close()?;
    assert!(context.allocation_stats()?.is_zero());
    stream.close()?;
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "remote-only pinned SmolLM2 reference-vs-online prefill benchmark on server-4096"]
fn benchmark_pinned_smollm2_reference_vs_online_prefill_execute_proxy_ttft() -> TestResult {
    let checkpoint = std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory");
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let token_ids = benchmark_tokens(MODEL_BENCHMARK_SEQUENCE_LENGTH);

    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote benchmark runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());

    let reference = run_llama_prefill_benchmark(
        &model,
        &context,
        &mut stream,
        LlamaPrefillBenchmarkSpec {
            backend_label: "reference",
            expected_implementation_id: "rustinfer.cuda.materialized-gqa-prefill.bf16",
            sequence_length: MODEL_BENCHMARK_SEQUENCE_LENGTH,
            token_ids: &token_ids,
            config: PreparedLlamaForwardConfig::default().with_reference_attention(),
        },
    )?;
    assert!(reference.selection.workspace_bytes() > 0);
    assert_eq!(
        reference.selection.workspace_bytes(),
        reference.selection.materialized_score_bytes()
    );

    let defaults = PreparedLlamaForwardConfig::default();
    let optimized_config = PreparedLlamaForwardConfig::new(
        defaults.upload_staging_bytes(),
        defaults.io_staging_bytes(),
        defaults.gemm_workspace_cap_bytes(),
        0,
    )
    .with_optimized_attention();
    let optimized = run_llama_prefill_benchmark(
        &model,
        &context,
        &mut stream,
        LlamaPrefillBenchmarkSpec {
            backend_label: "optimized",
            expected_implementation_id: "rustinfer.cuda.online-gqa-prefill.bf16.d64",
            sequence_length: MODEL_BENCHMARK_SEQUENCE_LENGTH,
            token_ids: &token_ids,
            config: optimized_config,
        },
    )?;
    assert_eq!(optimized.selection.workspace_bytes(), 0);
    assert_eq!(optimized.selection.materialized_score_bytes(), 0);
    assert_eq!(optimized.selection.layout_copy_bytes(), 0);
    assert_eq!(
        optimized.report.graph_bytes() + reference.selection.workspace_bytes(),
        reference.report.graph_bytes()
    );

    let metrics = numeric_metrics(&optimized.last_logits, &reference.last_logits);
    let reference_top1 = top_k(&reference.last_logits, 1)[0];
    let optimized_top1 = top_k(&optimized.last_logits, 1)[0];
    println!(
        "pr08-llama-prefill-benchmark-parity schema_version=1 sequence_length={} \
reference_top1={reference_top1} optimized_top1={optimized_top1} cosine={:.12} \
max_abs={:.9} mean_abs={:.9}",
        MODEL_BENCHMARK_SEQUENCE_LENGTH, metrics.cosine, metrics.max_abs, metrics.mean_abs,
    );
    assert_eq!(
        optimized_top1, reference_top1,
        "S=128 reference and online prefill must preserve the greedy next token"
    );
    assert!(
        metrics.cosine.is_finite() && metrics.max_abs.is_finite() && metrics.mean_abs.is_finite(),
        "S=128 online/reference parity metrics must be finite: {metrics:?}"
    );
    assert!(
        metrics.cosine >= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.cosine_min,
        "S=128 online/reference cosine failed the predeclared 3-metric gate: {metrics:?}"
    );
    assert!(
        metrics.max_abs <= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.max_abs_max,
        "S=128 online/reference max error failed the predeclared 3-metric gate: {metrics:?}"
    );
    assert!(
        metrics.mean_abs <= PR01_E0_V2_FINAL_LOGITS_TOLERANCE.mean_abs_max,
        "S=128 online/reference mean error failed the predeclared 3-metric gate: {metrics:?}"
    );

    stream.close()?;
    context.close()?;
    Ok(())
}
