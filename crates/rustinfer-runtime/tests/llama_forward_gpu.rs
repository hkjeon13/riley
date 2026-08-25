//! Remote-only end-to-end PR07 CUDA and pinned-golden validation.

#![cfg(feature = "cuda")]
#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

#[cfg(not(target_endian = "little"))]
compile_error!("the pinned safetensors golden gate requires a little-endian target");

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use rustinfer_cuda::{CudaAllocationStats, CudaErrorStage, CudaRuntime};
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
        LlamaTracePoint::Embedding => None,
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
        LlamaTracePoint::Layer14Output
        | LlamaTracePoint::FinalNormInput
        | LlamaTracePoint::FinalNormOutput => Some(cumulative_hidden),
        // Predeclared final-logits threshold from the immutable PR01 E0 v2 matrix.
        LlamaTracePoint::LastLogits => Some(NumericTolerance {
            cosine_min: 0.997_903_530_549_539_3,
            max_abs_max: 5.852_936_458_587_647,
            mean_abs_max: 1.151_280_319_263_363,
        }),
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
        if let Some(tolerance) = tolerance(point) {
            let metrics = numeric_metrics(actual, expected);
            println!(
                "pr07-trace point={} cosine={:.12} max_abs={:.9} mean_abs={:.9}",
                point.name(),
                metrics.cosine,
                metrics.max_abs,
                metrics.mean_abs
            );
            assert!(
                metrics.cosine >= tolerance.cosine_min
                    && metrics.max_abs <= tolerance.max_abs_max
                    && metrics.mean_abs <= tolerance.mean_abs_max,
                "first divergent PR07 checkpoint {}: metrics={metrics:?}, tolerance={tolerance:?}",
                point.name()
            );
        } else {
            assert_eq!(actual, expected, "{} must be byte-exact", point.name());
        }
    }

    let actual_logits = trace
        .tensor(LlamaTracePoint::LastLogits)
        .expect("last logits were captured");
    let expected_logits = golden.last().expect("golden trace contains last logits");
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
    );
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
        PreparedLlamaForwardConfig::default(),
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
