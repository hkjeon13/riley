//! Direct P2051 attention boundary qualifier against Hugging Face eager mode.
//!
//! This is an offline CUDA diagnostic. It consumes a create-only artifact made
//! by the actual Hugging Face Qwen eager module, then compares the native
//! materialized QK, scale/mask, softmax, and AV stages independently. It never
//! routes a request through Python or changes a serving selector.

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use riley_cuda::{
    AttentionBackend, AttentionBackendAvailability, AttentionPreference, AvGqaParams,
    CausalSoftmaxInPlaceParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaPinnedHostBuffer, CudaRuntime, CudaStream, PrefillAttentionParams,
    PrefillAttentionRequest, PreparedPrefillAttention, QkGqaParams, ScaleCausalMaskInPlaceParams,
    av_gqa, causal_softmax_in_place, qk_gqa, scale_causal_mask_in_place,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ARTIFACT_ROOT_ENV: &str = "RILEY_QWEN3B_P2051_ATTENTION_BOUNDARY_ROOT";
const OUTPUT_ENV: &str = "RILEY_QWEN3B_P2051_ATTENTION_BOUNDARY_OUTPUT";
const CUBLASLT_OUTPUT_ENV: &str = "RILEY_QWEN3B_P2051_CUBLASLT_PROBE_OUTPUT";
const MARKER_PREFIX: &str = "RILEY_QWEN3B_P2051_ATTENTION_BOUNDARY=";
const ARTIFACT_SCHEMA: &str = "riley.qwen3b-p2051-attention-boundary-probe.v1";
const ARTIFACT_KIND: &str = "offline-hf-qwen2-eager-attention-boundary-probe";
const RESULT_SCHEMA: &str = "riley.qwen3b-p2051-native-attention-boundary-comparison.v1";
const RESULT_KIND: &str = "qwen2.5-3b-riley-p2051-native-attention-boundary-comparison";
const CUBLASLT_RESULT_SCHEMA: &str = "riley.qwen3b-p2051-cublaslt-attention-probe-comparison.v1";
const CUBLASLT_RESULT_KIND: &str = "qwen2.5-3b-riley-p2051-cublaslt-attention-probe-comparison";
const INPUT_SHA256: &str = "850a1cb46f8fa5e98af1445a95d77af6621705afc14cb740ec6cb24a638f9c2c";
const HF_EAGER_SOURCE_SHA256: &str =
    "cb34ccea28710ae8d5d94a241b704ea3d974048f85372f0b1b3c2a8a2ef1ee20";
const S: u64 = 2_051;
const QH: u64 = 16;
const KVH: u64 = 2;
const D: u64 = 128;
const SCALE: f32 = 0.088_388_346;

const Q_BSHD: [u64; 4] = [1, S, QH, D];
const KV_BSHD: [u64; 4] = [1, S, KVH, D];
const LAST_SCORE_ROWS: [u64; 2] = [QH, S];

#[derive(Clone, Copy)]
struct TensorSpec {
    name: &'static str,
    shape: &'static [u64],
}

const INPUT_SPECS: &[TensorSpec] = &[
    TensorSpec {
        name: "query_bshd",
        shape: &Q_BSHD,
    },
    TensorSpec {
        name: "key_bshd",
        shape: &KV_BSHD,
    },
    TensorSpec {
        name: "value_bshd",
        shape: &KV_BSHD,
    },
];

const EXPECTED_SPECS: &[TensorSpec] = &[
    TensorSpec {
        name: "raw_qk_last",
        shape: &LAST_SCORE_ROWS,
    },
    TensorSpec {
        name: "scaled_masked_last",
        shape: &LAST_SCORE_ROWS,
    },
    TensorSpec {
        name: "probabilities_last",
        shape: &LAST_SCORE_ROWS,
    },
    TensorSpec {
        name: "context_bshd",
        shape: &Q_BSHD,
    },
];

#[derive(Debug)]
struct BoundaryArtifact {
    root: PathBuf,
    metadata_path: PathBuf,
    metadata_sha256: String,
    tensors: BTreeMap<String, Vec<u8>>,
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
            output
        },
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

fn required_path(variable: &str) -> TestResult<PathBuf> {
    let path = std::env::var_os(variable)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{variable} must be set"))?;
    if path.as_os_str().is_empty() {
        return Err(format!("{variable} must not be empty").into());
    }
    Ok(path)
}

fn regular_file(path: &Path, label: &str) -> TestResult<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file").into());
    }
    Ok(path.canonicalize()?)
}

fn regular_directory(path: &Path, label: &str) -> TestResult<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(format!("{label} must be a regular non-symlink directory").into());
    }
    Ok(path.canonicalize()?)
}

fn shape_byte_len(shape: &[u64]) -> TestResult<usize> {
    let elements = shape.iter().try_fold(1_u64, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or("attention boundary shape arithmetic overflow")
    })?;
    usize::try_from(
        elements
            .checked_mul(2)
            .ok_or("attention boundary byte arithmetic overflow")?,
    )
    .map_err(Into::into)
}

fn checked_tensor_file(root: &Path, relative: &str, label: &str) -> TestResult<PathBuf> {
    let path = Path::new(relative);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(format!("{label} path must be a basename").into());
    }
    let resolved = regular_file(&root.join(path), label)?;
    if !resolved.starts_with(root) {
        return Err(format!("{label} escapes the artifact root").into());
    }
    Ok(resolved)
}

fn load_tensor(
    root: &Path,
    records: &serde_json::Map<String, Value>,
    spec: TensorSpec,
) -> TestResult<Vec<u8>> {
    let record = records
        .get(spec.name)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("HF boundary tensor {} is missing", spec.name))?;
    let path = record
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("HF boundary tensor {} path is missing", spec.name))?;
    if record.get("dtype").and_then(Value::as_str) != Some("torch.bfloat16")
        || record.get("shape") != Some(&json!(spec.shape))
        || record.get("byte_len").and_then(Value::as_u64)
            != Some(u64::try_from(shape_byte_len(spec.shape)?)?)
    {
        return Err(format!("HF boundary tensor {} metadata differs", spec.name).into());
    }
    let expected_sha256 = record
        .get("sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("HF boundary tensor {} SHA-256 is missing", spec.name))?;
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!("HF boundary tensor {} SHA-256 differs", spec.name).into());
    }
    let source = checked_tensor_file(root, path, spec.name)?;
    let bytes = fs::read(&source)?;
    if bytes.len() != shape_byte_len(spec.shape)? || sha256_hex(&bytes) != expected_sha256 {
        return Err(format!("HF boundary tensor {} bytes differ", spec.name).into());
    }
    Ok(bytes)
}

fn load_artifact() -> TestResult<BoundaryArtifact> {
    let root = regular_directory(
        &required_path(ARTIFACT_ROOT_ENV)?,
        "HF boundary artifact root",
    )?;
    let metadata_path = regular_file(&root.join("metadata.json"), "HF boundary metadata")?;
    let bytes = fs::read(&metadata_path)?;
    let metadata_sha256 = sha256_hex(&bytes);
    let document: Value = serde_json::from_slice(&bytes)?;
    if document.get("schema_version").and_then(Value::as_str) != Some(ARTIFACT_SCHEMA)
        || document.get("kind").and_then(Value::as_str) != Some(ARTIFACT_KIND)
        || document["input"]["token_count"].as_u64() != Some(S)
        || document["input"]["sha256_le_u32"].as_str() != Some(INPUT_SHA256)
        || document["contract"]["query_heads"].as_u64() != Some(QH)
        || document["contract"]["key_value_heads"].as_u64() != Some(KVH)
        || document["contract"]["head_size"].as_u64() != Some(D)
        || document["contract"]["o_proj_preinput_exact"].as_bool() != Some(true)
        || document["hf_eager_source_sha256"].as_str() != Some(HF_EAGER_SOURCE_SHA256)
    {
        return Err("HF boundary artifact contract differs".into());
    }
    let scale = document["contract"]["scale"]
        .as_f64()
        .ok_or("HF boundary artifact scale is missing")? as f32;
    if (scale - SCALE).abs() > f32::EPSILON {
        return Err("HF boundary artifact scale differs".into());
    }
    let records = document["tensors"]
        .as_object()
        .ok_or("HF boundary tensor records are missing")?;
    let mut tensors = BTreeMap::new();
    for spec in INPUT_SPECS.iter().chain(EXPECTED_SPECS) {
        tensors.insert(spec.name.to_owned(), load_tensor(&root, records, *spec)?);
    }
    Ok(BoundaryArtifact {
        root,
        metadata_path,
        metadata_sha256,
        tensors,
    })
}

fn tensor<'a>(artifact: &'a BoundaryArtifact, name: &str) -> TestResult<&'a [u8]> {
    artifact
        .tensors
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("loaded boundary tensor {name} is missing").into())
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "runner has no CUDA device");
    let context = runtime.device(0)?.create_context()?;
    assert_eq!(context.compute_capability(), (8, 9));
    let stream = context.create_stream()?;
    Ok((context, stream))
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

fn download_full(
    stream: &mut CudaStream,
    buffer: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
) -> TestResult<Vec<u8>> {
    buffer
        .copy_to_pinned_async(0, staging, 0, buffer.byte_len(), stream)?
        .synchronize()?;
    Ok(staging.to_vec()?)
}

fn download_last_score_rows(
    stream: &mut CudaStream,
    scores: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
) -> TestResult<Vec<u8>> {
    let row_bytes = S.checked_mul(2).ok_or("last score row bytes overflow")?;
    assert_eq!(
        staging.byte_len(),
        row_bytes,
        "last-row staging shape differs"
    );
    let mut output = Vec::with_capacity(usize::try_from(QH * row_bytes)?);
    for head in 0..QH {
        let row_offset = head
            .checked_mul(S)
            .and_then(|value| value.checked_mul(S))
            .and_then(|value| value.checked_add((S - 1) * S))
            .and_then(|value| value.checked_mul(2))
            .ok_or("last score row offset overflow")?;
        scores
            .copy_to_pinned_async(row_offset, staging, 0, row_bytes, stream)?
            .synchronize()?;
        output.extend_from_slice(&staging.to_vec()?);
    }
    Ok(output)
}

fn qk(
    query: &CudaDeviceBuffer,
    key: &CudaDeviceBuffer,
    scores: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let score_bytes = scores.byte_len();
    let mut params = QkGqaParams {
        query: CudaBufferSpan::new(query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(key, CudaDType::BF16, 0, key.byte_len())?,
        output: CudaBufferSpanMut::new(scores, CudaDType::BF16, 0, score_bytes)?,
        token_count: S,
        query_head_count: QH,
        key_value_head_count: KVH,
        head_size: D,
    };
    qk_gqa(&mut params, stream)?;
    Ok(())
}

fn scale_mask(scores: &mut CudaDeviceBuffer, stream: &mut CudaStream) -> TestResult {
    let score_bytes = scores.byte_len();
    let mut params = ScaleCausalMaskInPlaceParams {
        scores: CudaBufferSpanMut::new(scores, CudaDType::BF16, 0, score_bytes)?,
        token_count: S,
        query_head_count: QH,
        scale: SCALE,
    };
    scale_causal_mask_in_place(&mut params, stream)?;
    Ok(())
}

fn softmax(scores: &mut CudaDeviceBuffer, stream: &mut CudaStream) -> TestResult {
    let score_bytes = scores.byte_len();
    let mut params = CausalSoftmaxInPlaceParams {
        scores: CudaBufferSpanMut::new(scores, CudaDType::BF16, 0, score_bytes)?,
        token_count: S,
        query_head_count: QH,
    };
    causal_softmax_in_place(&mut params, stream)?;
    Ok(())
}

fn av(
    scores: &CudaDeviceBuffer,
    value: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let output_bytes = output.byte_len();
    let mut params = AvGqaParams {
        probabilities: CudaBufferSpan::new(scores, CudaDType::BF16, 0, scores.byte_len())?,
        value: CudaBufferSpan::new(value, CudaDType::BF16, 0, value.byte_len())?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_bytes)?,
        token_count: S,
        query_head_count: QH,
        key_value_head_count: KVH,
        head_size: D,
    };
    av_gqa(&mut params, stream)?;
    Ok(())
}

fn execute_full_attention(
    query: &CudaDeviceBuffer,
    key: &CudaDeviceBuffer,
    value: &CudaDeviceBuffer,
    scores: &mut CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    qk(query, key, scores, stream)?;
    scale_mask(scores, stream)?;
    softmax(scores, stream)?;
    av(scores, value, output, stream)
}

fn bf16_metrics(expected: &[u8], actual: &[u8]) -> TestResult<Value> {
    if expected.len() != actual.len() || expected.len() % 2 != 0 {
        return Err("BF16 metric input byte lengths differ".into());
    }
    let mut mismatch_count = 0_u64;
    let mut first_mismatch = None;
    let mut max_abs = 0.0_f32;
    for (index, (expected_bits, actual_bits)) in expected
        .chunks_exact(2)
        .zip(actual.chunks_exact(2))
        .enumerate()
    {
        if expected_bits != actual_bits {
            mismatch_count += 1;
            first_mismatch.get_or_insert(u64::try_from(index)?);
        }
        let expected_value = f32::from_bits(
            u32::from(u16::from_le_bytes([expected_bits[0], expected_bits[1]])) << 16,
        );
        let actual_value =
            f32::from_bits(u32::from(u16::from_le_bytes([actual_bits[0], actual_bits[1]])) << 16);
        max_abs = max_abs.max((expected_value - actual_value).abs());
    }
    Ok(json!({
        "bf16_exact": mismatch_count == 0,
        "element_count": expected.len() / 2,
        "bf16_mismatch_count": mismatch_count,
        "first_bf16_mismatch_index": first_mismatch,
        "max_absolute_error": max_abs,
        "expected_bf16_le_sha256": sha256_hex(expected),
        "actual_bf16_le_sha256": sha256_hex(actual),
    }))
}

fn exact(metrics: &Value) -> TestResult<bool> {
    metrics
        .get("bf16_exact")
        .and_then(Value::as_bool)
        .ok_or("stage metric omits bf16_exact".into())
}

fn run_native_boundary(artifact: &BoundaryArtifact) -> TestResult<Value> {
    let query_bytes = tensor(artifact, "query_bshd")?;
    let key_bytes = tensor(artifact, "key_bshd")?;
    let value_bytes = tensor(artifact, "value_bshd")?;
    let expected_raw = tensor(artifact, "raw_qk_last")?;
    let expected_scaled = tensor(artifact, "scaled_masked_last")?;
    let expected_probabilities = tensor(artifact, "probabilities_last")?;
    let expected_context = tensor(artifact, "context_bshd")?;
    let score_bytes = QH
        .checked_mul(S)
        .and_then(|value| value.checked_mul(S))
        .and_then(|value| value.checked_mul(2))
        .ok_or("score byte arithmetic overflow")?;
    let row_bytes = S
        .checked_mul(2)
        .ok_or("score row byte arithmetic overflow")?;

    let (context, mut stream) = first_context()?;
    let mut transfer_staging =
        context.allocate_pinned_host_buffer(u64::try_from(query_bytes.len())?)?;
    let mut row_staging = context.allocate_pinned_host_buffer(row_bytes)?;
    let query = upload(&context, &mut stream, &mut transfer_staging, query_bytes)?;
    let key = upload(&context, &mut stream, &mut transfer_staging, key_bytes)?;
    let value = upload(&context, &mut stream, &mut transfer_staging, value_bytes)?;
    let mut scores = context.allocate_device_buffer(score_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(expected_context.len())?)?;
    let before = context.allocation_stats()?;

    qk(&query, &key, &mut scores, &mut stream)?;
    let raw_qk_last = download_last_score_rows(&mut stream, &mut scores, &mut row_staging)?;
    scale_mask(&mut scores, &mut stream)?;
    let scaled_masked_last = download_last_score_rows(&mut stream, &mut scores, &mut row_staging)?;
    softmax(&mut scores, &mut stream)?;
    let probabilities_last = download_last_score_rows(&mut stream, &mut scores, &mut row_staging)?;
    av(&scores, &value, &mut output, &mut stream)?;
    let context_bshd = download_full(&mut stream, &mut output, &mut transfer_staging)?;
    let after_first = context.allocation_stats()?;
    if after_first != before {
        return Err("native P2051 attention hot path changed allocations".into());
    }

    execute_full_attention(&query, &key, &value, &mut scores, &mut output, &mut stream)?;
    let repeated_context_bshd = download_full(&mut stream, &mut output, &mut transfer_staging)?;
    let after_repeat = context.allocation_stats()?;
    if after_repeat != before {
        return Err("repeated native P2051 attention changed allocations".into());
    }

    let result = (|| -> TestResult<Value> {
        let raw_qk = bf16_metrics(expected_raw, &raw_qk_last)?;
        let scaled_masked = bf16_metrics(expected_scaled, &scaled_masked_last)?;
        let probabilities = bf16_metrics(expected_probabilities, &probabilities_last)?;
        let context_metrics = bf16_metrics(expected_context, &context_bshd)?;
        let repeat = bf16_metrics(&context_bshd, &repeated_context_bshd)?;
        Ok(json!({
            "implementation_id": "riley.cuda.materialized-gqa-reference.bf16",
            "cache_free": true,
            "hot_execution_allocation_stable": true,
            "stages": {
                "raw_qk_last": raw_qk,
                "scaled_masked_last": scaled_masked,
                "probabilities_last": probabilities,
                "context_bshd": context_metrics,
            },
            "repeat_execution": {
                "context_bshd_bf16_exact": exact(&repeat)?,
                "context_bshd": repeat,
            },
        }))
    })();

    output.close()?;
    scores.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    row_staging.close()?;
    transfer_staging.close()?;
    stream.close()?;
    context.synchronize()?;
    let lifecycle_clean = context.allocation_stats()?.is_zero();
    context.close()?;
    let mut document = result?;
    document["lifecycle_clean_after_close"] = json!(lifecycle_clean);
    if !lifecycle_clean {
        return Err("native P2051 attention did not release every allocation".into());
    }
    Ok(document)
}

fn run_hf_eager_qwen_p2051_probe(artifact: &BoundaryArtifact) -> TestResult<Value> {
    let query_bytes = tensor(artifact, "query_bshd")?;
    let key_bytes = tensor(artifact, "key_bshd")?;
    let value_bytes = tensor(artifact, "value_bshd")?;
    let expected_probabilities = tensor(artifact, "probabilities_last")?;
    let expected_context = tensor(artifact, "context_bshd")?;
    let request =
        PrefillAttentionRequest::new(1, S, QH, KVH, D, SCALE, riley_cuda::AttentionMask::Causal);

    let (context, mut stream) = first_context()?;
    let prepared = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::HuggingFaceEagerQwenP2051Probe,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(
        prepared.backend(),
        AttentionBackend::HuggingFaceEagerQwenP2051Probe
    );
    let trace = prepared.selection_trace();
    let mut transfer_staging =
        context.allocate_pinned_host_buffer(u64::try_from(query_bytes.len())?)?;
    let mut row_staging = context.allocate_pinned_host_buffer(S * 2)?;
    let query = upload(&context, &mut stream, &mut transfer_staging, query_bytes)?;
    let key = upload(&context, &mut stream, &mut transfer_staging, key_bytes)?;
    let value = upload(&context, &mut stream, &mut transfer_staging, value_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(expected_context.len())?)?;
    let mut workspace = context.allocate_device_buffer(prepared.workspace_bytes())?;
    let before = context.allocation_stats()?;

    {
        let workspace_bytes = workspace.byte_len();
        let output_bytes = output.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
            workspace: Some(CudaBufferSpanMut::new(
                &mut workspace,
                CudaDType::BF16,
                0,
                workspace_bytes,
            )?),
        };
        prepared.execute(&mut params, &mut stream)?;
    }
    let probabilities_last =
        download_last_score_rows(&mut stream, &mut workspace, &mut row_staging)?;
    let context_bshd = download_full(&mut stream, &mut output, &mut transfer_staging)?;
    if context.allocation_stats()? != before {
        return Err("Qwen P2051 cuBLASLt probe hot path changed allocations".into());
    }

    {
        let workspace_bytes = workspace.byte_len();
        let output_bytes = output.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_bytes)?,
            workspace: Some(CudaBufferSpanMut::new(
                &mut workspace,
                CudaDType::BF16,
                0,
                workspace_bytes,
            )?),
        };
        prepared.execute(&mut params, &mut stream)?;
    }
    let repeated_context_bshd = download_full(&mut stream, &mut output, &mut transfer_staging)?;
    if context.allocation_stats()? != before {
        return Err("repeated Qwen P2051 cuBLASLt probe changed allocations".into());
    }

    let result = (|| -> TestResult<Value> {
        let probabilities = bf16_metrics(expected_probabilities, &probabilities_last)?;
        let context_metrics = bf16_metrics(expected_context, &context_bshd)?;
        let repeat = bf16_metrics(&context_bshd, &repeated_context_bshd)?;
        Ok(json!({
            "implementation_id": trace.implementation_id(),
            "implementation_version": trace.implementation_version(),
            "selection_reason": format!("{:?}", trace.reason()),
            "cache_free": true,
            "hot_execution_allocation_stable": true,
            "stages": {
                "probabilities_last": probabilities,
                "context_bshd": context_metrics,
            },
            "repeat_execution": {
                "context_bshd_bf16_exact": exact(&repeat)?,
                "context_bshd": repeat,
            },
        }))
    })();

    prepared.close()?;
    workspace.close()?;
    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    row_staging.close()?;
    transfer_staging.close()?;
    stream.close()?;
    context.synchronize()?;
    let lifecycle_clean = context.allocation_stats()?.is_zero();
    context.close()?;
    let mut document = result?;
    document["lifecycle_clean_after_close"] = json!(lifecycle_clean);
    if !lifecycle_clean {
        return Err("Qwen P2051 cuBLASLt probe did not release every allocation".into());
    }
    Ok(document)
}

fn output_path_for(variable: &str) -> TestResult<PathBuf> {
    let output = required_path(variable)?;
    if !output.is_absolute() || output.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err(format!("{variable} must be an absolute .json path").into());
    }
    let parent = output
        .parent()
        .ok_or("attention boundary output has no parent")?;
    fs::create_dir_all(parent)?;
    let parent = parent.canonicalize()?;
    Ok(parent.join(
        output
            .file_name()
            .ok_or("attention boundary output has no basename")?,
    ))
}

fn write_artifact_exclusive(path: &Path, document: &Value) -> TestResult {
    if fs::symlink_metadata(path).is_ok() {
        return Err("refusing to overwrite an attention boundary receipt".into());
    }
    let mut payload = serde_json::to_vec_pretty(document)?;
    payload.push(b'\n');
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(&payload)?;
    file.sync_all()?;
    Ok(())
}

fn unix_seconds() -> TestResult<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

#[test]
#[ignore = "requires a remote Qwen2.5-3B Hugging Face eager boundary artifact and CUDA GPU"]
fn qwen3b_p2051_native_attention_boundary_quality_gate() -> TestResult {
    let artifact = load_artifact()?;
    let native = run_native_boundary(&artifact)?;
    let stages = native
        .get("stages")
        .and_then(Value::as_object)
        .ok_or("native boundary stages are missing")?;
    let raw_qk_exact = exact(stages.get("raw_qk_last").ok_or("raw QK metric missing")?)?;
    let scaled_masked_exact = exact(
        stages
            .get("scaled_masked_last")
            .ok_or("scaled/masked metric missing")?,
    )?;
    let probabilities_exact = exact(
        stages
            .get("probabilities_last")
            .ok_or("probability metric missing")?,
    )?;
    let context_exact = exact(stages.get("context_bshd").ok_or("context metric missing")?)?;
    let repeat_exact = native["repeat_execution"]["context_bshd_bf16_exact"] == Value::Bool(true);
    let quality_pass = raw_qk_exact
        && scaled_masked_exact
        && probabilities_exact
        && context_exact
        && repeat_exact
        && native["hot_execution_allocation_stable"] == Value::Bool(true)
        && native["lifecycle_clean_after_close"] == Value::Bool(true);
    let receipt = json!({
        "schema_version": RESULT_SCHEMA,
        "artifact_kind": RESULT_KIND,
        "created_at_unix_seconds": unix_seconds()?,
        "performance_claim_eligible": false,
        "serving_selector_eligible": false,
        "contract": {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "token_count": S,
            "query_head_count": QH,
            "key_value_head_count": KVH,
            "head_size": D,
            "scale": SCALE,
            "input_token_ids_le_u32_sha256": INPUT_SHA256,
            "cache_free": true,
            "hf_eager_source_sha256": HF_EAGER_SOURCE_SHA256,
        },
        "hf_boundary_artifact": {
            "root": artifact.root,
            "metadata_path": artifact.metadata_path,
            "metadata_sha256": artifact.metadata_sha256,
        },
        "native": native,
        "quality_gate": {
            "raw_qk_last_bf16_exact": raw_qk_exact,
            "scaled_masked_last_bf16_exact": scaled_masked_exact,
            "probabilities_last_bf16_exact": probabilities_exact,
            "context_bshd_bf16_exact": context_exact,
            "repeat_context_bshd_bf16_exact": repeat_exact,
            "quality_pass": quality_pass,
            "corrected_cache_on_eligible": false,
            "serving_selector_eligible": false,
        },
    });
    let output = output_path_for(OUTPUT_ENV)?;
    write_artifact_exclusive(&output, &receipt)?;
    println!("{MARKER_PREFIX}{}", serde_json::to_string(&receipt)?);
    if !quality_pass {
        return Err(
            "native P2051 attention boundary quality gate failed; candidate design and serving promotion remain blocked"
                .into(),
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a remote Qwen2.5-3B Hugging Face eager boundary artifact and CUDA GPU"]
fn qwen3b_p2051_cublaslt_attention_probe_quality_gate() -> TestResult {
    let artifact = load_artifact()?;
    let candidate = run_hf_eager_qwen_p2051_probe(&artifact)?;
    let stages = candidate
        .get("stages")
        .and_then(Value::as_object)
        .ok_or("cuBLASLt probe stages are missing")?;
    let probabilities_exact = exact(
        stages
            .get("probabilities_last")
            .ok_or("cuBLASLt probability metric missing")?,
    )?;
    let context_exact = exact(
        stages
            .get("context_bshd")
            .ok_or("cuBLASLt context metric missing")?,
    )?;
    let repeat_exact =
        candidate["repeat_execution"]["context_bshd_bf16_exact"] == Value::Bool(true);
    let quality_pass = probabilities_exact
        && context_exact
        && repeat_exact
        && candidate["hot_execution_allocation_stable"] == Value::Bool(true)
        && candidate["lifecycle_clean_after_close"] == Value::Bool(true);
    let receipt = json!({
        "schema_version": CUBLASLT_RESULT_SCHEMA,
        "artifact_kind": CUBLASLT_RESULT_KIND,
        "created_at_unix_seconds": unix_seconds()?,
        "performance_claim_eligible": false,
        "serving_selector_eligible": false,
        "contract": {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "token_count": S,
            "query_head_count": QH,
            "key_value_head_count": KVH,
            "head_size": D,
            "scale": SCALE,
            "input_token_ids_le_u32_sha256": INPUT_SHA256,
            "cache_free": true,
            "hf_eager_source_sha256": HF_EAGER_SOURCE_SHA256,
            "candidate_opt_in_only": true,
        },
        "hf_boundary_artifact": {
            "root": artifact.root,
            "metadata_path": artifact.metadata_path,
            "metadata_sha256": artifact.metadata_sha256,
        },
        "candidate": candidate,
        "quality_gate": {
            "probabilities_last_bf16_exact": probabilities_exact,
            "context_bshd_bf16_exact": context_exact,
            "repeat_context_bshd_bf16_exact": repeat_exact,
            "quality_pass": quality_pass,
            "corrected_cache_on_eligible": false,
            "serving_selector_eligible": false,
        },
    });
    let output = output_path_for(CUBLASLT_OUTPUT_ENV)?;
    write_artifact_exclusive(&output, &receipt)?;
    println!("{MARKER_PREFIX}{}", serde_json::to_string(&receipt)?);
    if !quality_pass {
        return Err(
            "Qwen P2051 cuBLASLt attention probe quality gate failed; cache-on and serving promotion remain blocked"
                .into(),
        );
    }
    Ok(())
}
