//! Remote-only Qwen2.5-3B P2051 direct projection-boundary qualification.
//!
//! Hugging Face produces the immutable, cache-free full-sequence BF16 oracle
//! offline. This test consumes that artifact from Rust and runs the direct
//! CUDA operators only: a strict standalone Q GEMM, strict BF16 row-bias, and
//! cuBLASLt BIAS-epilogue Q/K/V GEMMs. It does not start Python, alter the
//! serving selector, or make a serving-performance claim.

#![cfg(feature = "cuda")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use riley_cuda::{
    BiasGemmParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaGemmAlgorithmMetadata, CudaGemmConfig, CudaPinnedHostBuffer, CudaPreparedBiasEpilogueGemm,
    CudaPreparedGemm, CudaRuntime, CudaStream, GemmParams, RowBiasAddInPlaceParams,
    row_bias_add_in_place,
};
use riley_model::{
    DecoderWeight, LoadLimits, LoadedModel, ModelArchitecture, ModelFamily, WeightSlot,
};
use riley_tensor::DType;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1024 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 1_048_576;
const MAX_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
const UPLOAD_STAGING_BYTES: u64 = 16 * 1024 * 1024;

const PROJECTION_SCHEMA_VERSION: &str = "riley.qwen3b-hf-eager-p2051-projection-trace.v1";
const PROJECTION_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-projection-trace";
const PROJECTION_TRACE_ID: &str = "qwen3b-p2051-cache-off-layer0-projection-boundary-v1";
const RESULT_SCHEMA_VERSION: &str = "riley.qwen3b-p2051-projection-boundary-comparison.v1";
const RESULT_ARTIFACT_KIND: &str = "qwen2.5-3b-riley-p2051-direct-projection-boundary-comparison";

const QWEN3B_MODEL_ID: &str = "Qwen/Qwen2.5-3B-Instruct";
const QWEN3B_REVISION: &str = "aa8e72537993ba99e69dfaafa59ed015b17504d1";
const QWEN3B_WORKLOAD_SCHEMA: &str = "riley.n06a-d128-serving-workload.v1";
const QWEN3B_WORKLOAD_CASE: &str = "qwen3b-c8-p2048-o128";
const QWEN3B_WORKLOAD_SHA256: &str =
    "7a0a8fec31d45e397e1ec57335fa1c9de2d3da7daa9a32e9002a62763c05261e";
const QWEN3B_PROMPT_TOKEN_SHA256: &str =
    "56619bc156fb385345c12e523c71604fa9ef8ad3c52d9c913d0f6ff09f1c1fd9";
const QWEN3B_PROMPT_TOKEN_ID: u32 = 3_409;
const QWEN3B_PROMPT_TOKEN_COUNT: usize = 2_048;
const TEACHER_PREFIX_TOKEN_COUNT: usize = 3;
const CONTEXT_TOKEN_COUNT: usize = QWEN3B_PROMPT_TOKEN_COUNT + TEACHER_PREFIX_TOKEN_COUNT;
const QWEN3B_LAYER_COUNT: usize = 36;
const QWEN3B_HIDDEN_SIZE: usize = 2_048;
const QWEN3B_KEY_VALUE_HEADS: usize = 2;
const QWEN3B_HEAD_DIMENSION: usize = 128;
const QWEN3B_VOCABULARY_SIZE: usize = 151_936;
const EXPECTED_SOURCE_ARCHITECTURE: &str = "Qwen2ForCausalLM";
const CHECKPOINT_RECEIPT_FILENAME: &str = "riley-checkpoint.json";

const TEACHER_ARTIFACT_SCHEMA: &str = "riley.qwen3b-hf-eager-teacher-forced-generation.v1";
const TEACHER_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2048-teacher-forced-generation";
const TEACHER_CACHE_OFF_SIDECAR_KEY: &str = "teacher_forced/logits";

const PROJECTION_TENSOR_NAMES: [&str; 5] = [
    "layer0.input_norm",
    "layer0.q_proj.unbiased_linear",
    "layer0.q_proj",
    "layer0.k_proj",
    "layer0.v_proj",
];

const EXPECTED_SOURCE_RECORDS: [(&str, &str); 13] = [
    (
        "qwen_serving_oracle",
        "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    ),
    (
        "teacher_forced_generation",
        "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    ),
    (
        "p2051_layer_stage_contract",
        "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    ),
    (
        "stage_trace_support",
        "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    ),
    (
        "p2051_projection_trace",
        "tools/python/reference/riley_reference/qwen3b_p2051_projection_trace.py",
    ),
    (
        "hf_calibration",
        "tools/python/reference/riley_reference/hf_calibration.py",
    ),
    (
        "rust_projection_consumer",
        "crates/riley-runtime/tests/qwen3b_p2051_bias_epilogue_gpu.rs",
    ),
    (
        "rust_projection_consumer_package",
        "crates/riley-runtime/Cargo.toml",
    ),
    (
        "rust_bias_epilogue_wrapper",
        "crates/riley-cuda/src/gemm.rs",
    ),
    ("rust_bias_epilogue_kernel", "kernels/src/gemm.cu"),
    ("reference_project", "tools/python/reference/pyproject.toml"),
    ("reference_lock", "tools/python/reference/uv.lock"),
    ("reference_python", "tools/python/reference/.python-version"),
];

#[derive(Clone, Copy, Debug)]
struct Projection {
    label: &'static str,
    output_name: &'static str,
    output_width: usize,
    weight: DecoderWeight,
    bias: DecoderWeight,
}

const PROJECTIONS: [Projection; 3] = [
    Projection {
        label: "q_proj",
        output_name: "layer0.q_proj",
        output_width: QWEN3B_HIDDEN_SIZE,
        weight: DecoderWeight::QueryWeight,
        bias: DecoderWeight::QueryBias,
    },
    Projection {
        label: "k_proj",
        output_name: "layer0.k_proj",
        output_width: QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
        weight: DecoderWeight::KeyWeight,
        bias: DecoderWeight::KeyBias,
    },
    Projection {
        label: "v_proj",
        output_name: "layer0.v_proj",
        output_width: QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
        weight: DecoderWeight::ValueWeight,
        bias: DecoderWeight::ValueBias,
    },
];

#[derive(Debug)]
struct Workload {
    prompt_token_ids: Vec<u32>,
}

#[derive(Debug)]
struct TeacherPrefix {
    artifact_sha256: String,
    cache_off_sidecar_sha256: String,
    full_teacher_token_ids_sha256: String,
    token_ids: Vec<u32>,
}

#[derive(Debug)]
struct HfProjectionArtifact {
    manifest_path: PathBuf,
    manifest_sha256: String,
    sidecar_path: PathBuf,
    sidecar_sha256: String,
    input_token_ids_le_sha256: String,
    teacher_artifact_sha256: String,
    teacher_sidecar_sha256: String,
    teacher_full_token_ids_sha256: String,
    teacher_prefix_token_ids: Vec<u32>,
    checkpoint_receipt_filename: String,
    checkpoint_receipt_sha256: String,
    tensors: BTreeMap<String, Vec<u8>>,
}

#[derive(Debug)]
struct ProjectionHostTensors {
    weight: Vec<u8>,
    bias: Vec<u8>,
    weight_provenance: Value,
    bias_provenance: Value,
}

#[derive(Debug)]
struct ExecutionReceipt {
    device: Value,
    projections: Map<String, Value>,
    endpoint_count: u64,
    exact_endpoint_count: u64,
    first_non_exact_endpoint: Option<String>,
    raw_q_shadow_bf16_exact: bool,
    strict_row_bias_q_bf16_exact: bool,
    fused_actual_endpoint_exact_count: u64,
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

fn sha256_file(path: &Path) -> TestResult<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1_048_576];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex_encode(&digest.finalize()))
}

fn regular_file(path: &Path, label: &str) -> TestResult<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label}: {path:?}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file").into());
    }
    Ok(path.canonicalize()?)
}

fn regular_directory(path: &Path, label: &str) -> TestResult<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label}: {path:?}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(format!("{label} must be a regular non-symlink directory").into());
    }
    Ok(path.canonicalize()?)
}

fn require_exact_fields(value: &Value, expected: &[&str], label: &str) -> TestResult {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    if actual != expected {
        return Err(
            format!("{label} fields differ: actual={actual:?} expected={expected:?}").into(),
        );
    }
    Ok(())
}

fn json_sha256(value: &Value, label: &str) -> TestResult<String> {
    let text = value
        .as_str()
        .ok_or_else(|| format!("{label} must be a SHA-256 string"))?;
    if text.len() != 64
        || !text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!("{label} must be 64 lowercase hexadecimal characters").into());
    }
    Ok(text.to_owned())
}

fn json_git_revision(value: &Value, label: &str) -> TestResult<String> {
    let text = value
        .as_str()
        .ok_or_else(|| format!("{label} must be a Git revision string"))?;
    if text.len() != 40
        || !text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!("{label} must be 40 lowercase hexadecimal characters").into());
    }
    Ok(text.to_owned())
}

fn json_u32(value: &Value, label: &str) -> TestResult<u32> {
    value
        .as_u64()
        .ok_or_else(|| format!("{label} must be an unsigned integer"))
        .and_then(|value| {
            u32::try_from(value).map_err(|error| format!("{label} does not fit u32: {error}"))
        })
        .map_err(Into::into)
}

fn json_u32_array(value: &Value, label: &str) -> TestResult<Vec<u32>> {
    value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?
        .iter()
        .map(|value| json_u32(value, label))
        .collect()
}

fn token_ids_sha256(token_ids: &[u32]) -> String {
    let bytes: Vec<_> = token_ids
        .iter()
        .flat_map(|token_id| token_id.to_le_bytes())
        .collect();
    sha256_hex(&bytes)
}

fn expected_shape(name: &str) -> TestResult<Vec<u64>> {
    let sequence = u64::try_from(CONTEXT_TOKEN_COUNT)?;
    let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE)?;
    let key_value = u64::try_from(QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION)?;
    match name {
        "layer0.input_norm" | "layer0.q_proj.unbiased_linear" | "layer0.q_proj" => {
            Ok(vec![sequence, hidden])
        }
        "layer0.k_proj" | "layer0.v_proj" => Ok(vec![sequence, key_value]),
        _ => Err(format!("unknown fixed P2051 projection tensor {name}").into()),
    }
}

fn shape_byte_len(shape: &[u64]) -> TestResult<usize> {
    let elements = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or("projection tensor shape element count overflows")
    })?;
    usize::try_from(
        elements
            .checked_mul(u64::try_from(BF16_BYTES)?)
            .ok_or("projection tensor byte count overflows")?,
    )
    .map_err(Into::into)
}

fn validate_finite_bf16(bytes: &[u8], label: &str) -> TestResult {
    if bytes.len() % BF16_BYTES != 0 {
        return Err(format!("{label} BF16 byte count differs").into());
    }
    for pair in bytes.chunks_exact(BF16_BYTES) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7f80 == 0x7f80 {
            return Err(format!("{label} contains non-finite BF16 values").into());
        }
    }
    Ok(())
}

fn load_workload() -> TestResult<Workload> {
    let path = regular_file(
        &required_path("RILEY_QWEN_SERVING_WORKLOAD")?,
        "Qwen workload",
    )?;
    let payload = fs::read(&path)?;
    if sha256_hex(&payload) != QWEN3B_WORKLOAD_SHA256 {
        return Err("Qwen workload SHA-256 differs from immutable P2048 case".into());
    }
    let document: Value = serde_json::from_slice(&payload)?;
    require_exact_fields(
        &document,
        &[
            "schema_version",
            "case",
            "model_id",
            "model_revision",
            "prompt",
            "prompt_token_ids",
            "output_token_ids",
            "output_text",
            "finish_reason",
            "sampling",
        ],
        "Qwen workload",
    )?;
    if document["schema_version"].as_str() != Some(QWEN3B_WORKLOAD_SCHEMA)
        || document["case"].as_str() != Some(QWEN3B_WORKLOAD_CASE)
        || document["model_id"].as_str() != Some(QWEN3B_MODEL_ID)
        || document["model_revision"].as_str() != Some(QWEN3B_REVISION)
    {
        return Err("Qwen workload identity differs".into());
    }
    let prompt_token_ids = json_u32_array(&document["prompt_token_ids"], "prompt token IDs")?;
    if prompt_token_ids.len() != QWEN3B_PROMPT_TOKEN_COUNT
        || !prompt_token_ids
            .iter()
            .all(|&token| token == QWEN3B_PROMPT_TOKEN_ID)
        || token_ids_sha256(&prompt_token_ids) != QWEN3B_PROMPT_TOKEN_SHA256
    {
        return Err("Qwen workload prompt IDs differ".into());
    }
    Ok(Workload { prompt_token_ids })
}

fn parse_teacher_cache_off_sidecar(path: &Path) -> TestResult {
    let source = regular_file(path, "HF cache-off sidecar")?;
    let bytes = fs::read(source)?;
    if bytes.len() < 8 {
        return Err("HF cache-off sidecar is too short".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES || 8 + header_len > bytes.len()
    {
        return Err("HF cache-off sidecar header differs".into());
    }
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
    let tensor = header
        .get(TEACHER_CACHE_OFF_SIDECAR_KEY)
        .and_then(Value::as_object)
        .ok_or("HF cache-off sidecar tensor is missing")?;
    if tensor.get("dtype").and_then(Value::as_str) != Some("BF16")
        || tensor.get("shape") != Some(&json!([128, QWEN3B_VOCABULARY_SIZE]))
    {
        return Err("HF cache-off sidecar tensor contract differs".into());
    }
    Ok(())
}

fn load_teacher_prefix() -> TestResult<TeacherPrefix> {
    let artifact_path = regular_file(
        &required_path("RILEY_QWEN_HF_TEACHER_FORCED_ORACLE")?,
        "HF teacher-forced artifact",
    )?;
    let cache_off_sidecar_path = regular_file(
        &required_path("RILEY_QWEN_HF_CACHE_OFF_SIDECAR")?,
        "HF cache-off sidecar",
    )?;
    let payload = fs::read(&artifact_path)?;
    let artifact_sha256 = sha256_hex(&payload);
    let document: Value = serde_json::from_slice(&payload)?;
    if document["schema_version"].as_str() != Some(TEACHER_ARTIFACT_SCHEMA)
        || document["artifact_kind"].as_str() != Some(TEACHER_ARTIFACT_KIND)
        || document["performance_claim_eligible"].as_bool() != Some(false)
    {
        return Err("HF teacher-forced artifact identity differs".into());
    }
    let generation = document["generation"]
        .as_object()
        .ok_or("HF teacher-forced generation is missing")?;
    let teacher_ids = json_u32_array(
        generation
            .get("teacher_token_ids")
            .ok_or("HF teacher token IDs are missing")?,
        "HF teacher token IDs",
    )?;
    if teacher_ids.len() != 128
        || token_ids_sha256(&teacher_ids)
            != json_sha256(
                generation
                    .get("teacher_token_ids_le_u32_sha256")
                    .ok_or("HF teacher token SHA-256 is missing")?,
                "HF teacher token SHA-256",
            )?
    {
        return Err("HF teacher token contract differs".into());
    }
    let cache_off = generation
        .get("cache_off")
        .and_then(Value::as_object)
        .ok_or("HF cache-off generation is missing")?;
    let sidecar = cache_off
        .get("sidecar")
        .and_then(Value::as_object)
        .ok_or("HF cache-off sidecar record is missing")?;
    let expected_name = cache_off_sidecar_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("HF cache-off sidecar basename is invalid")?;
    if sidecar.get("path").and_then(Value::as_str) != Some(expected_name)
        || sidecar.get("tensor_key").and_then(Value::as_str) != Some(TEACHER_CACHE_OFF_SIDECAR_KEY)
    {
        return Err("HF cache-off sidecar binding differs".into());
    }
    let cache_off_sidecar_sha256 = sha256_file(&cache_off_sidecar_path)?;
    if cache_off_sidecar_sha256
        != json_sha256(
            sidecar
                .get("sha256")
                .ok_or("HF cache-off sidecar SHA-256 is missing")?,
            "HF cache-off sidecar SHA-256",
        )?
    {
        return Err("HF cache-off sidecar SHA-256 differs".into());
    }
    parse_teacher_cache_off_sidecar(&cache_off_sidecar_path)?;
    let token_ids = teacher_ids
        .get(..TEACHER_PREFIX_TOKEN_COUNT)
        .ok_or("HF teacher token prefix is too short")?
        .to_vec();
    Ok(TeacherPrefix {
        artifact_sha256,
        cache_off_sidecar_sha256,
        full_teacher_token_ids_sha256: token_ids_sha256(&teacher_ids),
        token_ids,
    })
}

fn repository_root() -> TestResult<PathBuf> {
    let manifest_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_directory
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| std::io::Error::other("workspace root is unavailable"))?;
    Ok(workspace.canonicalize()?)
}

fn validate_source_record(
    root: &Path,
    value: &Value,
    expected_path: &str,
    label: &str,
) -> TestResult {
    require_exact_fields(value, &["path", "sha256"], label)?;
    let relative = value
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} path is missing"))?;
    let relative_path = Path::new(relative);
    if relative != expected_path
        || relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{label} path differs from the fixed source contract").into());
    }
    let expected_hash = json_sha256(
        value
            .get("sha256")
            .ok_or_else(|| format!("{label} SHA-256 is missing"))?,
        label,
    )?;
    let source = regular_file(&root.join(relative_path), label)?;
    if sha256_file(&source)? != expected_hash {
        return Err(format!("{label} SHA-256 differs from current source").into());
    }
    Ok(())
}

fn expected_rust_consumer() -> Value {
    json!({
        "api": "riley_cuda::CudaPreparedBiasEpilogueGemm",
        "execution": "direct-p2051-bf16-layer0-qkv-projection-qualification",
        "input_context_token_count": CONTEXT_TOKEN_COUNT,
        "input_tensor": "layer0.input_norm",
        "output_tensors": ["layer0.q_proj", "layer0.k_proj", "layer0.v_proj"],
        "shadow_unbiased_linear": {
            "input_tensor": "layer0.input_norm",
            "output_tensor": "layer0.q_proj.unbiased_linear",
            "operator": "torch.nn.functional.linear",
            "projection": "layer0.self_attn.q_proj",
            "bias": false,
            "dtype": "bfloat16",
        },
        "source_path": "crates/riley-runtime/tests/qwen3b_p2051_bias_epilogue_gpu.rs",
        "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
        "tensor_layout": "full-sequence-bf16",
    })
}

fn expected_execution_contract() -> Value {
    json!({
        "attention_implementation": "eager",
        "cache_free": true,
        "dtype": "bfloat16",
        "explicit_attention_mask": true,
        "explicit_input_ids": true,
        "explicit_position_ids": true,
        "inference_mode": true,
        "logits_to_keep": 1,
        "return_dict": true,
        "tf32_enabled": false,
        "use_cache": false,
    })
}

fn parse_projection_sidecar(
    manifest: &Value,
    path: &Path,
) -> TestResult<BTreeMap<String, Vec<u8>>> {
    let source = regular_file(path, "HF P2051 projection sidecar")?;
    let bytes = fs::read(&source)?;
    let sidecar = manifest["sidecar"]
        .as_object()
        .ok_or("HF P2051 projection sidecar record is missing")?;
    let expected_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("HF P2051 projection sidecar basename is invalid")?;
    if sidecar.get("path").and_then(Value::as_str) != Some(expected_name)
        || sidecar.get("format").and_then(Value::as_str) != Some("safetensors")
        || sidecar.get("tensor_count").and_then(Value::as_u64)
            != Some(u64::try_from(PROJECTION_TENSOR_NAMES.len())?)
        || sha256_hex(&bytes)
            != json_sha256(
                sidecar
                    .get("sha256")
                    .ok_or("HF P2051 projection sidecar SHA-256 is missing")?,
                "HF P2051 projection sidecar SHA-256",
            )?
    {
        return Err("HF P2051 projection sidecar binding differs".into());
    }
    if bytes.len() < 8 {
        return Err("HF P2051 projection sidecar is too short".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES || 8 + header_len > bytes.len()
    {
        return Err("HF P2051 projection sidecar header differs".into());
    }
    let data_start = 8_usize
        .checked_add(header_len)
        .ok_or("HF P2051 projection sidecar data offset overflows")?;
    let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
    let header = header
        .as_object()
        .ok_or("HF P2051 projection sidecar header is not an object")?;
    let expected_keys: BTreeSet<_> = PROJECTION_TENSOR_NAMES
        .iter()
        .map(|name| format!("trace/{}", name.replace('.', "/")))
        .collect();
    let actual_keys: BTreeSet<_> = header
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect();
    if actual_keys != expected_keys {
        return Err("HF P2051 projection sidecar tensor key set differs".into());
    }
    let manifest_tensors = manifest["tensors"]
        .as_object()
        .ok_or("HF P2051 projection manifest tensors are missing")?;
    let mut ranges = Vec::with_capacity(PROJECTION_TENSOR_NAMES.len());
    let mut output = BTreeMap::new();
    for name in PROJECTION_TENSOR_NAMES {
        let shape = expected_shape(name)?;
        let key = format!("trace/{}", name.replace('.', "/"));
        let reference = manifest_tensors
            .get(name)
            .ok_or("HF P2051 projection manifest tensor is missing")?;
        require_exact_fields(
            reference,
            &[
                "key",
                "shape",
                "dtype",
                "canonical_byte_order",
                "bf16_le_sha256",
                "bf16_le_bytes",
            ],
            "HF P2051 projection manifest tensor",
        )?;
        if reference.get("key").and_then(Value::as_str) != Some(key.as_str())
            || reference.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || reference
                .get("canonical_byte_order")
                .and_then(Value::as_str)
                != Some("little-endian-u16")
            || reference.get("shape")
                != Some(&Value::Array(
                    shape.iter().copied().map(Value::from).collect(),
                ))
            || reference.get("bf16_le_bytes").and_then(Value::as_u64)
                != Some(u64::try_from(shape_byte_len(&shape)?)?)
        {
            return Err(format!("HF P2051 projection tensor {name} metadata differs").into());
        }
        let expected_sha = json_sha256(
            reference
                .get("bf16_le_sha256")
                .ok_or("HF P2051 projection tensor SHA-256 is missing")?,
            "HF P2051 projection tensor SHA-256",
        )?;
        let entry = header
            .get(&key)
            .ok_or("HF P2051 projection sidecar tensor is missing")?;
        require_exact_fields(
            entry,
            &["dtype", "shape", "data_offsets"],
            "HF P2051 projection sidecar tensor",
        )?;
        if entry.get("dtype").and_then(Value::as_str) != Some("BF16")
            || entry.get("shape")
                != Some(&Value::Array(
                    shape.iter().copied().map(Value::from).collect(),
                ))
        {
            return Err(
                format!("HF P2051 projection sidecar tensor {name} metadata differs").into(),
            );
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("HF P2051 projection sidecar offsets are missing")?;
        if offsets.len() != 2 {
            return Err("HF P2051 projection sidecar offset count differs".into());
        }
        let start = usize::try_from(offsets[0].as_u64().ok_or("HF P2051 offset start")?)?;
        let end = usize::try_from(offsets[1].as_u64().ok_or("HF P2051 offset end")?)?;
        let expected_bytes = shape_byte_len(&shape)?;
        if end < start || end - start != expected_bytes || data_start + end > bytes.len() {
            return Err(format!("HF P2051 projection sidecar tensor {name} range differs").into());
        }
        let raw = bytes[data_start + start..data_start + end].to_vec();
        if sha256_hex(&raw) != expected_sha {
            return Err(format!("HF P2051 projection sidecar tensor {name} hash differs").into());
        }
        validate_finite_bf16(&raw, &format!("HF P2051 projection tensor {name}"))?;
        ranges.push((start, end));
        output.insert(name.to_owned(), raw);
    }
    let mut expected_start = 0_usize;
    ranges.sort_unstable();
    for (start, end) in ranges {
        if start != expected_start {
            return Err("HF P2051 projection sidecar offsets are non-contiguous".into());
        }
        expected_start = end;
    }
    if data_start + expected_start != bytes.len() {
        return Err("HF P2051 projection sidecar has trailing bytes".into());
    }
    Ok(output)
}

fn load_hf_projection_artifact(
    teacher: &TeacherPrefix,
    workload: &Workload,
) -> TestResult<HfProjectionArtifact> {
    let manifest_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_PROJECTION_MANIFEST")?,
        "HF P2051 projection manifest",
    )?;
    let sidecar_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_PROJECTION_SIDECAR")?,
        "HF P2051 projection sidecar",
    )?;
    let payload = fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_hex(&payload);
    let manifest: Value = serde_json::from_slice(&payload)?;
    require_exact_fields(
        &manifest,
        &[
            "schema_version",
            "artifact_kind",
            "trace_id",
            "performance_claim_eligible",
            "created_at",
            "producer",
            "trace_profile",
            "contract",
            "model",
            "provenance",
            "sidecar",
            "tensors",
        ],
        "HF P2051 projection manifest",
    )?;
    if manifest["schema_version"].as_str() != Some(PROJECTION_SCHEMA_VERSION)
        || manifest["artifact_kind"].as_str() != Some(PROJECTION_ARTIFACT_KIND)
        || manifest["trace_id"].as_str() != Some(PROJECTION_TRACE_ID)
        || manifest["performance_claim_eligible"].as_bool() != Some(false)
    {
        return Err("HF P2051 projection manifest identity differs".into());
    }
    let trace_profile = manifest["trace_profile"]
        .as_object()
        .ok_or("HF P2051 projection trace profile is missing")?;
    if trace_profile.get("capture_domain").and_then(Value::as_str)
        != Some("cache-free-p2051-layer0-full-projection-boundaries")
        || trace_profile.get("id").and_then(Value::as_str) != Some(PROJECTION_TRACE_ID)
        || trace_profile.get("tensor_count").and_then(Value::as_u64)
            != Some(u64::try_from(PROJECTION_TENSOR_NAMES.len())?)
        || trace_profile.get("rust_consumer") != Some(&expected_rust_consumer())
    {
        return Err("HF P2051 projection trace profile differs".into());
    }
    require_exact_fields(
        &manifest["model"],
        &[
            "checkpoint_path",
            "checkpoint_receipt_filename",
            "checkpoint_receipt_sha256",
        ],
        "HF P2051 projection model",
    )?;
    let model = manifest["model"]
        .as_object()
        .ok_or("HF P2051 projection model is missing")?;
    if model
        .get("checkpoint_path")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
        || model
            .get("checkpoint_receipt_filename")
            .and_then(Value::as_str)
            != Some(CHECKPOINT_RECEIPT_FILENAME)
    {
        return Err("HF P2051 projection checkpoint receipt identity differs".into());
    }
    let checkpoint_receipt_sha256 = json_sha256(
        model
            .get("checkpoint_receipt_sha256")
            .ok_or("HF P2051 projection checkpoint receipt SHA-256 is missing")?,
        "HF P2051 projection checkpoint receipt SHA-256",
    )?;
    let contract = manifest["contract"]
        .as_object()
        .ok_or("HF P2051 projection contract is missing")?;
    require_exact_fields(
        &manifest["contract"],
        &[
            "model_id",
            "model_revision",
            "workload",
            "execution",
            "input",
        ],
        "HF P2051 projection contract",
    )?;
    if contract.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
        || contract.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
        || contract.get("execution") != Some(&expected_execution_contract())
    {
        return Err("HF P2051 projection execution contract differs".into());
    }
    require_exact_fields(
        contract
            .get("workload")
            .ok_or("HF P2051 workload contract is missing")?,
        &[
            "schema_version",
            "case",
            "source_sha256",
            "prompt_token_count",
            "prompt_token_ids_le_u32_sha256",
        ],
        "HF P2051 projection workload contract",
    )?;
    let workload_contract = contract
        .get("workload")
        .and_then(Value::as_object)
        .ok_or("HF P2051 workload contract is missing")?;
    if workload_contract
        .get("schema_version")
        .and_then(Value::as_str)
        != Some(QWEN3B_WORKLOAD_SCHEMA)
        || workload_contract.get("case").and_then(Value::as_str) != Some(QWEN3B_WORKLOAD_CASE)
        || workload_contract
            .get("source_sha256")
            .and_then(Value::as_str)
            != Some(QWEN3B_WORKLOAD_SHA256)
        || workload_contract
            .get("prompt_token_count")
            .and_then(Value::as_u64)
            != Some(u64::try_from(workload.prompt_token_ids.len())?)
        || workload_contract
            .get("prompt_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(QWEN3B_PROMPT_TOKEN_SHA256)
    {
        return Err("HF P2051 projection workload contract differs".into());
    }
    let input = contract
        .get("input")
        .and_then(Value::as_object)
        .ok_or("HF P2051 projection input contract is missing")?;
    require_exact_fields(
        contract
            .get("input")
            .ok_or("HF P2051 projection input contract is missing")?,
        &[
            "construction",
            "context_token_count",
            "input_token_ids_le_u32_sha256",
            "prompt_token_count",
            "teacher_prefix_token_count",
            "teacher_prefix_token_ids",
            "teacher_prefix_token_ids_le_u32_sha256",
            "teacher_source",
        ],
        "HF P2051 projection input contract",
    )?;
    let expected_input = workload
        .prompt_token_ids
        .iter()
        .copied()
        .chain(teacher.token_ids.iter().copied())
        .collect::<Vec<_>>();
    let input_token_ids_le_sha256 = token_ids_sha256(&expected_input);
    let teacher_prefix_sha256 = token_ids_sha256(&teacher.token_ids);
    if input.get("construction").and_then(Value::as_str)
        != Some("workload.prompt_token_ids+verified_hf_cache_off.teacher_token_ids[:3]")
        || input.get("context_token_count").and_then(Value::as_u64)
            != Some(u64::try_from(CONTEXT_TOKEN_COUNT)?)
        || input.get("prompt_token_count").and_then(Value::as_u64)
            != Some(u64::try_from(QWEN3B_PROMPT_TOKEN_COUNT)?)
        || input
            .get("teacher_prefix_token_count")
            .and_then(Value::as_u64)
            != Some(u64::try_from(TEACHER_PREFIX_TOKEN_COUNT)?)
        || input
            .get("input_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(input_token_ids_le_sha256.as_str())
        || input
            .get("teacher_prefix_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(teacher_prefix_sha256.as_str())
        || json_u32_array(
            input
                .get("teacher_prefix_token_ids")
                .ok_or("HF P2051 teacher prefix is missing")?,
            "HF P2051 teacher prefix",
        )? != teacher.token_ids
    {
        return Err("HF P2051 projection input binding differs".into());
    }
    let teacher_source = input
        .get("teacher_source")
        .and_then(Value::as_object)
        .ok_or("HF P2051 projection teacher source is missing")?;
    require_exact_fields(
        input
            .get("teacher_source")
            .ok_or("HF P2051 projection teacher source is missing")?,
        &[
            "artifact_kind",
            "artifact_path",
            "artifact_schema_version",
            "artifact_sha256",
            "cache_mode",
            "cache_off_sidecar_path",
            "cache_off_sidecar_sha256",
            "cache_off_sidecar_tensor_key",
            "full_teacher_token_count",
            "full_teacher_token_ids_le_u32_sha256",
        ],
        "HF P2051 projection teacher source",
    )?;
    if teacher_source
        .get("artifact_schema_version")
        .and_then(Value::as_str)
        != Some(TEACHER_ARTIFACT_SCHEMA)
        || teacher_source.get("artifact_kind").and_then(Value::as_str)
            != Some(TEACHER_ARTIFACT_KIND)
        || teacher_source
            .get("artifact_sha256")
            .and_then(Value::as_str)
            != Some(teacher.artifact_sha256.as_str())
        || teacher_source.get("cache_mode").and_then(Value::as_str) != Some("cache-off")
        || teacher_source
            .get("cache_off_sidecar_sha256")
            .and_then(Value::as_str)
            != Some(teacher.cache_off_sidecar_sha256.as_str())
        || teacher_source
            .get("full_teacher_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(teacher.full_teacher_token_ids_sha256.as_str())
        || teacher_source
            .get("cache_off_sidecar_tensor_key")
            .and_then(Value::as_str)
            != Some(TEACHER_CACHE_OFF_SIDECAR_KEY)
        || teacher_source
            .get("full_teacher_token_count")
            .and_then(Value::as_u64)
            != Some(128)
        || teacher_source
            .get("artifact_path")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        || teacher_source
            .get("cache_off_sidecar_path")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    {
        return Err("HF P2051 projection teacher source binding differs".into());
    }
    let root = repository_root()?;
    require_exact_fields(
        &manifest["provenance"],
        &["source_repository"],
        "HF P2051 projection provenance",
    )?;
    let provenance = manifest["provenance"]["source_repository"]
        .as_object()
        .ok_or("HF P2051 projection source provenance is missing")?;
    require_exact_fields(
        &manifest["provenance"]["source_repository"],
        &[
            "git_revision",
            "source_dirty",
            "source_status_sha256",
            "sources",
        ],
        "HF P2051 projection source provenance",
    )?;
    if provenance.get("source_dirty").and_then(Value::as_bool) != Some(false) {
        return Err("HF P2051 projection source provenance is dirty".into());
    }
    let _revision = json_git_revision(
        provenance
            .get("git_revision")
            .ok_or("HF P2051 projection Git revision is missing")?,
        "HF P2051 projection Git revision",
    )?;
    let _source_status_sha256 = json_sha256(
        provenance
            .get("source_status_sha256")
            .ok_or("HF P2051 projection source status SHA-256 is missing")?,
        "HF P2051 projection source status SHA-256",
    )?;
    let sources = provenance
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("HF P2051 projection source records are missing")?;
    let expected_source_names: BTreeSet<_> = EXPECTED_SOURCE_RECORDS
        .iter()
        .map(|(name, _)| *name)
        .collect();
    let actual_source_names: BTreeSet<_> = sources.keys().map(String::as_str).collect();
    if actual_source_names != expected_source_names {
        return Err("HF P2051 projection source record set differs".into());
    }
    for (name, path) in EXPECTED_SOURCE_RECORDS {
        validate_source_record(
            &root,
            sources
                .get(name)
                .ok_or("HF P2051 projection source record is missing")?,
            path,
            &format!("HF P2051 projection source {name}"),
        )?;
    }
    let tensors = parse_projection_sidecar(&manifest, &sidecar_path)?;
    let sidecar_sha256 = sha256_file(&sidecar_path)?;
    Ok(HfProjectionArtifact {
        manifest_path,
        manifest_sha256,
        sidecar_path,
        sidecar_sha256,
        input_token_ids_le_sha256,
        teacher_artifact_sha256: teacher.artifact_sha256.clone(),
        teacher_sidecar_sha256: teacher.cache_off_sidecar_sha256.clone(),
        teacher_full_token_ids_sha256: teacher.full_teacher_token_ids_sha256.clone(),
        teacher_prefix_token_ids: teacher.token_ids.clone(),
        checkpoint_receipt_filename: CHECKPOINT_RECEIPT_FILENAME.to_owned(),
        checkpoint_receipt_sha256,
        tensors,
    })
}

fn load_model(hf: &HfProjectionArtifact) -> TestResult<LoadedModel> {
    let checkpoint = regular_directory(
        &required_path("RILEY_QWEN3B_CHECKPOINT")?,
        "Qwen checkpoint",
    )?;
    let receipt = regular_file(
        &checkpoint.join(&hf.checkpoint_receipt_filename),
        "Qwen checkpoint receipt",
    )?;
    if sha256_file(&receipt)? != hf.checkpoint_receipt_sha256 {
        return Err("Qwen checkpoint receipt differs from HF P2051 projection artifact".into());
    }
    let model = LoadedModel::load(
        &checkpoint,
        LoadLimits::default().with_weight_byte_limits(8 * ONE_GIB, 8 * ONE_GIB)?,
    )?;
    let spec = model.spec();
    if model.config().family() != ModelFamily::Qwen2
        || model.provenance().source_model() != QWEN3B_MODEL_ID
        || model.provenance().source_revision() != QWEN3B_REVISION
        || spec.architecture() != ModelArchitecture::Llama
        || spec.source_architecture() != EXPECTED_SOURCE_ARCHITECTURE
        || spec.dtype() != DType::BF16
        || spec.blocks().len() != QWEN3B_LAYER_COUNT
        || spec.embedding().hidden_size() != QWEN3B_HIDDEN_SIZE
        || spec.embedding().vocabulary_size() != QWEN3B_VOCABULARY_SIZE
    {
        return Err("loaded model differs from Qwen2.5-3B projection contract".into());
    }
    Ok(model)
}

fn bf16_le_to_native(bytes: &[u8]) -> TestResult<Vec<u8>> {
    if bytes.len() % BF16_BYTES != 0 {
        return Err("BF16 bytes must have an even length".into());
    }
    #[cfg(target_endian = "little")]
    {
        Ok(bytes.to_vec())
    }
    #[cfg(target_endian = "big")]
    {
        Ok(bytes
            .chunks_exact(BF16_BYTES)
            .flat_map(|pair| [pair[1], pair[0]])
            .collect())
    }
}

fn bf16_native_to_le(bytes: &[u8]) -> TestResult<Vec<u8>> {
    bf16_le_to_native(bytes)
}

fn decode_bf16_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn metrics(hf: &[u8], riley: &[u8]) -> TestResult<Value> {
    if hf.len() != riley.len() || hf.len() % BF16_BYTES != 0 {
        return Err("projection pair byte lengths differ".into());
    }
    let mut unequal = 0_u64;
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut hf_square = 0.0_f64;
    let mut riley_square = 0.0_f64;
    for (expected, actual) in hf
        .chunks_exact(BF16_BYTES)
        .zip(riley.chunks_exact(BF16_BYTES))
    {
        if expected != actual {
            unequal += 1;
        }
        let expected = f64::from(decode_bf16_le(expected));
        let actual = f64::from(decode_bf16_le(actual));
        if !expected.is_finite() || !actual.is_finite() {
            return Err("projection pair contains non-finite BF16 values".into());
        }
        let absolute = (expected - actual).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        dot += expected * actual;
        hf_square += expected * expected;
        riley_square += actual * actual;
    }
    let elements = u64::try_from(hf.len() / BF16_BYTES)?;
    let cosine = if hf_square == 0.0 || riley_square == 0.0 {
        Value::Null
    } else {
        json!(dot / (hf_square.sqrt() * riley_square.sqrt()))
    };
    Ok(json!({
        "element_count": elements,
        "bf16_exact": unequal == 0,
        "unequal_element_count": unequal,
        "max_abs_bf16_as_f32": max_abs,
        "mean_abs_bf16_as_f32": sum_abs / elements as f64,
        "cosine_similarity": cosine,
        "hf_bf16_le_sha256": sha256_hex(hf),
        "riley_bf16_le_sha256": sha256_hex(riley),
    }))
}

fn projection_host_tensors(
    model: &LoadedModel,
    projection: Projection,
) -> TestResult<ProjectionHostTensors> {
    let weight_slot = WeightSlot::Decoder {
        layer: 0,
        parameter: projection.weight,
    };
    let bias_slot = WeightSlot::Decoder {
        layer: 0,
        parameter: projection.bias,
    };
    let weight = model.weights().view(weight_slot)?;
    let bias = model.weights().view(bias_slot)?;
    let weight_view = weight.view();
    let bias_view = bias.view();
    let expected_weight_shape = [projection.output_width, QWEN3B_HIDDEN_SIZE];
    let expected_bias_shape = [projection.output_width];
    if weight_view.dtype() != DType::BF16
        || weight_view.shape().dimensions() != expected_weight_shape.as_slice()
        || weight_view.storage().len()
            != projection
                .output_width
                .checked_mul(QWEN3B_HIDDEN_SIZE)
                .and_then(|elements| elements.checked_mul(BF16_BYTES))
                .ok_or("Qwen projection weight byte count overflows")?
    {
        return Err(format!("{} weight checkpoint binding differs", projection.label).into());
    }
    if bias_view.dtype() != DType::BF16
        || bias_view.shape().dimensions() != expected_bias_shape.as_slice()
        || bias_view.storage().len()
            != projection
                .output_width
                .checked_mul(BF16_BYTES)
                .ok_or("Qwen projection bias byte count overflows")?
    {
        return Err(format!("{} bias checkpoint binding differs", projection.label).into());
    }
    let weight_bytes = weight_view.storage();
    let bias_bytes = bias_view.storage();
    Ok(ProjectionHostTensors {
        weight: bf16_le_to_native(weight_bytes)?,
        bias: bf16_le_to_native(bias_bytes)?,
        weight_provenance: json!({
            "slot": weight_slot.name(),
            "source_tensor": weight.source().tensor_name(),
            "source_shard": weight.source().shard_path().display().to_string(),
            "shape": weight_view.shape().dimensions(),
            "raw_checkpoint_storage_sha256": sha256_hex(weight_bytes),
        }),
        bias_provenance: json!({
            "slot": bias_slot.name(),
            "source_tensor": bias.source().tensor_name(),
            "source_shard": bias.source().shard_path().display().to_string(),
            "shape": bias_view.shape().dimensions(),
            "raw_checkpoint_storage_sha256": sha256_hex(bias_bytes),
        }),
    })
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

fn execute_strict(
    plan: &mut CudaPreparedGemm,
    config: CudaGemmConfig,
    metadata: CudaGemmAlgorithmMetadata,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    stream: &mut CudaStream,
) -> TestResult {
    let mut params = GemmParams {
        input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
        weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
        workspace: workspace_span(workspace, metadata.workspace_bytes())?,
    };
    plan.execute(&mut params, stream)?;
    Ok(())
}

fn execute_bias_epilogue(
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

fn execute_row_bias(
    config: CudaGemmConfig,
    output: &mut CudaDeviceBuffer,
    bias: &CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let mut params = RowBiasAddInPlaceParams {
        matrix: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
        bias: CudaBufferSpan::new(bias, CudaDType::BF16, 0, config.bias_bytes())?,
        row_count: config.m(),
        column_count: config.n(),
    };
    row_bias_add_in_place(&mut params, stream)?;
    Ok(())
}

fn metadata_json(metadata: CudaGemmAlgorithmMetadata) -> Value {
    json!({
        "backend_id": metadata.backend_id(),
        "algorithm_id": metadata.algorithm_id(),
        "tile_id": metadata.tile_id(),
        "stages_id": metadata.stages_id(),
        "split_k": metadata.split_k(),
        "reduction_scheme": metadata.reduction_scheme(),
        "workspace_bytes": metadata.workspace_bytes(),
        "deterministic": metadata.deterministic(),
        "compute_capability": metadata.compute_capability(),
        "runtime_version": metadata.runtime_version(),
        "cublaslt_version": metadata.cublaslt_version(),
    })
}

fn validate_metadata(
    metadata: CudaGemmAlgorithmMetadata,
    config: CudaGemmConfig,
    compute_capability: (u32, u32),
    label: &str,
) -> TestResult {
    if metadata.backend_id() != CudaGemmAlgorithmMetadata::CUBLASLT_BACKEND_ID
        || !metadata.deterministic()
        || metadata.dimensions() != (config.m(), config.n(), config.k())
        || metadata.compute_capability() != compute_capability
        || metadata.workspace_bytes() > config.max_workspace_bytes()
        || metadata.split_k() > 1
        || metadata.reduction_scheme() != 0
    {
        return Err(format!("{label} selected an ineligible CUDA GEMM algorithm").into());
    }
    Ok(())
}

fn record_endpoint(
    metrics: &Value,
    label: &str,
    endpoint_count: &mut u64,
    exact_endpoint_count: &mut u64,
    first_non_exact_endpoint: &mut Option<String>,
) -> TestResult {
    let exact = metrics
        .get("bf16_exact")
        .and_then(Value::as_bool)
        .ok_or("projection metric lacks bf16_exact")?;
    *endpoint_count += 1;
    if exact {
        *exact_endpoint_count += 1;
    } else if first_non_exact_endpoint.is_none() {
        *first_non_exact_endpoint = Some(label.to_owned());
    }
    Ok(())
}

fn bf16_exact(metrics: &Value, label: &str) -> TestResult<bool> {
    metrics
        .get("bf16_exact")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} metric lacks bf16_exact"))
        .map_err(Into::into)
}

fn close_context(stream: CudaStream, context: CudaContext) -> TestResult {
    let mut failures = Vec::new();
    if let Err(error) = context.synchronize() {
        failures.push(format!("CUDA context synchronize failed: {error}"));
    }
    match context.allocation_stats() {
        Ok(stats) if stats.is_zero() => {}
        Ok(stats) => failures.push(format!(
            "CUDA allocation accounting is non-zero after direct qualification: {stats:?}"
        )),
        Err(error) => failures.push(format!("CUDA allocation accounting failed: {error}")),
    }
    if let Err(error) = stream.close() {
        failures.push(format!("CUDA stream close failed: {error}"));
    }
    if let Err(error) = context.close() {
        failures.push(format!("CUDA context close failed: {error}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; ").into())
    }
}

fn run_direct_qualification(
    model: &LoadedModel,
    hf: &HfProjectionArtifact,
) -> TestResult<ExecutionReceipt> {
    let input_hf_le = hf
        .tensors
        .get("layer0.input_norm")
        .ok_or("HF P2051 input_norm tensor is missing")?;
    if input_hf_le.len() != shape_byte_len(&expected_shape("layer0.input_norm")?)? {
        return Err("HF P2051 input_norm byte count differs".into());
    }
    let input_native = bf16_le_to_native(input_hf_le)?;
    let runtime = CudaRuntime::initialize()?;
    if runtime.device_count() == 0 {
        return Err("remote GPU runner has no CUDA device".into());
    }
    let gpu = runtime.device(0)?;
    let properties = gpu.properties().clone();
    let context = gpu.create_context()?;
    let mut stream = context.create_stream()?;
    let device = json!({
        "ordinal": properties.ordinal(),
        "name": properties.name(),
        "total_memory_bytes": properties.total_memory_bytes(),
        "compute_capability": properties.compute_capability(),
        "multiprocessor_count": properties.multiprocessor_count(),
        "driver_version": properties.driver_version(),
        "runtime_version": properties.runtime_version(),
    });

    let result = (|| -> TestResult<ExecutionReceipt> {
        let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
        let input = upload(&context, &mut stream, &mut staging, &input_native)?;
        let mut projections = Map::new();
        let mut endpoint_count = 0_u64;
        let mut exact_endpoint_count = 0_u64;
        let mut first_non_exact_endpoint = None;
        let mut raw_q_shadow_bf16_exact = false;
        let mut strict_row_bias_q_bf16_exact = false;
        let mut fused_actual_endpoint_exact_count = 0_u64;

        for projection in PROJECTIONS {
            let config = CudaGemmConfig::new(
                u64::try_from(CONTEXT_TOKEN_COUNT)?,
                u64::try_from(projection.output_width)?,
                u64::try_from(QWEN3B_HIDDEN_SIZE)?,
                MAX_WORKSPACE_BYTES,
            )?;
            let expected_hf_le = hf
                .tensors
                .get(projection.output_name)
                .ok_or("HF P2051 actual projection tensor is missing")?;
            if expected_hf_le.len() != shape_byte_len(&expected_shape(projection.output_name)?)? {
                return Err(
                    format!("HF P2051 {} byte count differs", projection.output_name).into(),
                );
            }
            let host = projection_host_tensors(model, projection)?;
            let weight = upload(&context, &mut stream, &mut staging, &host.weight)?;
            let bias = upload(&context, &mut stream, &mut staging, &host.bias)?;
            let mut record = Map::new();
            record.insert("output_tensor".to_owned(), json!(projection.output_name));
            record.insert(
                "dimensions".to_owned(),
                json!({"m": config.m(), "n": config.n(), "k": config.k()}),
            );
            record.insert("weight".to_owned(), host.weight_provenance);
            record.insert("bias".to_owned(), host.bias_provenance);

            if projection.label == "q_proj" {
                let shadow_hf_le = hf
                    .tensors
                    .get("layer0.q_proj.unbiased_linear")
                    .ok_or("HF P2051 Q unbiased shadow tensor is missing")?;
                let mut strict_plan = context.prepare_gemm(config)?;
                let strict_metadata = strict_plan.algorithm_metadata();
                if strict_plan.config() != config {
                    return Err("strict Q plan configuration differs from direct contract".into());
                }
                validate_metadata(
                    strict_metadata,
                    config,
                    properties.compute_capability(),
                    "strict Q",
                )?;
                let mut strict_output = context.allocate_device_buffer(config.output_bytes())?;
                let mut strict_workspace = if strict_metadata.workspace_bytes() == 0 {
                    None
                } else {
                    Some(context.allocate_device_buffer(strict_metadata.workspace_bytes())?)
                };
                let allocations_before = context.allocation_stats()?;

                execute_strict(
                    &mut strict_plan,
                    config,
                    strict_metadata,
                    &input,
                    &weight,
                    &mut strict_output,
                    strict_workspace.as_mut(),
                    &mut stream,
                )?;
                let raw_first_native = download(&context, &mut stream, &mut strict_output)?;
                execute_strict(
                    &mut strict_plan,
                    config,
                    strict_metadata,
                    &input,
                    &weight,
                    &mut strict_output,
                    strict_workspace.as_mut(),
                    &mut stream,
                )?;
                let raw_repeated_native = download(&context, &mut stream, &mut strict_output)?;
                if raw_first_native != raw_repeated_native {
                    return Err("strict Q raw GEMM output changed on repetition".into());
                }
                execute_strict(
                    &mut strict_plan,
                    config,
                    strict_metadata,
                    &input,
                    &weight,
                    &mut strict_output,
                    strict_workspace.as_mut(),
                    &mut stream,
                )?;
                execute_row_bias(config, &mut strict_output, &bias, &mut stream)?;
                let biased_first_native = download(&context, &mut stream, &mut strict_output)?;
                execute_strict(
                    &mut strict_plan,
                    config,
                    strict_metadata,
                    &input,
                    &weight,
                    &mut strict_output,
                    strict_workspace.as_mut(),
                    &mut stream,
                )?;
                execute_row_bias(config, &mut strict_output, &bias, &mut stream)?;
                let biased_repeated_native = download(&context, &mut stream, &mut strict_output)?;
                if biased_first_native != biased_repeated_native {
                    return Err("strict Q row-bias output changed on repetition".into());
                }
                if context.allocation_stats()? != allocations_before {
                    return Err("strict Q execution changed allocation accounting".into());
                }
                let raw_metrics = metrics(shadow_hf_le, &bf16_native_to_le(&raw_first_native)?)?;
                let biased_metrics =
                    metrics(expected_hf_le, &bf16_native_to_le(&biased_first_native)?)?;
                raw_q_shadow_bf16_exact = bf16_exact(&raw_metrics, "strict Q raw")?;
                strict_row_bias_q_bf16_exact = bf16_exact(&biased_metrics, "strict Q row-bias")?;
                record_endpoint(
                    &raw_metrics,
                    "q_proj.strict_gemm_no_bias",
                    &mut endpoint_count,
                    &mut exact_endpoint_count,
                    &mut first_non_exact_endpoint,
                )?;
                record_endpoint(
                    &biased_metrics,
                    "q_proj.strict_gemm_plus_row_bias",
                    &mut endpoint_count,
                    &mut exact_endpoint_count,
                    &mut first_non_exact_endpoint,
                )?;
                record.insert(
                    "strict_gemm_no_bias".to_owned(),
                    json!({
                        "algorithm": metadata_json(strict_metadata),
                        "repeated_output_bf16_exact": true,
                        "allocation_accounting_unchanged": true,
                        "hf_shadow_unbiased_linear_comparison": raw_metrics,
                    }),
                );
                record.insert(
                    "strict_gemm_plus_row_bias".to_owned(),
                    json!({
                        "row_bias_operator": "riley_cuda::row_bias_add_in_place",
                        "row_bias_contract": "BF16 expand-to-F32 add then BF16 round-to-nearest-even",
                        "algorithm": metadata_json(strict_metadata),
                        "repeated_output_bf16_exact": true,
                        "allocation_accounting_unchanged": true,
                        "hf_module_output_comparison": biased_metrics,
                    }),
                );
                strict_plan.close()?;
                strict_output.close()?;
                if let Some(workspace) = strict_workspace {
                    workspace.close()?;
                }
            }

            let mut fused_plan = context.prepare_bias_epilogue_gemm(
                config,
                CudaBufferSpan::new(&bias, CudaDType::BF16, 0, config.bias_bytes())?,
            )?;
            let fused_metadata = fused_plan.algorithm_metadata();
            if fused_plan.config() != config {
                return Err(format!(
                    "fused {} plan configuration differs from direct contract",
                    projection.label
                )
                .into());
            }
            validate_metadata(
                fused_metadata,
                config,
                properties.compute_capability(),
                &format!("fused {}", projection.label),
            )?;
            let mut fused_output = context.allocate_device_buffer(config.output_bytes())?;
            let mut fused_workspace = if fused_metadata.workspace_bytes() == 0 {
                None
            } else {
                Some(context.allocate_device_buffer(fused_metadata.workspace_bytes())?)
            };
            let allocations_before = context.allocation_stats()?;
            execute_bias_epilogue(
                &mut fused_plan,
                config,
                fused_metadata,
                &input,
                &weight,
                &bias,
                &mut fused_output,
                fused_workspace.as_mut(),
                &mut stream,
            )?;
            let fused_first_native = download(&context, &mut stream, &mut fused_output)?;
            execute_bias_epilogue(
                &mut fused_plan,
                config,
                fused_metadata,
                &input,
                &weight,
                &bias,
                &mut fused_output,
                fused_workspace.as_mut(),
                &mut stream,
            )?;
            let fused_repeated_native = download(&context, &mut stream, &mut fused_output)?;
            if fused_first_native != fused_repeated_native {
                return Err(
                    format!("fused {} output changed on repetition", projection.label).into(),
                );
            }
            if context.allocation_stats()? != allocations_before {
                return Err(format!(
                    "fused {} execution changed allocation accounting",
                    projection.label
                )
                .into());
            }
            let fused_metrics = metrics(expected_hf_le, &bf16_native_to_le(&fused_first_native)?)?;
            if bf16_exact(&fused_metrics, "fused projection")? {
                fused_actual_endpoint_exact_count += 1;
            }
            record_endpoint(
                &fused_metrics,
                &format!("{}.cublaslt_bias_epilogue", projection.label),
                &mut endpoint_count,
                &mut exact_endpoint_count,
                &mut first_non_exact_endpoint,
            )?;
            println!(
                "qwen3b-p2051-projection projection={} m={} n={} k={} fused_bf16_exact={} unequal_elements={} workspace_bytes={} algorithm_id={} tile_id={} stages_id={} split_k={} reduction_scheme={} repeated_output_bf16_exact=true allocation_accounting_unchanged=true performance_claim_eligible=false",
                projection.label,
                config.m(),
                config.n(),
                config.k(),
                fused_metrics["bf16_exact"],
                fused_metrics["unequal_element_count"],
                fused_metadata.workspace_bytes(),
                fused_metadata.algorithm_id(),
                fused_metadata.tile_id(),
                fused_metadata.stages_id(),
                fused_metadata.split_k(),
                fused_metadata.reduction_scheme(),
            );
            record.insert(
                "cublaslt_bias_epilogue".to_owned(),
                json!({
                    "algorithm": metadata_json(fused_metadata),
                    "repeated_output_bf16_exact": true,
                    "allocation_accounting_unchanged": true,
                    "hf_module_output_comparison": fused_metrics,
                }),
            );
            fused_plan.close()?;
            fused_output.close()?;
            if let Some(workspace) = fused_workspace {
                workspace.close()?;
            }
            weight.close()?;
            bias.close()?;
            projections.insert(projection.label.to_owned(), Value::Object(record));
        }

        input.close()?;
        staging.close()?;
        if !context.allocation_stats()?.is_zero() {
            return Err(
                "direct P2051 qualification left a CUDA allocation before context close".into(),
            );
        }
        Ok(ExecutionReceipt {
            device,
            projections,
            endpoint_count,
            exact_endpoint_count,
            first_non_exact_endpoint,
            raw_q_shadow_bf16_exact,
            strict_row_bias_q_bf16_exact,
            fused_actual_endpoint_exact_count,
        })
    })();
    let cleanup = close_context(stream, context);
    match (result, cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => Err(format!(
            "direct P2051 projection qualification failed: {run_error}; cleanup also failed: {cleanup_error}"
        )
        .into()),
    }
}

fn validate_output_destination(path: &Path) -> TestResult<PathBuf> {
    if !path.is_absolute() || path.extension().and_then(|value| value.to_str()) != Some("json") {
        return Err("P2051 projection output must be an absolute .json path".into());
    }
    let root = repository_root()?;
    let parent_input = path
        .parent()
        .ok_or("P2051 projection output has no parent")?;
    fs::create_dir_all(parent_input)?;
    let parent = parent_input.canonicalize()?;
    let output = parent.join(
        path.file_name()
            .ok_or("P2051 projection output has no basename")?,
    );
    if output == root || output.starts_with(&root) {
        return Err("P2051 projection output must be outside the repository".into());
    }
    if fs::symlink_metadata(&output).is_ok() {
        return Err("refusing to overwrite existing P2051 projection artifact".into());
    }
    Ok(output)
}

fn write_artifact_exclusive(path: &Path, document: &Value) -> TestResult {
    let output = validate_output_destination(path)?;
    let mut payload = serde_json::to_vec_pretty(document)?;
    payload.push(b'\n');
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    file.write_all(&payload)?;
    file.sync_all()?;
    Ok(())
}

fn unix_seconds() -> TestResult<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

#[test]
#[ignore = "remote-only Qwen2.5-3B P2051 direct Q/K/V projection-boundary qualifier"]
fn qwen3b_p2051_direct_projection_boundary_qualifies_strict_and_fused_qkv() -> TestResult {
    let output = required_path("RILEY_QWEN3B_P2051_PROJECTION_OUTPUT")?;
    validate_output_destination(&output)?;
    let workload = load_workload()?;
    let teacher = load_teacher_prefix()?;
    let hf = load_hf_projection_artifact(&teacher, &workload)?;
    let input = workload
        .prompt_token_ids
        .iter()
        .copied()
        .chain(teacher.token_ids.iter().copied())
        .collect::<Vec<_>>();
    if input.len() != CONTEXT_TOKEN_COUNT
        || token_ids_sha256(&input) != hf.input_token_ids_le_sha256
    {
        return Err("P2051 input does not bind the HF projection artifact".into());
    }
    let model = load_model(&hf)?;
    let execution = run_direct_qualification(&model, &hf)?;
    let all_direct_endpoints_bf16_exact =
        execution.exact_endpoint_count == execution.endpoint_count;
    let fused_actual_qkv_bf16_exact =
        execution.fused_actual_endpoint_exact_count == u64::try_from(PROJECTIONS.len())?;
    let projection_boundary_candidate_eligible =
        execution.raw_q_shadow_bf16_exact && fused_actual_qkv_bf16_exact;
    let exact_endpoint_count = execution.exact_endpoint_count;
    let receipt = json!({
        "schema_version": RESULT_SCHEMA_VERSION,
        "artifact_kind": RESULT_ARTIFACT_KIND,
        "performance_claim_eligible": false,
        "created_at_unix_seconds": unix_seconds()?,
        "hf_projection_trace": {
            "manifest_path": hf.manifest_path,
            "manifest_sha256": hf.manifest_sha256,
            "sidecar_path": hf.sidecar_path,
            "sidecar_sha256": hf.sidecar_sha256,
        },
        "contract": {
            "trace_id": PROJECTION_TRACE_ID,
            "model_id": QWEN3B_MODEL_ID,
            "model_revision": QWEN3B_REVISION,
            "workload_case": QWEN3B_WORKLOAD_CASE,
            "workload_sha256": QWEN3B_WORKLOAD_SHA256,
            "prompt_token_count": QWEN3B_PROMPT_TOKEN_COUNT,
            "teacher_prefix_token_count": TEACHER_PREFIX_TOKEN_COUNT,
            "teacher_prefix_token_ids": hf.teacher_prefix_token_ids,
            "input_token_count": CONTEXT_TOKEN_COUNT,
            "input_token_ids_le_u32_sha256": hf.input_token_ids_le_sha256,
            "teacher_artifact_sha256": hf.teacher_artifact_sha256,
            "teacher_cache_off_sidecar_sha256": hf.teacher_sidecar_sha256,
            "teacher_full_token_ids_le_u32_sha256": hf.teacher_full_token_ids_sha256,
            "checkpoint_receipt_filename": hf.checkpoint_receipt_filename,
            "checkpoint_receipt_sha256": hf.checkpoint_receipt_sha256,
            "input": "HF eager cache-free layer0.input_norm full BF16 sidecar",
            "q_unbiased_shadow": "HF torch.nn.functional.linear(input, q_weight, bias=None) full BF16 sidecar",
            "strict_operator": "cuBLASLt BF16 GEMM with F32 compute followed by riley_cuda::row_bias_add_in_place",
            "fused_operator": "cuBLASLt BF16 BIAS epilogue with F32 compute",
            "reduction_policy": "strict-no-split-v1",
            "cuda_graph": false,
            "python_in_hot_path": false,
            "serving_selector_changed": false,
        },
        "device": execution.device,
        "summary": {
            "direct_endpoint_count": execution.endpoint_count,
            "bf16_exact_direct_endpoint_count": execution.exact_endpoint_count,
            "first_non_exact_direct_endpoint": execution.first_non_exact_endpoint,
            "all_direct_endpoints_bf16_exact": all_direct_endpoints_bf16_exact,
            "raw_q_shadow_bf16_exact": execution.raw_q_shadow_bf16_exact,
            "strict_row_bias_q_bf16_exact": execution.strict_row_bias_q_bf16_exact,
            "fused_actual_qkv_bf16_exact_count": execution.fused_actual_endpoint_exact_count,
            "fused_actual_qkv_bf16_exact": fused_actual_qkv_bf16_exact,
            "projection_boundary_candidate_eligible": projection_boundary_candidate_eligible,
            "serving_selector_changed": false,
        },
        "projections": execution.projections,
    });
    // Write a create-only receipt before returning a numerical mismatch so the
    // next diagnosis has the exact algorithm and endpoint evidence.
    write_artifact_exclusive(&output, &receipt)?;
    if !projection_boundary_candidate_eligible {
        return Err(format!(
            "P2051 raw-Q or fused-Q/K/V projection candidate differs; first direct mismatch is {}",
            receipt["summary"]["first_non_exact_direct_endpoint"]
                .as_str()
                .unwrap_or("an unspecified endpoint")
        )
        .into());
    }
    println!(
        "QWEN3B_P2051_PROJECTION trace_id={} exact_endpoints={} projection_boundary_candidate_eligible=true strict_row_bias_q_bf16_exact={} performance_claim_eligible=false",
        PROJECTION_TRACE_ID,
        exact_endpoint_count,
        receipt["summary"]["strict_row_bias_q_bf16_exact"],
    );
    Ok(())
}
