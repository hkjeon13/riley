//! Remote-only Qwen2.5-3B P2051 selected-layer numerical discriminator.
//!
//! Hugging Face writes the paired BF16 sidecar offline.  This Rust-only test
//! replays the same cache-free P2051 input through the candidate execution
//! stack and compares every final-token boundary inside one selected decoder
//! layer.  It is diagnostic evidence, never a serving performance result.

#![cfg(all(feature = "cuda", feature = "cuda-cublas-gemm-probe"))]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use riley_model::{LoadLimits, LoadedModel, ModelArchitecture, ModelFamily};
use riley_runtime::llama::{
    LlamaLastTokenLayerStage, LlamaProjectionBiasMode, PreparedLlamaForward,
    PreparedLlamaForwardConfig,
};
use riley_runtime::{CudaContext, CudaRuntime, CudaStream};
use riley_tensor::DType;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1024 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 1_048_576;
const UPLOAD_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const IO_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const HF_COMPAT_GEMM_WORKSPACE_CAP_BYTES: u64 = 1024 * 1024;
const REFERENCE_ATTENTION_BUDGET_BYTES: u64 = 1_342_177_280;

const WINDOW_SCHEMA_VERSION: &str = "riley.qwen3b-hf-eager-p2051-cache-off-layer-window-trace.v1";
const WINDOW_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-layer-window-trace";
const WINDOW_TRACE_ID: &str = "qwen3b-p2051-cache-off-last-token-layer-window-v1";
const WINDOW_IMPLEMENTATION_ID: &str = "riley-python-qwen3b-hf-eager-p2051-layer-window-trace-v1";
const RESULT_SCHEMA_VERSION: &str =
    "riley.qwen3b-p2051-hf-compatible-cache-free-layer-window-comparison.v1";
const RESULT_ARTIFACT_KIND: &str =
    "qwen2.5-3b-riley-p2051-hf-compatible-cache-free-layer-window-comparison";
const MARKER_PREFIX: &str = "RILEY_QWEN3B_P2051_LAYER_WINDOW=";

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
const LAST_TOKEN_ROW_INDEX: usize = CONTEXT_TOKEN_COUNT - 1;
const QWEN3B_LAYER_COUNT: usize = 36;
const QWEN3B_HIDDEN_SIZE: usize = 2_048;
const QWEN3B_INTERMEDIATE_SIZE: usize = 11_008;
const QWEN3B_QUERY_HEADS: usize = 16;
const QWEN3B_KEY_VALUE_HEADS: usize = 2;
const QWEN3B_HEAD_DIMENSION: usize = 128;
const QWEN3B_VOCABULARY_SIZE: usize = 151_936;
const EXPECTED_SOURCE_ARCHITECTURE: &str = "Qwen2ForCausalLM";
const TEACHER_ARTIFACT_SCHEMA: &str = "riley.qwen3b-hf-eager-teacher-forced-generation.v1";
const TEACHER_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2048-teacher-forced-generation";
const TEACHER_CACHE_OFF_SIDECAR_KEY: &str = "teacher_forced/logits";
const CHECKPOINT_RECEIPT_FILENAME: &str = "riley-checkpoint.json";

const HF_EAGER_QWEN_P2051_ATTENTION_BACKEND_ID: &str =
    "riley.cuda.hf-eager-cublaslt-qwen-p2051-probe.bf16";
const HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID: &str =
    "hf-eager-qwen-p2051-direct-cublas-probe-v1";
const HF_COMPATIBLE_BIAS_BACKEND_ID: &str = "hf-compatible-bias-epilogue-probe-v1";

const WINDOW_SOURCE_RECORDS: [(&str, &str); 13] = [
    (
        "qwen_serving_oracle",
        "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    ),
    (
        "teacher_forced_generation",
        "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    ),
    (
        "stage_trace_support",
        "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    ),
    (
        "p2051_layer_stage_contract",
        "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    ),
    (
        "p2051_layer_window_trace",
        "tools/python/reference/riley_reference/qwen3b_p2051_layer_window_trace.py",
    ),
    (
        "hf_calibration",
        "tools/python/reference/riley_reference/hf_calibration.py",
    ),
    (
        "rust_layer_window_forward_api",
        "crates/riley-runtime/src/llama/forward.rs",
    ),
    (
        "rust_layer_window_module",
        "crates/riley-runtime/src/llama/mod.rs",
    ),
    (
        "rust_layer_window_consumer",
        "crates/riley-runtime/tests/qwen3b_p2051_layer_window_gpu.rs",
    ),
    (
        "rust_layer_window_package",
        "crates/riley-runtime/Cargo.toml",
    ),
    ("reference_project", "tools/python/reference/pyproject.toml"),
    ("reference_lock", "tools/python/reference/uv.lock"),
    ("reference_python", "tools/python/reference/.python-version"),
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

#[derive(Clone, Debug)]
struct StageSpec {
    name: String,
    shape: Vec<u64>,
    stage: LlamaLastTokenLayerStage,
}

#[derive(Debug)]
struct HfWindowArtifact {
    manifest_path: PathBuf,
    manifest_sha256: String,
    sidecar_path: PathBuf,
    sidecar_sha256: String,
    layer_index: usize,
    input_token_ids_le_sha256: String,
    checkpoint_receipt_filename: String,
    checkpoint_receipt_sha256: String,
    tensors: BTreeMap<String, Vec<u8>>,
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

fn stage_specs(layer_index: usize) -> TestResult<Vec<StageSpec>> {
    if layer_index >= QWEN3B_LAYER_COUNT {
        return Err("window layer index is outside Qwen2.5-3B".into());
    }
    let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE)?;
    let intermediate = u64::try_from(QWEN3B_INTERMEDIATE_SIZE)?;
    let key_value_width = u64::try_from(QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION)?;
    let query_heads = u64::try_from(QWEN3B_QUERY_HEADS)?;
    let key_value_heads = u64::try_from(QWEN3B_KEY_VALUE_HEADS)?;
    let head_dimension = u64::try_from(QWEN3B_HEAD_DIMENSION)?;
    let mut specs = Vec::with_capacity(LlamaLastTokenLayerStage::ALL.len());
    for stage in LlamaLastTokenLayerStage::ALL {
        let shape = match stage {
            LlamaLastTokenLayerStage::KeyProjection | LlamaLastTokenLayerStage::ValueProjection => {
                vec![key_value_width]
            }
            LlamaLastTokenLayerStage::QueryRotary => vec![query_heads, head_dimension],
            LlamaLastTokenLayerStage::KeyRotary => vec![key_value_heads, head_dimension],
            LlamaLastTokenLayerStage::GateProjection
            | LlamaLastTokenLayerStage::UpProjection
            | LlamaLastTokenLayerStage::Gated => vec![intermediate],
            _ => vec![hidden],
        };
        specs.push(StageSpec {
            name: format!("layer{layer_index}.{}.last", stage.name()),
            shape,
            stage,
        });
    }
    Ok(specs)
}

fn shape_byte_len(shape: &[u64]) -> TestResult<usize> {
    let elements = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or("stage tensor shape element count overflows")
    })?;
    usize::try_from(
        elements
            .checked_mul(u64::try_from(BF16_BYTES)?)
            .ok_or("stage tensor byte count overflows")?,
    )
    .map_err(Into::into)
}

fn repository_root() -> TestResult<PathBuf> {
    let manifest_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_directory
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| std::io::Error::other("workspace root is unavailable"))?;
    Ok(workspace.canonicalize()?)
}

fn validate_source_record(root: &Path, value: &Value, label: &str) -> TestResult {
    let record = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let relative = record
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} path is missing"))?;
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{label} path is not a repository-relative regular path").into());
    }
    let expected_hash = json_sha256(
        record
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

fn validate_finite_bf16(bytes: &[u8], label: &str) -> TestResult {
    if bytes.len() % BF16_BYTES != 0 {
        return Err(format!("{label} byte count differs").into());
    }
    for pair in bytes.chunks_exact(BF16_BYTES) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7f80 == 0x7f80 {
            return Err(format!("{label} contains non-finite BF16").into());
        }
    }
    Ok(())
}

fn parse_window_sidecar(
    manifest: &Value,
    path: &Path,
    specs: &[StageSpec],
) -> TestResult<BTreeMap<String, Vec<u8>>> {
    let source = regular_file(path, "HF P2051 layer-window sidecar")?;
    let bytes = fs::read(&source)?;
    let sidecar = manifest["sidecar"]
        .as_object()
        .ok_or("HF P2051 layer-window sidecar record is missing")?;
    let expected_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("HF P2051 layer-window sidecar basename is invalid")?;
    if sidecar.get("path").and_then(Value::as_str) != Some(expected_name)
        || sidecar.get("format").and_then(Value::as_str) != Some("safetensors")
        || sidecar.get("tensor_count").and_then(Value::as_u64) != Some(u64::try_from(specs.len())?)
        || sha256_hex(&bytes)
            != json_sha256(
                sidecar
                    .get("sha256")
                    .ok_or("HF P2051 layer-window sidecar SHA-256 is missing")?,
                "HF P2051 layer-window sidecar SHA-256",
            )?
    {
        return Err("HF P2051 layer-window sidecar binding differs".into());
    }
    if bytes.len() < 8 {
        return Err("HF P2051 layer-window sidecar is too short".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES || 8 + header_len > bytes.len()
    {
        return Err("HF P2051 layer-window sidecar header differs".into());
    }
    let data_start = 8_usize
        .checked_add(header_len)
        .ok_or("HF P2051 layer-window sidecar data offset overflows")?;
    let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
    let header = header
        .as_object()
        .ok_or("HF P2051 layer-window sidecar header is not an object")?;
    let expected_keys: BTreeSet<_> = specs
        .iter()
        .map(|spec| format!("trace/{}", spec.name.replace('.', "/")))
        .collect();
    let actual_keys: BTreeSet<_> = header
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect();
    if actual_keys != expected_keys {
        return Err("HF P2051 layer-window sidecar tensor key set differs".into());
    }
    let manifest_tensors = manifest["tensors"]
        .as_object()
        .ok_or("HF P2051 layer-window manifest tensors are missing")?;
    let mut ranges = Vec::with_capacity(specs.len());
    let mut output = BTreeMap::new();
    for spec in specs {
        let reference = manifest_tensors
            .get(&spec.name)
            .and_then(Value::as_object)
            .ok_or("HF P2051 layer-window manifest tensor is missing")?;
        let key = format!("trace/{}", spec.name.replace('.', "/"));
        if reference.get("key").and_then(Value::as_str) != Some(key.as_str())
            || reference.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || reference
                .get("canonical_byte_order")
                .and_then(Value::as_str)
                != Some("little-endian-u16")
            || reference["shape"]
                .as_array()
                .ok_or("HF P2051 layer-window tensor shape is missing")?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or("HF P2051 stage shape must be unsigned")
                })
                .collect::<Result<Vec<_>, _>>()?
                != spec.shape
            || reference.get("bf16_le_bytes").and_then(Value::as_u64)
                != Some(u64::try_from(shape_byte_len(&spec.shape)?)?)
        {
            return Err(format!(
                "HF P2051 layer-window tensor {} metadata differs",
                spec.name
            )
            .into());
        }
        let entry = header
            .get(&key)
            .and_then(Value::as_object)
            .ok_or("HF P2051 layer-window sidecar tensor is missing")?;
        if entry.get("dtype").and_then(Value::as_str) != Some("BF16")
            || entry.get("shape")
                != Some(&Value::Array(
                    spec.shape.iter().copied().map(Value::from).collect(),
                ))
        {
            return Err(format!(
                "HF P2051 layer-window sidecar tensor {} metadata differs",
                spec.name
            )
            .into());
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("HF P2051 layer-window sidecar offsets are missing")?;
        if offsets.len() != 2 {
            return Err("HF P2051 layer-window sidecar offset count differs".into());
        }
        let start = usize::try_from(offsets[0].as_u64().ok_or("HF P2051 offset start")?)?;
        let end = usize::try_from(offsets[1].as_u64().ok_or("HF P2051 offset end")?)?;
        let expected_bytes = shape_byte_len(&spec.shape)?;
        if end < start || end - start != expected_bytes || data_start + end > bytes.len() {
            return Err(format!("HF P2051 layer-window tensor {} range differs", spec.name).into());
        }
        let raw = bytes[data_start + start..data_start + end].to_vec();
        if sha256_hex(&raw)
            != json_sha256(
                reference
                    .get("bf16_le_sha256")
                    .ok_or("HF P2051 layer-window tensor SHA-256 is missing")?,
                "HF P2051 layer-window tensor SHA-256",
            )?
        {
            return Err(format!("HF P2051 layer-window tensor {} hash differs", spec.name).into());
        }
        validate_finite_bf16(&raw, &format!("HF P2051 layer-window tensor {}", spec.name))?;
        ranges.push((start, end));
        output.insert(spec.name.clone(), raw);
    }
    let mut expected_start = 0_usize;
    ranges.sort_unstable();
    for (start, end) in ranges {
        if start != expected_start {
            return Err("HF P2051 layer-window sidecar offsets are non-contiguous".into());
        }
        expected_start = end;
    }
    if data_start + expected_start != bytes.len() {
        return Err("HF P2051 layer-window sidecar has trailing bytes".into());
    }
    Ok(output)
}

fn window_rust_consumer(layer_index: usize) -> Value {
    json!({
        "api": "riley_runtime::llama::PreparedLlamaForward::prepare_last_token_layer_stage_trace+execute_last_token_layer_stage_traced",
        "attention_backend": HF_EAGER_QWEN_P2051_ATTENTION_BACKEND_ID,
        "cache": false,
        "input_context_token_count": CONTEXT_TOKEN_COUNT,
        "last_token_row_index": LAST_TOKEN_ROW_INDEX,
        "layer_index": layer_index,
        "mlp_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID,
        "output_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID,
        "projection_bias_backend": HF_COMPATIBLE_BIAS_BACKEND_ID,
        "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
        "trace_row_layout": "last-token-row-only",
    })
}

fn load_hf_window_artifact(
    teacher: &TeacherPrefix,
    workload: &Workload,
) -> TestResult<HfWindowArtifact> {
    let manifest_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_LAYER_WINDOW_MANIFEST")?,
        "HF P2051 layer-window manifest",
    )?;
    let sidecar_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_LAYER_WINDOW_SIDECAR")?,
        "HF P2051 layer-window sidecar",
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
        "HF P2051 layer-window manifest",
    )?;
    if manifest["schema_version"].as_str() != Some(WINDOW_SCHEMA_VERSION)
        || manifest["artifact_kind"].as_str() != Some(WINDOW_ARTIFACT_KIND)
        || manifest["trace_id"].as_str() != Some(WINDOW_TRACE_ID)
        || manifest["performance_claim_eligible"].as_bool() != Some(false)
    {
        return Err("HF P2051 layer-window manifest identity differs".into());
    }
    require_exact_fields(
        &manifest["trace_profile"],
        &[
            "capture_domain",
            "id",
            "layer_index",
            "tensor_names",
            "tensor_count",
            "rust_consumer",
        ],
        "HF P2051 layer-window trace profile",
    )?;
    let profile = manifest["trace_profile"]
        .as_object()
        .ok_or("HF P2051 layer-window trace profile is missing")?;
    let layer_index = usize::try_from(
        profile
            .get("layer_index")
            .and_then(Value::as_u64)
            .ok_or("HF P2051 layer-window layer index is missing")?,
    )?;
    let specs = stage_specs(layer_index)?;
    let expected_names = Value::Array(specs.iter().map(|spec| json!(spec.name)).collect());
    if profile.get("capture_domain").and_then(Value::as_str)
        != Some("cache-free-p2051-selected-layer-last-token-boundaries")
        || profile.get("id").and_then(Value::as_str) != Some(WINDOW_TRACE_ID)
        || profile.get("tensor_names") != Some(&expected_names)
        || profile.get("tensor_count").and_then(Value::as_u64) != Some(u64::try_from(specs.len())?)
        || profile.get("rust_consumer") != Some(&window_rust_consumer(layer_index))
    {
        return Err("HF P2051 layer-window trace profile differs".into());
    }
    let producer = manifest["producer"]
        .as_object()
        .ok_or("HF P2051 layer-window producer is missing")?;
    if producer.get("implementation_id").and_then(Value::as_str) != Some(WINDOW_IMPLEMENTATION_ID) {
        return Err("HF P2051 layer-window producer identity differs".into());
    }
    for field in [
        "runtime_dependency_class",
        "torch_version",
        "transformers_version",
    ] {
        if producer
            .get(field)
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err("HF P2051 layer-window producer metadata differs".into());
        }
    }
    require_exact_fields(
        &manifest["model"],
        &[
            "checkpoint_path",
            "checkpoint_receipt_filename",
            "checkpoint_receipt_sha256",
        ],
        "HF P2051 layer-window model",
    )?;
    let model = manifest["model"]
        .as_object()
        .ok_or("HF P2051 layer-window model is missing")?;
    if model
        .get("checkpoint_path")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
        || model
            .get("checkpoint_receipt_filename")
            .and_then(Value::as_str)
            != Some(CHECKPOINT_RECEIPT_FILENAME)
    {
        return Err("HF P2051 layer-window checkpoint receipt identity differs".into());
    }
    let checkpoint_receipt_sha256 = json_sha256(
        model
            .get("checkpoint_receipt_sha256")
            .ok_or("HF P2051 layer-window checkpoint receipt SHA-256 is missing")?,
        "HF P2051 layer-window checkpoint receipt SHA-256",
    )?;
    require_exact_fields(
        &manifest["contract"],
        &[
            "model_id",
            "model_revision",
            "workload",
            "execution",
            "input",
        ],
        "HF P2051 layer-window contract",
    )?;
    let contract = manifest["contract"]
        .as_object()
        .ok_or("HF P2051 layer-window contract is missing")?;
    let expected_execution = json!({
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
    });
    if contract.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
        || contract.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
        || contract.get("execution") != Some(&expected_execution)
    {
        return Err("HF P2051 layer-window execution contract differs".into());
    }
    let workload_contract = contract
        .get("workload")
        .and_then(Value::as_object)
        .ok_or("HF P2051 layer-window workload contract is missing")?;
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
        return Err("HF P2051 layer-window workload contract differs".into());
    }
    let input = contract
        .get("input")
        .and_then(Value::as_object)
        .ok_or("HF P2051 layer-window input contract is missing")?;
    let expected_input = workload
        .prompt_token_ids
        .iter()
        .copied()
        .chain(teacher.token_ids.iter().copied())
        .collect::<Vec<_>>();
    let input_token_ids_le_sha256 = token_ids_sha256(&expected_input);
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
        || json_u32_array(
            input
                .get("teacher_prefix_token_ids")
                .ok_or("HF P2051 layer-window teacher prefix is missing")?,
            "HF P2051 layer-window teacher prefix",
        )? != teacher.token_ids
    {
        return Err("HF P2051 layer-window input binding differs".into());
    }
    let teacher_source = input
        .get("teacher_source")
        .and_then(Value::as_object)
        .ok_or("HF P2051 layer-window teacher source is missing")?;
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
    {
        return Err("HF P2051 layer-window teacher source binding differs".into());
    }
    require_exact_fields(
        &manifest["provenance"],
        &["source_repository"],
        "HF P2051 layer-window provenance",
    )?;
    let provenance = manifest["provenance"]["source_repository"]
        .as_object()
        .ok_or("HF P2051 layer-window source provenance is missing")?;
    require_exact_fields(
        &manifest["provenance"]["source_repository"],
        &[
            "git_revision",
            "source_dirty",
            "source_status_sha256",
            "sources",
        ],
        "HF P2051 layer-window source provenance",
    )?;
    if provenance.get("source_dirty").and_then(Value::as_bool) != Some(false) {
        return Err("HF P2051 layer-window source provenance is dirty".into());
    }
    let _revision = provenance
        .get("git_revision")
        .and_then(Value::as_str)
        .filter(|revision| {
            revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        .ok_or("HF P2051 layer-window Git revision differs")?;
    let _status_sha256 = json_sha256(
        provenance
            .get("source_status_sha256")
            .ok_or("HF P2051 layer-window source status SHA-256 is missing")?,
        "HF P2051 layer-window source status SHA-256",
    )?;
    let sources = provenance
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("HF P2051 layer-window source records are missing")?;
    let expected_source_names: BTreeSet<_> = WINDOW_SOURCE_RECORDS
        .iter()
        .map(|(name, _)| *name)
        .collect();
    let actual_source_names: BTreeSet<_> = sources.keys().map(String::as_str).collect();
    if actual_source_names != expected_source_names {
        return Err("HF P2051 layer-window source record names differ".into());
    }
    let root = repository_root()?;
    for (name, expected_path) in WINDOW_SOURCE_RECORDS {
        let record = sources
            .get(name)
            .ok_or("HF P2051 layer-window source record is missing")?;
        if record.get("path").and_then(Value::as_str) != Some(expected_path) {
            return Err("HF P2051 layer-window source record path differs".into());
        }
        validate_source_record(&root, record, &format!("HF P2051 source {name}"))?;
    }
    let tensors = parse_window_sidecar(&manifest, &sidecar_path, &specs)?;
    let sidecar_sha256 = sha256_file(&sidecar_path)?;
    Ok(HfWindowArtifact {
        manifest_path,
        manifest_sha256,
        sidecar_path,
        sidecar_sha256,
        layer_index,
        input_token_ids_le_sha256,
        checkpoint_receipt_filename: CHECKPOINT_RECEIPT_FILENAME.to_owned(),
        checkpoint_receipt_sha256,
        tensors,
    })
}

fn load_model(hf: &HfWindowArtifact) -> TestResult<LoadedModel> {
    let checkpoint = regular_directory(
        &required_path("RILEY_QWEN3B_CHECKPOINT")?,
        "Qwen checkpoint",
    )?;
    let receipt = regular_file(
        &checkpoint.join(&hf.checkpoint_receipt_filename),
        "Qwen checkpoint receipt",
    )?;
    if sha256_file(&receipt)? != hf.checkpoint_receipt_sha256 {
        return Err("Qwen checkpoint receipt differs from HF P2051 layer-window artifact".into());
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
        return Err("loaded model differs from Qwen2.5-3B layer-window contract".into());
    }
    Ok(model)
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    Ok((context, stream))
}

fn close_resources(
    forward: Option<PreparedLlamaForward>,
    stream: CudaStream,
    context: CudaContext,
) -> TestResult {
    let mut failures = Vec::new();
    if let Some(forward) = forward {
        if let Err(error) = forward.close() {
            failures.push(format!("forward close failed: {error}"));
        }
    }
    if let Err(error) = context.synchronize() {
        failures.push(format!("context synchronize failed: {error}"));
    }
    match context.allocation_stats() {
        Ok(stats) if stats.is_zero() => {}
        Ok(stats) => failures.push(format!(
            "CUDA allocation accounting is non-zero after close: {stats:?}"
        )),
        Err(error) => failures.push(format!("CUDA allocation accounting failed: {error}")),
    }
    if let Err(error) = stream.close() {
        failures.push(format!("stream close failed: {error}"));
    }
    if let Err(error) = context.close() {
        failures.push(format!("context close failed: {error}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; ").into())
    }
}

fn canonical_bf16_le(bytes: &[u8]) -> TestResult<Vec<u8>> {
    if bytes.len() % BF16_BYTES != 0 {
        return Err("BF16 byte count must be even".into());
    }
    #[cfg(target_endian = "little")]
    {
        Ok(bytes.to_vec())
    }
    #[cfg(target_endian = "big")]
    {
        Ok(bytes
            .chunks_exact(BF16_BYTES)
            .flat_map(|chunk| [chunk[1], chunk[0]])
            .collect())
    }
}

fn decode_bf16_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn metrics(hf: &[u8], riley: &[u8]) -> TestResult<Value> {
    if hf.len() != riley.len() || hf.len() % BF16_BYTES != 0 {
        return Err("stage pair byte lengths differ".into());
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
            return Err("stage pair contains non-finite BF16 values".into());
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

fn run_candidate(model: &LoadedModel, input: &[u32], hf: &HfWindowArtifact) -> TestResult<Value> {
    let (context, mut stream) = first_context()?;
    let config = PreparedLlamaForwardConfig::new(
        UPLOAD_STAGING_BYTES,
        IO_STAGING_BYTES,
        HF_COMPAT_GEMM_WORKSPACE_CAP_BYTES,
        REFERENCE_ATTENTION_BUDGET_BYTES,
    )
    .with_projection_bias_mode(LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1)
    .with_hugging_face_eager_qwen_p2051_probe_attention()
    .with_hf_eager_qwen_p2051_direct_cublas_output_projection_probe()
    .with_hf_eager_qwen_p2051_direct_cublas_mlp_projection_probe()
    .with_hugging_face_cuda_qwen_p2051_rope_table_probe();
    let mut forward = match PreparedLlamaForward::prepare(
        model,
        &context,
        &mut stream,
        input.len(),
        config,
    ) {
        Ok(forward) => forward,
        Err(error) => {
            let cleanup = close_resources(None, stream, context);
            return match cleanup {
                Ok(()) => Err(error.into()),
                Err(cleanup_error) => Err(format!(
                    "layer-window forward preparation failed: {error}; cleanup also failed: {cleanup_error}"
                )
                .into()),
            };
        }
    };
    let result = (|| -> TestResult<Value> {
        if forward.projection_bias_mode()
            != LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1
            || forward.attention_selection().implementation_id()
                != HF_EAGER_QWEN_P2051_ATTENTION_BACKEND_ID
            || forward.output_projection_backend_id()
                != HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID
            || forward.mlp_projection_backend_id()
                != HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID
        {
            return Err("layer-window forward selected an unexpected numerical backend".into());
        }
        let specs = stage_specs(hf.layer_index)?;
        let mut trace = forward.prepare_last_token_layer_stage_trace(hf.layer_index)?;
        if trace.layer_index() != hf.layer_index || trace.captured_stage_count() != 0 {
            return Err("selected-layer trace preparation contract differs".into());
        }
        forward.upload_tokens(input, &mut stream)?;
        forward.execute_last_token_layer_stage_traced(&mut stream, &mut trace)?;
        if !trace.is_complete() || trace.captured_stage_count() != u32::try_from(specs.len())? {
            return Err("selected-layer trace did not capture every boundary".into());
        }
        let mut first_execution = BTreeMap::new();
        for spec in &specs {
            let row = canonical_bf16_le(
                trace
                    .tensor(spec.stage)
                    .ok_or("selected-layer trace tensor is missing")?,
            )?;
            if row.len() != shape_byte_len(&spec.shape)? {
                return Err(format!("selected-layer {} byte length differs", spec.name).into());
            }
            first_execution.insert(spec.name.clone(), row);
        }
        forward.execute_last_token_layer_stage_traced(&mut stream, &mut trace)?;
        if !trace.is_complete() || trace.captured_stage_count() != u32::try_from(specs.len())? {
            return Err("repeat selected-layer trace did not capture every boundary".into());
        }
        for spec in &specs {
            let repeated = canonical_bf16_le(
                trace
                    .tensor(spec.stage)
                    .ok_or("repeat selected-layer trace tensor is missing")?,
            )?;
            if first_execution.get(&spec.name) != Some(&repeated) {
                return Err(format!("repeat {} BF16 row differs", spec.name).into());
            }
        }
        let mut stages = Map::new();
        let mut exact_count = 0_u64;
        let mut first_non_exact = None;
        for spec in &specs {
            let hf_bytes = hf
                .tensors
                .get(&spec.name)
                .ok_or("HF layer-window tensor is missing")?;
            let riley_bytes = first_execution
                .get(&spec.name)
                .ok_or("Riley layer-window tensor is missing")?;
            let stage_metrics = metrics(hf_bytes, riley_bytes)?;
            if stage_metrics["bf16_exact"] == true {
                exact_count += 1;
            } else if first_non_exact.is_none() {
                first_non_exact = Some(spec.name.clone());
            }
            stages.insert(spec.name.clone(), stage_metrics);
        }
        let exact = exact_count == u64::try_from(specs.len())? && first_non_exact.is_none();
        Ok(json!({
            "profile_id": "hf-compatible-bias+hf-eager-p2051-attention+direct-cublas-projections-v1",
            "projection_bias_backend": HF_COMPATIBLE_BIAS_BACKEND_ID,
            "attention_backend": HF_EAGER_QWEN_P2051_ATTENTION_BACKEND_ID,
            "output_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID,
            "mlp_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID,
            "use_cache": false,
            "same_scheduler_engine": false,
            "repeat_execution": {
                "reused_prepared_owner": true,
                "all_stage_rows_bf16_exact": true,
            },
            "summary": {
                "stage_count": stages.len(),
                "bf16_exact_stage_count": exact_count,
                "first_non_exact_stage": first_non_exact,
                "cache_off_selected_layer_bf16_exact": exact,
            },
            "stages": stages,
        }))
    })();
    let cleanup = close_resources(Some(forward), stream, context);
    match (result, cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => Err(format!(
            "layer-window execution failed: {run_error}; cleanup also failed: {cleanup_error}"
        )
        .into()),
    }
}

fn write_artifact_exclusive(path: &Path, document: &Value) -> TestResult {
    if !path.is_absolute() || path.extension().and_then(|value| value.to_str()) != Some("json") {
        return Err("layer-window output must be an absolute .json path".into());
    }
    let root = repository_root()?;
    fs::create_dir_all(path.parent().ok_or("layer-window output has no parent")?)?;
    let parent = path
        .parent()
        .ok_or("layer-window output has no parent")?
        .canonicalize()?;
    let output = parent.join(
        path.file_name()
            .ok_or("layer-window output has no basename")?,
    );
    if output == root || output.starts_with(&root) {
        return Err("layer-window output must be outside the repository".into());
    }
    if fs::symlink_metadata(&output).is_ok() {
        return Err("refusing to overwrite existing layer-window output".into());
    }
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
#[ignore = "remote-only Qwen2.5-3B P2051 selected-layer cache-free discriminator"]
fn qwen3b_p2051_hf_compatible_cache_free_selected_layer_quality_gate() -> TestResult {
    let workload = load_workload()?;
    let teacher = load_teacher_prefix()?;
    let hf = load_hf_window_artifact(&teacher, &workload)?;
    let input = workload
        .prompt_token_ids
        .iter()
        .copied()
        .chain(teacher.token_ids.iter().copied())
        .collect::<Vec<_>>();
    if input.len() != CONTEXT_TOKEN_COUNT
        || token_ids_sha256(&input) != hf.input_token_ids_le_sha256
    {
        return Err("P2051 input does not bind the HF layer-window artifact".into());
    }
    let model = load_model(&hf)?;
    let profile = run_candidate(&model, &input, &hf)?;
    let summary = profile
        .get("summary")
        .and_then(Value::as_object)
        .ok_or("layer-window summary is missing")?;
    let cache_off_selected_layer_bf16_exact = summary
        .get("cache_off_selected_layer_bf16_exact")
        .and_then(Value::as_bool)
        .ok_or("layer-window exactness is missing")?;
    let output = required_path("RILEY_QWEN3B_P2051_LAYER_WINDOW_OUTPUT")?;
    let receipt = json!({
        "schema_version": RESULT_SCHEMA_VERSION,
        "artifact_kind": RESULT_ARTIFACT_KIND,
        "performance_claim_eligible": false,
        "created_at_unix_seconds": unix_seconds()?,
        "contract": {
            "model_id": QWEN3B_MODEL_ID,
            "model_revision": QWEN3B_REVISION,
            "input_token_count": CONTEXT_TOKEN_COUNT,
            "input_token_ids_le_u32_sha256": hf.input_token_ids_le_sha256,
            "use_cache": false,
            "same_scheduler_engine": false,
            "layer_index": hf.layer_index,
            "last_token_row_index": LAST_TOKEN_ROW_INDEX,
            "candidate_attention_backend": HF_EAGER_QWEN_P2051_ATTENTION_BACKEND_ID,
            "candidate_output_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID,
            "candidate_mlp_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_PROJECTION_BACKEND_ID,
        },
        "hf_layer_window_artifact": {
            "manifest_path": hf.manifest_path,
            "manifest_sha256": hf.manifest_sha256,
            "sidecar_path": hf.sidecar_path,
            "sidecar_sha256": hf.sidecar_sha256,
        },
        "profile": profile,
        "quality_gate": {
            "required_stage_count": LlamaLastTokenLayerStage::ALL.len(),
            "cache_off_selected_layer_bf16_exact": cache_off_selected_layer_bf16_exact,
            "corrected_cache_on_eligible": false,
            "serving_selector_eligible": false,
            "performance_claim_eligible": false,
        },
    });
    write_artifact_exclusive(&output, &receipt)?;
    println!("{MARKER_PREFIX}{}", serde_json::to_string(&receipt)?);
    if !cache_off_selected_layer_bf16_exact {
        return Err(
            "HF-compatible cache-off selected-layer quality gate failed; receipt was written"
                .into(),
        );
    }
    Ok(())
}
