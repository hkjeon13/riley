//! Remote-only Qwen2.5-3B P2048 QKV explicit staged-bias discriminator.
//!
//! V2 established that Riley's three Q/K/V standalone GEMM outputs are BF16
//! exact to a separately evaluated Hugging Face no-bias endpoint, while the
//! post-bias endpoints differ.  This V2b diagnostic keeps the serving path
//! unchanged and compares those existing Riley post-bias captures with an
//! explicit Hugging Face BF16 -> FP32 bias add -> BF16 round reference.
#![cfg(feature = "cuda")]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use riley_cuda::CudaRuntime;
use riley_model::{LoadLimits, LoadedModel, ModelArchitecture, ModelFamily};
use riley_runtime::llama::{LlamaTracePoint, PreparedLlamaForward, PreparedLlamaForwardConfig};
use riley_tensor::DType;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1024 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 1_048_576;
const TRACE_SCHEMA_VERSION: &str = "1.0.0";
const HF_ARTIFACT_KIND: &str = "qwen3b-hf-eager-layer0-qkv-staged-bias-trace";
const RESULT_ARTIFACT_KIND: &str = "qwen3b-riley-layer0-qkv-staged-bias-comparison";
const TRACE_ID: &str = "qwen3b-p2048-layer0-qkv-staged-bias-v2b";
const HF_IMPLEMENTATION_ID: &str = "hf-transformers-qwen2-hooks-qkv-staged-bias-v2b";
const QWEN3B_MODEL_ID: &str = "Qwen/Qwen2.5-3B-Instruct";
const QWEN3B_REVISION: &str = "aa8e72537993ba99e69dfaafa59ed015b17504d1";
const QWEN3B_WORKLOAD_SHA256: &str =
    "7a0a8fec31d45e397e1ec57335fa1c9de2d3da7daa9a32e9002a62763c05261e";
const QWEN3B_WORKLOAD_SCHEMA_VERSION: &str = "riley.n06a-d128-serving-workload.v1";
const QWEN3B_WORKLOAD_CASE: &str = "qwen3b-c8-p2048-o128";
const QWEN3B_PROMPT_TOKEN_SHA256: &str =
    "56619bc156fb385345c12e523c71604fa9ef8ad3c52d9c913d0f6ff09f1c1fd9";
const QWEN3B_PROMPT_TOKEN_ID: u32 = 3_409;
const QWEN3B_PROMPT_TOKEN_COUNT: usize = 2_048;
const QWEN3B_LAYER_COUNT: usize = 36;
const QWEN3B_HIDDEN_SIZE: usize = 2_048;
const QWEN3B_KEY_VALUE_HEADS: usize = 2;
const QWEN3B_HEAD_DIMENSION: usize = 128;
const QWEN3B_VOCABULARY_SIZE: usize = 151_936;
const EXPECTED_SOURCE_ARCHITECTURE: &str = "Qwen2ForCausalLM";

/// Every sidecar tensor in immutable semantic order.  The first tensor of
/// each trio is an independently evaluated no-bias `F.linear` endpoint.  The
/// second is its explicit staged bias calculation, and the third is the
/// unmodified Hugging Face module-hook output.
const TRACE_TENSORS: [(&str, &str); 9] = [
    (
        "layer0.q_proj.unbiased_linear",
        "trace/layer0/q_proj/unbiased_linear",
    ),
    (
        "layer0.q_proj.staged_bias_bf16",
        "trace/layer0/q_proj/staged_bias_bf16",
    ),
    ("layer0.q_proj", "trace/layer0/q_proj"),
    (
        "layer0.k_proj.unbiased_linear",
        "trace/layer0/k_proj/unbiased_linear",
    ),
    (
        "layer0.k_proj.staged_bias_bf16",
        "trace/layer0/k_proj/staged_bias_bf16",
    ),
    ("layer0.k_proj", "trace/layer0/k_proj"),
    (
        "layer0.v_proj.unbiased_linear",
        "trace/layer0/v_proj/unbiased_linear",
    ),
    (
        "layer0.v_proj.staged_bias_bf16",
        "trace/layer0/v_proj/staged_bias_bf16",
    ),
    ("layer0.v_proj", "trace/layer0/v_proj"),
];

/// Riley already captures both endpoints needed here.  V2b deliberately adds
/// no trace point and changes no serving operator: it compares the existing
/// standalone GEMM and post-bias captures with a more explicit HF reference.
const PROJECTIONS: [(LlamaTracePoint, LlamaTracePoint, &str, &str, &str); 3] = [
    (
        LlamaTracePoint::Layer0QueryProjectionUnbiasedLinear,
        LlamaTracePoint::Layer0QueryProjection,
        "layer0.q_proj.unbiased_linear",
        "layer0.q_proj.staged_bias_bf16",
        "layer0.q_proj",
    ),
    (
        LlamaTracePoint::Layer0KeyProjectionUnbiasedLinear,
        LlamaTracePoint::Layer0KeyProjection,
        "layer0.k_proj.unbiased_linear",
        "layer0.k_proj.staged_bias_bf16",
        "layer0.k_proj",
    ),
    (
        LlamaTracePoint::Layer0ValueProjectionUnbiasedLinear,
        LlamaTracePoint::Layer0ValueProjection,
        "layer0.v_proj.unbiased_linear",
        "layer0.v_proj.staged_bias_bf16",
        "layer0.v_proj",
    ),
];

const SOURCE_PATHS: [(&str, &str); 8] = [
    (
        "qwen_serving_oracle",
        "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    ),
    (
        "stage_trace_support",
        "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    ),
    (
        "qkv_bias_boundary_trace",
        "tools/python/reference/riley_reference/qwen3b_qkv_bias_boundary_trace.py",
    ),
    (
        "qkv_staged_bias_trace",
        "tools/python/reference/riley_reference/qwen3b_qkv_staged_bias_trace.py",
    ),
    (
        "rust_trace_driver",
        "crates/riley-runtime/tests/qwen3b_qkv_staged_bias_gpu.rs",
    ),
    ("reference_project", "tools/python/reference/pyproject.toml"),
    ("reference_lock", "tools/python/reference/uv.lock"),
    ("reference_python", "tools/python/reference/.python-version"),
];

struct TraceInputs {
    checkpoint: PathBuf,
    workload: PathBuf,
    manifest: PathBuf,
    sidecar: PathBuf,
    output: PathBuf,
}

#[derive(Clone, Debug)]
struct SidecarTensor {
    data_start: u64,
    data_end: u64,
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

fn trace_inputs() -> TestResult<TraceInputs> {
    Ok(TraceInputs {
        checkpoint: required_path("RILEY_QWEN3B_CHECKPOINT")?,
        workload: required_path("RILEY_QWEN_SERVING_WORKLOAD")?,
        manifest: required_path("RILEY_QWEN3B_QKV_STAGED_BIAS_TRACE_MANIFEST")?,
        sidecar: required_path("RILEY_QWEN3B_QKV_STAGED_BIAS_TRACE_SIDECAR")?,
        output: required_path("RILEY_QWEN3B_QKV_STAGED_BIAS_TRACE_OUTPUT")?,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().fold(
        String::with_capacity(digest.len() * 2),
        |mut output, byte| {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
            output
        },
    )
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
    let digest = digest.finalize();
    Ok(digest
        .iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
            output
        }))
}

fn regular_file(path: &Path, label: &str) -> TestResult<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label}: {path:?}: {error}"))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file").into());
    }
    Ok(path.canonicalize()?)
}

fn require_object<'a>(value: &'a Value, label: &str) -> TestResult<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object").into())
}

fn require_exact_keys(object: &Map<String, Value>, expected: &[&str], label: &str) -> TestResult {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!("{label} fields differ").into());
    }
    Ok(())
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> TestResult<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string").into())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn required_sha256(object: &Map<String, Value>, field: &str, label: &str) -> TestResult<String> {
    let value = required_string(object, field, label)?;
    if !valid_sha256(value) {
        return Err(format!("{label}.{field} must be lowercase SHA-256").into());
    }
    Ok(value.to_owned())
}

fn valid_git_revision(value: &str) -> bool {
    (40..=64).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn load_prompt_ids(path: &Path) -> TestResult<Vec<u32>> {
    let workload = regular_file(path, "Qwen workload")?;
    let raw = fs::read(&workload)?;
    if sha256_hex(&raw) != QWEN3B_WORKLOAD_SHA256 {
        return Err("Qwen workload SHA-256 differs from the immutable P2048 case".into());
    }
    let document: Value = serde_json::from_slice(&raw)?;
    if document["schema_version"] != QWEN3B_WORKLOAD_SCHEMA_VERSION
        || document["case"] != QWEN3B_WORKLOAD_CASE
        || document["model_id"] != QWEN3B_MODEL_ID
        || document["model_revision"] != QWEN3B_REVISION
    {
        return Err("Qwen workload identity differs".into());
    }
    let values = document["prompt_token_ids"]
        .as_array()
        .ok_or("Qwen workload prompt_token_ids must be an array")?;
    if values.len() != QWEN3B_PROMPT_TOKEN_COUNT {
        return Err("Qwen workload prompt token count differs".into());
    }
    let mut tokens = Vec::with_capacity(values.len());
    for value in values {
        tokens.push(u32::try_from(
            value
                .as_u64()
                .ok_or("Qwen workload prompt token must be unsigned")?,
        )?);
    }
    if tokens.iter().any(|&token| token != QWEN3B_PROMPT_TOKEN_ID) {
        return Err("Qwen workload prompt IDs differ from the fixed repeated token".into());
    }
    let token_bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    if sha256_hex(&token_bytes) != QWEN3B_PROMPT_TOKEN_SHA256 {
        return Err("Qwen workload prompt ID byte hash differs".into());
    }
    Ok(tokens)
}

fn load_qwen3b(checkpoint: &Path) -> TestResult<LoadedModel> {
    let model = LoadedModel::load(
        checkpoint,
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
        return Err("loaded checkpoint differs from Qwen2.5-3B trace contract".into());
    }
    Ok(model)
}

fn expected_shape(name: &str) -> Vec<u64> {
    let sequence = u64::try_from(QWEN3B_PROMPT_TOKEN_COUNT).expect("sequence fits");
    let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE).expect("hidden fits");
    let kv = u64::try_from(QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION).expect("KV width fits");
    match name {
        "layer0.q_proj.unbiased_linear" | "layer0.q_proj.staged_bias_bf16" | "layer0.q_proj" => {
            vec![sequence, hidden]
        }
        "layer0.k_proj.unbiased_linear"
        | "layer0.k_proj.staged_bias_bf16"
        | "layer0.k_proj"
        | "layer0.v_proj.unbiased_linear"
        | "layer0.v_proj.staged_bias_bf16"
        | "layer0.v_proj" => vec![sequence, kv],
        _ => unreachable!("fixed trace tensor"),
    }
}

fn shape_byte_len(shape: &[u64]) -> TestResult<usize> {
    let elements = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or("trace tensor shape element count overflow")
    })?;
    usize::try_from(
        elements
            .checked_mul(u64::try_from(BF16_BYTES)?)
            .ok_or("trace tensor byte count overflow")?,
    )
    .map_err(Into::into)
}

fn validate_source_provenance(document: &Value) -> TestResult {
    let provenance = require_object(&document["provenance"], "HF trace provenance")?;
    require_exact_keys(provenance, &["source_repository"], "HF trace provenance")?;
    let source = require_object(
        provenance
            .get("source_repository")
            .ok_or("HF trace source provenance is missing")?,
        "HF trace source provenance",
    )?;
    require_exact_keys(
        source,
        &[
            "git_revision",
            "source_dirty",
            "source_status_sha256",
            "sources",
        ],
        "HF trace source provenance",
    )?;
    let revision = required_string(source, "git_revision", "HF trace source provenance")?;
    if !valid_git_revision(revision) {
        return Err("HF trace Git revision is malformed".into());
    }
    if source
        .get("source_dirty")
        .and_then(Value::as_bool)
        .is_none()
    {
        return Err("HF trace source_dirty must be a boolean".into());
    }
    required_sha256(source, "source_status_sha256", "HF trace source provenance")?;
    let sources = require_object(
        source
            .get("sources")
            .ok_or("HF trace source records are missing")?,
        "HF trace source records",
    )?;
    let expected_names = SOURCE_PATHS
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    let actual_names = sources.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        return Err("HF trace source record set differs".into());
    }
    for (name, path) in SOURCE_PATHS {
        let record = require_object(
            sources
                .get(name)
                .ok_or("HF trace source record is missing")?,
            "HF trace source record",
        )?;
        require_exact_keys(record, &["path", "sha256"], "HF trace source record")?;
        if required_string(record, "path", "HF trace source record")? != path {
            return Err("HF trace source path differs".into());
        }
        required_sha256(record, "sha256", "HF trace source record")?;
    }
    Ok(())
}

fn validate_current_source_provenance(document: &Value) -> TestResult {
    validate_source_provenance(document)?;
    let root = repository_root()?;
    let source = require_object(
        require_object(&document["provenance"], "HF trace provenance")?
            .get("source_repository")
            .ok_or("HF trace source provenance is missing")?,
        "HF trace source provenance",
    )?;
    let sources = require_object(
        source
            .get("sources")
            .ok_or("HF trace source records are missing")?,
        "HF trace source records",
    )?;
    let revision_output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()?;
    if !revision_output.status.success() {
        return Err("cannot read current Git revision for HF trace provenance".into());
    }
    let revision = std::str::from_utf8(&revision_output.stdout)?.trim();
    if revision != required_string(source, "git_revision", "HF trace source provenance")? {
        return Err("HF trace Git revision differs from comparator checkout".into());
    }
    let source_paths = SOURCE_PATHS
        .iter()
        .map(|(_, path)| *path)
        .collect::<Vec<_>>();
    let status_output = Command::new("git")
        .args(["status", "--porcelain=v1", "--untracked-files=all", "--"])
        .args(&source_paths)
        .current_dir(&root)
        .output()?;
    if !status_output.status.success() {
        return Err("cannot read current Git source status for HF trace provenance".into());
    }
    let source_dirty = source
        .get("source_dirty")
        .and_then(Value::as_bool)
        .ok_or("HF trace source_dirty must be a boolean")?;
    if source_dirty != !status_output.stdout.is_empty()
        || sha256_hex(&status_output.stdout)
            != required_sha256(source, "source_status_sha256", "HF trace source provenance")?
    {
        return Err("HF trace source status differs from comparator checkout".into());
    }
    for (name, relative) in SOURCE_PATHS {
        let record = require_object(
            sources
                .get(name)
                .ok_or("HF trace source record is missing")?,
            "HF trace source record",
        )?;
        let actual = sha256_file(&regular_file(&root.join(relative), "current trace source")?)?;
        if actual != required_sha256(record, "sha256", "HF trace source record")? {
            return Err("HF trace source hash differs from comparator checkout".into());
        }
    }
    Ok(())
}

fn read_manifest(path: &Path) -> TestResult<Value> {
    let source = regular_file(path, "HF staged-bias trace manifest")?;
    let raw = fs::read(&source)?;
    let document: Value = serde_json::from_slice(&raw)?;
    let root = require_object(&document, "HF staged-bias trace manifest")?;
    require_exact_keys(
        root,
        &[
            "schema_version",
            "artifact_kind",
            "trace_id",
            "performance_claim_eligible",
            "created_at_unix_seconds",
            "producer",
            "trace_profile",
            "contract",
            "model",
            "provenance",
            "sidecar",
            "tensors",
        ],
        "HF staged-bias trace manifest",
    )?;
    if document["schema_version"] != TRACE_SCHEMA_VERSION
        || document["artifact_kind"] != HF_ARTIFACT_KIND
        || document["trace_id"] != TRACE_ID
        || document["performance_claim_eligible"] != false
        || document["created_at_unix_seconds"]
            .as_u64()
            .is_none_or(|value| value == 0)
    {
        return Err("HF staged-bias trace manifest identity differs".into());
    }

    let producer = require_object(&document["producer"], "HF staged-bias trace producer")?;
    if producer.get("implementation_id").and_then(Value::as_str) != Some(HF_IMPLEMENTATION_ID) {
        return Err("HF staged-bias trace producer implementation differs".into());
    }
    for field in [
        "runtime_dependency_class",
        "torch_version",
        "transformers_version",
    ] {
        required_string(producer, field, "HF staged-bias trace producer")?;
    }
    let qwen_source = require_object(
        producer
            .get("transformers_qwen2_source")
            .ok_or("HF staged-bias trace producer Qwen source is missing")?,
        "HF staged-bias trace producer Qwen source",
    )?;
    require_exact_keys(
        qwen_source,
        &["path", "sha256"],
        "HF staged-bias trace producer Qwen source",
    )?;
    if required_string(
        qwen_source,
        "path",
        "HF staged-bias trace producer Qwen source",
    )? != "transformers/models/qwen2/modeling_qwen2.py"
        || required_sha256(
            qwen_source,
            "sha256",
            "HF staged-bias trace producer Qwen source",
        )? != "34390fea7648b444d5727abef0b85acceba28f856aedec6c335d51253806bfee"
    {
        return Err("HF staged-bias trace producer Qwen source differs".into());
    }

    let expected_profile = json!({
        "capture_domain": "all-2048-token-positions",
        "id": TRACE_ID,
        "pair_order": "unbiased-linear-then-explicit-fp32-bias-add-bf16-round-then-unmodified-module-output",
        "tensor_count": TRACE_TENSORS.len(),
        "unbiased_linear_semantics": "post-module-hook:torch.nn.functional.linear(input,weight,bias=None)",
        "staged_bias_bf16_semantics": "post-module-hook:bf16(no_bias.to(float32)+bias.to(float32))",
        "post_bias_semantics": "unmodified-module-forward-hook",
    });
    if document["trace_profile"] != expected_profile {
        return Err("HF staged-bias trace profile differs".into());
    }

    let contract = require_object(&document["contract"], "HF staged-bias trace contract")?;
    require_exact_keys(
        contract,
        &["model_id", "model_revision", "workload", "execution"],
        "HF staged-bias trace contract",
    )?;
    if contract.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
        || contract.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
    {
        return Err("HF staged-bias trace model identity differs".into());
    }
    let workload = require_object(
        contract
            .get("workload")
            .ok_or("HF staged-bias trace workload is missing")?,
        "HF staged-bias trace workload",
    )?;
    require_exact_keys(
        workload,
        &[
            "schema_version",
            "case",
            "source_sha256",
            "prompt_token_count",
            "prompt_token_ids_le_u32_sha256",
        ],
        "HF staged-bias trace workload",
    )?;
    if workload.get("schema_version").and_then(Value::as_str)
        != Some(QWEN3B_WORKLOAD_SCHEMA_VERSION)
        || workload.get("case").and_then(Value::as_str) != Some(QWEN3B_WORKLOAD_CASE)
        || workload.get("source_sha256").and_then(Value::as_str) != Some(QWEN3B_WORKLOAD_SHA256)
        || workload.get("prompt_token_count").and_then(Value::as_u64)
            != Some(u64::try_from(QWEN3B_PROMPT_TOKEN_COUNT)?)
        || workload
            .get("prompt_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(QWEN3B_PROMPT_TOKEN_SHA256)
    {
        return Err("HF staged-bias trace workload contract differs".into());
    }
    let expected_execution = json!({
        "attention_implementation": "eager",
        "dtype": "bfloat16",
        "explicit_attention_mask": true,
        "explicit_position_ids": true,
        "logits_to_keep": 1,
        "tf32_enabled": false,
        "trace_capture": "post-module-hooks+separate-functional-linear-bias-none+explicit-fp32-bias-add-bf16-round",
        "use_cache": false,
    });
    if contract.get("execution") != Some(&expected_execution) {
        return Err("HF staged-bias trace execution contract differs".into());
    }

    let model = require_object(&document["model"], "HF staged-bias trace model")?;
    require_exact_keys(
        model,
        &[
            "checkpoint_path",
            "checkpoint_receipt_filename",
            "checkpoint_receipt_sha256",
        ],
        "HF staged-bias trace model",
    )?;
    required_string(model, "checkpoint_path", "HF staged-bias trace model")?;
    if model
        .get("checkpoint_receipt_filename")
        .and_then(Value::as_str)
        != Some("riley-checkpoint.json")
    {
        return Err("HF staged-bias trace checkpoint receipt filename differs".into());
    }
    required_sha256(
        model,
        "checkpoint_receipt_sha256",
        "HF staged-bias trace model",
    )?;

    validate_current_source_provenance(&document)?;

    let sidecar = require_object(&document["sidecar"], "HF staged-bias trace sidecar")?;
    require_exact_keys(
        sidecar,
        &["path", "sha256", "format", "tensor_count"],
        "HF staged-bias trace sidecar",
    )?;
    let name = required_string(sidecar, "path", "HF staged-bias trace sidecar")?;
    if Path::new(name).file_name().and_then(|value| value.to_str()) != Some(name)
        || !name.ends_with(".safetensors")
        || sidecar.get("format").and_then(Value::as_str) != Some("safetensors")
        || sidecar.get("tensor_count").and_then(Value::as_u64)
            != Some(u64::try_from(TRACE_TENSORS.len())?)
    {
        return Err("HF staged-bias trace sidecar metadata differs".into());
    }
    required_sha256(sidecar, "sha256", "HF staged-bias trace sidecar")?;

    let tensors = require_object(&document["tensors"], "HF staged-bias trace tensors")?;
    let expected_names = TRACE_TENSORS
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    let actual_names = tensors.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        return Err("HF staged-bias trace tensor set differs".into());
    }
    for (name, key) in TRACE_TENSORS {
        let tensor = require_object(
            tensors
                .get(name)
                .ok_or("HF staged-bias trace tensor is missing")?,
            "HF staged-bias trace tensor",
        )?;
        require_exact_keys(
            tensor,
            &[
                "key",
                "shape",
                "dtype",
                "canonical_byte_order",
                "bf16_le_sha256",
                "bf16_le_bytes",
            ],
            "HF staged-bias trace tensor",
        )?;
        if tensor.get("key").and_then(Value::as_str) != Some(key)
            || tensor.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || tensor.get("canonical_byte_order").and_then(Value::as_str)
                != Some("little-endian-u16")
        {
            return Err("HF staged-bias trace tensor metadata differs".into());
        }
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("HF staged-bias trace tensor shape must be an array")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or("HF staged-bias trace shape must be unsigned")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape != expected_shape(name) {
            return Err("HF staged-bias trace tensor shape differs".into());
        }
        let expected_bytes = u64::try_from(shape_byte_len(&shape)?)?;
        if tensor.get("bf16_le_bytes").and_then(Value::as_u64) != Some(expected_bytes) {
            return Err("HF staged-bias trace tensor byte size differs".into());
        }
        required_sha256(tensor, "bf16_le_sha256", "HF staged-bias trace tensor")?;
    }
    Ok(document)
}

fn parse_sidecar_header(path: &Path) -> TestResult<(Map<String, Value>, u64, u64)> {
    let source = regular_file(path, "HF staged-bias trace sidecar")?;
    let mut file = File::open(&source)?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)?;
    let header_length = usize::try_from(u64::from_le_bytes(length_bytes))?;
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES {
        return Err("HF staged-bias trace sidecar header length differs".into());
    }
    let mut raw_header = vec![0_u8; header_length];
    file.read_exact(&mut raw_header)?;
    let header = serde_json::from_slice::<Value>(&raw_header)?
        .as_object()
        .cloned()
        .ok_or("HF staged-bias trace sidecar header must be an object")?;
    let data_start = u64::try_from(8 + header_length)?;
    let size = file.metadata()?.len();
    if data_start > size {
        return Err("HF staged-bias trace sidecar header is truncated".into());
    }
    Ok((header, data_start, size))
}

fn sidecar_tensors(
    manifest: &Value,
    sidecar: &Path,
) -> TestResult<BTreeMap<String, SidecarTensor>> {
    let metadata = require_object(&manifest["sidecar"], "HF staged-bias trace sidecar")?;
    let actual_sidecar = regular_file(sidecar, "HF staged-bias trace sidecar")?;
    let actual_sidecar_sha = sha256_file(&actual_sidecar)?;
    if metadata.get("path").and_then(Value::as_str)
        != actual_sidecar.file_name().and_then(|name| name.to_str())
        || metadata.get("sha256").and_then(Value::as_str) != Some(actual_sidecar_sha.as_str())
        || metadata.get("format").and_then(Value::as_str) != Some("safetensors")
    {
        return Err("HF staged-bias trace sidecar binding differs".into());
    }
    let (header, data_start, sidecar_size) = parse_sidecar_header(&actual_sidecar)?;
    let expected_keys = TRACE_TENSORS
        .iter()
        .map(|(_, key)| *key)
        .collect::<BTreeSet<_>>();
    let actual_keys = header
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual_keys != expected_keys {
        return Err("HF staged-bias trace sidecar tensor set differs".into());
    }
    let manifest_tensors = require_object(&manifest["tensors"], "HF staged-bias trace tensors")?;
    let mut output = BTreeMap::new();
    let mut ranges = Vec::with_capacity(TRACE_TENSORS.len());
    for (name, key) in TRACE_TENSORS {
        let entry = require_object(
            header
                .get(key)
                .ok_or("HF staged-bias trace sidecar tensor is missing")?,
            "HF staged-bias trace sidecar tensor",
        )?;
        if entry.get("dtype").and_then(Value::as_str) != Some("BF16") {
            return Err("HF staged-bias trace sidecar tensor dtype differs".into());
        }
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("HF staged-bias trace sidecar tensor shape must be an array")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or("HF staged-bias trace sidecar shape must be unsigned")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape != expected_shape(name) {
            return Err("HF staged-bias trace sidecar tensor shape differs".into());
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("HF staged-bias trace sidecar offsets must be an array")?;
        if offsets.len() != 2 {
            return Err("HF staged-bias trace sidecar offset count differs".into());
        }
        let start = offsets[0]
            .as_u64()
            .ok_or("HF staged-bias trace sidecar start differs")?;
        let end = offsets[1]
            .as_u64()
            .ok_or("HF staged-bias trace sidecar end differs")?;
        let expected_bytes = u64::try_from(shape_byte_len(&shape)?)?;
        let data_end = data_start
            .checked_add(end)
            .ok_or("HF staged-bias trace sidecar data range overflows")?;
        if end < start || end - start != expected_bytes || data_end > sidecar_size {
            return Err("HF staged-bias trace sidecar tensor range differs".into());
        }
        let manifest_tensor = require_object(
            manifest_tensors
                .get(name)
                .ok_or("HF staged-bias trace manifest tensor is missing")?,
            "HF staged-bias trace manifest tensor",
        )?;
        if manifest_tensor.get("key").and_then(Value::as_str) != Some(key) {
            return Err("HF staged-bias trace manifest tensor key differs".into());
        }
        ranges.push((start, end));
        output.insert(
            name.to_owned(),
            SidecarTensor {
                data_start: data_start
                    .checked_add(start)
                    .ok_or("HF staged-bias trace sidecar data range overflows")?,
                data_end,
            },
        );
    }
    ranges.sort_unstable();
    let mut expected_start = 0_u64;
    for (start, end) in ranges {
        if start != expected_start {
            return Err("HF staged-bias trace sidecar ranges are non-contiguous".into());
        }
        expected_start = end;
    }
    if data_start
        .checked_add(expected_start)
        .ok_or("HF staged-bias trace sidecar data range overflows")?
        != sidecar_size
    {
        return Err("HF staged-bias trace sidecar payload size differs".into());
    }
    Ok(output)
}

fn read_sidecar_tensor(
    sidecar: &Path,
    tensor: &SidecarTensor,
    expected_sha256: &str,
) -> TestResult<Vec<u8>> {
    let length = usize::try_from(tensor.data_end - tensor.data_start)?;
    let mut bytes = vec![0_u8; length];
    let mut file = File::open(sidecar)?;
    file.seek(SeekFrom::Start(tensor.data_start))?;
    file.read_exact(&mut bytes)?;
    if sha256_hex(&bytes) != expected_sha256 {
        return Err("HF staged-bias trace sidecar raw BF16 SHA-256 differs".into());
    }
    Ok(bytes)
}

fn canonical_riley_bf16(bytes: &[u8]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytes.to_vec()
    }
    #[cfg(target_endian = "big")]
    {
        bytes
            .chunks_exact(BF16_BYTES)
            .flat_map(|element| [element[1], element[0]])
            .collect()
    }
}

fn decode_bf16_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn pair_metrics(reference: &[u8], candidate: &[u8]) -> TestResult<Value> {
    if reference.len() != candidate.len() || reference.len() % BF16_BYTES != 0 {
        return Err("trace tensor byte sizes differ".into());
    }
    let mut unequal_elements = 0_u64;
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut reference_square = 0.0_f64;
    let mut candidate_square = 0.0_f64;
    for (expected, actual) in reference
        .chunks_exact(BF16_BYTES)
        .zip(candidate.chunks_exact(BF16_BYTES))
    {
        if expected != actual {
            unequal_elements += 1;
        }
        let expected_value = f64::from(decode_bf16_le(expected));
        let actual_value = f64::from(decode_bf16_le(actual));
        if !expected_value.is_finite() || !actual_value.is_finite() {
            return Err("trace tensor contains non-finite BF16 values".into());
        }
        let absolute = (expected_value - actual_value).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        dot += expected_value * actual_value;
        reference_square += expected_value * expected_value;
        candidate_square += actual_value * actual_value;
    }
    let elements = u64::try_from(reference.len() / BF16_BYTES)?;
    let cosine = if reference_square == 0.0 || candidate_square == 0.0 {
        None
    } else {
        Some(dot / (reference_square.sqrt() * candidate_square.sqrt()))
    };
    Ok(json!({
        "element_count": elements,
        "bf16_exact": unequal_elements == 0,
        "unequal_element_count": unequal_elements,
        "max_abs_bf16_as_f32": max_abs,
        "mean_abs_bf16_as_f32": sum_abs / elements as f64,
        "cosine_similarity": cosine,
        "reference_bf16_le_sha256": sha256_hex(reference),
        "candidate_bf16_le_sha256": sha256_hex(candidate),
    }))
}

fn repository_root() -> TestResult<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .canonicalize()?)
}

fn write_artifact_exclusive(path: &Path, document: &Value) -> TestResult {
    if !path.is_absolute() || path.extension().and_then(|value| value.to_str()) != Some("json") {
        return Err("staged-bias comparison output must be an absolute .json path".into());
    }
    let root = repository_root()?;
    fs::create_dir_all(
        path.parent()
            .ok_or("staged-bias comparison output has no parent")?,
    )?;
    let parent = path
        .parent()
        .ok_or("staged-bias comparison output has no parent")?
        .canonicalize()?;
    let output = parent.join(
        path.file_name()
            .ok_or("staged-bias comparison output has no basename")?,
    );
    if output == root || output.starts_with(&root) {
        return Err("staged-bias comparison output must be outside the repository".into());
    }
    if fs::symlink_metadata(&output).is_ok() {
        return Err("refusing to overwrite existing staged-bias comparison artifact".into());
    }
    let mut payload = serde_json::to_vec_pretty(document)?;
    payload.push(10);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)?;
    if let Err(error) = file.write_all(&payload).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&output);
        return Err(error.into());
    }
    Ok(())
}

fn unix_seconds() -> TestResult<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

#[test]
#[ignore = "remote-only Qwen2.5-3B P2048 HF/Rust QKV explicit staged-bias discriminator"]
fn qwen3b_p2048_layer0_qkv_staged_bias_trace_matches_riley_post_bias() -> TestResult {
    let inputs = trace_inputs()?;
    let prompt_tokens = load_prompt_ids(&inputs.workload)?;
    let manifest = read_manifest(&inputs.manifest)?;
    let sidecar = regular_file(&inputs.sidecar, "HF staged-bias trace sidecar")?;
    let sidecar_tensors = sidecar_tensors(&manifest, &sidecar)?;
    let model = load_qwen3b(&inputs.checkpoint)?;

    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut forward = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        prompt_tokens.len(),
        PreparedLlamaForwardConfig::default().with_reference_attention(),
    )?;
    let points = PROJECTIONS
        .iter()
        .flat_map(|(gemm_point, post_bias_point, _, _, _)| [*gemm_point, *post_bias_point])
        .collect::<Vec<_>>();
    let mut trace = forward.prepare_trace_points(&points)?;
    forward.upload_tokens(&prompt_tokens, &mut stream)?;
    forward.execute_traced(&mut stream, &mut trace)?;
    if trace.captured_count() != u32::try_from(points.len())? {
        return Err("Riley staged-bias trace did not capture every requested stage".into());
    }

    let manifest_tensors = require_object(&manifest["tensors"], "HF staged-bias trace tensors")?;
    let mut projections = Map::new();
    let mut staged_exact_count = 0_u64;
    let mut unbiased_exact_count = 0_u64;
    let mut first_non_exact_staged: Option<&str> = None;
    for (gemm_point, post_bias_point, unbiased_name, staged_name, module_output_name) in PROJECTIONS
    {
        let riley_gemm = canonical_riley_bf16(
            trace
                .tensor(gemm_point)
                .ok_or("Riley staged-bias GEMM trace tensor is missing")?,
        );
        let riley_post_bias = canonical_riley_bf16(
            trace
                .tensor(post_bias_point)
                .ok_or("Riley staged-bias post-bias trace tensor is missing")?,
        );
        let expected_shape = expected_shape(module_output_name);
        let expected_bytes = shape_byte_len(&expected_shape)?;
        if riley_gemm.len() != expected_bytes || riley_post_bias.len() != expected_bytes {
            return Err("Riley staged-bias trace tensor byte count differs from contract".into());
        }
        let read_hf = |name: &str| -> TestResult<Vec<u8>> {
            let tensor = require_object(
                manifest_tensors
                    .get(name)
                    .ok_or("HF staged-bias manifest tensor is missing")?,
                "HF staged-bias manifest tensor",
            )?;
            let hash = required_sha256(tensor, "bf16_le_sha256", "HF staged-bias manifest tensor")?;
            let metadata = sidecar_tensors
                .get(name)
                .ok_or("HF staged-bias sidecar tensor is missing")?;
            read_sidecar_tensor(&sidecar, metadata, &hash)
        };
        let hf_unbiased = read_hf(unbiased_name)?;
        let hf_staged = read_hf(staged_name)?;
        let hf_module_output = read_hf(module_output_name)?;
        let unbiased_vs_riley_gemm = pair_metrics(&hf_unbiased, &riley_gemm)?;
        let staged_vs_riley_post_bias = pair_metrics(&hf_staged, &riley_post_bias)?;
        let staged_vs_hf_module_output = pair_metrics(&hf_staged, &hf_module_output)?;
        let hf_module_output_vs_riley_post_bias =
            pair_metrics(&hf_module_output, &riley_post_bias)?;
        if unbiased_vs_riley_gemm["bf16_exact"] == true {
            unbiased_exact_count += 1;
        }
        if staged_vs_riley_post_bias["bf16_exact"] == true {
            staged_exact_count += 1;
        } else if first_non_exact_staged.is_none() {
            first_non_exact_staged = Some(module_output_name);
        }
        projections.insert(
            module_output_name.to_owned(),
            json!({
                "hf_unbiased_linear_tensor": unbiased_name,
                "hf_staged_bias_bf16_tensor": staged_name,
                "hf_module_output_tensor": module_output_name,
                "riley_gemm_capture": LlamaTracePoint::name(gemm_point),
                "riley_post_bias_capture": LlamaTracePoint::name(post_bias_point),
                "bf16_hashes": {
                    "hf_unbiased_linear_le_sha256": sha256_hex(&hf_unbiased),
                    "hf_staged_bias_bf16_le_sha256": sha256_hex(&hf_staged),
                    "hf_module_output_le_sha256": sha256_hex(&hf_module_output),
                    "riley_gemm_le_sha256": sha256_hex(&riley_gemm),
                    "riley_post_bias_le_sha256": sha256_hex(&riley_post_bias),
                },
                "unbiased_linear_vs_riley_gemm": unbiased_vs_riley_gemm,
                "staged_bias_bf16_vs_riley_post_bias": staged_vs_riley_post_bias,
                "staged_bias_bf16_vs_hf_module_output": staged_vs_hf_module_output,
                "hf_module_output_vs_riley_post_bias": hf_module_output_vs_riley_post_bias,
            }),
        );
    }

    forward.close()?;
    drop(trace);
    context.synchronize()?;
    if !context.allocation_stats()?.is_zero() {
        return Err("Riley staged-bias trace left a CUDA allocation after close".into());
    }
    stream.close()?;
    context.close()?;

    let result = json!({
        "schema_version": TRACE_SCHEMA_VERSION,
        "artifact_kind": RESULT_ARTIFACT_KIND,
        "performance_claim_eligible": false,
        "created_at_unix_seconds": unix_seconds()?,
        "hf_trace": {
            "manifest_path": inputs.manifest,
            "manifest_sha256": sha256_file(&inputs.manifest)?,
            "sidecar_path": sidecar,
            "sidecar_sha256": sha256_file(&sidecar)?,
        },
        "contract": {
            "trace_id": TRACE_ID,
            "model_id": QWEN3B_MODEL_ID,
            "model_revision": QWEN3B_REVISION,
            "workload_sha256": QWEN3B_WORKLOAD_SHA256,
            "prompt_token_ids_le_u32_sha256": QWEN3B_PROMPT_TOKEN_SHA256,
            "attention_backend": "materialized-reference",
            "use_cache": false,
            "dtype": "bfloat16",
            "unbiased_linear_semantics": "HF separate no-bias shadow endpoint vs Riley standalone GEMM before its separate bias operator",
            "staged_bias_bf16_semantics": "HF explicit bf16(no_bias.to(float32)+bias.to(float32)) vs Riley staged BF16-to-FP32 bias-add-to-BF16 result",
            "post_bias_semantics": "HF unmodified module output vs Riley staged BF16-to-FP32 bias-add-to-BF16 result",
        },
        "summary": {
            "projection_count": PROJECTIONS.len(),
            "bf16_exact_unbiased_linear_vs_riley_gemm_count": unbiased_exact_count,
            "bf16_exact_staged_bias_vs_riley_post_bias_count": staged_exact_count,
            "first_non_exact_staged_bias_vs_riley_post_bias": first_non_exact_staged,
        },
        "projections": projections,
    });
    write_artifact_exclusive(&inputs.output, &result)?;
    println!(
        "QWEN3B_QKV_STAGED_BIAS trace_id={} first_non_exact_staged_bias_vs_riley_post_bias={} staged_exact_projections={} unbiased_exact_projections={} performance_claim_eligible=false",
        TRACE_ID,
        first_non_exact_staged.unwrap_or("none"),
        staged_exact_count,
        unbiased_exact_count,
    );
    Ok(())
}
