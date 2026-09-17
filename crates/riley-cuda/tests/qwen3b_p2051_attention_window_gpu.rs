//! Source-bound P2051 Qwen eager-attention quality gate.
//!
//! Hugging Face creates the paired BF16 safetensors artifact offline. This
//! Rust-only CUDA diagnostic then replays the selected layer's actual Q/K/V
//! tensors through the opt-in cuBLASLt candidate and compares every numerical
//! boundary that the candidate controls. It never starts a serving runtime.

#![cfg(feature = "cuda")]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use riley_cuda::{
    AttentionBackend, AttentionBackendAvailability, AttentionMask, AttentionPreference,
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, HfEagerQwenP2051LastRowTrace,
    PrefillAttentionParams, PrefillAttentionRequest, PreparedPrefillAttention,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const MANIFEST_ENV: &str = "RILEY_QWEN3B_P2051_ATTENTION_WINDOW_MANIFEST";
const SIDECAR_ENV: &str = "RILEY_QWEN3B_P2051_ATTENTION_WINDOW_SIDECAR";
const OUTPUT_ENV: &str = "RILEY_QWEN3B_P2051_ATTENTION_WINDOW_OUTPUT";
const MARKER_PREFIX: &str = "RILEY_QWEN3B_P2051_ATTENTION_WINDOW=";

const ARTIFACT_SCHEMA: &str = "riley.qwen3b-hf-eager-p2051-cache-off-attention-window-trace.v1";
const ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-attention-window-trace";
const TRACE_ID: &str = "qwen3b-p2051-cache-off-eager-attention-window-v1";
const RESULT_SCHEMA: &str = "riley.qwen3b-p2051-hf-eager-attention-window-comparison.v1";
const RESULT_KIND: &str = "qwen2.5-3b-riley-p2051-hf-eager-attention-window-comparison";
const IMPLEMENTATION_ID: &str = "riley.cuda.hf-eager-cublaslt-qwen-p2051-probe.bf16";

const MODEL_ID: &str = "Qwen/Qwen2.5-3B-Instruct";
const MODEL_REVISION: &str = "aa8e72537993ba99e69dfaafa59ed015b17504d1";
const WORKLOAD_CASE: &str = "qwen3b-c8-p2048-o128";
const S: u64 = 2_051;
const QH: u64 = 16;
const KVH: u64 = 2;
const D: u64 = 128;
const SCALE: f32 = 0.088_388_346;
const BF16_BYTES: u64 = 2;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 1_048_576;
const MODEL_LAYER_COUNT: usize = 36;

const Q_BSHD: [u64; 4] = [1, S, QH, D];
const KV_BSHD: [u64; 4] = [1, S, KVH, D];
const SCORE_LAST: [u64; 2] = [QH, S];
const MASK_LAST: [u64; 1] = [S];

const SOURCE_RECORDS: [(&str, &str); 17] = [
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
        "p2051_attention_window_trace",
        "tools/python/reference/riley_reference/qwen3b_p2051_attention_window_trace.py",
    ),
    (
        "hf_calibration",
        "tools/python/reference/riley_reference/hf_calibration.py",
    ),
    (
        "native_attention_trace",
        "kernels/src/attention_cublaslt.cu",
    ),
    (
        "native_attention_trace_header",
        "kernels/include/riley_cuda.h",
    ),
    ("rust_attention_trace_ffi", "crates/riley-cuda/src/ffi.rs"),
    (
        "rust_attention_trace_api",
        "crates/riley-cuda/src/prefill.rs",
    ),
    (
        "rust_attention_trace_exports",
        "crates/riley-cuda/src/lib.rs",
    ),
    (
        "rust_attention_trace_consumer",
        "crates/riley-cuda/tests/qwen3b_p2051_attention_window_gpu.rs",
    ),
    (
        "rust_attention_trace_package",
        "crates/riley-cuda/Cargo.toml",
    ),
    ("reference_project", "tools/python/reference/pyproject.toml"),
    ("reference_lock", "tools/python/reference/uv.lock"),
    ("reference_python", "tools/python/reference/.python-version"),
];

#[derive(Clone, Copy)]
struct TensorSpec {
    suffix: &'static str,
    shape: &'static [u64],
}

const TENSOR_SPECS: [TensorSpec; 8] = [
    TensorSpec {
        suffix: "attention.query_bshd",
        shape: &Q_BSHD,
    },
    TensorSpec {
        suffix: "attention.key_bshd",
        shape: &KV_BSHD,
    },
    TensorSpec {
        suffix: "attention.value_bshd",
        shape: &KV_BSHD,
    },
    TensorSpec {
        suffix: "attention.raw_qk_last",
        shape: &SCORE_LAST,
    },
    TensorSpec {
        suffix: "attention.scaled_masked_last",
        shape: &SCORE_LAST,
    },
    TensorSpec {
        suffix: "attention.probabilities_last",
        shape: &SCORE_LAST,
    },
    TensorSpec {
        suffix: "attention.context_bshd",
        shape: &Q_BSHD,
    },
    TensorSpec {
        suffix: "attention.mask_last",
        shape: &MASK_LAST,
    },
];

#[derive(Debug)]
struct AttentionArtifact {
    layer_index: usize,
    manifest_path: PathBuf,
    manifest_sha256: String,
    sidecar_path: PathBuf,
    sidecar_sha256: String,
    tensors: BTreeMap<String, Vec<u8>>,
}

fn tensor_name(layer_index: usize, spec: TensorSpec) -> String {
    format!("layer{layer_index}.{}", spec.suffix)
}

fn tensor_key(name: &str) -> String {
    format!("trace/{}", name.replace('.', "/"))
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
    let value = value
        .as_str()
        .ok_or_else(|| format!("{label} must be a SHA-256 string"))?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!("{label} must be lowercase SHA-256").into());
    }
    Ok(value.to_owned())
}

fn shape_byte_len(shape: &[u64]) -> TestResult<usize> {
    let elements = shape.iter().try_fold(1_u64, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or("attention trace shape arithmetic overflow")
    })?;
    usize::try_from(
        elements
            .checked_mul(BF16_BYTES)
            .ok_or("attention trace byte arithmetic overflow")?,
    )
    .map_err(Into::into)
}

fn shape_from_json(value: &Value, label: &str) -> TestResult<Vec<u64>> {
    value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .ok_or_else(|| format!("{label} has a non-u64 dimension"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn repository_root() -> TestResult<PathBuf> {
    let manifest_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_directory
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| std::io::Error::other("workspace root is unavailable"))
        .and_then(|root| root.canonicalize())
        .map_err(Into::into)
}

fn validate_source_record(
    root: &Path,
    value: &Value,
    expected_path: &str,
    label: &str,
) -> TestResult {
    require_exact_fields(value, &["path", "sha256"], label)?;
    if value.get("path").and_then(Value::as_str) != Some(expected_path) {
        return Err(format!("{label} path differs").into());
    }
    let relative = Path::new(expected_path);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{label} path is not repository-relative").into());
    }
    let expected_hash = json_sha256(
        value
            .get("sha256")
            .ok_or_else(|| format!("{label} SHA-256 is missing"))?,
        label,
    )?;
    let path = regular_file(&root.join(relative), label)?;
    if sha256_file(&path)? != expected_hash {
        return Err(format!("{label} SHA-256 differs from current source").into());
    }
    Ok(())
}

fn validate_source_provenance(value: &Value) -> TestResult {
    require_exact_fields(
        value,
        &[
            "git_revision",
            "source_dirty",
            "source_status_sha256",
            "sources",
        ],
        "attention trace source provenance",
    )?;
    if value.get("source_dirty").and_then(Value::as_bool) != Some(false) {
        return Err("attention trace source provenance is dirty".into());
    }
    let revision = value
        .get("git_revision")
        .and_then(Value::as_str)
        .filter(|revision| {
            revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        .ok_or("attention trace Git revision differs")?;
    let _ = revision;
    let _ = json_sha256(
        value
            .get("source_status_sha256")
            .ok_or("attention trace source status hash is missing")?,
        "attention trace source status hash",
    )?;
    let sources = value
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("attention trace source records are missing")?;
    let expected_names: BTreeSet<_> = SOURCE_RECORDS.iter().map(|(name, _)| *name).collect();
    let actual_names: BTreeSet<_> = sources.keys().map(String::as_str).collect();
    if actual_names != expected_names {
        return Err("attention trace source record names differ".into());
    }
    let root = repository_root()?;
    for (name, path) in SOURCE_RECORDS {
        validate_source_record(
            &root,
            sources
                .get(name)
                .ok_or("attention trace source record is missing")?,
            path,
            &format!("attention trace source {name}"),
        )?;
    }
    Ok(())
}

fn validate_manifest_contract(manifest: &Value) -> TestResult<usize> {
    require_exact_fields(
        manifest,
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
        "HF P2051 attention-window manifest",
    )?;
    if manifest.get("schema_version").and_then(Value::as_str) != Some(ARTIFACT_SCHEMA)
        || manifest.get("artifact_kind").and_then(Value::as_str) != Some(ARTIFACT_KIND)
        || manifest.get("trace_id").and_then(Value::as_str) != Some(TRACE_ID)
        || manifest
            .get("performance_claim_eligible")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err("HF P2051 attention-window manifest identity differs".into());
    }
    let profile = manifest
        .get("trace_profile")
        .ok_or("HF P2051 attention-window trace profile is missing")?;
    require_exact_fields(
        profile,
        &[
            "capture_domain",
            "id",
            "layer_index",
            "tensor_names",
            "tensor_count",
            "rust_consumer",
        ],
        "HF P2051 attention-window trace profile",
    )?;
    let layer_index = usize::try_from(
        profile
            .get("layer_index")
            .and_then(Value::as_u64)
            .ok_or("HF P2051 attention-window layer index is missing")?,
    )?;
    if layer_index >= MODEL_LAYER_COUNT {
        return Err("HF P2051 attention-window layer index is out of range".into());
    }
    let expected_names: Vec<_> = TENSOR_SPECS
        .iter()
        .copied()
        .map(|spec| tensor_name(layer_index, spec))
        .collect();
    if profile.get("capture_domain").and_then(Value::as_str)
        != Some("cache-free-p2051-selected-layer-hf-eager-attention-boundaries")
        || profile.get("id").and_then(Value::as_str) != Some(TRACE_ID)
        || profile
            .get("tensor_names")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            != Some(expected_names.iter().map(String::as_str).collect())
        || profile.get("tensor_count").and_then(Value::as_u64)
            != Some(u64::try_from(TENSOR_SPECS.len())?)
    {
        return Err("HF P2051 attention-window trace profile differs".into());
    }
    let expected_consumer = json!({
        "api": "riley_cuda::PreparedPrefillAttention::execute_hf_eager_qwen_p2051_last_row_traced",
        "attention_backend": IMPLEMENTATION_ID,
        "cache": false,
        "input_context_token_count": S,
        "last_token_row_index": S - 1,
        "layer_index": layer_index,
        "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
        "trace_row_layout": "full-qkv-context-plus-last-score-row",
    });
    if profile.get("rust_consumer") != Some(&expected_consumer) {
        return Err("HF P2051 attention-window Rust consumer contract differs".into());
    }
    let contract = manifest
        .get("contract")
        .and_then(Value::as_object)
        .ok_or("HF P2051 attention-window contract is missing")?;
    if contract.get("model_id").and_then(Value::as_str) != Some(MODEL_ID)
        || contract.get("model_revision").and_then(Value::as_str) != Some(MODEL_REVISION)
        || contract
            .get("workload")
            .and_then(Value::as_object)
            .and_then(|workload| workload.get("case"))
            .and_then(Value::as_str)
            != Some(WORKLOAD_CASE)
    {
        return Err("HF P2051 attention-window workload contract differs".into());
    }
    let producer = manifest
        .get("producer")
        .and_then(Value::as_object)
        .ok_or("HF P2051 attention-window producer is missing")?;
    if producer.get("implementation_id").and_then(Value::as_str)
        != Some("riley-python-qwen3b-hf-eager-p2051-attention-window-trace-v1")
    {
        return Err("HF P2051 attention-window producer differs".into());
    }
    let provenance = manifest
        .get("provenance")
        .and_then(Value::as_object)
        .and_then(|provenance| provenance.get("source_repository"))
        .ok_or("HF P2051 attention-window provenance is missing")?;
    validate_source_provenance(provenance)?;
    Ok(layer_index)
}

fn validate_finite_bf16(bytes: &[u8], label: &str) -> TestResult {
    if bytes.len() % BF16_BYTES as usize != 0 {
        return Err(format!("{label} byte count differs").into());
    }
    for pair in bytes.chunks_exact(BF16_BYTES as usize) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7f80 == 0x7f80 {
            return Err(format!("{label} contains non-finite BF16").into());
        }
    }
    Ok(())
}

fn parse_sidecar(
    manifest: &Value,
    sidecar_path: &Path,
    layer_index: usize,
) -> TestResult<BTreeMap<String, Vec<u8>>> {
    let source = regular_file(sidecar_path, "HF P2051 attention-window sidecar")?;
    let bytes = fs::read(&source)?;
    let sidecar = manifest
        .get("sidecar")
        .and_then(Value::as_object)
        .ok_or("HF P2051 attention-window sidecar metadata is missing")?;
    let expected_basename = source
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("HF P2051 attention-window sidecar basename is invalid")?;
    if sidecar.get("path").and_then(Value::as_str) != Some(expected_basename)
        || sidecar.get("format").and_then(Value::as_str) != Some("safetensors")
        || sidecar.get("tensor_count").and_then(Value::as_u64)
            != Some(u64::try_from(TENSOR_SPECS.len())?)
        || sha256_hex(&bytes)
            != json_sha256(
                sidecar
                    .get("sha256")
                    .ok_or("HF P2051 attention-window sidecar SHA-256 is missing")?,
                "HF P2051 attention-window sidecar SHA-256",
            )?
    {
        return Err("HF P2051 attention-window sidecar binding differs".into());
    }
    if bytes.len() < 8 {
        return Err("HF P2051 attention-window sidecar is too short".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    if header_len == 0
        || header_len > MAX_SAFETENSORS_HEADER_BYTES
        || 8_usize
            .checked_add(header_len)
            .is_none_or(|end| end > bytes.len())
    {
        return Err("HF P2051 attention-window sidecar header differs".into());
    }
    let data_start = 8_usize
        .checked_add(header_len)
        .ok_or("HF P2051 attention-window sidecar data offset overflows")?;
    let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
    let header = header
        .as_object()
        .ok_or("HF P2051 attention-window sidecar header is not an object")?;
    let expected_keys: BTreeSet<_> = TENSOR_SPECS
        .iter()
        .copied()
        .map(|spec| tensor_name(layer_index, spec))
        .map(|name| tensor_key(&name))
        .collect();
    let actual_keys: BTreeSet<_> = header
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect();
    if actual_keys != expected_keys {
        return Err("HF P2051 attention-window sidecar tensor keys differ".into());
    }
    let tensors = manifest
        .get("tensors")
        .and_then(Value::as_object)
        .ok_or("HF P2051 attention-window tensor manifest is missing")?;
    let mut ranges = Vec::with_capacity(TENSOR_SPECS.len());
    let mut output = BTreeMap::new();
    for spec in TENSOR_SPECS {
        let name = tensor_name(layer_index, spec);
        let key = tensor_key(&name);
        let expected_shape = spec.shape.to_vec();
        let expected_bytes = shape_byte_len(spec.shape)?;
        let reference = tensors
            .get(&name)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("HF P2051 attention-window tensor {name} is missing"))?;
        if reference.get("key").and_then(Value::as_str) != Some(key.as_str())
            || reference.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || reference
                .get("canonical_byte_order")
                .and_then(Value::as_str)
                != Some("little-endian-u16")
            || shape_from_json(
                reference
                    .get("shape")
                    .ok_or("HF P2051 attention-window tensor shape is missing")?,
                "HF P2051 attention-window tensor shape",
            )? != expected_shape
            || reference.get("bf16_le_bytes").and_then(Value::as_u64)
                != Some(u64::try_from(expected_bytes)?)
        {
            return Err(format!("HF P2051 attention-window tensor {name} metadata differs").into());
        }
        let entry = header
            .get(&key)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("HF P2051 attention-window sidecar tensor {name} is missing"))?;
        if entry.get("dtype").and_then(Value::as_str) != Some("BF16")
            || shape_from_json(
                entry
                    .get("shape")
                    .ok_or("HF P2051 attention-window sidecar tensor shape is missing")?,
                "HF P2051 attention-window sidecar tensor shape",
            )? != expected_shape
        {
            return Err(format!(
                "HF P2051 attention-window sidecar tensor {name} metadata differs"
            )
            .into());
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("HF P2051 attention-window sidecar data offsets are missing")?;
        if offsets.len() != 2 {
            return Err("HF P2051 attention-window sidecar offset count differs".into());
        }
        let start = usize::try_from(offsets[0].as_u64().ok_or("HF P2051 sidecar offset start")?)?;
        let end = usize::try_from(offsets[1].as_u64().ok_or("HF P2051 sidecar offset end")?)?;
        if end < start || end - start != expected_bytes || data_start + end > bytes.len() {
            return Err(format!("HF P2051 attention-window tensor {name} range differs").into());
        }
        let raw = bytes[data_start + start..data_start + end].to_vec();
        if sha256_hex(&raw)
            != json_sha256(
                reference
                    .get("bf16_le_sha256")
                    .ok_or("HF P2051 attention-window tensor SHA-256 is missing")?,
                "HF P2051 attention-window tensor SHA-256",
            )?
        {
            return Err(format!("HF P2051 attention-window tensor {name} hash differs").into());
        }
        validate_finite_bf16(&raw, &format!("HF P2051 attention-window tensor {name}"))?;
        ranges.push((start, end));
        output.insert(name, raw);
    }
    ranges.sort_unstable();
    let mut expected_start = 0_usize;
    for (start, end) in ranges {
        if start != expected_start {
            return Err("HF P2051 attention-window sidecar offsets are non-contiguous".into());
        }
        expected_start = end;
    }
    if data_start + expected_start != bytes.len() {
        return Err("HF P2051 attention-window sidecar has trailing bytes".into());
    }
    Ok(output)
}

fn load_artifact() -> TestResult<AttentionArtifact> {
    let manifest_path = regular_file(
        &required_path(MANIFEST_ENV)?,
        "HF P2051 attention-window manifest",
    )?;
    let sidecar_path = regular_file(
        &required_path(SIDECAR_ENV)?,
        "HF P2051 attention-window sidecar",
    )?;
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let layer_index = validate_manifest_contract(&manifest)?;
    let tensors = parse_sidecar(&manifest, &sidecar_path, layer_index)?;
    Ok(AttentionArtifact {
        layer_index,
        manifest_path,
        manifest_sha256: sha256_hex(&manifest_bytes),
        sidecar_sha256: sha256_file(&sidecar_path)?,
        sidecar_path,
        tensors,
    })
}

fn tensor<'a>(artifact: &'a AttentionArtifact, suffix: &str) -> TestResult<&'a [u8]> {
    artifact
        .tensors
        .get(&format!("layer{}.{suffix}", artifact.layer_index))
        .map(Vec::as_slice)
        .ok_or_else(|| format!("attention trace tensor {suffix} is missing").into())
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
    if staging.byte_len() != buffer.byte_len() {
        return Err("download staging and device buffer byte lengths differ".into());
    }
    buffer
        .copy_to_pinned_async(0, staging, 0, buffer.byte_len(), stream)?
        .synchronize()?;
    Ok(staging.to_vec()?)
}

fn bf16_metrics(expected: &[u8], actual: &[u8]) -> TestResult<Value> {
    if expected.len() != actual.len() || expected.len() % BF16_BYTES as usize != 0 {
        return Err("BF16 metric input byte lengths differ".into());
    }
    let mut mismatch_count = 0_u64;
    let mut first_mismatch = None;
    let mut max_abs = 0.0_f32;
    for (index, (expected_bits, actual_bits)) in expected
        .chunks_exact(BF16_BYTES as usize)
        .zip(actual.chunks_exact(BF16_BYTES as usize))
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
        "element_count": expected.len() / BF16_BYTES as usize,
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

fn execute_once(
    prepared: &PreparedPrefillAttention,
    query: &CudaDeviceBuffer,
    key: &CudaDeviceBuffer,
    value: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: &mut CudaDeviceBuffer,
    raw_qk_last: &mut CudaDeviceBuffer,
    scaled_masked_last: &mut CudaDeviceBuffer,
    probabilities_last: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let output_bytes = output.byte_len();
    let workspace_bytes = workspace.byte_len();
    let raw_bytes = raw_qk_last.byte_len();
    let scaled_bytes = scaled_masked_last.byte_len();
    let probabilities_bytes = probabilities_last.byte_len();
    let mut params = PrefillAttentionParams {
        query: CudaBufferSpan::new(query, CudaDType::BF16, 0, query.byte_len())?,
        key: CudaBufferSpan::new(key, CudaDType::BF16, 0, key.byte_len())?,
        value: CudaBufferSpan::new(value, CudaDType::BF16, 0, value.byte_len())?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_bytes)?,
        workspace: Some(CudaBufferSpanMut::new(
            workspace,
            CudaDType::BF16,
            0,
            workspace_bytes,
        )?),
    };
    let trace = HfEagerQwenP2051LastRowTrace {
        raw_qk_last: CudaBufferSpanMut::new(raw_qk_last, CudaDType::BF16, 0, raw_bytes)?,
        scaled_masked_last: CudaBufferSpanMut::new(
            scaled_masked_last,
            CudaDType::BF16,
            0,
            scaled_bytes,
        )?,
        probabilities_last: CudaBufferSpanMut::new(
            probabilities_last,
            CudaDType::BF16,
            0,
            probabilities_bytes,
        )?,
    };
    prepared.execute_hf_eager_qwen_p2051_last_row_traced(&mut params, trace, stream)?;
    Ok(())
}

fn run_candidate(artifact: &AttentionArtifact) -> TestResult<Value> {
    let query_bytes = tensor(artifact, "attention.query_bshd")?;
    let key_bytes = tensor(artifact, "attention.key_bshd")?;
    let value_bytes = tensor(artifact, "attention.value_bshd")?;
    let expected_raw = tensor(artifact, "attention.raw_qk_last")?;
    let expected_scaled = tensor(artifact, "attention.scaled_masked_last")?;
    let expected_probabilities = tensor(artifact, "attention.probabilities_last")?;
    let expected_context = tensor(artifact, "attention.context_bshd")?;
    let expected_mask = tensor(artifact, "attention.mask_last")?;
    let trace_bytes = u64::try_from(expected_raw.len())?;
    if expected_scaled.len() != expected_raw.len()
        || expected_probabilities.len() != expected_raw.len()
    {
        return Err("attention trace row byte lengths differ".into());
    }

    let request = PrefillAttentionRequest::new(1, S, QH, KVH, D, SCALE, AttentionMask::Causal);
    let (context, mut stream) = first_context()?;
    let prepared = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::HuggingFaceEagerQwenP2051Probe,
        AttentionBackendAvailability::linked(),
    )?;
    if prepared.backend() != AttentionBackend::HuggingFaceEagerQwenP2051Probe {
        return Err("attention trace selected a non-HF-eager P2051 backend".into());
    }
    let selection = prepared.selection_trace();
    if selection.implementation_id() != IMPLEMENTATION_ID {
        return Err("attention trace selected an unexpected implementation".into());
    }
    let mut transfer_staging =
        context.allocate_pinned_host_buffer(u64::try_from(query_bytes.len())?)?;
    let mut trace_staging = context.allocate_pinned_host_buffer(trace_bytes)?;
    let query = upload(&context, &mut stream, &mut transfer_staging, query_bytes)?;
    let key = upload(&context, &mut stream, &mut transfer_staging, key_bytes)?;
    let value = upload(&context, &mut stream, &mut transfer_staging, value_bytes)?;
    let mut output = context.allocate_device_buffer(u64::try_from(expected_context.len())?)?;
    let mut workspace = context.allocate_device_buffer(prepared.workspace_bytes())?;
    let mut raw_qk_last = context.allocate_device_buffer(trace_bytes)?;
    let mut scaled_masked_last = context.allocate_device_buffer(trace_bytes)?;
    let mut probabilities_last = context.allocate_device_buffer(trace_bytes)?;
    let before = context.allocation_stats()?;

    let result = (|| -> TestResult<Value> {
        execute_once(
            &prepared,
            &query,
            &key,
            &value,
            &mut output,
            &mut workspace,
            &mut raw_qk_last,
            &mut scaled_masked_last,
            &mut probabilities_last,
            &mut stream,
        )?;
        let raw = download_full(&mut stream, &mut raw_qk_last, &mut trace_staging)?;
        let scaled = download_full(&mut stream, &mut scaled_masked_last, &mut trace_staging)?;
        let probabilities =
            download_full(&mut stream, &mut probabilities_last, &mut trace_staging)?;
        let output_context = download_full(&mut stream, &mut output, &mut transfer_staging)?;
        if context.allocation_stats()? != before {
            return Err("P2051 last-row trace hot path changed allocations".into());
        }
        execute_once(
            &prepared,
            &query,
            &key,
            &value,
            &mut output,
            &mut workspace,
            &mut raw_qk_last,
            &mut scaled_masked_last,
            &mut probabilities_last,
            &mut stream,
        )?;
        let repeated_context = download_full(&mut stream, &mut output, &mut transfer_staging)?;
        if context.allocation_stats()? != before {
            return Err("repeated P2051 last-row trace changed allocations".into());
        }
        let repeat = bf16_metrics(&output_context, &repeated_context)?;
        let raw_metrics = bf16_metrics(expected_raw, &raw)?;
        let scaled_metrics = bf16_metrics(expected_scaled, &scaled)?;
        let probability_metrics = bf16_metrics(expected_probabilities, &probabilities)?;
        let context_metrics = bf16_metrics(expected_context, &output_context)?;
        Ok(json!({
            "implementation_id": selection.implementation_id(),
            "implementation_version": selection.implementation_version(),
            "selection_reason": format!("{:?}", selection.reason()),
            "cache_free": true,
            "hot_execution_allocation_stable": true,
            "hf_mask_last_bf16_le_sha256": sha256_hex(expected_mask),
            "stages": {
                "raw_qk_last": raw_metrics,
                "scaled_masked_last": scaled_metrics,
                "probabilities_last": probability_metrics,
                "context_bshd": context_metrics,
            },
            "repeat_execution": {
                "context_bshd_bf16_exact": exact(&repeat)?,
                "context_bshd": repeat,
            },
        }))
    })();

    let mut cleanup_failures = Vec::new();
    for (label, close) in [
        ("prepared attention plan", prepared.close()),
        ("probability trace", probabilities_last.close()),
        ("scaled/masked trace", scaled_masked_last.close()),
        ("raw QK trace", raw_qk_last.close()),
        ("workspace", workspace.close()),
        ("output", output.close()),
        ("value", value.close()),
        ("key", key.close()),
        ("query", query.close()),
        ("trace staging", trace_staging.close()),
        ("transfer staging", transfer_staging.close()),
    ] {
        if let Err(error) = close {
            cleanup_failures.push(format!("{label} close failed: {error}"));
        }
    }
    if let Err(error) = stream.close() {
        cleanup_failures.push(format!("stream close failed: {error}"));
    }
    if let Err(error) = context.synchronize() {
        cleanup_failures.push(format!("context synchronize failed: {error}"));
    }
    let lifecycle_clean = match context.allocation_stats() {
        Ok(stats) if stats.is_zero() => true,
        Ok(stats) => {
            cleanup_failures.push(format!("CUDA allocation accounting is non-zero: {stats:?}"));
            false
        }
        Err(error) => {
            cleanup_failures.push(format!("CUDA allocation accounting failed: {error}"));
            false
        }
    };
    if let Err(error) = context.close() {
        cleanup_failures.push(format!("context close failed: {error}"));
    }
    let mut result = result?;
    result["lifecycle_clean_after_close"] = json!(lifecycle_clean);
    if cleanup_failures.is_empty() {
        Ok(result)
    } else {
        Err(cleanup_failures.join("; ").into())
    }
}

fn output_path() -> TestResult<PathBuf> {
    let output = required_path(OUTPUT_ENV)?;
    if !output.is_absolute() || output.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err(format!("{OUTPUT_ENV} must be an absolute .json path").into());
    }
    let parent = output
        .parent()
        .ok_or("attention-window output has no parent")?;
    fs::create_dir_all(parent)?;
    let parent = parent.canonicalize()?;
    Ok(parent.join(
        output
            .file_name()
            .ok_or("attention-window output has no basename")?,
    ))
}

fn write_artifact_exclusive(path: &Path, document: &Value) -> TestResult {
    if fs::symlink_metadata(path).is_ok() {
        return Err("refusing to overwrite an attention-window receipt".into());
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
#[ignore = "requires a remote source-bound Qwen2.5-3B P2051 eager attention artifact and CUDA GPU"]
fn qwen3b_p2051_hf_eager_attention_window_quality_gate() -> TestResult {
    let artifact = load_artifact()?;
    let candidate = run_candidate(&artifact)?;
    let stages = candidate
        .get("stages")
        .and_then(Value::as_object)
        .ok_or("candidate attention stages are missing")?;
    let raw_exact = exact(stages.get("raw_qk_last").ok_or("raw QK stage is missing")?)?;
    let scaled_exact = exact(
        stages
            .get("scaled_masked_last")
            .ok_or("scaled/masked stage is missing")?,
    )?;
    let probabilities_exact = exact(
        stages
            .get("probabilities_last")
            .ok_or("probability stage is missing")?,
    )?;
    let context_exact = exact(
        stages
            .get("context_bshd")
            .ok_or("context stage is missing")?,
    )?;
    let repeat_exact =
        candidate["repeat_execution"]["context_bshd_bf16_exact"] == Value::Bool(true);
    let quality_pass = raw_exact
        && scaled_exact
        && probabilities_exact
        && context_exact
        && repeat_exact
        && candidate["hot_execution_allocation_stable"] == Value::Bool(true)
        && candidate["lifecycle_clean_after_close"] == Value::Bool(true);
    let receipt = json!({
        "schema_version": RESULT_SCHEMA,
        "artifact_kind": RESULT_KIND,
        "created_at_unix_seconds": unix_seconds()?,
        "performance_claim_eligible": false,
        "serving_selector_eligible": false,
        "contract": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layer_index": artifact.layer_index,
            "token_count": S,
            "query_head_count": QH,
            "key_value_head_count": KVH,
            "head_size": D,
            "scale": SCALE,
            "cache_free": true,
            "candidate_opt_in_only": true,
        },
        "hf_attention_window_artifact": {
            "manifest_path": artifact.manifest_path,
            "manifest_sha256": artifact.manifest_sha256,
            "sidecar_path": artifact.sidecar_path,
            "sidecar_sha256": artifact.sidecar_sha256,
        },
        "candidate": candidate,
        "quality_gate": {
            "raw_qk_last_bf16_exact": raw_exact,
            "scaled_masked_last_bf16_exact": scaled_exact,
            "probabilities_last_bf16_exact": probabilities_exact,
            "context_bshd_bf16_exact": context_exact,
            "repeat_context_bshd_bf16_exact": repeat_exact,
            "quality_pass": quality_pass,
            "corrected_cache_on_eligible": false,
            "serving_selector_eligible": false,
        },
    });
    let output = output_path()?;
    write_artifact_exclusive(&output, &receipt)?;
    println!("{MARKER_PREFIX}{}", serde_json::to_string(&receipt)?);
    if !quality_pass {
        return Err(
            "source-bound Q/K/V P2051 attention quality gate failed; cache-on and serving promotion remain blocked"
                .into(),
        );
    }
    Ok(())
}
