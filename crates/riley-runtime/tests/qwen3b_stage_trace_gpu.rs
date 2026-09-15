//! Remote-only Qwen2.5-3B P2048 pre-attention numerical discriminator.
#![cfg(feature = "cuda")]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
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
const HF_ARTIFACT_KIND: &str = "qwen3b-hf-eager-layer0-pre-attention-trace";
const RESULT_ARTIFACT_KIND: &str = "qwen3b-riley-layer0-pre-attention-comparison";
const FUSED_QKV_RESULT_ARTIFACT_KIND: &str =
    "qwen3b-riley-cublaslt-bias-epilogue-layer0-qkv-comparison";
const TRACE_ID: &str = "qwen3b-p2048-layer0-pre-attention-v1";
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
const QWEN3B_QUERY_HEADS: usize = 16;
const QWEN3B_KEY_VALUE_HEADS: usize = 2;
const QWEN3B_HEAD_DIMENSION: usize = 128;
const QWEN3B_VOCABULARY_SIZE: usize = 151_936;
const EXPECTED_SOURCE_ARCHITECTURE: &str = "Qwen2ForCausalLM";

const TRACE_STAGES: [(LlamaTracePoint, &str, &str); 7] = [
    (LlamaTracePoint::Embedding, "embedding", "trace/embedding"),
    (
        LlamaTracePoint::Layer0InputNorm,
        "layer0.input_norm",
        "trace/layer0/input_norm",
    ),
    (
        LlamaTracePoint::Layer0QueryProjection,
        "layer0.q_proj",
        "trace/layer0/q_proj",
    ),
    (
        LlamaTracePoint::Layer0KeyProjection,
        "layer0.k_proj",
        "trace/layer0/k_proj",
    ),
    (
        LlamaTracePoint::Layer0ValueProjection,
        "layer0.v_proj",
        "trace/layer0/v_proj",
    ),
    (
        LlamaTracePoint::Layer0QueryRotary,
        "layer0.q_rope",
        "trace/layer0/q_rope",
    ),
    (
        LlamaTracePoint::Layer0KeyRotary,
        "layer0.k_rope",
        "trace/layer0/k_rope",
    ),
];

const FUSED_QKV_STAGES: [(LlamaTracePoint, &str, &str); 3] = [
    (
        LlamaTracePoint::Layer0QueryProjection,
        "layer0.q_proj",
        "trace/layer0/q_proj",
    ),
    (
        LlamaTracePoint::Layer0KeyProjection,
        "layer0.k_proj",
        "trace/layer0/k_proj",
    ),
    (
        LlamaTracePoint::Layer0ValueProjection,
        "layer0.v_proj",
        "trace/layer0/v_proj",
    ),
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

fn trace_inputs(output_variable: &str) -> TestResult<TraceInputs> {
    Ok(TraceInputs {
        checkpoint: required_path("RILEY_QWEN3B_CHECKPOINT")?,
        workload: required_path("RILEY_QWEN_SERVING_WORKLOAD")?,
        manifest: required_path("RILEY_QWEN3B_STAGE_TRACE_MANIFEST")?,
        sidecar: required_path("RILEY_QWEN3B_STAGE_TRACE_SIDECAR")?,
        output: required_path(output_variable)?,
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
        "embedding" | "layer0.input_norm" | "layer0.q_proj" => vec![sequence, hidden],
        "layer0.k_proj" | "layer0.v_proj" => vec![sequence, kv],
        "layer0.q_rope" => vec![
            sequence,
            u64::try_from(QWEN3B_QUERY_HEADS).expect("heads fit"),
            u64::try_from(QWEN3B_HEAD_DIMENSION).expect("head dimension fits"),
        ],
        "layer0.k_rope" => vec![
            sequence,
            u64::try_from(QWEN3B_KEY_VALUE_HEADS).expect("KV heads fit"),
            u64::try_from(QWEN3B_HEAD_DIMENSION).expect("head dimension fits"),
        ],
        _ => unreachable!("fixed trace stage"),
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

fn read_manifest(path: &Path) -> TestResult<Value> {
    let source = regular_file(path, "HF trace manifest")?;
    let raw = fs::read(&source)?;
    let document: Value = serde_json::from_slice(&raw)?;
    if document["schema_version"] != TRACE_SCHEMA_VERSION
        || document["artifact_kind"] != HF_ARTIFACT_KIND
        || document["trace_id"] != TRACE_ID
        || document["performance_claim_eligible"] != false
    {
        return Err("HF trace manifest identity differs".into());
    }
    if document["trace_profile"]["id"] != TRACE_ID
        || document["trace_profile"]["capture_domain"] != "all-2048-token-positions"
        || document["trace_profile"]["tensor_count"] != TRACE_STAGES.len()
    {
        return Err("HF trace manifest profile differs".into());
    }
    if document["contract"]["model_id"] != QWEN3B_MODEL_ID
        || document["contract"]["model_revision"] != QWEN3B_REVISION
        || document["contract"]["workload"]["source_sha256"] != QWEN3B_WORKLOAD_SHA256
        || document["contract"]["workload"]["prompt_token_ids_le_u32_sha256"]
            != QWEN3B_PROMPT_TOKEN_SHA256
        || document["contract"]["execution"]["attention_implementation"] != "eager"
        || document["contract"]["execution"]["dtype"] != "bfloat16"
        || document["contract"]["execution"]["use_cache"] != false
    {
        return Err("HF trace manifest numerical contract differs".into());
    }
    let tensors = document["tensors"]
        .as_object()
        .ok_or("HF trace manifest tensors must be an object")?;
    if tensors.len() != TRACE_STAGES.len() {
        return Err("HF trace manifest tensor count differs".into());
    }
    for (_, name, key) in TRACE_STAGES {
        let tensor = tensors
            .get(name)
            .and_then(Value::as_object)
            .ok_or("HF trace manifest tensor is missing")?;
        if tensor.get("key").and_then(Value::as_str) != Some(key)
            || tensor.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || tensor.get("canonical_byte_order").and_then(Value::as_str)
                != Some("little-endian-u16")
        {
            return Err("HF trace manifest tensor metadata differs".into());
        }
        let shape = tensor["shape"]
            .as_array()
            .ok_or("HF trace manifest tensor shape must be an array")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or("HF trace tensor shape must be unsigned")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape != expected_shape(name) {
            return Err("HF trace manifest tensor shape differs".into());
        }
        let expected_bytes = shape_byte_len(&shape)?;
        if tensor["bf16_le_bytes"].as_u64() != Some(u64::try_from(expected_bytes)?)
            || tensor["bf16_le_sha256"]
                .as_str()
                .is_none_or(|hash| hash.len() != 64)
        {
            return Err("HF trace manifest tensor byte contract differs".into());
        }
    }
    Ok(document)
}

fn parse_sidecar_header(path: &Path) -> TestResult<(Map<String, Value>, u64, u64)> {
    let source = regular_file(path, "HF trace sidecar")?;
    let mut file = File::open(&source)?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)?;
    let header_length = usize::try_from(u64::from_le_bytes(length_bytes))?;
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES {
        return Err("HF trace sidecar header length differs".into());
    }
    let mut raw_header = vec![0_u8; header_length];
    file.read_exact(&mut raw_header)?;
    let header = serde_json::from_slice::<Value>(&raw_header)?
        .as_object()
        .cloned()
        .ok_or("HF trace sidecar header must be an object")?;
    let data_start = u64::try_from(8 + header_length)?;
    Ok((header, data_start, file.metadata()?.len()))
}

fn sidecar_tensors(
    manifest: &Value,
    sidecar: &Path,
) -> TestResult<BTreeMap<String, SidecarTensor>> {
    let metadata = manifest["sidecar"]
        .as_object()
        .ok_or("HF trace manifest sidecar must be an object")?;
    let actual_sidecar = regular_file(sidecar, "HF trace sidecar")?;
    let actual_sidecar_sha = sha256_file(&actual_sidecar)?;
    if metadata.get("path").and_then(Value::as_str)
        != actual_sidecar.file_name().and_then(|name| name.to_str())
        || metadata.get("sha256").and_then(Value::as_str) != Some(actual_sidecar_sha.as_str())
        || metadata.get("format").and_then(Value::as_str) != Some("safetensors")
    {
        return Err("HF trace sidecar binding differs".into());
    }
    let (header, data_start, sidecar_size) = parse_sidecar_header(&actual_sidecar)?;
    let mut output = BTreeMap::new();
    let manifest_tensors = manifest["tensors"]
        .as_object()
        .ok_or("HF trace manifest tensors must be an object")?;
    for (_, name, key) in TRACE_STAGES {
        let entry = header
            .get(key)
            .and_then(Value::as_object)
            .ok_or("HF trace sidecar tensor is missing")?;
        if entry.get("dtype").and_then(Value::as_str) != Some("BF16") {
            return Err("HF trace sidecar tensor dtype differs".into());
        }
        let shape = entry["shape"]
            .as_array()
            .ok_or("HF trace sidecar tensor shape must be an array")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or("HF trace sidecar shape must be unsigned")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape != expected_shape(name) {
            return Err("HF trace sidecar tensor shape differs".into());
        }
        let offsets = entry["data_offsets"]
            .as_array()
            .ok_or("HF trace sidecar offsets must be an array")?;
        if offsets.len() != 2 {
            return Err("HF trace sidecar offset count differs".into());
        }
        let start = offsets[0]
            .as_u64()
            .ok_or("HF trace sidecar start differs")?;
        let end = offsets[1].as_u64().ok_or("HF trace sidecar end differs")?;
        let expected_bytes = u64::try_from(shape_byte_len(&shape)?)?;
        let data_end = data_start
            .checked_add(end)
            .ok_or("HF trace sidecar data range overflows")?;
        if end < start || end - start != expected_bytes || data_end > sidecar_size {
            return Err("HF trace sidecar tensor range differs".into());
        }
        let manifest_tensor = manifest_tensors
            .get(name)
            .and_then(Value::as_object)
            .ok_or("HF trace manifest tensor missing")?;
        if manifest_tensor.get("key").and_then(Value::as_str) != Some(key) {
            return Err("HF trace manifest tensor key differs".into());
        }
        output.insert(
            name.to_owned(),
            SidecarTensor {
                data_start: data_start
                    .checked_add(start)
                    .ok_or("HF trace sidecar data range overflows")?,
                data_end,
            },
        );
    }
    if header
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .count()
        != TRACE_STAGES.len()
    {
        return Err("HF trace sidecar tensor set differs".into());
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
        return Err("HF trace sidecar tensor raw BF16 SHA-256 differs".into());
    }
    Ok(bytes)
}

fn decode_hf_bf16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn decode_riley_bf16(bytes: &[u8]) -> f32 {
    let bits = u16::from_ne_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn stage_metrics(hf: &[u8], riley: &[u8]) -> TestResult<Value> {
    if hf.len() != riley.len() || hf.len() % BF16_BYTES != 0 {
        return Err("trace tensor byte sizes differ".into());
    }
    let mut unequal_elements = 0_u64;
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
            unequal_elements += 1;
        }
        let expected_value = f64::from(decode_hf_bf16(expected));
        let actual_value = f64::from(decode_riley_bf16(actual));
        if !expected_value.is_finite() || !actual_value.is_finite() {
            return Err("trace tensor contains non-finite BF16 values".into());
        }
        let absolute = (expected_value - actual_value).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        dot += expected_value * actual_value;
        hf_square += expected_value * expected_value;
        riley_square += actual_value * actual_value;
    }
    let elements = u64::try_from(hf.len() / BF16_BYTES)?;
    let cosine = if hf_square == 0.0 || riley_square == 0.0 {
        None
    } else {
        Some(dot / (hf_square.sqrt() * riley_square.sqrt()))
    };
    Ok(json!({
        "element_count": elements,
        "bf16_exact": unequal_elements == 0,
        "unequal_element_count": unequal_elements,
        "max_abs_bf16_as_f32": max_abs,
        "mean_abs_bf16_as_f32": sum_abs / elements as f64,
        "cosine_similarity": cosine,
        "hf_bf16_le_sha256": sha256_hex(hf),
        "riley_bf16_le_sha256": sha256_hex(riley),
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
        return Err("trace comparison output must be an absolute .json path".into());
    }
    let root = repository_root()?;
    fs::create_dir_all(
        path.parent()
            .ok_or("trace comparison output has no parent")?,
    )?;
    let parent = path
        .parent()
        .ok_or("trace comparison output has no parent")?
        .canonicalize()?;
    let output = parent.join(
        path.file_name()
            .ok_or("trace comparison output has no basename")?,
    );
    if output == root || output.starts_with(&root) {
        return Err("trace comparison output must be outside the repository".into());
    }
    if fs::symlink_metadata(&output).is_ok() {
        return Err("refusing to overwrite existing trace comparison artifact".into());
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
#[ignore = "remote-only Qwen2.5-3B P2048 HF/Rust pre-attention discriminator"]
fn qwen3b_p2048_layer0_pre_attention_trace_matches_hf_artifact() -> TestResult {
    let inputs = trace_inputs("RILEY_QWEN3B_STAGE_TRACE_OUTPUT")?;
    let prompt_tokens = load_prompt_ids(&inputs.workload)?;
    let manifest = read_manifest(&inputs.manifest)?;
    let sidecar = regular_file(&inputs.sidecar, "HF trace sidecar")?;
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
    let points: Vec<_> = TRACE_STAGES.iter().map(|(point, _, _)| *point).collect();
    let mut trace = forward.prepare_trace_points(&points)?;
    forward.upload_tokens(&prompt_tokens, &mut stream)?;
    forward.execute_traced(&mut stream, &mut trace)?;
    if trace.captured_count() != u32::try_from(TRACE_STAGES.len())? {
        return Err("Riley trace did not capture every requested stage".into());
    }

    let manifest_tensors = manifest["tensors"]
        .as_object()
        .ok_or("HF trace manifest tensors must be an object")?;
    let mut stages = Map::new();
    let mut first_non_exact: Option<&str> = None;
    let mut exact_count = 0_u64;
    for (point, name, _) in TRACE_STAGES {
        let riley = trace.tensor(point).ok_or("Riley trace tensor is missing")?;
        let expected_shape = expected_shape(name);
        if riley.len() != shape_byte_len(&expected_shape)? {
            return Err("Riley trace tensor byte count differs from the trace contract".into());
        }
        let manifest_tensor = manifest_tensors
            .get(name)
            .and_then(Value::as_object)
            .ok_or("HF trace manifest tensor is missing")?;
        let expected_sha = manifest_tensor["bf16_le_sha256"]
            .as_str()
            .ok_or("HF trace manifest raw BF16 SHA-256 is missing")?;
        let hf_metadata = sidecar_tensors
            .get(name)
            .ok_or("HF trace sidecar tensor is missing")?;
        let hf = read_sidecar_tensor(&sidecar, hf_metadata, expected_sha)?;
        let metrics = stage_metrics(&hf, riley)?;
        if metrics["bf16_exact"] == true {
            exact_count += 1;
        } else if first_non_exact.is_none() {
            first_non_exact = Some(name);
        }
        stages.insert(name.to_owned(), metrics);
    }

    forward.close()?;
    drop(trace);
    context.synchronize()?;
    if !context.allocation_stats()?.is_zero() {
        return Err("Riley trace left a CUDA allocation after close".into());
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
        },
        "summary": {
            "stage_count": TRACE_STAGES.len(),
            "bf16_exact_stage_count": exact_count,
            "first_non_exact_stage": first_non_exact,
        },
        "stages": stages,
    });
    write_artifact_exclusive(&inputs.output, &result)?;
    println!(
        "QWEN3B_STAGE0_TRACE trace_id={} first_non_exact_stage={} exact_stages={} performance_claim_eligible=false",
        TRACE_ID,
        first_non_exact.unwrap_or("none"),
        exact_count,
    );
    Ok(())
}

#[test]
#[ignore = "remote-only Qwen2.5-3B P2048 integrated fused-QKV Hugging Face gate"]
fn qwen3b_p2048_fused_qkv_trace_matches_hf_module_output() -> TestResult {
    let inputs = trace_inputs("RILEY_QWEN3B_FUSED_QKV_TRACE_OUTPUT")?;
    let prompt_tokens = load_prompt_ids(&inputs.workload)?;
    let manifest = read_manifest(&inputs.manifest)?;
    let sidecar = regular_file(&inputs.sidecar, "HF trace sidecar")?;
    let sidecar_tensors = sidecar_tensors(&manifest, &sidecar)?;
    let model = load_qwen3b(&inputs.checkpoint)?;

    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let config = PreparedLlamaForwardConfig::default()
        .with_reference_attention()
        .with_cublaslt_bias_epilogue_projection_bias();
    let mut forward =
        PreparedLlamaForward::prepare(&model, &context, &mut stream, prompt_tokens.len(), config)?;
    let points: Vec<_> = FUSED_QKV_STAGES
        .iter()
        .map(|(point, _, _)| *point)
        .collect();
    let mut trace = forward.prepare_trace_points(&points)?;
    forward.upload_tokens(&prompt_tokens, &mut stream)?;
    forward.execute_traced(&mut stream, &mut trace)?;
    if trace.captured_count() != u32::try_from(FUSED_QKV_STAGES.len())? {
        return Err("Riley fused trace did not capture every Q/K/V stage".into());
    }

    let manifest_tensors = manifest["tensors"]
        .as_object()
        .ok_or("HF trace manifest tensors must be an object")?;
    let mut stages = Map::new();
    for (point, name, _) in FUSED_QKV_STAGES {
        let riley = trace
            .tensor(point)
            .ok_or("Riley fused trace tensor is missing")?;
        let expected_shape = expected_shape(name);
        if riley.len() != shape_byte_len(&expected_shape)? {
            return Err(
                "Riley fused trace tensor byte count differs from the trace contract".into(),
            );
        }
        let manifest_tensor = manifest_tensors
            .get(name)
            .and_then(Value::as_object)
            .ok_or("HF trace manifest tensor is missing")?;
        let expected_sha = manifest_tensor["bf16_le_sha256"]
            .as_str()
            .ok_or("HF trace manifest raw BF16 SHA-256 is missing")?;
        let hf_metadata = sidecar_tensors
            .get(name)
            .ok_or("HF trace sidecar tensor is missing")?;
        let hf = read_sidecar_tensor(&sidecar, hf_metadata, expected_sha)?;
        let metrics = stage_metrics(&hf, riley)?;
        if metrics["bf16_exact"] != true {
            return Err(
                format!("fused {name} differs from the unmodified HF module output").into(),
            );
        }
        stages.insert(name.to_owned(), metrics);
    }

    let selected_mode = forward.projection_bias_mode().id();
    forward.close()?;
    drop(trace);
    context.synchronize()?;
    if !context.allocation_stats()?.is_zero() {
        return Err("Riley fused trace left a CUDA allocation after close".into());
    }
    stream.close()?;
    context.close()?;

    let result = json!({
        "schema_version": TRACE_SCHEMA_VERSION,
        "artifact_kind": FUSED_QKV_RESULT_ARTIFACT_KIND,
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
            "projection_bias_mode": selected_mode,
            "use_cache": false,
            "dtype": "bfloat16",
        },
        "summary": {
            "stage_count": FUSED_QKV_STAGES.len(),
            "bf16_exact_stage_count": FUSED_QKV_STAGES.len(),
        },
        "stages": stages,
    });
    write_artifact_exclusive(&inputs.output, &result)?;
    println!(
        "QWEN3B_FUSED_QKV_TRACE trace_id={} exact_stages={} performance_claim_eligible=false",
        TRACE_ID,
        FUSED_QKV_STAGES.len(),
    );
    Ok(())
}
