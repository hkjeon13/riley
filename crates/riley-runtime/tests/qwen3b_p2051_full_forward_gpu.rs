//! Remote-only Qwen2.5-3B P2051 full-forward HF-compatible-bias qualifier.
//!
//! The input is the immutable P2048 workload followed by the first three
//! cache-off teacher tokens from an externally validated Hugging Face artifact.
//! Hugging Face produces the paired last-token BF16 sidecar offline. This test
//! consumes that sidecar from Rust, runs no Python, and never represents a
//! serving performance result.
//!
//! The baseline strict staged-bias profile is compared with the feature-gated
//! HF-compatible cuBLASLt BIAS candidate. The candidate is then repeated with
//! the opt-in Qwen P2051 cuBLASLt attention probe, followed by the paired
//! direct-cuBLAS output-projection probe, then with the paired direct
//! MLP-projection probe, and finally with a last-token direct-cuBLAS LM head
//! matching Hugging Face's `logits_to_keep=1` invocation. Layer-zero
//! boundaries are captured with the existing
//! selective trace API; every decoder residual output is captured as one
//! last-token row through the bounded layer-trace API.

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
    LlamaLastTokenLayerStage, LlamaProjectionBiasMode, LlamaTracePoint, PreparedLlamaDecode,
    PreparedLlamaDecodeConfig, PreparedLlamaDecodeM1Trace, PreparedLlamaForward,
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
const FULL_FORWARD_UPLOAD_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const FULL_FORWARD_IO_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const STRICT_GEMM_WORKSPACE_CAP_BYTES: u64 = 16 * 1024 * 1024;
// The direct P2051 discriminator established this as the maximum workspace
// allowance under which PyTorch's actual `torch.addmm` Q/K/V profiles are
// reproduced. It is a qualifier input, not a serving workspace policy.
const HF_COMPAT_GEMM_WORKSPACE_CAP_BYTES: u64 = 1024 * 1024;
const REFERENCE_ATTENTION_BUDGET_BYTES: u64 = 1_342_177_280;
const STAGE_SCHEMA_VERSION: &str = "riley.qwen3b-hf-eager-p2051-cache-off-layer-stage-trace.v1";
const STAGE_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-layer-stage-trace";
const STAGE_TRACE_ID: &str = "qwen3b-p2051-cache-off-last-token-layer-stage-v1";
const FULL_SEQUENCE_STAGE_SCHEMA_VERSION: &str =
    "riley.qwen3b-hf-eager-p2051-cache-off-full-sequence-layer-stage-trace.v3";
const FULL_SEQUENCE_STAGE_ARTIFACT_KIND: &str =
    "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-full-sequence-selected-layer-stage-rope-table-trace";
const FULL_SEQUENCE_STAGE_LAYER_INDEX: usize = 2;
const RESULT_SCHEMA_VERSION: &str =
    "riley.qwen3b-p2051-hf-compatible-cache-free-full-forward-comparison.v2";
const RESULT_ARTIFACT_KIND: &str =
    "qwen2.5-3b-riley-p2051-hf-compatible-cache-free-full-forward-comparison";
const MARKER_PREFIX: &str = "RILEY_QWEN3B_P2051_HF_COMPAT_FULL_FORWARD=";
const FULL_SEQUENCE_RESULT_SCHEMA_VERSION: &str =
    "riley.qwen3b-p2051-hf-compatible-cache-free-full-sequence-layer-stage-comparison.v1";
const FULL_SEQUENCE_RESULT_ARTIFACT_KIND: &str =
    "qwen2.5-3b-riley-p2051-hf-compatible-cache-free-full-sequence-layer-stage-comparison";
const FULL_SEQUENCE_MARKER_PREFIX: &str = "RILEY_QWEN3B_P2051_HF_COMPAT_FULL_SEQUENCE_LAYER_STAGE=";
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
const DENSE_REFERENCE_ATTENTION_BACKEND_ID: &str = "riley.cuda.materialized-gqa-prefill.bf16";
const HF_EAGER_QWEN_P2051_PROBE_ATTENTION_BACKEND_ID: &str =
    "riley.cuda.hf-eager-cublaslt-qwen-p2051-probe.bf16";
const STRICT_OUTPUT_PROJECTION_BACKEND_ID: &str = "strict-hidden-gemm-v1";
const HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID: &str =
    "hf-eager-qwen-p2051-direct-cublas-probe-v1";
const STRICT_MLP_PROJECTION_BACKEND_ID: &str = "strict-staged-v1";
const HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID: &str =
    "hf-eager-qwen-p2051-direct-cublas-probe-v1";
const STRICT_LM_HEAD_BACKEND_ID: &str = "strict-full-sequence-gemm-v1";
const HF_EAGER_QWEN_P2051_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID: &str =
    "hf-eager-qwen-p2051-last-token-direct-cublas-probe-v1";
const HF_EAGER_QWEN_P2051_RMS_NORM_BACKEND_ID: &str = "hf-cuda-qwen-p2051-rmsnorm-probe-v1";
const TEACHER_ARTIFACT_SCHEMA: &str = "riley.qwen3b-hf-eager-teacher-forced-generation.v1";
const TEACHER_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2048-teacher-forced-generation";
const TEACHER_CACHE_OFF_SIDECAR_KEY: &str = "teacher_forced/logits";
const CHECKPOINT_RECEIPT_FILENAME: &str = "riley-checkpoint.json";
const CACHE_ON_STAGE_SCHEMA_VERSION: &str =
    "riley.qwen3b-hf-eager-p2048-cache-on-layer-stage-trace.v1";
const CACHE_ON_STAGE_ARTIFACT_KIND: &str =
    "qwen2.5-3b-hf-eager-bf16-p2048-cache-on-layer-stage-trace";
const CACHE_ON_STAGE_TRACE_ID: &str = "qwen3b-p2048-cache-on-prefill-m1-layer-stage-v1";
const CACHE_ON_PREFILL_RESULT_SCHEMA_VERSION: &str =
    "riley.qwen3b-p2048-hf-compatible-cache-on-prefill-comparison.v1";
const CACHE_ON_PREFILL_RESULT_ARTIFACT_KIND: &str =
    "qwen2.5-3b-riley-p2048-hf-compatible-cache-on-prefill-comparison";
const CACHE_ON_PREFILL_MARKER_PREFIX: &str = "RILEY_QWEN3B_P2048_CACHE_ON_PREFILL=";
const CACHE_ON_M1_RESULT_SCHEMA_VERSION: &str =
    "riley.qwen3b-p2048-hf-compatible-cache-on-m1-reference-trace.v1";
const CACHE_ON_M1_RESULT_ARTIFACT_KIND: &str =
    "qwen2.5-3b-riley-p2048-hf-compatible-cache-on-m1-reference-trace";
const CACHE_ON_M1_MARKER_PREFIX: &str = "RILEY_QWEN3B_P2048_CACHE_ON_M1=";
const HF_EAGER_QWEN_P2048_CACHE_ON_PROBE_ATTENTION_BACKEND_ID: &str =
    "riley.cuda.hf-eager-cublaslt-qwen-p2048-cache-on-probe.bf16";
const HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID: &str =
    "hf-eager-qwen-p2048-cache-on-direct-cublas-probe-v1";
const HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID: &str =
    "hf-eager-qwen-p2048-cache-on-direct-cublas-probe-v1";
const HF_EAGER_QWEN_P2048_CACHE_ON_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID: &str =
    "hf-eager-qwen-p2048-cache-on-last-token-direct-cublas-probe-v1";
const HF_EAGER_QWEN_P2048_CACHE_ON_RMS_NORM_BACKEND_ID: &str =
    "hf-cuda-qwen-p2048-cache-on-rmsnorm-probe-v1";

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
struct TeacherCacheOn {
    artifact_sha256: String,
    cache_on_sidecar_sha256: String,
    full_teacher_token_ids_sha256: String,
    token_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
enum StageSource {
    Static(LlamaTracePoint, usize),
    LayerOutput,
    FinalNormOutput,
    LastLogits,
    FullSequence(LlamaLastTokenLayerStage),
    HfEagerRopeTableCosine,
    HfEagerRopeTableSine,
}

#[derive(Clone, Copy, Debug)]
enum RopeTableProfile {
    Default,
    HuggingFaceCudaQwenP2051ProbeV1,
}

impl RopeTableProfile {
    const fn id(self) -> &'static str {
        match self {
            Self::Default => "default-rope-table-v1",
            Self::HuggingFaceCudaQwenP2051ProbeV1 => "hf-cuda-rope-table-probe-v1",
        }
    }

    const fn configure(self, config: PreparedLlamaForwardConfig) -> PreparedLlamaForwardConfig {
        match self {
            Self::Default => config,
            Self::HuggingFaceCudaQwenP2051ProbeV1 => {
                config.with_hugging_face_cuda_qwen_p2051_rope_table_probe()
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RmsNormProfile {
    Default,
    HuggingFaceCudaQwenP2051ProbeV1,
}

impl RmsNormProfile {
    const fn id(self) -> &'static str {
        match self {
            Self::Default => "default-rmsnorm-v1",
            Self::HuggingFaceCudaQwenP2051ProbeV1 => "hf-cuda-qwen-p2051-rmsnorm-probe-v1",
        }
    }

    const fn backend_id(self) -> &'static str {
        match self {
            Self::Default => "canonical-rmsnorm-v1",
            Self::HuggingFaceCudaQwenP2051ProbeV1 => HF_EAGER_QWEN_P2051_RMS_NORM_BACKEND_ID,
        }
    }

    const fn configure(self, config: PreparedLlamaForwardConfig) -> PreparedLlamaForwardConfig {
        match self {
            Self::Default => config,
            Self::HuggingFaceCudaQwenP2051ProbeV1 => {
                config.with_hugging_face_cuda_qwen_p2051_rms_norm_probe()
            }
        }
    }
}

#[derive(Clone, Debug)]
struct StageSpec {
    name: String,
    shape: Vec<u64>,
    source: StageSource,
}

#[derive(Clone, Copy, Debug)]
enum AttentionProfile {
    Reference,
    HfEagerQwenP2051Probe,
}

impl AttentionProfile {
    const fn id(self) -> &'static str {
        match self {
            Self::Reference => "reference-attention-v1",
            Self::HfEagerQwenP2051Probe => "hf-eager-qwen-p2051-probe-v1",
        }
    }

    const fn backend_id(self) -> &'static str {
        match self {
            Self::Reference => DENSE_REFERENCE_ATTENTION_BACKEND_ID,
            Self::HfEagerQwenP2051Probe => HF_EAGER_QWEN_P2051_PROBE_ATTENTION_BACKEND_ID,
        }
    }

    const fn configure(self, config: PreparedLlamaForwardConfig) -> PreparedLlamaForwardConfig {
        match self {
            Self::Reference => config.with_reference_attention(),
            Self::HfEagerQwenP2051Probe => {
                config.with_hugging_face_eager_qwen_p2051_probe_attention()
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OutputProjectionProfile {
    StrictHiddenGemmV1,
    HfEagerQwenP2051DirectCublasProbeV1,
}

impl OutputProjectionProfile {
    const fn id(self) -> &'static str {
        match self {
            Self::StrictHiddenGemmV1 => "strict-hidden-gemm-v1",
            Self::HfEagerQwenP2051DirectCublasProbeV1 => {
                "hf-eager-qwen-p2051-direct-cublas-probe-v1"
            }
        }
    }

    const fn backend_id(self) -> &'static str {
        match self {
            Self::StrictHiddenGemmV1 => STRICT_OUTPUT_PROJECTION_BACKEND_ID,
            Self::HfEagerQwenP2051DirectCublasProbeV1 => {
                HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID
            }
        }
    }

    const fn configure(self, config: PreparedLlamaForwardConfig) -> PreparedLlamaForwardConfig {
        match self {
            Self::StrictHiddenGemmV1 => config,
            Self::HfEagerQwenP2051DirectCublasProbeV1 => {
                config.with_hf_eager_qwen_p2051_direct_cublas_output_projection_probe()
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum MlpProjectionProfile {
    StrictStagedV1,
    HfEagerQwenP2051DirectCublasProbeV1,
}

impl MlpProjectionProfile {
    const fn id(self) -> &'static str {
        match self {
            Self::StrictStagedV1 => "strict-staged-v1",
            Self::HfEagerQwenP2051DirectCublasProbeV1 => {
                "hf-eager-qwen-p2051-direct-cublas-probe-v1"
            }
        }
    }

    const fn backend_id(self) -> &'static str {
        match self {
            Self::StrictStagedV1 => STRICT_MLP_PROJECTION_BACKEND_ID,
            Self::HfEagerQwenP2051DirectCublasProbeV1 => {
                HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID
            }
        }
    }

    const fn configure(self, config: PreparedLlamaForwardConfig) -> PreparedLlamaForwardConfig {
        match self {
            Self::StrictStagedV1 => config,
            Self::HfEagerQwenP2051DirectCublasProbeV1 => {
                config.with_hf_eager_qwen_p2051_direct_cublas_mlp_projection_probe()
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum LmHeadProfile {
    StrictFullSequenceGemmV1,
    HfEagerQwenP2051LastTokenDirectCublasProbeV1,
}

impl LmHeadProfile {
    const fn id(self) -> &'static str {
        match self {
            Self::StrictFullSequenceGemmV1 => "strict-full-sequence-gemm-v1",
            Self::HfEagerQwenP2051LastTokenDirectCublasProbeV1 => {
                "hf-eager-qwen-p2051-last-token-direct-cublas-probe-v1"
            }
        }
    }

    const fn backend_id(self) -> &'static str {
        match self {
            Self::StrictFullSequenceGemmV1 => STRICT_LM_HEAD_BACKEND_ID,
            Self::HfEagerQwenP2051LastTokenDirectCublasProbeV1 => {
                HF_EAGER_QWEN_P2051_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID
            }
        }
    }

    const fn configure(self, config: PreparedLlamaForwardConfig) -> PreparedLlamaForwardConfig {
        match self {
            Self::StrictFullSequenceGemmV1 => config,
            Self::HfEagerQwenP2051LastTokenDirectCublasProbeV1 => {
                config.with_hf_eager_qwen_p2051_last_token_direct_cublas_lm_head_probe()
            }
        }
    }
}

#[derive(Debug)]
struct HfStageArtifact {
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
struct HfFullSequenceStageArtifact {
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
struct HfCacheOnPrefillStageArtifact {
    manifest_path: PathBuf,
    manifest_sha256: String,
    sidecar_path: PathBuf,
    sidecar_sha256: String,
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
    let digest = digest.finalize();
    Ok(hex_encode(&digest))
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

fn parse_teacher_cache_on_sidecar(path: &Path) -> TestResult {
    let source = regular_file(path, "HF cache-on sidecar")?;
    let bytes = fs::read(source)?;
    if bytes.len() < 8 {
        return Err("HF cache-on sidecar is too short".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES || 8 + header_len > bytes.len()
    {
        return Err("HF cache-on sidecar header differs".into());
    }
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
    let tensor = header
        .get(TEACHER_CACHE_OFF_SIDECAR_KEY)
        .and_then(Value::as_object)
        .ok_or("HF cache-on sidecar tensor is missing")?;
    if tensor.get("dtype").and_then(Value::as_str) != Some("BF16")
        || tensor.get("shape") != Some(&json!([128, QWEN3B_VOCABULARY_SIZE]))
    {
        return Err("HF cache-on sidecar tensor contract differs".into());
    }
    Ok(())
}

fn load_teacher_cache_on() -> TestResult<TeacherCacheOn> {
    let prefix = load_teacher_prefix()?;
    let cache_on_sidecar_path = regular_file(
        &required_path("RILEY_QWEN_HF_CACHE_ON_SIDECAR")?,
        "HF cache-on sidecar",
    )?;
    parse_teacher_cache_on_sidecar(&cache_on_sidecar_path)?;
    let token_ids = prefix
        .token_ids
        .get(..2)
        .ok_or("HF teacher cache-on token prefix is too short")?
        .to_vec();
    Ok(TeacherCacheOn {
        artifact_sha256: prefix.artifact_sha256,
        cache_on_sidecar_sha256: sha256_file(&cache_on_sidecar_path)?,
        full_teacher_token_ids_sha256: prefix.full_teacher_token_ids_sha256,
        token_ids,
    })
}

fn expected_stage_specs() -> Vec<StageSpec> {
    let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE).expect("hidden fits");
    let intermediate = u64::try_from(QWEN3B_INTERMEDIATE_SIZE).expect("intermediate fits");
    let query_heads = u64::try_from(QWEN3B_QUERY_HEADS).expect("heads fit");
    let key_value_heads = u64::try_from(QWEN3B_KEY_VALUE_HEADS).expect("KV heads fit");
    let head_dimension = u64::try_from(QWEN3B_HEAD_DIMENSION).expect("head dimension fits");
    let hidden_stage = |name: &str, point| StageSpec {
        name: name.to_owned(),
        shape: vec![hidden],
        source: StageSource::Static(point, QWEN3B_HIDDEN_SIZE),
    };
    let mut stages = vec![
        hidden_stage("embedding.last", LlamaTracePoint::Embedding),
        hidden_stage("layer0.input_norm.last", LlamaTracePoint::Layer0InputNorm),
        hidden_stage("layer0.q_proj.last", LlamaTracePoint::Layer0QueryProjection),
        StageSpec {
            name: "layer0.k_proj.last".to_owned(),
            shape: vec![key_value_heads * head_dimension],
            source: StageSource::Static(
                LlamaTracePoint::Layer0KeyProjection,
                QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            ),
        },
        StageSpec {
            name: "layer0.v_proj.last".to_owned(),
            shape: vec![key_value_heads * head_dimension],
            source: StageSource::Static(
                LlamaTracePoint::Layer0ValueProjection,
                QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            ),
        },
        StageSpec {
            name: "layer0.q_rope.last".to_owned(),
            shape: vec![query_heads, head_dimension],
            source: StageSource::Static(LlamaTracePoint::Layer0QueryRotary, QWEN3B_HIDDEN_SIZE),
        },
        StageSpec {
            name: "layer0.k_rope.last".to_owned(),
            shape: vec![key_value_heads, head_dimension],
            source: StageSource::Static(
                LlamaTracePoint::Layer0KeyRotary,
                QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            ),
        },
        hidden_stage(
            "layer0.attention_context.last",
            LlamaTracePoint::Layer0AttentionContext,
        ),
        hidden_stage(
            "layer0.after_attention_residual.last",
            LlamaTracePoint::Layer0AfterAttentionResidual,
        ),
        hidden_stage(
            "layer0.post_attention_norm.last",
            LlamaTracePoint::Layer0PostAttentionNorm,
        ),
        StageSpec {
            name: "layer0.gate_proj.last".to_owned(),
            shape: vec![intermediate],
            source: StageSource::Static(
                LlamaTracePoint::Layer0GateProjection,
                QWEN3B_INTERMEDIATE_SIZE,
            ),
        },
        StageSpec {
            name: "layer0.up_proj.last".to_owned(),
            shape: vec![intermediate],
            source: StageSource::Static(
                LlamaTracePoint::Layer0UpProjection,
                QWEN3B_INTERMEDIATE_SIZE,
            ),
        },
        StageSpec {
            name: "layer0.gated.last".to_owned(),
            shape: vec![intermediate],
            source: StageSource::Static(LlamaTracePoint::Layer0Gated, QWEN3B_INTERMEDIATE_SIZE),
        },
        hidden_stage(
            "layer0.down_proj.last",
            LlamaTracePoint::Layer0DownProjection,
        ),
        hidden_stage("layer0.output.last", LlamaTracePoint::Layer0Output),
    ];
    for layer_index in 1..QWEN3B_LAYER_COUNT {
        stages.push(StageSpec {
            name: format!("layer{layer_index}.output.last"),
            shape: vec![hidden],
            source: StageSource::LayerOutput,
        });
    }
    stages.push(StageSpec {
        name: "final_norm.output.last".to_owned(),
        shape: vec![hidden],
        source: StageSource::FinalNormOutput,
    });
    stages.push(StageSpec {
        name: "last_logits".to_owned(),
        shape: vec![u64::try_from(QWEN3B_VOCABULARY_SIZE).expect("vocabulary fits")],
        source: StageSource::LastLogits,
    });
    assert_eq!(stages.len(), 52);
    stages
}

fn expected_cache_on_stage_specs() -> Vec<StageSpec> {
    let base = expected_stage_specs();
    let mut stages = Vec::with_capacity(base.len() * 3);
    for step in ["prefill", "decode_step_1", "decode_step_2"] {
        for spec in &base {
            stages.push(StageSpec {
                name: format!("{step}.{}", spec.name),
                shape: spec.shape.clone(),
                source: spec.source,
            });
        }
    }
    assert_eq!(stages.len(), 156);
    stages
}

fn expected_cache_on_prefill_stage_specs() -> Vec<StageSpec> {
    expected_cache_on_stage_specs()
        .into_iter()
        .filter(|spec| spec.name.starts_with("prefill."))
        .collect()
}

fn full_sequence_stage_trace_id(layer_index: usize) -> String {
    format!("qwen3b-p2051-cache-off-full-sequence-layer{layer_index}-stage-rope-table-v3")
}

fn expected_full_sequence_stage_specs(layer_index: usize) -> Vec<StageSpec> {
    let sequence = u64::try_from(CONTEXT_TOKEN_COUNT).expect("context token count fits");
    let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE).expect("hidden size fits");
    let key_value_width = u64::try_from(QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION)
        .expect("key/value width fits");
    let hidden_stage = |name: &str, stage| StageSpec {
        name: name.to_owned(),
        shape: vec![sequence, hidden],
        source: StageSource::FullSequence(stage),
    };
    vec![
        hidden_stage(
            &format!("layer{layer_index}.input_norm.full"),
            LlamaLastTokenLayerStage::InputNorm,
        ),
        hidden_stage(
            &format!("layer{layer_index}.q_proj.full"),
            LlamaLastTokenLayerStage::QueryProjection,
        ),
        StageSpec {
            name: format!("layer{layer_index}.k_proj.full"),
            shape: vec![sequence, key_value_width],
            source: StageSource::FullSequence(LlamaLastTokenLayerStage::KeyProjection),
        },
        StageSpec {
            name: format!("layer{layer_index}.v_proj.full"),
            shape: vec![sequence, key_value_width],
            source: StageSource::FullSequence(LlamaLastTokenLayerStage::ValueProjection),
        },
        StageSpec {
            name: format!("layer{layer_index}.rope_cos.full"),
            shape: vec![
                sequence,
                u64::try_from(QWEN3B_HEAD_DIMENSION).expect("head dimension fits"),
            ],
            source: StageSource::HfEagerRopeTableCosine,
        },
        StageSpec {
            name: format!("layer{layer_index}.rope_sin.full"),
            shape: vec![
                sequence,
                u64::try_from(QWEN3B_HEAD_DIMENSION).expect("head dimension fits"),
            ],
            source: StageSource::HfEagerRopeTableSine,
        },
        StageSpec {
            name: format!("layer{layer_index}.q_rope.full"),
            shape: vec![
                sequence,
                u64::try_from(QWEN3B_QUERY_HEADS).expect("query heads fit"),
                u64::try_from(QWEN3B_HEAD_DIMENSION).expect("head dimension fits"),
            ],
            source: StageSource::FullSequence(LlamaLastTokenLayerStage::QueryRotary),
        },
        StageSpec {
            name: format!("layer{layer_index}.k_rope.full"),
            shape: vec![
                sequence,
                u64::try_from(QWEN3B_KEY_VALUE_HEADS).expect("key/value heads fit"),
                u64::try_from(QWEN3B_HEAD_DIMENSION).expect("head dimension fits"),
            ],
            source: StageSource::FullSequence(LlamaLastTokenLayerStage::KeyRotary),
        },
        hidden_stage(
            &format!("layer{layer_index}.attention_context.full"),
            LlamaLastTokenLayerStage::AttentionContext,
        ),
        hidden_stage(
            &format!("layer{layer_index}.after_attention_residual.full"),
            LlamaLastTokenLayerStage::AfterAttentionResidual,
        ),
        hidden_stage(
            &format!("layer{layer_index}.output.full"),
            LlamaLastTokenLayerStage::Output,
        ),
    ]
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

fn parse_stage_sidecar(
    manifest: &Value,
    path: &Path,
    specs: &[StageSpec],
) -> TestResult<BTreeMap<String, Vec<u8>>> {
    let source = regular_file(path, "HF P2051 stage sidecar")?;
    let bytes = fs::read(&source)?;
    let sidecar = manifest["sidecar"]
        .as_object()
        .ok_or("HF P2051 stage sidecar record is missing")?;
    let expected_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("HF P2051 stage sidecar basename is invalid")?;
    if sidecar.get("path").and_then(Value::as_str) != Some(expected_name)
        || sidecar.get("format").and_then(Value::as_str) != Some("safetensors")
        || sidecar.get("tensor_count").and_then(Value::as_u64) != Some(u64::try_from(specs.len())?)
        || sha256_hex(&bytes)
            != json_sha256(
                sidecar
                    .get("sha256")
                    .ok_or("HF P2051 stage sidecar SHA-256 is missing")?,
                "HF P2051 stage sidecar SHA-256",
            )?
    {
        return Err("HF P2051 stage sidecar binding differs".into());
    }
    if bytes.len() < 8 {
        return Err("HF P2051 stage sidecar is too short".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES || 8 + header_len > bytes.len()
    {
        return Err("HF P2051 stage sidecar header differs".into());
    }
    let data_start = 8_usize
        .checked_add(header_len)
        .ok_or("HF P2051 stage sidecar data offset overflows")?;
    let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
    let header = header
        .as_object()
        .ok_or("HF P2051 stage sidecar header is not an object")?;
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
        return Err("HF P2051 stage sidecar tensor key set differs".into());
    }
    let manifest_tensors = manifest["tensors"]
        .as_object()
        .ok_or("HF P2051 stage manifest tensors are missing")?;
    let mut ranges = Vec::with_capacity(specs.len());
    let mut output = BTreeMap::new();
    for spec in specs {
        let reference = manifest_tensors
            .get(&spec.name)
            .and_then(Value::as_object)
            .ok_or("HF P2051 stage manifest tensor is missing")?;
        let key = format!("trace/{}", spec.name.replace('.', "/"));
        if reference.get("key").and_then(Value::as_str) != Some(key.as_str())
            || reference.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || reference
                .get("canonical_byte_order")
                .and_then(Value::as_str)
                != Some("little-endian-u16")
            || reference["shape"]
                .as_array()
                .ok_or("HF P2051 stage tensor shape is missing")?
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
            return Err(format!("HF P2051 stage tensor {} metadata differs", spec.name).into());
        }
        let entry = header
            .get(&key)
            .and_then(Value::as_object)
            .ok_or("HF P2051 stage sidecar tensor is missing")?;
        if entry.get("dtype").and_then(Value::as_str) != Some("BF16")
            || entry.get("shape")
                != Some(&Value::Array(
                    spec.shape.iter().copied().map(Value::from).collect(),
                ))
        {
            return Err(format!("HF P2051 sidecar tensor {} metadata differs", spec.name).into());
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("HF P2051 stage sidecar offsets are missing")?;
        if offsets.len() != 2 {
            return Err("HF P2051 stage sidecar offsets differ".into());
        }
        let start = usize::try_from(offsets[0].as_u64().ok_or("HF P2051 offset start")?)?;
        let end = usize::try_from(offsets[1].as_u64().ok_or("HF P2051 offset end")?)?;
        let expected_bytes = shape_byte_len(&spec.shape)?;
        if end < start || end - start != expected_bytes || data_start + end > bytes.len() {
            return Err(format!("HF P2051 sidecar tensor {} range differs", spec.name).into());
        }
        let raw = bytes[data_start + start..data_start + end].to_vec();
        if sha256_hex(&raw)
            != json_sha256(
                reference
                    .get("bf16_le_sha256")
                    .ok_or("HF P2051 stage tensor SHA-256 is missing")?,
                "HF P2051 stage tensor SHA-256",
            )?
        {
            return Err(format!("HF P2051 sidecar tensor {} hash differs", spec.name).into());
        }
        ranges.push((start, end));
        output.insert(spec.name.clone(), raw);
    }
    let mut expected_start = 0_usize;
    ranges.sort_unstable();
    for (start, end) in ranges {
        if start != expected_start {
            return Err("HF P2051 sidecar offsets are non-contiguous".into());
        }
        expected_start = end;
    }
    if data_start + expected_start != bytes.len() {
        return Err("HF P2051 sidecar has trailing bytes".into());
    }
    Ok(output)
}

fn load_hf_stage_artifact(
    teacher: &TeacherPrefix,
    workload: &Workload,
) -> TestResult<HfStageArtifact> {
    let manifest_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_STAGE_MANIFEST")?,
        "HF P2051 stage manifest",
    )?;
    let sidecar_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_STAGE_SIDECAR")?,
        "HF P2051 stage sidecar",
    )?;
    let payload = fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_hex(&payload);
    let manifest: Value = serde_json::from_slice(&payload)?;
    if manifest["schema_version"].as_str() != Some(STAGE_SCHEMA_VERSION)
        || manifest["artifact_kind"].as_str() != Some(STAGE_ARTIFACT_KIND)
        || manifest["trace_id"].as_str() != Some(STAGE_TRACE_ID)
        || manifest["performance_claim_eligible"].as_bool() != Some(false)
    {
        return Err("HF P2051 stage manifest identity differs".into());
    }
    require_exact_fields(
        &manifest["model"],
        &[
            "checkpoint_path",
            "checkpoint_receipt_filename",
            "checkpoint_receipt_sha256",
        ],
        "HF P2051 stage model",
    )?;
    let model = manifest["model"]
        .as_object()
        .ok_or("HF P2051 stage model is missing")?;
    if model
        .get("checkpoint_path")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
        || model
            .get("checkpoint_receipt_filename")
            .and_then(Value::as_str)
            != Some(CHECKPOINT_RECEIPT_FILENAME)
    {
        return Err("HF P2051 stage checkpoint receipt identity differs".into());
    }
    let checkpoint_receipt_filename = CHECKPOINT_RECEIPT_FILENAME.to_owned();
    let checkpoint_receipt_sha256 = json_sha256(
        model
            .get("checkpoint_receipt_sha256")
            .ok_or("HF P2051 stage checkpoint receipt SHA-256 is missing")?,
        "HF P2051 stage checkpoint receipt SHA-256",
    )?;
    let contract = manifest["contract"]
        .as_object()
        .ok_or("HF P2051 stage contract is missing")?;
    if contract.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
        || contract.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
        || contract.get("execution")
            != Some(&json!({
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
            }))
    {
        return Err("HF P2051 stage execution contract differs".into());
    }
    let workload_contract = contract
        .get("workload")
        .and_then(Value::as_object)
        .ok_or("HF P2051 workload contract is missing")?;
    if workload_contract
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
        return Err("HF P2051 workload contract differs".into());
    }
    let input = contract
        .get("input")
        .and_then(Value::as_object)
        .ok_or("HF P2051 input contract is missing")?;
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
                .ok_or("HF P2051 teacher prefix missing")?,
            "HF P2051 teacher prefix",
        )? != teacher.token_ids
    {
        return Err("HF P2051 input binding differs".into());
    }
    let teacher_source = input
        .get("teacher_source")
        .and_then(Value::as_object)
        .ok_or("HF P2051 teacher source is missing")?;
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
        return Err("HF P2051 teacher source binding differs".into());
    }
    let root = repository_root()?;
    let provenance = manifest["provenance"]["source_repository"]
        .as_object()
        .ok_or("HF P2051 source provenance is missing")?;
    if provenance.get("source_dirty").and_then(Value::as_bool) != Some(false) {
        return Err("HF P2051 source provenance is dirty".into());
    }
    let sources = provenance
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("HF P2051 source records are missing")?;
    if !sources.contains_key("rust_trace_point_api")
        || !sources.contains_key("rust_stage_discriminator")
    {
        return Err("HF P2051 source records omit Rust consumer bindings".into());
    }
    for (name, record) in sources {
        validate_source_record(&root, record, &format!("HF P2051 source {name}"))?;
    }
    let specs = expected_stage_specs();
    let tensors = parse_stage_sidecar(&manifest, &sidecar_path, &specs)?;
    let sidecar_sha256 = sha256_file(&sidecar_path)?;
    Ok(HfStageArtifact {
        manifest_path,
        manifest_sha256,
        sidecar_path,
        sidecar_sha256,
        input_token_ids_le_sha256,
        teacher_artifact_sha256: teacher.artifact_sha256.clone(),
        teacher_sidecar_sha256: teacher.cache_off_sidecar_sha256.clone(),
        teacher_full_token_ids_sha256: teacher.full_teacher_token_ids_sha256.clone(),
        teacher_prefix_token_ids: teacher.token_ids.clone(),
        checkpoint_receipt_filename,
        checkpoint_receipt_sha256,
        tensors,
    })
}

fn load_hf_cache_on_prefill_stage_artifact(
    teacher: &TeacherCacheOn,
    workload: &Workload,
) -> TestResult<HfCacheOnPrefillStageArtifact> {
    let manifest_path = regular_file(
        &required_path("RILEY_QWEN3B_P2048_CACHE_ON_STAGE_MANIFEST")?,
        "HF P2048 cache-on stage manifest",
    )?;
    let sidecar_path = regular_file(
        &required_path("RILEY_QWEN3B_P2048_CACHE_ON_STAGE_SIDECAR")?,
        "HF P2048 cache-on stage sidecar",
    )?;
    let payload = fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_hex(&payload);
    let manifest: Value = serde_json::from_slice(&payload)?;
    if manifest["schema_version"].as_str() != Some(CACHE_ON_STAGE_SCHEMA_VERSION)
        || manifest["artifact_kind"].as_str() != Some(CACHE_ON_STAGE_ARTIFACT_KIND)
        || manifest["trace_id"].as_str() != Some(CACHE_ON_STAGE_TRACE_ID)
        || manifest["performance_claim_eligible"].as_bool() != Some(false)
    {
        return Err("HF P2048 cache-on stage manifest identity differs".into());
    }
    require_exact_fields(
        &manifest["model"],
        &[
            "checkpoint_path",
            "checkpoint_receipt_filename",
            "checkpoint_receipt_sha256",
        ],
        "HF P2048 cache-on stage model",
    )?;
    let model = manifest["model"]
        .as_object()
        .ok_or("HF P2048 cache-on stage model is missing")?;
    if model
        .get("checkpoint_path")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
        || model
            .get("checkpoint_receipt_filename")
            .and_then(Value::as_str)
            != Some(CHECKPOINT_RECEIPT_FILENAME)
    {
        return Err("HF P2048 cache-on stage checkpoint receipt identity differs".into());
    }
    let checkpoint_receipt_filename = CHECKPOINT_RECEIPT_FILENAME.to_owned();
    let checkpoint_receipt_sha256 = json_sha256(
        model
            .get("checkpoint_receipt_sha256")
            .ok_or("HF P2048 cache-on checkpoint receipt SHA-256 is missing")?,
        "HF P2048 cache-on checkpoint receipt SHA-256",
    )?;

    let contract = manifest["contract"]
        .as_object()
        .ok_or("HF P2048 cache-on stage contract is missing")?;
    if contract.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
        || contract.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
        || contract.get("execution")
            != Some(&json!({
                "attention_implementation": "eager",
                "batch_size": 1,
                "cache_position_argument": "omitted-transformers-5.15.1",
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": true,
                "dtype": "bfloat16",
                "explicit_attention_mask": true,
                "explicit_input_ids": true,
                "explicit_position_ids": true,
                "hf_hub_offline": true,
                "inference_mode": true,
                "local_files_only": true,
                "logits_to_keep": 1,
                "return_dict": true,
                "sampling_applied": false,
                "tf32_enabled": false,
                "transformers_offline": true,
                "trust_remote_code": false,
                "use_cache": true,
            }))
    {
        return Err("HF P2048 cache-on stage execution contract differs".into());
    }
    let workload_contract = contract
        .get("workload")
        .and_then(Value::as_object)
        .ok_or("HF P2048 cache-on workload contract is missing")?;
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
        return Err("HF P2048 cache-on workload contract differs".into());
    }
    let input = contract
        .get("input")
        .and_then(Value::as_object)
        .ok_or("HF P2048 cache-on input contract is missing")?;
    if input.get("construction").and_then(Value::as_str)
        != Some("verified_hf_cache_on.P2048_prefill_then_teacher_token_ids[:2]_M1_decodes")
        || input
            .get("prefill_prompt_token_count")
            .and_then(Value::as_u64)
            != Some(u64::try_from(QWEN3B_PROMPT_TOKEN_COUNT)?)
        || input
            .get("prefill_prompt_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(QWEN3B_PROMPT_TOKEN_SHA256)
        || input
            .get("teacher_decode_token_count")
            .and_then(Value::as_u64)
            != Some(2)
        || json_u32_array(
            input
                .get("teacher_decode_token_ids")
                .ok_or("HF P2048 cache-on teacher decode token IDs are missing")?,
            "HF P2048 cache-on teacher decode token IDs",
        )? != teacher.token_ids
        || input
            .get("teacher_decode_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(token_ids_sha256(&teacher.token_ids).as_str())
    {
        return Err("HF P2048 cache-on input binding differs".into());
    }
    let teacher_source = input
        .get("teacher_source")
        .and_then(Value::as_object)
        .ok_or("HF P2048 cache-on teacher source is missing")?;
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
        || teacher_source.get("cache_mode").and_then(Value::as_str) != Some("cache-on")
        || teacher_source
            .get("cache_on_sidecar_sha256")
            .and_then(Value::as_str)
            != Some(teacher.cache_on_sidecar_sha256.as_str())
        || teacher_source
            .get("full_teacher_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(teacher.full_teacher_token_ids_sha256.as_str())
        || teacher_source
            .get("cache_on_sidecar_tensor_key")
            .and_then(Value::as_str)
            != Some(TEACHER_CACHE_OFF_SIDECAR_KEY)
    {
        return Err("HF P2048 cache-on teacher source binding differs".into());
    }

    let trace_profile = manifest["trace_profile"]
        .as_object()
        .ok_or("HF P2048 cache-on trace profile is missing")?;
    if trace_profile.get("capture_domain").and_then(Value::as_str)
        != Some("cache-on-p2048-prefill-and-m1-decode-last-token-rows")
        || trace_profile.get("id").and_then(Value::as_str) != Some(CACHE_ON_STAGE_TRACE_ID)
        || trace_profile
            .get("captured_step_count")
            .and_then(Value::as_u64)
            != Some(3)
        || trace_profile
            .get("tensor_count_per_step")
            .and_then(Value::as_u64)
            != Some(52)
        || trace_profile.get("tensor_count").and_then(Value::as_u64) != Some(156)
    {
        return Err("HF P2048 cache-on trace profile differs".into());
    }
    let steps = trace_profile
        .get("selected_steps")
        .and_then(Value::as_array)
        .ok_or("HF P2048 cache-on trace steps are missing")?;
    if steps
        != &vec![
            json!({
                "name": "prefill",
                "source_logit_row": 0,
                "input_token_count": 2048,
                "attention_mask_token_count": 2048,
                "position_start": 0,
                "position_end": 2047,
                "cache_length_before": 0,
                "cache_length_after": 2048,
                "teacher_decode_token_index": null,
            }),
            json!({
                "name": "decode_step_1",
                "source_logit_row": 1,
                "input_token_count": 1,
                "attention_mask_token_count": 2049,
                "position_start": 2048,
                "position_end": 2048,
                "cache_length_before": 2048,
                "cache_length_after": 2049,
                "teacher_decode_token_index": 0,
            }),
            json!({
                "name": "decode_step_2",
                "source_logit_row": 2,
                "input_token_count": 1,
                "attention_mask_token_count": 2050,
                "position_start": 2049,
                "position_end": 2049,
                "cache_length_before": 2049,
                "cache_length_after": 2050,
                "teacher_decode_token_index": 1,
            }),
        ]
    {
        return Err("HF P2048 cache-on trace steps differ".into());
    }

    let root = repository_root()?;
    let provenance = manifest["provenance"]["source_repository"]
        .as_object()
        .ok_or("HF P2048 cache-on source provenance is missing")?;
    if provenance.get("source_dirty").and_then(Value::as_bool) != Some(false) {
        return Err("HF P2048 cache-on source provenance is dirty".into());
    }
    let sources = provenance
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("HF P2048 cache-on source records are missing")?;
    for required in [
        "rust_forward",
        "rust_decode",
        "rust_cache_on_trace",
        "rust_p2051_quality_gate",
    ] {
        if !sources.contains_key(required) {
            return Err(format!("HF P2048 cache-on source records omit {required}").into());
        }
    }
    for (name, record) in sources {
        validate_source_record(&root, record, &format!("HF P2048 cache-on source {name}"))?;
    }
    let specs = expected_cache_on_stage_specs();
    let tensors = parse_stage_sidecar(&manifest, &sidecar_path, &specs)?;
    let sidecar_sha256 = sha256_file(&sidecar_path)?;
    Ok(HfCacheOnPrefillStageArtifact {
        manifest_path,
        manifest_sha256,
        sidecar_path,
        sidecar_sha256,
        checkpoint_receipt_filename,
        checkpoint_receipt_sha256,
        tensors,
    })
}

fn load_hf_full_sequence_stage_artifact(
    teacher: &TeacherPrefix,
    workload: &Workload,
) -> TestResult<HfFullSequenceStageArtifact> {
    let manifest_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_FULL_SEQUENCE_STAGE_MANIFEST")?,
        "HF P2051 full-sequence stage manifest",
    )?;
    let sidecar_path = regular_file(
        &required_path("RILEY_QWEN3B_P2051_FULL_SEQUENCE_STAGE_SIDECAR")?,
        "HF P2051 full-sequence stage sidecar",
    )?;
    let payload = fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_hex(&payload);
    let manifest: Value = serde_json::from_slice(&payload)?;
    let full_sequence_trace_id = full_sequence_stage_trace_id(FULL_SEQUENCE_STAGE_LAYER_INDEX);
    if manifest["schema_version"].as_str() != Some(FULL_SEQUENCE_STAGE_SCHEMA_VERSION)
        || manifest["artifact_kind"].as_str() != Some(FULL_SEQUENCE_STAGE_ARTIFACT_KIND)
        || manifest["trace_id"].as_str() != Some(full_sequence_trace_id.as_str())
        || manifest["performance_claim_eligible"].as_bool() != Some(false)
        || manifest["trace_profile"]
            != json!({
                "capture_domain": "cache-free-p2051-full-sequence-layer-boundaries",
                "id": full_sequence_trace_id,
                "layer_index": FULL_SEQUENCE_STAGE_LAYER_INDEX,
                "tensor_count": 11,
                "rust_consumer": {
                    "api": "riley_runtime::llama::PreparedLlamaForward::prepare_full_sequence_layer_stage_trace+execute_full_sequence_layer_stage_traced",
                    "attention_backend": "hf-eager-cublaslt-qwen-p2051-probe",
                    "cache": false,
                    "input_context_token_count": CONTEXT_TOKEN_COUNT,
                    "rope_table_capture": "PreparedLlamaForward::download_hugging_face_bf16_rope_table_trace",
                    "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
                    "trace_row_layout": "full-sequence-token-major",
                },
            })
    {
        return Err("HF P2051 full-sequence stage manifest identity differs".into());
    }
    require_exact_fields(
        &manifest["model"],
        &[
            "checkpoint_path",
            "checkpoint_receipt_filename",
            "checkpoint_receipt_sha256",
        ],
        "HF P2051 full-sequence stage model",
    )?;
    let model = manifest["model"]
        .as_object()
        .ok_or("HF P2051 full-sequence stage model is missing")?;
    if model
        .get("checkpoint_path")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
        || model
            .get("checkpoint_receipt_filename")
            .and_then(Value::as_str)
            != Some(CHECKPOINT_RECEIPT_FILENAME)
    {
        return Err("HF P2051 full-sequence stage checkpoint receipt identity differs".into());
    }
    let checkpoint_receipt_filename = CHECKPOINT_RECEIPT_FILENAME.to_owned();
    let checkpoint_receipt_sha256 = json_sha256(
        model
            .get("checkpoint_receipt_sha256")
            .ok_or("HF P2051 full-sequence stage checkpoint receipt SHA-256 is missing")?,
        "HF P2051 full-sequence stage checkpoint receipt SHA-256",
    )?;
    let contract = manifest["contract"]
        .as_object()
        .ok_or("HF P2051 full-sequence stage contract is missing")?;
    if contract.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
        || contract.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
        || contract.get("execution")
            != Some(&json!({
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
            }))
    {
        return Err("HF P2051 full-sequence stage execution contract differs".into());
    }
    let workload_contract = contract
        .get("workload")
        .and_then(Value::as_object)
        .ok_or("HF P2051 full-sequence workload contract is missing")?;
    if workload_contract
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
        return Err("HF P2051 full-sequence workload contract differs".into());
    }
    let input = contract
        .get("input")
        .and_then(Value::as_object)
        .ok_or("HF P2051 full-sequence input contract is missing")?;
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
        || input
            .get("input_token_ids_le_u32_sha256")
            .and_then(Value::as_str)
            != Some(input_token_ids_le_sha256.as_str())
        || json_u32_array(
            input
                .get("teacher_prefix_token_ids")
                .ok_or("HF P2051 full-sequence teacher prefix missing")?,
            "HF P2051 full-sequence teacher prefix",
        )? != teacher.token_ids
    {
        return Err("HF P2051 full-sequence input binding differs".into());
    }
    let teacher_source = input
        .get("teacher_source")
        .and_then(Value::as_object)
        .ok_or("HF P2051 full-sequence teacher source is missing")?;
    if teacher_source
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
        return Err("HF P2051 full-sequence teacher source binding differs".into());
    }
    let root = repository_root()?;
    let provenance = manifest["provenance"]["source_repository"]
        .as_object()
        .ok_or("HF P2051 full-sequence source provenance is missing")?;
    if provenance.get("source_dirty").and_then(Value::as_bool) != Some(false) {
        return Err("HF P2051 full-sequence source provenance is dirty".into());
    }
    let sources = provenance
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("HF P2051 full-sequence source records are missing")?;
    if !sources.contains_key("p2051_full_sequence_layer_stage_trace")
        || !sources.contains_key("rust_full_sequence_discriminator")
        || !sources.contains_key("rust_trace_point_api")
    {
        return Err("HF P2051 full-sequence source records omit consumer bindings".into());
    }
    for (name, record) in sources {
        validate_source_record(
            &root,
            record,
            &format!("HF P2051 full-sequence source {name}"),
        )?;
    }
    let specs = expected_full_sequence_stage_specs(FULL_SEQUENCE_STAGE_LAYER_INDEX);
    let tensors = parse_stage_sidecar(&manifest, &sidecar_path, &specs)?;
    let sidecar_sha256 = sha256_file(&sidecar_path)?;
    Ok(HfFullSequenceStageArtifact {
        manifest_path,
        manifest_sha256,
        sidecar_path,
        sidecar_sha256,
        input_token_ids_le_sha256,
        teacher_artifact_sha256: teacher.artifact_sha256.clone(),
        teacher_sidecar_sha256: teacher.cache_off_sidecar_sha256.clone(),
        teacher_full_token_ids_sha256: teacher.full_teacher_token_ids_sha256.clone(),
        teacher_prefix_token_ids: teacher.token_ids.clone(),
        checkpoint_receipt_filename,
        checkpoint_receipt_sha256,
        tensors,
    })
}

fn load_model(hf: &HfStageArtifact) -> TestResult<LoadedModel> {
    load_model_from_checkpoint_receipt(
        &hf.checkpoint_receipt_filename,
        &hf.checkpoint_receipt_sha256,
    )
}

fn load_cache_on_prefill_model(hf: &HfCacheOnPrefillStageArtifact) -> TestResult<LoadedModel> {
    load_model_from_checkpoint_receipt(
        &hf.checkpoint_receipt_filename,
        &hf.checkpoint_receipt_sha256,
    )
}

fn load_full_sequence_model(hf: &HfFullSequenceStageArtifact) -> TestResult<LoadedModel> {
    load_model_from_checkpoint_receipt(
        &hf.checkpoint_receipt_filename,
        &hf.checkpoint_receipt_sha256,
    )
}

fn load_model_from_checkpoint_receipt(
    checkpoint_receipt_filename: &str,
    checkpoint_receipt_sha256: &str,
) -> TestResult<LoadedModel> {
    let checkpoint = regular_directory(
        &required_path("RILEY_QWEN3B_CHECKPOINT")?,
        "Qwen checkpoint",
    )?;
    let receipt = regular_file(
        &checkpoint.join(checkpoint_receipt_filename),
        "Qwen checkpoint receipt",
    )?;
    if sha256_file(&receipt)? != checkpoint_receipt_sha256 {
        return Err("Qwen checkpoint receipt differs from HF P2051 stage artifact".into());
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
        return Err("loaded model differs from Qwen2.5-3B stage contract".into());
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

fn close_decode_resources(
    decode: Option<PreparedLlamaDecode>,
    stream: CudaStream,
    context: CudaContext,
) -> TestResult {
    let mut failures = Vec::new();
    if let Some(decode) = decode {
        if let Err(error) = decode.close() {
            failures.push(format!("decode close failed: {error}"));
        }
    }
    if let Err(error) = context.synchronize() {
        failures.push(format!("context synchronize failed: {error}"));
    }
    match context.allocation_stats() {
        Ok(stats) if stats.is_zero() => {}
        Ok(stats) => failures.push(format!(
            "CUDA allocation accounting is non-zero after decode close: {stats:?}"
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

fn last_row(bytes: &[u8], element_count: usize) -> TestResult<Vec<u8>> {
    let byte_count = element_count
        .checked_mul(BF16_BYTES)
        .ok_or("last-stage row byte count overflows")?;
    if bytes.len() < byte_count || bytes.len() % byte_count != 0 {
        return Err("static stage buffer length differs from its row contract".into());
    }
    canonical_bf16_le(&bytes[bytes.len() - byte_count..])
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

fn static_points() -> Vec<LlamaTracePoint> {
    vec![
        LlamaTracePoint::Embedding,
        LlamaTracePoint::Layer0InputNorm,
        LlamaTracePoint::Layer0QueryProjection,
        LlamaTracePoint::Layer0KeyProjection,
        LlamaTracePoint::Layer0ValueProjection,
        LlamaTracePoint::Layer0QueryRotary,
        LlamaTracePoint::Layer0KeyRotary,
        LlamaTracePoint::Layer0AttentionContext,
        LlamaTracePoint::Layer0AfterAttentionResidual,
        LlamaTracePoint::Layer0PostAttentionNorm,
        LlamaTracePoint::Layer0GateProjection,
        LlamaTracePoint::Layer0UpProjection,
        LlamaTracePoint::Layer0Gated,
        LlamaTracePoint::Layer0DownProjection,
        LlamaTracePoint::Layer0Output,
    ]
}

fn profile_workspace_cap_bytes(mode: LlamaProjectionBiasMode) -> u64 {
    if mode == LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1 {
        HF_COMPAT_GEMM_WORKSPACE_CAP_BYTES
    } else {
        STRICT_GEMM_WORKSPACE_CAP_BYTES
    }
}

fn candidate_quality_gate(profile: &Value) -> TestResult<Value> {
    let stages = profile
        .get("stages")
        .and_then(Value::as_object)
        .ok_or("HF-compatible profile stages are missing")?;
    let summary = profile
        .get("summary")
        .and_then(Value::as_object)
        .ok_or("HF-compatible profile summary is missing")?;
    let exact_stage_count = summary
        .get("bf16_exact_stage_count")
        .and_then(Value::as_u64)
        .ok_or("HF-compatible exact stage count is missing")?;
    let expected_stage_count = u64::try_from(expected_stage_specs().len())?;
    let qkv_names = [
        "layer0.q_proj.last",
        "layer0.k_proj.last",
        "layer0.v_proj.last",
    ];
    let qkv_exact = qkv_names.into_iter().try_fold(true, |all_exact, name| {
        let exact = stages
            .get(name)
            .and_then(Value::as_object)
            .and_then(|stage| stage.get("bf16_exact"))
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("HF-compatible {name} exactness is missing"))?;
        Ok::<_, Box<dyn Error>>(all_exact && exact)
    })?;
    let attention_context_exact = stages
        .get("layer0.attention_context.last")
        .and_then(Value::as_object)
        .and_then(|stage| stage.get("bf16_exact"))
        .and_then(Value::as_bool)
        .ok_or("HF-compatible layer0 attention context exactness is missing")?;
    let direct_output_projection_selected = profile
        .get("output_projection_backend")
        .and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID);
    let direct_mlp_projection_selected = profile
        .get("mlp_projection_backend")
        .and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID);
    let direct_last_token_lm_head_selected = profile.get("lm_head_backend").and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2051_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID);
    let p2051_rms_norm_selected = profile.get("rms_norm_backend").and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2051_RMS_NORM_BACKEND_ID);
    let full_forward_exact = direct_output_projection_selected
        && direct_mlp_projection_selected
        && direct_last_token_lm_head_selected
        && p2051_rms_norm_selected
        && qkv_exact
        && attention_context_exact
        && exact_stage_count == expected_stage_count
        && summary
            .get("first_non_exact_stage")
            .is_some_and(Value::is_null);
    Ok(json!({
        "required_stage_count": expected_stage_count,
        "candidate_exact_stage_count": exact_stage_count,
        "qkv_last_rows_bf16_exact": qkv_exact,
        "attention_context_last_row_bf16_exact": attention_context_exact,
        "direct_cublas_output_projection_selected": direct_output_projection_selected,
        "direct_cublas_mlp_projection_selected": direct_mlp_projection_selected,
        "direct_cublas_last_token_lm_head_selected": direct_last_token_lm_head_selected,
        "p2051_rms_norm_selected": p2051_rms_norm_selected,
        "cache_off_full_forward_bf16_exact": full_forward_exact,
        "corrected_cache_on_eligible": full_forward_exact,
        "serving_selector_eligible": full_forward_exact,
        "performance_claim_eligible": false,
    }))
}

fn run_profile(
    model: &LoadedModel,
    input: &[u32],
    hf: &HfStageArtifact,
    mode: LlamaProjectionBiasMode,
    attention_profile: AttentionProfile,
    output_projection_profile: OutputProjectionProfile,
    mlp_projection_profile: MlpProjectionProfile,
    lm_head_profile: LmHeadProfile,
    rope_table_profile: RopeTableProfile,
    rms_norm_profile: RmsNormProfile,
) -> TestResult<Value> {
    let (context, mut stream) = first_context()?;
    let config = PreparedLlamaForwardConfig::new(
        FULL_FORWARD_UPLOAD_STAGING_BYTES,
        FULL_FORWARD_IO_STAGING_BYTES,
        profile_workspace_cap_bytes(mode),
        REFERENCE_ATTENTION_BUDGET_BYTES,
    )
    .with_projection_bias_mode(mode);
    let config = attention_profile.configure(config);
    let config = output_projection_profile.configure(config);
    let config = mlp_projection_profile.configure(config);
    let config = lm_head_profile.configure(config);
    let config = rope_table_profile.configure(config);
    let config = rms_norm_profile.configure(config);
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
                Err(cleanup_error) => Err(format!("stage forward preparation failed: {error}; cleanup also failed: {cleanup_error}").into()),
            };
        }
    };
    let result = (|| -> TestResult<Value> {
        if forward.projection_bias_mode() != mode
            || forward.attention_selection().implementation_id() != attention_profile.backend_id()
            || forward.output_projection_backend_id() != output_projection_profile.backend_id()
            || forward.mlp_projection_backend_id() != mlp_projection_profile.backend_id()
            || forward.lm_head_backend_id() != lm_head_profile.backend_id()
            || forward.rms_norm_backend_id() != rms_norm_profile.backend_id()
        {
            return Err("stage forward selected an unexpected numerical backend".into());
        }
        let points = static_points();
        let mut static_trace = forward.prepare_trace_points(&points)?;
        forward.upload_tokens(input, &mut stream)?;
        forward.execute_traced(&mut stream, &mut static_trace)?;
        if static_trace.captured_count() != u32::try_from(points.len())? {
            return Err("stage static trace did not capture every requested point".into());
        }
        let mut observed = BTreeMap::new();
        for spec in expected_stage_specs() {
            if let StageSource::Static(point, elements) = spec.source {
                let source = static_trace
                    .tensor(point)
                    .ok_or("stage static trace tensor is missing")?;
                observed.insert(spec.name, last_row(source, elements)?);
            }
        }
        let layer_indices = (0..QWEN3B_LAYER_COUNT).collect::<Vec<_>>();
        let mut layer_trace = forward.prepare_last_token_layer_trace(&layer_indices, true)?;
        if layer_trace.requested_layer_indices() != layer_indices.as_slice()
            || layer_trace.row_byte_len() != QWEN3B_HIDDEN_SIZE * BF16_BYTES
            || !layer_trace.requests_final_norm_output()
        {
            return Err("last-token layer trace preparation contract differs".into());
        }
        forward.execute_last_token_layer_traced(&mut stream, &mut layer_trace)?;
        if layer_trace.captured_layer_count() != QWEN3B_LAYER_COUNT {
            return Err("last-token layer trace did not capture every decoder layer".into());
        }
        for layer_index in 0..QWEN3B_LAYER_COUNT {
            let row = canonical_bf16_le(
                layer_trace
                    .layer_residual_output(layer_index)
                    .ok_or("last-token layer row is missing")?,
            )?;
            let name = format!("layer{layer_index}.output.last");
            if layer_index == 0 {
                let static_row = observed
                    .get(&name)
                    .ok_or("static layer0 output row is missing")?;
                if static_row != &row {
                    return Err("independent static/dynamic layer0 output rows differ".into());
                }
            } else {
                observed.insert(name, row);
            }
        }
        observed.insert(
            "final_norm.output.last".to_owned(),
            canonical_bf16_le(
                layer_trace
                    .final_norm_output()
                    .ok_or("last-token final norm output is missing")?,
            )?,
        );
        let mut logits = vec![0_u8; QWEN3B_VOCABULARY_SIZE * BF16_BYTES];
        forward.download_last_logits(&mut logits, &mut stream)?;
        observed.insert("last_logits".to_owned(), canonical_bf16_le(&logits)?);

        // Reuse the same fully prepared owner without another upload. This
        // catches an accidental split-K or workspace lifetime dependence that
        // a one-shot comparison cannot observe.
        let mut repeat_trace = forward.prepare_trace_points(&points)?;
        forward.execute_traced(&mut stream, &mut repeat_trace)?;
        if repeat_trace.captured_count() != u32::try_from(points.len())? {
            return Err("repeat static trace did not capture every requested point".into());
        }
        for (name, point, elements) in [
            (
                "layer0.q_proj.last",
                LlamaTracePoint::Layer0QueryProjection,
                QWEN3B_HIDDEN_SIZE,
            ),
            (
                "layer0.k_proj.last",
                LlamaTracePoint::Layer0KeyProjection,
                QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            ),
            (
                "layer0.v_proj.last",
                LlamaTracePoint::Layer0ValueProjection,
                QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            ),
            (
                "layer0.output.last",
                LlamaTracePoint::Layer0Output,
                QWEN3B_HIDDEN_SIZE,
            ),
        ] {
            let repeated = last_row(
                repeat_trace
                    .tensor(point)
                    .ok_or("repeat static trace tensor is missing")?,
                elements,
            )?;
            if observed.get(name) != Some(&repeated) {
                return Err(format!("repeat {name} BF16 row differs").into());
            }
        }
        let mut repeat_logits = vec![0_u8; QWEN3B_VOCABULARY_SIZE * BF16_BYTES];
        forward.download_last_logits(&mut repeat_logits, &mut stream)?;
        if observed.get("last_logits") != Some(&canonical_bf16_le(&repeat_logits)?) {
            return Err("repeat last logits BF16 row differs".into());
        }

        let mut stages = Map::new();
        let mut exact_count = 0_u64;
        let mut first_non_exact = None;
        for spec in expected_stage_specs() {
            let hf_bytes = hf
                .tensors
                .get(&spec.name)
                .ok_or("HF P2051 stage tensor is missing")?;
            let riley_bytes = observed
                .get(&spec.name)
                .ok_or("Riley P2051 stage tensor is missing")?;
            if hf_bytes.len() != shape_byte_len(&spec.shape)? || riley_bytes.len() != hf_bytes.len()
            {
                return Err(
                    format!("stage {} byte length differs from contract", spec.name).into(),
                );
            }
            let stage_metrics = metrics(hf_bytes, riley_bytes)?;
            if stage_metrics["bf16_exact"] == true {
                exact_count += 1;
            } else if first_non_exact.is_none() {
                first_non_exact = Some(spec.name.clone());
            }
            stages.insert(spec.name, stage_metrics);
        }
        Ok(json!({
            "profile_id": format!(
                "{}+{}+{}+{}+{}+{}+{}",
                mode.id(),
                attention_profile.id(),
                output_projection_profile.id(),
                mlp_projection_profile.id(),
                lm_head_profile.id(),
                rope_table_profile.id(),
                rms_norm_profile.id(),
            ),
            "projection_bias_backend": mode.id(),
            "attention_backend": attention_profile.backend_id(),
            "output_projection_backend": forward.output_projection_backend_id(),
            "mlp_projection_backend": forward.mlp_projection_backend_id(),
            "lm_head_backend": forward.lm_head_backend_id(),
            "rope_table_backend": rope_table_profile.id(),
            "rms_norm_backend": forward.rms_norm_backend_id(),
            "use_cache": false,
            "same_scheduler_engine": false,
            "repeat_execution": {
                "reused_prepared_owner": true,
                "qkv_last_rows_bf16_exact": true,
                "layer0_output_last_row_bf16_exact": true,
                "last_logits_bf16_exact": true,
            },
            "static_layer0_capture": true,
            "last_token_layer_capture": {
                "layer_indices": layer_indices,
                "last_token_row_index": LAST_TOKEN_ROW_INDEX,
                "final_norm_output": true,
            },
            "summary": {
                "stage_count": stages.len(),
                "bf16_exact_stage_count": exact_count,
                "first_non_exact_stage": first_non_exact,
            },
            "stages": stages,
        }))
    })();
    let cleanup = close_resources(Some(forward), stream, context);
    match (result, cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => Err(format!(
            "stage execution failed: {run_error}; cleanup also failed: {cleanup_error}"
        )
        .into()),
    }
}

fn cache_on_prefill_quality_gate(profile: &Value) -> TestResult<Value> {
    let stages = profile
        .get("stages")
        .and_then(Value::as_object)
        .ok_or("P2048 cache-on prefill profile stages are missing")?;
    let summary = profile
        .get("summary")
        .and_then(Value::as_object)
        .ok_or("P2048 cache-on prefill profile summary is missing")?;
    let exact_stage_count = summary
        .get("bf16_exact_stage_count")
        .and_then(Value::as_u64)
        .ok_or("P2048 cache-on prefill exact stage count is missing")?;
    let expected_stage_count = u64::try_from(expected_cache_on_prefill_stage_specs().len())?;
    let qkv_exact = [
        "prefill.layer0.q_proj.last",
        "prefill.layer0.k_proj.last",
        "prefill.layer0.v_proj.last",
    ]
    .into_iter()
    .try_fold(true, |all_exact, name| {
        let exact = stages
            .get(name)
            .and_then(Value::as_object)
            .and_then(|stage| stage.get("bf16_exact"))
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("P2048 cache-on prefill {name} exactness is missing"))?;
        Ok::<_, Box<dyn Error>>(all_exact && exact)
    })?;
    let attention_context_exact = stages
        .get("prefill.layer0.attention_context.last")
        .and_then(Value::as_object)
        .and_then(|stage| stage.get("bf16_exact"))
        .and_then(Value::as_bool)
        .ok_or("P2048 cache-on prefill attention context exactness is missing")?;
    let direct_output_projection_selected = profile
        .get("output_projection_backend")
        .and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID);
    let direct_mlp_projection_selected = profile
        .get("mlp_projection_backend")
        .and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID);
    let direct_last_token_lm_head_selected = profile.get("lm_head_backend").and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2048_CACHE_ON_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID);
    let p2048_rms_norm_selected = profile.get("rms_norm_backend").and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2048_CACHE_ON_RMS_NORM_BACKEND_ID);
    let prefill_exact = direct_output_projection_selected
        && direct_mlp_projection_selected
        && direct_last_token_lm_head_selected
        && p2048_rms_norm_selected
        && qkv_exact
        && attention_context_exact
        && exact_stage_count == expected_stage_count
        && summary
            .get("first_non_exact_stage")
            .is_some_and(Value::is_null);
    Ok(json!({
        "required_prefill_stage_count": expected_stage_count,
        "candidate_exact_prefill_stage_count": exact_stage_count,
        "qkv_last_rows_bf16_exact": qkv_exact,
        "attention_context_last_row_bf16_exact": attention_context_exact,
        "direct_cublas_output_projection_selected": direct_output_projection_selected,
        "direct_cublas_mlp_projection_selected": direct_mlp_projection_selected,
        "direct_cublas_last_token_lm_head_selected": direct_last_token_lm_head_selected,
        "p2048_cache_on_rms_norm_selected": p2048_rms_norm_selected,
        "cache_on_prefill_bf16_exact": prefill_exact,
        "cache_on_m1_decode_bf16_exact": false,
        "cache_on_full_forward_bf16_exact": false,
        "corrected_cache_on_eligible": false,
        "serving_selector_eligible": false,
        "performance_claim_eligible": false,
    }))
}

fn run_cache_on_prefill_profile(
    model: &LoadedModel,
    input: &[u32],
    hf: &HfCacheOnPrefillStageArtifact,
) -> TestResult<Value> {
    let (context, mut stream) = first_context()?;
    let config = PreparedLlamaForwardConfig::new(
        FULL_FORWARD_UPLOAD_STAGING_BYTES,
        FULL_FORWARD_IO_STAGING_BYTES,
        HF_COMPAT_GEMM_WORKSPACE_CAP_BYTES,
        REFERENCE_ATTENTION_BUDGET_BYTES,
    )
    .with_hf_eager_qwen_p2048_cache_on_prefill_probe();
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
                    "P2048 cache-on prefill preparation failed: {error}; cleanup also failed: {cleanup_error}"
                )
                .into()),
            };
        }
    };
    let result = (|| -> TestResult<Value> {
        if forward.projection_bias_mode()
            != LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1
            || forward.attention_selection().implementation_id()
                != HF_EAGER_QWEN_P2048_CACHE_ON_PROBE_ATTENTION_BACKEND_ID
            || forward.output_projection_backend_id()
                != HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID
            || forward.mlp_projection_backend_id()
                != HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID
            || forward.lm_head_backend_id()
                != HF_EAGER_QWEN_P2048_CACHE_ON_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID
            || forward.rms_norm_backend_id() != HF_EAGER_QWEN_P2048_CACHE_ON_RMS_NORM_BACKEND_ID
        {
            return Err("P2048 cache-on prefill selected an unexpected numerical backend".into());
        }
        let points = static_points();
        let mut static_trace = forward.prepare_trace_points(&points)?;
        forward.upload_tokens(input, &mut stream)?;
        forward.execute_traced(&mut stream, &mut static_trace)?;
        if static_trace.captured_count() != u32::try_from(points.len())? {
            return Err("P2048 cache-on static trace did not capture every requested point".into());
        }
        let mut observed = BTreeMap::new();
        for spec in expected_stage_specs() {
            if let StageSource::Static(point, elements) = spec.source {
                let source = static_trace
                    .tensor(point)
                    .ok_or("P2048 cache-on static trace tensor is missing")?;
                observed.insert(spec.name, last_row(source, elements)?);
            }
        }
        let layer_indices = (0..QWEN3B_LAYER_COUNT).collect::<Vec<_>>();
        let mut layer_trace = forward.prepare_last_token_layer_trace(&layer_indices, true)?;
        if layer_trace.requested_layer_indices() != layer_indices.as_slice()
            || layer_trace.row_byte_len() != QWEN3B_HIDDEN_SIZE * BF16_BYTES
            || !layer_trace.requests_final_norm_output()
        {
            return Err("P2048 cache-on last-token layer trace preparation differs".into());
        }
        forward.execute_last_token_layer_traced(&mut stream, &mut layer_trace)?;
        if layer_trace.captured_layer_count() != QWEN3B_LAYER_COUNT {
            return Err("P2048 cache-on layer trace did not capture every decoder layer".into());
        }
        for layer_index in 0..QWEN3B_LAYER_COUNT {
            let row = canonical_bf16_le(
                layer_trace
                    .layer_residual_output(layer_index)
                    .ok_or("P2048 cache-on last-token layer row is missing")?,
            )?;
            let name = format!("layer{layer_index}.output.last");
            if layer_index == 0 {
                let static_row = observed
                    .get(&name)
                    .ok_or("P2048 cache-on static layer0 output is missing")?;
                if static_row != &row {
                    return Err("P2048 cache-on static/dynamic layer0 output differs".into());
                }
            } else {
                observed.insert(name, row);
            }
        }
        observed.insert(
            "final_norm.output.last".to_owned(),
            canonical_bf16_le(
                layer_trace
                    .final_norm_output()
                    .ok_or("P2048 cache-on final norm output is missing")?,
            )?,
        );
        let mut logits = vec![0_u8; QWEN3B_VOCABULARY_SIZE * BF16_BYTES];
        forward.download_last_logits(&mut logits, &mut stream)?;
        observed.insert("last_logits".to_owned(), canonical_bf16_le(&logits)?);

        let mut repeat_trace = forward.prepare_trace_points(&points)?;
        forward.execute_traced(&mut stream, &mut repeat_trace)?;
        if repeat_trace.captured_count() != u32::try_from(points.len())? {
            return Err("P2048 cache-on repeat trace did not capture every requested point".into());
        }
        for (name, point, elements) in [
            (
                "layer0.q_proj.last",
                LlamaTracePoint::Layer0QueryProjection,
                QWEN3B_HIDDEN_SIZE,
            ),
            (
                "layer0.k_proj.last",
                LlamaTracePoint::Layer0KeyProjection,
                QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            ),
            (
                "layer0.v_proj.last",
                LlamaTracePoint::Layer0ValueProjection,
                QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            ),
            (
                "layer0.output.last",
                LlamaTracePoint::Layer0Output,
                QWEN3B_HIDDEN_SIZE,
            ),
        ] {
            let repeated = last_row(
                repeat_trace
                    .tensor(point)
                    .ok_or("P2048 cache-on repeat static trace tensor is missing")?,
                elements,
            )?;
            if observed.get(name) != Some(&repeated) {
                return Err(format!("P2048 cache-on repeat {name} BF16 row differs").into());
            }
        }
        let mut repeat_logits = vec![0_u8; QWEN3B_VOCABULARY_SIZE * BF16_BYTES];
        forward.download_last_logits(&mut repeat_logits, &mut stream)?;
        if observed.get("last_logits") != Some(&canonical_bf16_le(&repeat_logits)?) {
            return Err("P2048 cache-on repeat last logits BF16 row differs".into());
        }

        let mut stages = Map::new();
        let mut exact_count = 0_u64;
        let mut first_non_exact = None;
        for spec in expected_cache_on_prefill_stage_specs() {
            let base_name = spec
                .name
                .strip_prefix("prefill.")
                .ok_or("P2048 cache-on prefill stage lacks its prefix")?;
            let hf_bytes = hf
                .tensors
                .get(&spec.name)
                .ok_or("HF P2048 cache-on prefill stage is missing")?;
            let riley_bytes = observed
                .get(base_name)
                .ok_or("Riley P2048 cache-on prefill stage is missing")?;
            if hf_bytes.len() != shape_byte_len(&spec.shape)? || riley_bytes.len() != hf_bytes.len()
            {
                return Err(format!(
                    "P2048 cache-on prefill stage {} byte length differs",
                    spec.name
                )
                .into());
            }
            let stage_metrics = metrics(hf_bytes, riley_bytes)?;
            if stage_metrics["bf16_exact"] == true {
                exact_count += 1;
            } else if first_non_exact.is_none() {
                first_non_exact = Some(spec.name.clone());
            }
            stages.insert(spec.name, stage_metrics);
        }
        Ok(json!({
            "profile_id": "hf-eager-qwen-p2048-cache-on-prefill-probe-v1",
            "projection_bias_backend": forward.projection_bias_mode().id(),
            "attention_backend": forward.attention_selection().implementation_id(),
            "output_projection_backend": forward.output_projection_backend_id(),
            "mlp_projection_backend": forward.mlp_projection_backend_id(),
            "lm_head_backend": forward.lm_head_backend_id(),
            "rms_norm_backend": forward.rms_norm_backend_id(),
            "cache_mode": "cache-on-prefill-arithmetic-only",
            "same_scheduler_engine": false,
            "repeat_execution": {
                "reused_prepared_owner": true,
                "qkv_last_rows_bf16_exact": true,
                "layer0_output_last_row_bf16_exact": true,
                "last_logits_bf16_exact": true,
            },
            "static_layer0_capture": true,
            "last_token_layer_capture": {
                "layer_indices": layer_indices,
                "last_token_row_index": QWEN3B_PROMPT_TOKEN_COUNT - 1,
                "final_norm_output": true,
            },
            "summary": {
                "stage_count": stages.len(),
                "bf16_exact_stage_count": exact_count,
                "first_non_exact_stage": first_non_exact,
            },
            "stages": stages,
        }))
    })();
    let cleanup = close_resources(Some(forward), stream, context);
    match (result, cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => Err(format!(
            "P2048 cache-on prefill execution failed: {run_error}; cleanup also failed: {cleanup_error}"
        )
        .into()),
    }
}

fn collect_cache_on_m1_trace(
    trace: &PreparedLlamaDecodeM1Trace,
) -> TestResult<BTreeMap<String, Vec<u8>>> {
    if !trace.is_complete() {
        return Err("P2048 cache-on M1 trace did not capture every boundary".into());
    }
    let mut observed = BTreeMap::new();
    observed.insert(
        "embedding.last".to_owned(),
        canonical_bf16_le(
            trace
                .embedding()
                .ok_or("P2048 cache-on M1 embedding trace is missing")?,
        )?,
    );
    for stage in LlamaLastTokenLayerStage::ALL {
        let name = format!("layer0.{}.last", stage.name());
        observed.insert(
            name,
            canonical_bf16_le(
                trace
                    .layer_zero(stage)
                    .ok_or("P2048 cache-on M1 layer-zero trace is missing")?,
            )?,
        );
    }
    for layer_index in 0..QWEN3B_LAYER_COUNT {
        let name = format!("layer{layer_index}.output.last");
        let row = canonical_bf16_le(
            trace
                .layer_output(layer_index)
                .ok_or("P2048 cache-on M1 layer output trace is missing")?,
        )?;
        if layer_index == 0 {
            if observed.get(&name) != Some(&row) {
                return Err(
                    "P2048 cache-on M1 layer-zero output trace disagrees with residual trace"
                        .into(),
                );
            }
        } else {
            observed.insert(name, row);
        }
    }
    observed.insert(
        "final_norm.output.last".to_owned(),
        canonical_bf16_le(
            trace
                .final_norm_output()
                .ok_or("P2048 cache-on M1 final norm trace is missing")?,
        )?,
    );
    observed.insert(
        "last_logits".to_owned(),
        canonical_bf16_le(
            trace
                .logits()
                .ok_or("P2048 cache-on M1 logits trace is missing")?,
        )?,
    );
    if observed.len() != expected_stage_specs().len() {
        return Err("P2048 cache-on M1 trace stage count differs from the HF artifact".into());
    }
    Ok(observed)
}

fn run_cache_on_m1_reference_trace_profile(
    model: &LoadedModel,
    input: &[u32],
    m1_token: u32,
    hf: &HfCacheOnPrefillStageArtifact,
) -> TestResult<Value> {
    let (context, mut stream) = first_context()?;
    let config = PreparedLlamaDecodeConfig::new(PreparedLlamaForwardConfig::new(
        FULL_FORWARD_UPLOAD_STAGING_BYTES,
        FULL_FORWARD_IO_STAGING_BYTES,
        HF_COMPAT_GEMM_WORKSPACE_CAP_BYTES,
        REFERENCE_ATTENTION_BUDGET_BYTES,
    ))
    .with_hf_eager_qwen_p2048_cache_on_m1_trace_probe();
    let mut decode = match PreparedLlamaDecode::prepare(
        model,
        &context,
        &mut stream,
        input.len(),
        QWEN3B_PROMPT_TOKEN_COUNT + 2,
        config,
    ) {
        Ok(decode) => decode,
        Err(error) => {
            let cleanup = close_decode_resources(None, stream, context);
            return match cleanup {
                Ok(()) => Err(error.into()),
                Err(cleanup_error) => Err(format!(
                    "P2048 cache-on M1 decode preparation failed: {error}; cleanup also failed: {cleanup_error}"
                )
                .into()),
            };
        }
    };
    let result = (|| -> TestResult<Value> {
        if !matches!(
            decode.cache_layout(),
            riley_runtime::llama::LlamaKvCacheStorageLayout::Contiguous(_)
        ) {
            return Err("P2048 cache-on M1 trace did not select contiguous KV storage".into());
        }
        let selection = decode.prepared_attention().selection_trace();
        if selection.implementation_id() != "riley.cuda.materialized-gqa-decode.bf16" {
            return Err(
                "P2048 cache-on M1 trace did not select materialized reference decode attention"
                    .into(),
            );
        }

        decode.prefill(input, &mut stream)?;
        let mut trace = decode.prepare_hf_eager_qwen_p2048_cache_on_m1_trace()?;
        decode.decode_hf_eager_qwen_p2048_cache_on_m1_traced(m1_token, &mut trace, &mut stream)?;
        let observed = collect_cache_on_m1_trace(&trace)?;

        // Reset, rebuild the P2048 cache, and repeat with the same cold owner
        // and host trace storage. This verifies lifecycle reuse separately
        // from the offline oracle and is not a serving timing measurement.
        decode.reset()?;
        decode.prefill(input, &mut stream)?;
        decode.decode_hf_eager_qwen_p2048_cache_on_m1_traced(m1_token, &mut trace, &mut stream)?;
        let repeated = collect_cache_on_m1_trace(&trace)?;
        if observed != repeated {
            return Err(
                "P2048 cache-on M1 repeated trace differs with the same prepared owner".into(),
            );
        }

        let mut stages = Map::new();
        let mut exact_count = 0_u64;
        let mut first_non_exact = None;
        for spec in expected_stage_specs() {
            let hf_name = format!("decode_step_1.{}", spec.name);
            let hf_bytes = hf
                .tensors
                .get(&hf_name)
                .ok_or("HF P2048 cache-on M1 stage is missing")?;
            let riley_bytes = observed
                .get(&spec.name)
                .ok_or("Riley P2048 cache-on M1 stage is missing")?;
            if hf_bytes.len() != shape_byte_len(&spec.shape)? || riley_bytes.len() != hf_bytes.len()
            {
                return Err(
                    format!("P2048 cache-on M1 stage {} byte length differs", spec.name).into(),
                );
            }
            let stage_metrics = metrics(hf_bytes, riley_bytes)?;
            if stage_metrics["bf16_exact"] == true {
                exact_count += 1;
            } else if first_non_exact.is_none() {
                first_non_exact = Some(spec.name.clone());
            }
            stages.insert(spec.name, stage_metrics);
        }
        Ok(json!({
            "profile_id": "hf-eager-qwen-p2048-cache-on-m1-reference-trace-v1",
            "cache_mode": "cache-on-m1-diagnostic",
            "same_scheduler_engine": false,
            "prefill_profile": "hf-eager-qwen-p2048-cache-on-prefill-probe-v1",
            "m1_execution": {
                "teacher_forced_token_id": m1_token,
                "position": QWEN3B_PROMPT_TOKEN_COUNT,
                "logical_cache_length": QWEN3B_PROMPT_TOKEN_COUNT + 1,
                "kv_layout": "contiguous-head-major",
                "decode_attention_backend": selection.implementation_id(),
                "decode_attention_selection_reason": format!("{:?}", selection.reason()),
                "projection_path": "HF-compatible M1 Q/K/V cuBLASLt bias epilogues plus direct-cuBLAS O/MLP/LM-head candidate",
                "rms_norm_path": HF_EAGER_QWEN_P2048_CACHE_ON_RMS_NORM_BACKEND_ID,
            },
            "repeat_execution": {
                "reused_prepared_owner": true,
                "reused_trace_storage": true,
                "all_m1_trace_tensors_bf16_identical": true,
            },
            "summary": {
                "stage_count": stages.len(),
                "bf16_exact_stage_count": exact_count,
                "first_non_exact_stage": first_non_exact,
            },
            "stages": stages,
        }))
    })();
    let cleanup = close_decode_resources(Some(decode), stream, context);
    match (result, cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => Err(format!(
            "P2048 cache-on M1 trace execution failed: {run_error}; cleanup also failed: {cleanup_error}"
        )
        .into()),
    }
}

fn run_full_sequence_layer_stage_profile(
    model: &LoadedModel,
    input: &[u32],
    hf: &HfFullSequenceStageArtifact,
) -> TestResult<Value> {
    let (context, mut stream) = first_context()?;
    let config = PreparedLlamaForwardConfig::new(
        FULL_FORWARD_UPLOAD_STAGING_BYTES,
        FULL_FORWARD_IO_STAGING_BYTES,
        HF_COMPAT_GEMM_WORKSPACE_CAP_BYTES,
        REFERENCE_ATTENTION_BUDGET_BYTES,
    )
    .with_hf_compatible_bias_epilogue_projection_probe()
    .with_hugging_face_eager_qwen_p2051_probe_attention()
    .with_hf_eager_qwen_p2051_direct_cublas_output_projection_probe()
    .with_hf_eager_qwen_p2051_direct_cublas_mlp_projection_probe()
    .with_hugging_face_cuda_qwen_p2051_rope_table_probe()
    .with_hugging_face_cuda_qwen_p2051_rms_norm_probe();
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
                    "full-sequence forward preparation failed: {error}; cleanup also failed: {cleanup_error}"
                )
                .into()),
            };
        }
    };
    let result = (|| -> TestResult<Value> {
        if forward.projection_bias_mode()
            != LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1
            || forward.attention_selection().implementation_id()
                != HF_EAGER_QWEN_P2051_PROBE_ATTENTION_BACKEND_ID
            || forward.output_projection_backend_id()
                != HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID
            || forward.mlp_projection_backend_id()
                != HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID
            || forward.rms_norm_backend_id() != HF_EAGER_QWEN_P2051_RMS_NORM_BACKEND_ID
        {
            return Err("full-sequence forward selected an unexpected numerical backend".into());
        }
        let specs = expected_full_sequence_stage_specs(FULL_SEQUENCE_STAGE_LAYER_INDEX);
        let selected = specs
            .iter()
            .filter_map(|spec| match spec.source {
                StageSource::FullSequence(stage) => Some(stage),
                StageSource::HfEagerRopeTableCosine | StageSource::HfEagerRopeTableSine => None,
                _ => panic!("full-sequence stage source is malformed"),
            })
            .collect::<Vec<_>>();
        let mut trace = forward
            .prepare_full_sequence_layer_stage_trace(FULL_SEQUENCE_STAGE_LAYER_INDEX, &selected)?;
        if trace.layer_index() != FULL_SEQUENCE_STAGE_LAYER_INDEX
            || trace.requested_stage_count() != u32::try_from(selected.len())?
        {
            return Err("full-sequence layer trace preparation contract differs".into());
        }
        for spec in &specs {
            let StageSource::FullSequence(stage) = spec.source else {
                continue;
            };
            if trace.tensor_byte_len(stage) != shape_byte_len(&spec.shape)? {
                return Err(format!("full-sequence {} byte reservation differs", spec.name).into());
            }
        }
        let allocation_report = forward.allocation_report();
        forward.upload_tokens(input, &mut stream)?;
        forward.execute_full_sequence_layer_stage_traced(&mut stream, &mut trace)?;
        if !trace.is_complete() || trace.captured_stage_count() != u32::try_from(selected.len())? {
            return Err("full-sequence layer trace did not capture every requested stage".into());
        }
        let rope_tables = forward.download_hugging_face_bf16_rope_table_trace(&mut stream)?;
        let mut observed = BTreeMap::new();
        for spec in &specs {
            let row_major = match spec.source {
                StageSource::FullSequence(stage) => canonical_bf16_le(
                    trace
                        .tensor(stage)
                        .ok_or("full-sequence layer trace tensor is missing")?,
                )?,
                StageSource::HfEagerRopeTableCosine => canonical_bf16_le(rope_tables.cosine())?,
                StageSource::HfEagerRopeTableSine => canonical_bf16_le(rope_tables.sine())?,
                _ => return Err("full-sequence stage source is malformed".into()),
            };
            if row_major.len() != shape_byte_len(&spec.shape)? {
                return Err(format!("full-sequence {} byte length differs", spec.name).into());
            }
            observed.insert(spec.name.clone(), row_major);
        }

        // Re-run with the same prepared owner and the same trace storage. This
        // catches reuse or workspace-lifetime sensitivity without treating a
        // trace D2H path as a serving measurement.
        forward.execute_full_sequence_layer_stage_traced(&mut stream, &mut trace)?;
        if !trace.is_complete() || trace.captured_stage_count() != u32::try_from(selected.len())? {
            return Err("repeat full-sequence layer trace is incomplete".into());
        }
        let repeated_rope_tables =
            forward.download_hugging_face_bf16_rope_table_trace(&mut stream)?;
        for spec in &specs {
            let repeated = match spec.source {
                StageSource::FullSequence(stage) => canonical_bf16_le(
                    trace
                        .tensor(stage)
                        .ok_or("repeat full-sequence layer trace tensor is missing")?,
                )?,
                StageSource::HfEagerRopeTableCosine => {
                    canonical_bf16_le(repeated_rope_tables.cosine())?
                }
                StageSource::HfEagerRopeTableSine => {
                    canonical_bf16_le(repeated_rope_tables.sine())?
                }
                _ => return Err("full-sequence stage source is malformed".into()),
            };
            if observed.get(&spec.name) != Some(&repeated) {
                return Err(format!("repeat {} BF16 tensor differs", spec.name).into());
            }
        }

        let mut stages = Map::new();
        let mut exact_count = 0_u64;
        let mut first_non_exact = None;
        for spec in &specs {
            let hf_bytes = hf
                .tensors
                .get(&spec.name)
                .ok_or("HF full-sequence stage tensor is missing")?;
            let riley_bytes = observed
                .get(&spec.name)
                .ok_or("Riley full-sequence stage tensor is missing")?;
            if hf_bytes.len() != shape_byte_len(&spec.shape)? || riley_bytes.len() != hf_bytes.len()
            {
                return Err(format!(
                    "full-sequence {} byte length differs from contract",
                    spec.name
                )
                .into());
            }
            let stage_metrics = metrics(hf_bytes, riley_bytes)?;
            if stage_metrics["bf16_exact"] == true {
                exact_count += 1;
            } else if first_non_exact.is_none() {
                first_non_exact = Some(spec.name.clone());
            }
            stages.insert(spec.name.clone(), stage_metrics);
        }
        Ok(json!({
            "profile_id": "hf-compatible-bias-epilogue-probe-v1+hf-eager-qwen-p2051-probe-v1+hf-eager-qwen-p2051-direct-cublas-probe-v1+hf-eager-qwen-p2051-direct-cublas-probe-v1+hf-cuda-rope-table-probe-v1+hf-cuda-qwen-p2051-rmsnorm-probe-v1",
            "projection_bias_backend": forward.projection_bias_mode().id(),
            "attention_backend": forward.attention_selection().implementation_id(),
            "output_projection_backend": forward.output_projection_backend_id(),
            "mlp_projection_backend": forward.mlp_projection_backend_id(),
            "rms_norm_backend": forward.rms_norm_backend_id(),
            "use_cache": false,
            "same_scheduler_engine": false,
            "full_sequence_layer_capture": {
                "layer_index": FULL_SEQUENCE_STAGE_LAYER_INDEX,
                "token_count": CONTEXT_TOKEN_COUNT,
                "stage_count": selected.len(),
                "token_major_bf16": true,
            },
            "hf_eager_rope_table_capture": {
                "token_count": CONTEXT_TOKEN_COUNT,
                "head_dimension": QWEN3B_HEAD_DIMENSION,
                "table_dtype": "bfloat16",
                "table_layout": "token-major-duplicated-half",
            },
            "repeat_execution": {
                "reused_prepared_owner": true,
                "all_selected_full_sequence_tensors_bf16_exact": true,
            },
            "allocation_report": {
                "weight_bytes": allocation_report.weight_bytes(),
                "graph_bytes": allocation_report.graph_bytes(),
                "gemm_workspace_bytes": allocation_report.gemm_workspace_bytes(),
                "total_device_bytes": allocation_report.total_device_bytes(),
                "device_allocation_count": allocation_report.device_allocation_count(),
                "pinned_host_bytes": allocation_report.pinned_host_bytes(),
                "pinned_host_allocation_count": allocation_report.pinned_host_allocation_count(),
            },
            "summary": {
                "stage_count": stages.len(),
                "bf16_exact_stage_count": exact_count,
                "first_non_exact_stage": first_non_exact,
            },
            "stages": stages,
        }))
    })();
    let cleanup = close_resources(Some(forward), stream, context);
    match (result, cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => Err(format!(
            "full-sequence stage execution failed: {run_error}; cleanup also failed: {cleanup_error}"
        )
        .into()),
    }
}

fn full_sequence_quality_gate(profile: &Value) -> TestResult<Value> {
    let summary = profile
        .get("summary")
        .and_then(Value::as_object)
        .ok_or("full-sequence stage summary is missing")?;
    let exact_stage_count = summary
        .get("bf16_exact_stage_count")
        .and_then(Value::as_u64)
        .ok_or("full-sequence exact stage count is missing")?;
    let required_stage_count =
        u64::try_from(expected_full_sequence_stage_specs(FULL_SEQUENCE_STAGE_LAYER_INDEX).len())?;
    let p2051_rms_norm_selected = profile.get("rms_norm_backend").and_then(Value::as_str)
        == Some(HF_EAGER_QWEN_P2051_RMS_NORM_BACKEND_ID);
    let full_sequence_exact = p2051_rms_norm_selected
        && exact_stage_count == required_stage_count
        && summary
            .get("first_non_exact_stage")
            .is_some_and(Value::is_null);
    Ok(json!({
        "required_stage_count": required_stage_count,
        "candidate_exact_stage_count": exact_stage_count,
        "p2051_rms_norm_selected": p2051_rms_norm_selected,
        "layer2_full_sequence_bf16_exact": full_sequence_exact,
        "cache_on_eligible": false,
        "serving_selector_eligible": false,
        "performance_claim_eligible": false,
    }))
}

fn write_artifact_exclusive(path: &Path, document: &Value) -> TestResult {
    if !path.is_absolute() || path.extension().and_then(|value| value.to_str()) != Some("json") {
        return Err("P2051 stage output must be an absolute .json path".into());
    }
    let root = repository_root()?;
    fs::create_dir_all(path.parent().ok_or("P2051 stage output has no parent")?)?;
    let parent = path
        .parent()
        .ok_or("P2051 stage output has no parent")?
        .canonicalize()?;
    let output = parent.join(
        path.file_name()
            .ok_or("P2051 stage output has no basename")?,
    );
    if output == root || output.starts_with(&root) {
        return Err("P2051 stage output must be outside the repository".into());
    }
    if fs::symlink_metadata(&output).is_ok() {
        return Err("refusing to overwrite existing P2051 stage output".into());
    }
    let mut payload = serde_json::to_vec_pretty(document)?;
    payload.push(b'\n');
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    if let Err(error) = file.write_all(&payload).and_then(|_| file.sync_all()) {
        return Err(error.into());
    }
    Ok(())
}

fn unix_seconds() -> TestResult<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

#[test]
#[ignore = "remote-only Qwen2.5-3B P2048 cache-on prefill HF/Rust layer-stage discriminator"]
fn qwen3b_p2048_hf_compatible_cache_on_prefill_quality_gate() -> TestResult {
    let workload = load_workload()?;
    let teacher = load_teacher_cache_on()?;
    let hf = load_hf_cache_on_prefill_stage_artifact(&teacher, &workload)?;
    if workload.prompt_token_ids.len() != QWEN3B_PROMPT_TOKEN_COUNT
        || token_ids_sha256(&workload.prompt_token_ids) != QWEN3B_PROMPT_TOKEN_SHA256
    {
        return Err("P2048 cache-on prefill input does not bind the HF stage artifact".into());
    }
    let model = load_cache_on_prefill_model(&hf)?;
    let candidate = run_cache_on_prefill_profile(&model, &workload.prompt_token_ids, &hf)?;
    let quality_gate = cache_on_prefill_quality_gate(&candidate)?;
    let prefill_exact = quality_gate
        .get("cache_on_prefill_bf16_exact")
        .and_then(Value::as_bool)
        .ok_or("P2048 cache-on prefill quality gate is missing")?;
    let output = required_path("RILEY_QWEN3B_P2048_CACHE_ON_PREFILL_STAGE_OUTPUT")?;
    let receipt = json!({
        "schema_version": CACHE_ON_PREFILL_RESULT_SCHEMA_VERSION,
        "artifact_kind": CACHE_ON_PREFILL_RESULT_ARTIFACT_KIND,
        "performance_claim_eligible": false,
        "created_at_unix_seconds": unix_seconds()?,
        "contract": {
            "model_id": QWEN3B_MODEL_ID,
            "model_revision": QWEN3B_REVISION,
            "prefill_input_token_count": QWEN3B_PROMPT_TOKEN_COUNT,
            "prefill_input_token_ids_le_u32_sha256": QWEN3B_PROMPT_TOKEN_SHA256,
            "teacher_decode_token_ids": teacher.token_ids,
            "teacher_decode_token_ids_le_u32_sha256": token_ids_sha256(&teacher.token_ids),
            "teacher_forced_artifact_sha256": teacher.artifact_sha256,
            "teacher_cache_on_sidecar_sha256": teacher.cache_on_sidecar_sha256,
            "teacher_full_token_ids_le_u32_sha256": teacher.full_teacher_token_ids_sha256,
            "checkpoint_receipt_filename": hf.checkpoint_receipt_filename,
            "checkpoint_receipt_sha256": hf.checkpoint_receipt_sha256,
            "hf_execution": "P2048 cache-building prefill followed by future M1 decode",
            "riley_execution": "P2048 prefill arithmetic only; M1 cache-on decode remains separately gated",
            "candidate_attention_backend": HF_EAGER_QWEN_P2048_CACHE_ON_PROBE_ATTENTION_BACKEND_ID,
            "candidate_output_projection_backend": HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID,
            "candidate_mlp_projection_backend": HF_EAGER_QWEN_P2048_CACHE_ON_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID,
            "candidate_lm_head_backend": HF_EAGER_QWEN_P2048_CACHE_ON_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID,
            "candidate_rms_norm_backend": HF_EAGER_QWEN_P2048_CACHE_ON_RMS_NORM_BACKEND_ID,
        },
        "hf_stage_artifact": {
            "manifest_path": hf.manifest_path,
            "manifest_sha256": hf.manifest_sha256,
            "sidecar_path": hf.sidecar_path,
            "sidecar_sha256": hf.sidecar_sha256,
        },
        "profiles": [candidate],
        "quality_gate": quality_gate,
    });
    write_artifact_exclusive(&output, &receipt)?;
    println!(
        "{CACHE_ON_PREFILL_MARKER_PREFIX}{}",
        serde_json::to_string(&receipt)?
    );
    if !prefill_exact {
        return Err(
            "P2048 cache-on prefill quality gate failed; M1 decode and serving-selector promotion remain blocked"
                .into(),
        );
    }
    Ok(())
}

#[test]
#[ignore = "remote-only Qwen2.5-3B P2048 cache-on M1 HF/Rust layer-stage trace"]
fn qwen3b_p2048_hf_compatible_cache_on_m1_reference_trace() -> TestResult {
    let workload = load_workload()?;
    let teacher = load_teacher_cache_on()?;
    let hf = load_hf_cache_on_prefill_stage_artifact(&teacher, &workload)?;
    let m1_token = *teacher
        .token_ids
        .first()
        .ok_or("P2048 cache-on teacher artifact has no M1 token")?;
    if workload.prompt_token_ids.len() != QWEN3B_PROMPT_TOKEN_COUNT
        || token_ids_sha256(&workload.prompt_token_ids) != QWEN3B_PROMPT_TOKEN_SHA256
    {
        return Err("P2048 cache-on M1 input does not bind the HF stage artifact".into());
    }
    let model = load_cache_on_prefill_model(&hf)?;
    let candidate =
        run_cache_on_m1_reference_trace_profile(&model, &workload.prompt_token_ids, m1_token, &hf)?;
    let summary = candidate
        .get("summary")
        .and_then(Value::as_object)
        .ok_or("P2048 cache-on M1 candidate summary is missing")?;
    let exact_count = summary
        .get("bf16_exact_stage_count")
        .and_then(Value::as_u64)
        .ok_or("P2048 cache-on M1 exact stage count is missing")?;
    let stage_count = u64::try_from(expected_stage_specs().len())?;
    let m1_exact = exact_count == stage_count
        && summary
            .get("first_non_exact_stage")
            .is_some_and(Value::is_null);
    let output = required_path("RILEY_QWEN3B_P2048_CACHE_ON_M1_STAGE_OUTPUT")?;
    let receipt = json!({
        "schema_version": CACHE_ON_M1_RESULT_SCHEMA_VERSION,
        "artifact_kind": CACHE_ON_M1_RESULT_ARTIFACT_KIND,
        "performance_claim_eligible": false,
        "created_at_unix_seconds": unix_seconds()?,
        "contract": {
            "model_id": QWEN3B_MODEL_ID,
            "model_revision": QWEN3B_REVISION,
            "prefill_input_token_count": QWEN3B_PROMPT_TOKEN_COUNT,
            "prefill_input_token_ids_le_u32_sha256": QWEN3B_PROMPT_TOKEN_SHA256,
            "teacher_m1_token_id": m1_token,
            "teacher_decode_token_ids": teacher.token_ids,
            "teacher_forced_artifact_sha256": teacher.artifact_sha256,
            "teacher_cache_on_sidecar_sha256": teacher.cache_on_sidecar_sha256,
            "teacher_full_token_ids_le_u32_sha256": teacher.full_teacher_token_ids_sha256,
            "checkpoint_receipt_filename": hf.checkpoint_receipt_filename,
            "checkpoint_receipt_sha256": hf.checkpoint_receipt_sha256,
            "hf_execution": "P2048 cache-building prefill followed by teacher-forced M1",
            "riley_execution": "P2048 exact-prefill candidate plus source-bound M1 reference-decode trace",
            "trace_stage_count": stage_count,
        },
        "hf_stage_artifact": {
            "manifest_path": hf.manifest_path,
            "manifest_sha256": hf.manifest_sha256,
            "sidecar_path": hf.sidecar_path,
            "sidecar_sha256": hf.sidecar_sha256,
        },
        "candidate": candidate,
        "quality_gate": {
            "required_m1_stage_count": stage_count,
            "candidate_exact_m1_stage_count": exact_count,
            "cache_on_m1_decode_bf16_exact": m1_exact,
            "cache_on_full_forward_bf16_exact": false,
            "corrected_cache_on_eligible": false,
            "serving_selector_eligible": false,
            "performance_claim_eligible": false,
        },
    });
    write_artifact_exclusive(&output, &receipt)?;
    println!(
        "{CACHE_ON_M1_MARKER_PREFIX}{}",
        serde_json::to_string(&receipt)?
    );
    Ok(())
}

#[test]
#[ignore = "remote-only Qwen2.5-3B P2051 cache-free HF/Rust layer-stage discriminator"]
fn qwen3b_p2051_hf_compatible_cache_free_full_forward_quality_gate() -> TestResult {
    let workload = load_workload()?;
    let teacher = load_teacher_prefix()?;
    let hf = load_hf_stage_artifact(&teacher, &workload)?;
    let input = workload
        .prompt_token_ids
        .iter()
        .copied()
        .chain(teacher.token_ids.iter().copied())
        .collect::<Vec<_>>();
    if input.len() != CONTEXT_TOKEN_COUNT
        || token_ids_sha256(&input) != hf.input_token_ids_le_sha256
    {
        return Err("P2051 input does not bind the HF stage artifact".into());
    }
    let model = load_model(&hf)?;
    let strict = run_profile(
        &model,
        &input,
        &hf,
        LlamaProjectionBiasMode::StrictStagedV1,
        AttentionProfile::Reference,
        OutputProjectionProfile::StrictHiddenGemmV1,
        MlpProjectionProfile::StrictStagedV1,
        LmHeadProfile::StrictFullSequenceGemmV1,
        RopeTableProfile::Default,
        RmsNormProfile::Default,
    )?;
    let hf_compatible = run_profile(
        &model,
        &input,
        &hf,
        LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1,
        AttentionProfile::Reference,
        OutputProjectionProfile::StrictHiddenGemmV1,
        MlpProjectionProfile::StrictStagedV1,
        LmHeadProfile::StrictFullSequenceGemmV1,
        RopeTableProfile::Default,
        RmsNormProfile::Default,
    )?;
    let hf_compatible_qwen_p2051 = run_profile(
        &model,
        &input,
        &hf,
        LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1,
        AttentionProfile::HfEagerQwenP2051Probe,
        OutputProjectionProfile::StrictHiddenGemmV1,
        MlpProjectionProfile::StrictStagedV1,
        LmHeadProfile::StrictFullSequenceGemmV1,
        RopeTableProfile::Default,
        RmsNormProfile::Default,
    )?;
    let hf_compatible_qwen_p2051_o_projection = run_profile(
        &model,
        &input,
        &hf,
        LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1,
        AttentionProfile::HfEagerQwenP2051Probe,
        OutputProjectionProfile::HfEagerQwenP2051DirectCublasProbeV1,
        MlpProjectionProfile::StrictStagedV1,
        LmHeadProfile::StrictFullSequenceGemmV1,
        RopeTableProfile::Default,
        RmsNormProfile::Default,
    )?;
    let hf_compatible_qwen_p2051_all_projections = run_profile(
        &model,
        &input,
        &hf,
        LlamaProjectionBiasMode::HfCompatibleBiasEpilogueProbeV1,
        AttentionProfile::HfEagerQwenP2051Probe,
        OutputProjectionProfile::HfEagerQwenP2051DirectCublasProbeV1,
        MlpProjectionProfile::HfEagerQwenP2051DirectCublasProbeV1,
        LmHeadProfile::HfEagerQwenP2051LastTokenDirectCublasProbeV1,
        RopeTableProfile::HuggingFaceCudaQwenP2051ProbeV1,
        RmsNormProfile::HuggingFaceCudaQwenP2051ProbeV1,
    )?;
    let quality_gate = candidate_quality_gate(&hf_compatible_qwen_p2051_all_projections)?;
    let cache_off_full_forward_bf16_exact = quality_gate
        .get("cache_off_full_forward_bf16_exact")
        .and_then(Value::as_bool)
        .ok_or("HF-compatible cache-off gate is missing")?;
    let output = required_path("RILEY_QWEN3B_P2051_STAGE_OUTPUT")?;
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
            "teacher_prefix_token_count": TEACHER_PREFIX_TOKEN_COUNT,
            "teacher_prefix_token_ids": hf.teacher_prefix_token_ids,
            "teacher_forced_artifact_sha256": hf.teacher_artifact_sha256,
            "teacher_cache_off_sidecar_sha256": hf.teacher_sidecar_sha256,
            "teacher_full_token_ids_le_u32_sha256": hf.teacher_full_token_ids_sha256,
            "checkpoint_receipt_filename": hf.checkpoint_receipt_filename,
            "checkpoint_receipt_sha256": hf.checkpoint_receipt_sha256,
            "use_cache": false,
            "same_scheduler_engine": false,
            "baseline_attention_backend": DENSE_REFERENCE_ATTENTION_BACKEND_ID,
            "candidate_attention_backend": HF_EAGER_QWEN_P2051_PROBE_ATTENTION_BACKEND_ID,
            "baseline_output_projection_backend": STRICT_OUTPUT_PROJECTION_BACKEND_ID,
            "candidate_output_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID,
            "baseline_mlp_projection_backend": STRICT_MLP_PROJECTION_BACKEND_ID,
            "candidate_mlp_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID,
            "baseline_lm_head_backend": STRICT_LM_HEAD_BACKEND_ID,
            "candidate_lm_head_backend": HF_EAGER_QWEN_P2051_LAST_TOKEN_DIRECT_CUBLAS_LM_HEAD_BACKEND_ID,
            "candidate_rope_table_backend": "hf-cuda-rope-table-probe-v1",
            "candidate_rms_norm_backend": HF_EAGER_QWEN_P2051_RMS_NORM_BACKEND_ID,
            "last_token_row_index": LAST_TOKEN_ROW_INDEX,
        },
        "hf_stage_artifact": {
            "manifest_path": hf.manifest_path,
            "manifest_sha256": hf.manifest_sha256,
            "sidecar_path": hf.sidecar_path,
            "sidecar_sha256": hf.sidecar_sha256,
        },
        "profiles": [
            strict,
            hf_compatible,
            hf_compatible_qwen_p2051,
            hf_compatible_qwen_p2051_o_projection,
            hf_compatible_qwen_p2051_all_projections,
        ],
        "quality_gate": quality_gate,
    });
    write_artifact_exclusive(&output, &receipt)?;
    println!("{MARKER_PREFIX}{}", serde_json::to_string(&receipt)?);
    if !cache_off_full_forward_bf16_exact {
        return Err(
            "HF-compatible cache-off full-forward quality gate failed; receipt was written and cache-on/selector promotion remains blocked"
                .into(),
        );
    }
    Ok(())
}

#[test]
#[ignore = "remote-only Qwen2.5-3B P2051 full-sequence layer-stage discriminator"]
fn qwen3b_p2051_hf_compatible_full_sequence_layer2_quality_gate() -> TestResult {
    let workload = load_workload()?;
    let teacher = load_teacher_prefix()?;
    let hf = load_hf_full_sequence_stage_artifact(&teacher, &workload)?;
    let input = workload
        .prompt_token_ids
        .iter()
        .copied()
        .chain(teacher.token_ids.iter().copied())
        .collect::<Vec<_>>();
    if input.len() != CONTEXT_TOKEN_COUNT
        || token_ids_sha256(&input) != hf.input_token_ids_le_sha256
    {
        return Err("P2051 input does not bind the HF full-sequence stage artifact".into());
    }
    let model = load_full_sequence_model(&hf)?;
    let candidate = run_full_sequence_layer_stage_profile(&model, &input, &hf)?;
    let quality_gate = full_sequence_quality_gate(&candidate)?;
    let full_sequence_exact = quality_gate
        .get("layer2_full_sequence_bf16_exact")
        .and_then(Value::as_bool)
        .ok_or("full-sequence quality gate is missing")?;
    let output = required_path("RILEY_QWEN3B_P2051_FULL_SEQUENCE_STAGE_OUTPUT")?;
    let receipt = json!({
        "schema_version": FULL_SEQUENCE_RESULT_SCHEMA_VERSION,
        "artifact_kind": FULL_SEQUENCE_RESULT_ARTIFACT_KIND,
        "performance_claim_eligible": false,
        "created_at_unix_seconds": unix_seconds()?,
        "contract": {
            "model_id": QWEN3B_MODEL_ID,
            "model_revision": QWEN3B_REVISION,
            "input_token_count": CONTEXT_TOKEN_COUNT,
            "input_token_ids_le_u32_sha256": hf.input_token_ids_le_sha256,
            "teacher_prefix_token_count": TEACHER_PREFIX_TOKEN_COUNT,
            "teacher_prefix_token_ids": hf.teacher_prefix_token_ids,
            "teacher_forced_artifact_sha256": hf.teacher_artifact_sha256,
            "teacher_cache_off_sidecar_sha256": hf.teacher_sidecar_sha256,
            "teacher_full_token_ids_le_u32_sha256": hf.teacher_full_token_ids_sha256,
            "checkpoint_receipt_filename": hf.checkpoint_receipt_filename,
            "checkpoint_receipt_sha256": hf.checkpoint_receipt_sha256,
            "use_cache": false,
            "same_scheduler_engine": false,
            "layer_index": FULL_SEQUENCE_STAGE_LAYER_INDEX,
            "candidate_attention_backend": HF_EAGER_QWEN_P2051_PROBE_ATTENTION_BACKEND_ID,
            "candidate_output_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_OUTPUT_PROJECTION_BACKEND_ID,
            "candidate_mlp_projection_backend": HF_EAGER_QWEN_P2051_DIRECT_CUBLAS_MLP_PROJECTION_BACKEND_ID,
            "candidate_rope_table_backend": "hf-cuda-rope-table-probe-v1",
            "candidate_rms_norm_backend": HF_EAGER_QWEN_P2051_RMS_NORM_BACKEND_ID,
            "diagnostic_only": true,
        },
        "hf_full_sequence_stage_artifact": {
            "manifest_path": hf.manifest_path,
            "manifest_sha256": hf.manifest_sha256,
            "sidecar_path": hf.sidecar_path,
            "sidecar_sha256": hf.sidecar_sha256,
        },
        "candidate": candidate,
        "quality_gate": quality_gate,
    });
    write_artifact_exclusive(&output, &receipt)?;
    println!(
        "{FULL_SEQUENCE_MARKER_PREFIX}{}",
        serde_json::to_string(&receipt)?
    );
    if !full_sequence_exact {
        return Err(
            "HF-compatible full-sequence layer2 quality gate failed; receipt was written and cache-on/selector promotion remains blocked"
                .into(),
        );
    }
    Ok(())
}
