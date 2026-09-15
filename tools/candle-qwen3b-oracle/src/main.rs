//! CUDA-only Candle execution for a fixed Qwen2.5-3B diagnostic.
//!
//! The one `unsafe` call below is Candle's multi-file mmap constructor. It is
//! intentionally confined here, after the library verified a pinned receipt,
//! regular non-symlink shard paths, and the expected shard sizes. The caller
//! must keep the checkpoint immutable until the process exits.

#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use half::bf16;
use riley_candle_qwen3b_oracle::{
    CheckpointBinding, FileRecord, LogitCapture, OracleError, OracleResult, ServingWorkload,
    capture_logits, load_workload, publish_create_only, source_provenance, verify_checkpoint,
};
use serde::Serialize;

const SCHEMA_VERSION: &str = "riley.qwen3b-candle-bf16-raw-logits.v1";
const ARTIFACT_KIND: &str = "qwen2.5-3b-candle-0.9.1-cuda-bf16-p2048-numeric-triangulation";
const IMPLEMENTATION_ID: &str = "riley-candle-qwen3b-oracle-v1";

#[derive(Debug)]
struct Arguments {
    checkpoint: PathBuf,
    workload: PathBuf,
    output: PathBuf,
    repo_root: PathBuf,
    device: usize,
}

#[derive(Debug, Serialize)]
struct Artifact {
    schema_version: &'static str,
    artifact_kind: &'static str,
    performance_claim_eligible: bool,
    numeric_equivalence_gate: bool,
    contract: Contract,
    producer: Producer,
    checkpoint: CheckpointArtifact,
    execution: Execution,
    last_logits: LogitCapture,
}

#[derive(Debug, Serialize)]
struct Contract {
    workload: WorkloadArtifact,
    model: ModelArtifact,
}

#[derive(Debug, Serialize)]
struct WorkloadArtifact {
    schema_version: &'static str,
    case: &'static str,
    source_bytes: u64,
    source_sha256: String,
    prompt_token_count: usize,
    prompt_token_ids_le_u32_sha256: &'static str,
}

#[derive(Debug, Serialize)]
struct ModelArtifact {
    model_id: &'static str,
    model_revision: &'static str,
    vocabulary_size: usize,
    dtype: &'static str,
}

#[derive(Debug, Serialize)]
struct Producer {
    implementation_id: &'static str,
    runtime_dependency_class: &'static str,
    candle_version: &'static str,
    rustc_version: String,
    source: riley_candle_qwen3b_oracle::SourceProvenance,
}

#[derive(Debug, Serialize)]
struct CheckpointArtifact {
    receipt: FileRecord,
    files: Vec<FileRecord>,
    shard_content_rehashed: bool,
}

#[derive(Debug, Serialize)]
struct Execution {
    backend: &'static str,
    selected_device: GpuInfo,
    nvcc_version: String,
    dtype: &'static str,
    input_mode: &'static str,
    attention_cache_contract: &'static str,
    numerical_contract: &'static str,
}

#[derive(Debug, Serialize)]
struct GpuInfo {
    cuda_ordinal: usize,
    visible_devices: Option<String>,
    name: String,
    compute_capability: String,
    driver_version: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("riley-candle-qwen3b-oracle: {error}");
        process::exit(2);
    }
}

fn run() -> OracleResult<()> {
    let arguments = parse_arguments(env::args().skip(1))?;
    let repo_root = canonical_directory(&arguments.repo_root, "--repo-root")?;
    let workload = load_workload(&arguments.workload)?;
    let checkpoint = verify_checkpoint(&arguments.checkpoint)?;
    let source = source_provenance(&repo_root)?;
    let gpu = probe_gpu(arguments.device)?;
    let nvcc_version = command_stdout("nvcc", &["--version"])?;
    let rustc_version = command_stdout("rustc", &["--version"])?;

    let device = Device::new_cuda(arguments.device).map_err(|error| {
        OracleError::new(format!("open CUDA device {}: {error}", arguments.device))
    })?;
    if !device.supports_bf16() {
        return Err(OracleError::new(
            "Candle CUDA device does not report BF16 support",
        ));
    }
    let config: Config = serde_json::from_slice(&checkpoint.config_bytes)
        .map_err(|error| OracleError::new(format!("parse Candle Qwen2 config: {error}")))?;
    validate_candle_config(&config)?;
    let last_logits = execute_candle(&checkpoint, &workload, &config, &device)?;

    let artifact = Artifact {
        schema_version: SCHEMA_VERSION,
        artifact_kind: ARTIFACT_KIND,
        performance_claim_eligible: false,
        numeric_equivalence_gate: false,
        contract: Contract {
            workload: WorkloadArtifact {
                schema_version: riley_candle_qwen3b_oracle::WORKLOAD_SCHEMA,
                case: riley_candle_qwen3b_oracle::WORKLOAD_CASE,
                source_bytes: workload.source_bytes,
                source_sha256: workload.source_sha256.clone(),
                prompt_token_count: workload.prompt_token_ids.len(),
                prompt_token_ids_le_u32_sha256: riley_candle_qwen3b_oracle::PROMPT_TOKENS_SHA256,
            },
            model: ModelArtifact {
                model_id: riley_candle_qwen3b_oracle::MODEL_ID,
                model_revision: riley_candle_qwen3b_oracle::MODEL_REVISION,
                vocabulary_size: riley_candle_qwen3b_oracle::VOCABULARY_SIZE,
                dtype: "bfloat16",
            },
        },
        producer: Producer {
            implementation_id: IMPLEMENTATION_ID,
            runtime_dependency_class: "offline-rust-candle-numerical-triangulation",
            candle_version: "0.9.1",
            rustc_version,
            source,
        },
        checkpoint: CheckpointArtifact {
            receipt: checkpoint.receipt,
            files: checkpoint.files,
            shard_content_rehashed: false,
        },
        execution: Execution {
            backend: "candle-transformers-qwen2-cuda",
            selected_device: gpu,
            nvcc_version,
            dtype: "bfloat16",
            input_mode: "direct-immutable-u32-token-ids",
            attention_cache_contract: "Candle Qwen2 populates an internal KV cache during this first forward; it is cleared after capture and is not equivalent to Transformers use_cache=false.",
            numerical_contract: "Candle 0.9.1 Qwen2 uses BF16 RoPE tables, BF16 softmax, and Candle CUDA RMSNorm; this artifact is numerical triangulation, not an HF eager byte-equality gate.",
        },
        last_logits,
    };
    let mut bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| OracleError::new(format!("serialize artifact: {error}")))?;
    bytes.push(b'\n');
    let output = publish_create_only(&arguments.output, &repo_root, &bytes)?;
    println!("published {}", output.display());
    Ok(())
}

fn execute_candle(
    checkpoint: &CheckpointBinding,
    workload: &ServingWorkload,
    config: &Config,
    device: &Device,
) -> OracleResult<LogitCapture> {
    // SAFETY: `verify_checkpoint` accepted only the two pinned, regular,
    // non-symlink shard paths after receipt and size validation. The caller's
    // immutable-checkpoint contract keeps those files unchanged through model
    // construction and the one forward below; Candle documents mmap as unsafe
    // because it cannot enforce that external filesystem invariant itself.
    let variables = unsafe {
        VarBuilder::from_mmaped_safetensors(checkpoint.weight_paths(), DType::BF16, device)
    }
    .map_err(|error| OracleError::new(format!("mmap Qwen safetensors shards: {error}")))?;
    let mut model = ModelForCausalLM::new(config, variables)
        .map_err(|error| OracleError::new(format!("construct Candle Qwen2 model: {error}")))?;
    let input_ids = Tensor::from_vec(
        workload.prompt_token_ids.clone(),
        (1, workload.prompt_token_ids.len()),
        device,
    )
    .map_err(|error| OracleError::new(format!("create Candle input IDs: {error}")))?;
    let row = model
        .forward(&input_ids, 0)
        .and_then(|logits| logits.squeeze(0)?.squeeze(0))
        .map_err(|error| OracleError::new(format!("run Candle Qwen2 first forward: {error}")))?;
    let values = row
        .to_vec1::<bf16>()
        .map_err(|error| OracleError::new(format!("download Candle BF16 logits: {error}")))?;
    model.clear_kv_cache();
    let bits: Vec<u16> = values.iter().map(|value| value.to_bits()).collect();
    let floats: Vec<f32> = values.iter().map(|value| value.to_f32()).collect();
    capture_logits(&bits, &floats)
}

fn validate_candle_config(config: &Config) -> OracleResult<()> {
    let valid = config.vocab_size == riley_candle_qwen3b_oracle::VOCABULARY_SIZE
        && config.hidden_size == 2_048
        && config.intermediate_size == 11_008
        && config.num_hidden_layers == 36
        && config.num_attention_heads == 16
        && config.num_key_value_heads == 2
        && config.max_position_embeddings == 32_768
        && config.sliding_window == 32_768
        && config.max_window_layers == 70
        && config.tie_word_embeddings
        && config.rope_theta == 1_000_000.0
        && config.rms_norm_eps == 1e-6
        && !config.use_sliding_window;
    if !valid {
        return Err(OracleError::new(
            "Candle Qwen2 config differs from the pinned Qwen2.5-3B geometry",
        ));
    }
    Ok(())
}

fn probe_gpu(ordinal: usize) -> OracleResult<GpuInfo> {
    let ordinal_text = ordinal.to_string();
    let query = command_stdout(
        "nvidia-smi",
        &[
            "-i",
            &ordinal_text,
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let columns: Vec<&str> = query.trim().split(',').map(str::trim).collect();
    if columns.len() != 3 || columns.iter().any(|column| column.is_empty()) {
        return Err(OracleError::new(
            "could not parse nvidia-smi CUDA device identity",
        ));
    }
    let capability: f32 = columns[1]
        .parse()
        .map_err(|_| OracleError::new("nvidia-smi compute capability is invalid"))?;
    if capability < 8.0 {
        return Err(OracleError::new(
            "Candle diagnostic requires a CUDA compute capability of at least 8.0 for BF16",
        ));
    }
    Ok(GpuInfo {
        cuda_ordinal: ordinal,
        visible_devices: env::var("CUDA_VISIBLE_DEVICES").ok(),
        name: columns[0].to_owned(),
        compute_capability: columns[1].to_owned(),
        driver_version: columns[2].to_owned(),
    })
}

fn parse_arguments<I>(arguments: I) -> OracleResult<Arguments>
where
    I: IntoIterator<Item = String>,
{
    let values: Vec<String> = arguments.into_iter().collect();
    if values.len() == 1 && matches!(values[0].as_str(), "--help" | "-h") {
        print_help();
        process::exit(0);
    }
    let mut named = BTreeMap::new();
    let mut cursor = 0;
    while cursor < values.len() {
        let name = values[cursor].clone();
        if !matches!(
            name.as_str(),
            "--checkpoint" | "--workload" | "--output" | "--repo-root" | "--device"
        ) {
            return Err(OracleError::new(format!("unknown argument {name}")));
        }
        let value = values
            .get(cursor + 1)
            .ok_or_else(|| OracleError::new(format!("{name} requires a value")))?
            .clone();
        if named.insert(name.clone(), value).is_some() {
            return Err(OracleError::new(format!(
                "{name} was supplied more than once"
            )));
        }
        cursor += 2;
    }
    let path = |name: &str| -> OracleResult<PathBuf> {
        let value = named
            .get(name)
            .ok_or_else(|| OracleError::new(format!("{name} is required")))?;
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(OracleError::new(format!("{name} must be an absolute path")));
        }
        Ok(path)
    };
    let device = named
        .get("--device")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| OracleError::new("--device must be an unsigned integer"))
        })
        .transpose()?
        .unwrap_or(0);
    Ok(Arguments {
        checkpoint: path("--checkpoint")?,
        workload: path("--workload")?,
        output: path("--output")?,
        repo_root: path("--repo-root")?,
        device,
    })
}

fn canonical_directory(path: &Path, label: &str) -> OracleResult<PathBuf> {
    let canonical = path
        .canonicalize()
        .map_err(|error| OracleError::new(format!("canonicalize {label}: {error}")))?;
    if !canonical.is_dir() {
        return Err(OracleError::new(format!("{label} must name a directory")));
    }
    Ok(canonical)
}

fn command_stdout(program: &str, arguments: &[&str]) -> OracleResult<String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| OracleError::new(format!("run {program}: {error}")))?;
    if !output.status.success() {
        return Err(OracleError::new(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|_| OracleError::new(format!("{program} returned non-UTF-8 output")))
}

fn print_help() {
    println!(
        "Usage: riley-candle-qwen3b-oracle --checkpoint ABSOLUTE_DIR --workload ABSOLUTE_JSON --output ABSOLUTE_JSON --repo-root ABSOLUTE_DIR [--device N]"
    );
}
