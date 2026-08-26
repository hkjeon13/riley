use std::collections::BTreeMap;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use rustinfer_cuda::{CudaContext, CudaDevice, CudaRuntime, CudaStream};
use rustinfer_model::{EncodeOptions, LoadLimits, LoadedModel};
use rustinfer_runtime::generation::{FinishReason, GenerationRequest, GenerationState};
use rustinfer_runtime::llama::{
    LlamaReductionProfile, LlamaTracePoint, PreparedLlamaForward, PreparedLlamaForwardConfig,
    PreparedLlamaGeneration,
};
use rustinfer_runtime::sampling::SamplingParams;
use serde_json::{Map, Value, json};

use crate::contract::{
    ATTENTION_BACKEND, CALIBRATION_GATE_ID, CALIBRATION_PROMPT_COUNT, CALIBRATION_SCHEMA_VERSION,
    CALIBRATION_TOP_K, CANDIDATE_KIND, CROSS_CACHE_EXACT_WINDOW, EOS_TOKEN_ID,
    GATE_MANIFEST_SHA256, LOG_PROB_PIPELINE, MAX_CONTEXT_TOKENS, MODEL_CONFIG_SHA256, MODEL_ID,
    MODEL_REVISION, MODEL_WEIGHTS_SHA256, NATIVE_SOURCE_PATHS, PRIMARY_ENVIRONMENT_ID,
    PromptRecord, SEMANTIC_GENERATION_STEPS, TOKENIZER_FILES_SHA256, TOKENIZER_SHA256,
    candidate_reduction_variant, load_prompts, oracle_reduction_variant, sha256_bytes, sha256_file,
    token_ids_sha256, utc_now,
};
use crate::environment::{AcceleratorObservation, probe_primary_environment};
use crate::numeric::{
    canonical_log_softmax_bf16, exact_window_match, first_divergence, ranked_top_k_bf16,
};
use crate::sidecar::{SafeTensorWriter, TensorDType, TensorSpec};
use crate::{
    CalibrationArgs, NATIVE_BUILD_ARGV, NATIVE_ENGINE_REVISION, NATIVE_EXECUTABLE_FILENAME,
    ReductionVariant,
};

const HIDDEN_SIZE: usize = 576;
const VOCABULARY_SIZE: usize = 49_152;
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const BUILD_GIT_REVISION: &str = env!("RUSTINFER_NATIVE_BUILD_GIT_REVISION");
const BUILD_PROFILE: &str = env!("RUSTINFER_NATIVE_BUILD_PROFILE");

#[derive(Debug)]
pub struct NativeCalibrationError {
    message: String,
}

impl NativeCalibrationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn context(error: impl fmt::Display, context: &str) -> Self {
        Self::new(format!("{context}: {error}"))
    }
}

impl fmt::Display for NativeCalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for NativeCalibrationError {}

type ProducerResult<T> = Result<T, NativeCalibrationError>;

fn finish_with_cleanup<const N: usize>(
    operation: ProducerResult<()>,
    cleanups: [ProducerResult<()>; N],
) -> ProducerResult<()> {
    let mut failure = operation.err();
    for cleanup in cleanups {
        if let Err(cleanup) = cleanup {
            if let Some(primary) = failure.as_mut() {
                primary
                    .message
                    .push_str(&format!("; cleanup also failed: {cleanup}"));
            } else {
                failure = Some(cleanup);
            }
        }
    }
    failure.map_or(Ok(()), Err)
}

fn capture_phase(
    variant: ReductionVariant,
    phase: &'static str,
    operation: impl FnOnce() -> ProducerResult<()>,
) -> ProducerResult<()> {
    eprintln!(
        "rustinfer-native-calibration event=start variant={} phase={phase}",
        variant.id()
    );
    match operation() {
        Ok(()) => {
            eprintln!(
                "rustinfer-native-calibration event=complete variant={} phase={phase}",
                variant.id()
            );
            Ok(())
        }
        Err(error) => {
            eprintln!(
                "rustinfer-native-calibration event=failed variant={} phase={phase}",
                variant.id()
            );
            Err(NativeCalibrationError::context(
                error,
                &format!("capture phase={phase} variant={}", variant.id()),
            ))
        }
    }
}

fn require_cache_off_exact_token(
    prompt_id: &str,
    variant: ReductionVariant,
    step: usize,
    cache_on_token_ids: &[u32],
    observed: u32,
) -> ProducerResult<()> {
    if step >= CROSS_CACHE_EXACT_WINDOW {
        return Ok(());
    }
    let expected = cache_on_token_ids.get(step).copied();
    if expected == Some(observed) {
        return Ok(());
    }
    Err(NativeCalibrationError::new(format!(
        "phase=cache-off prompt_id={prompt_id} variant={} step={step} \
expected={expected:?} observed={observed} exact_window={CROSS_CACHE_EXACT_WINDOW}",
        variant.id()
    )))
}

#[derive(Clone, Debug)]
struct CaseState {
    prompt: PromptRecord,
    input_token_ids: Vec<u32>,
    variants: BTreeMap<&'static str, VariantState>,
}

#[derive(Clone, Debug, Default)]
struct VariantState {
    top_1_token_id: Option<u32>,
    top_k_token_id_set: Option<Vec<u32>>,
    cache_on: Option<SemanticPath>,
    cache_off: Option<SemanticPath>,
}

#[derive(Clone, Debug)]
struct SemanticPath {
    generated_token_ids: Vec<u32>,
    stop_reason: &'static str,
}

struct CacheOffStep {
    variant: ReductionVariant,
    profile: LlamaReductionProfile,
    base_length: usize,
    generated_count: usize,
    active: Vec<usize>,
}

#[derive(Clone, Copy)]
struct CacheOffPosition {
    variant: ReductionVariant,
    sequence_length: usize,
    generated_count: usize,
}

struct OutputTransaction {
    created: Vec<PathBuf>,
    committed: bool,
}

impl OutputTransaction {
    fn new() -> Self {
        Self {
            created: Vec::new(),
            committed: false,
        }
    }

    fn record(&mut self, path: PathBuf) {
        self.created.push(path);
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for OutputTransaction {
    fn drop(&mut self) {
        if !self.committed {
            for path in self.created.iter().rev() {
                let _ = fs::remove_file(path);
            }
        }
    }
}

struct CalibrationPaths {
    repository_root: PathBuf,
    model_root: PathBuf,
    manifest: PathBuf,
    sidecar: PathBuf,
    executable: PathBuf,
}

/// Produces one create-only native candidate manifest, sidecar, and executable bundle.
///
/// # Errors
///
/// Returns when any contract preflight, CUDA/model execution, provenance
/// recheck, cleanup, or durable artifact write fails.
pub fn run_calibration(args: &CalibrationArgs) -> Result<(), NativeCalibrationError> {
    let paths = resolve_calibration_paths(args)?;
    let prompts = load_calibration_prompts(args, &paths.repository_root)?;
    let source_hashes = hash_sources(&paths.repository_root)?;
    let git = crate::git::require_clean_repository(&paths.repository_root)
        .map_err(|error| NativeCalibrationError::context(error, "Git preflight"))?;
    validate_release_provenance(&git)?;

    let (device, observed_environment) = probe_calibration_device()?;
    let created_at =
        utc_now().map_err(|error| NativeCalibrationError::context(error, "capture timestamp"))?;

    let model = LoadedModel::load(&paths.model_root, LoadLimits::default())
        .map_err(|error| NativeCalibrationError::context(error, "load pinned checkpoint"))?;
    validate_model(&model)?;
    validate_model_root_files(&paths.model_root)?;
    let mut cases = materialize_cases(&model, prompts)?;
    let specs = tensor_specs(&cases)?;
    if specs.len() != CALIBRATION_PROMPT_COUNT * crate::REQUIRED_REDUCTION_VARIANTS.len() * 3 {
        return Err(NativeCalibrationError::new(
            "candidate tensor inventory is not exactly 93 tensors",
        ));
    }

    let mut transaction = OutputTransaction::new();
    bundle_executable(&paths.executable)?;
    transaction.record(paths.executable.clone());
    let executable_sha256 = sha256_file(&paths.executable)
        .map_err(|error| NativeCalibrationError::context(error, "hash bundled executable"))?;
    capture_sidecar(&paths.sidecar, &specs, &device, &model, &mut cases)?;
    transaction.record(paths.sidecar.clone());

    let post_git = crate::git::require_clean_repository(&paths.repository_root)
        .map_err(|error| NativeCalibrationError::context(error, "post-capture Git check"))?;
    if post_git != git || hash_sources(&paths.repository_root)? != source_hashes {
        return Err(NativeCalibrationError::new(
            "repository provenance changed during calibration",
        ));
    }
    validate_model_files(&model)?;
    validate_model_root_files(&paths.model_root)?;
    let sidecar_sha256 = sha256_file(&paths.sidecar)
        .map_err(|error| NativeCalibrationError::context(error, "hash sidecar"))?;
    let executable_post_sha256 = sha256_file(&paths.executable)
        .map_err(|error| NativeCalibrationError::context(error, "rehash bundled executable"))?;
    if executable_post_sha256 != executable_sha256 {
        return Err(NativeCalibrationError::new(
            "bundled executable changed during calibration",
        ));
    }
    let manifest = build_manifest(
        args,
        &cases,
        &created_at,
        &git.revision,
        &source_hashes,
        &observed_environment,
        &executable_sha256,
        &sidecar_sha256,
    )?;
    write_json_exclusive(&paths.manifest, &manifest)?;
    transaction.record(paths.manifest);
    transaction.commit();
    Ok(())
}

fn resolve_calibration_paths(args: &CalibrationArgs) -> ProducerResult<CalibrationPaths> {
    let repository_root = args
        .repository_root()
        .canonicalize()
        .map_err(|error| NativeCalibrationError::context(error, "canonicalize repository root"))?;
    if repository_root != args.repository_root() {
        return Err(NativeCalibrationError::new(
            "--repository-root must already be its canonical filesystem path",
        ));
    }
    let model_root = args
        .model()
        .canonicalize()
        .map_err(|error| NativeCalibrationError::context(error, "canonicalize model root"))?;
    if model_root != args.model() {
        return Err(NativeCalibrationError::new(
            "--model must already be its canonical filesystem path",
        ));
    }
    let output_root = std::env::current_dir()
        .and_then(|path| path.canonicalize())
        .map_err(|error| NativeCalibrationError::context(error, "canonicalize output directory"))?;
    if output_root == repository_root || output_root.starts_with(&repository_root) {
        return Err(NativeCalibrationError::new(
            "candidate artifacts must be written outside the repository",
        ));
    }
    let paths = CalibrationPaths {
        repository_root,
        model_root,
        manifest: output_root.join(args.manifest()),
        sidecar: output_root.join(args.sidecar()),
        executable: output_root.join(NATIVE_EXECUTABLE_FILENAME),
    };
    if paths.manifest == paths.sidecar
        || paths.manifest == paths.executable
        || paths.sidecar == paths.executable
    {
        return Err(NativeCalibrationError::new(
            "manifest, sidecar, and executable sibling names must be distinct",
        ));
    }
    for path in [&paths.manifest, &paths.sidecar, &paths.executable] {
        if path.exists() {
            return Err(NativeCalibrationError::new(format!(
                "refusing to overwrite existing output {}",
                path.display()
            )));
        }
    }
    Ok(paths)
}

fn load_calibration_prompts(
    args: &CalibrationArgs,
    repository_root: &Path,
) -> ProducerResult<Vec<PromptRecord>> {
    validate_gate(&repository_root.join(args.gate_manifest()))?;
    load_prompts(&repository_root.join(args.prompts()))
        .map_err(|error| NativeCalibrationError::context(error, "load prompt corpus"))
}

fn validate_release_provenance(git: &crate::git::CleanGitProvenance) -> ProducerResult<()> {
    if cfg!(debug_assertions)
        || !cfg!(panic = "abort")
        || BUILD_PROFILE != "release-cuda-linux-x86_64"
    {
        return Err(NativeCalibrationError::new(
            "native calibration requires the reviewed release profile",
        ));
    }
    if git.revision != BUILD_GIT_REVISION {
        return Err(NativeCalibrationError::new(format!(
            "runtime Git revision {} differs from build revision {BUILD_GIT_REVISION}",
            git.revision
        )));
    }
    if git.status_sha256 != EMPTY_SHA256 {
        return Err(NativeCalibrationError::new(
            "clean Git status digest is not SHA256(empty)",
        ));
    }
    Ok(())
}

fn probe_calibration_device() -> ProducerResult<(CudaDevice, Value)> {
    let runtime = CudaRuntime::initialize()
        .map_err(|error| NativeCalibrationError::context(error, "initialize CUDA runtime"))?;
    if runtime.device_count() != 1 {
        return Err(NativeCalibrationError::new(format!(
            "calibration requires one visible GPU, found {}",
            runtime.device_count()
        )));
    }
    let device = runtime
        .device(0)
        .map_err(|error| NativeCalibrationError::context(error, "read CUDA device properties"))?;
    let nvml = rustinfer_cuda::probe_nvidia_environment()
        .map_err(|error| NativeCalibrationError::context(error, "probe NVIDIA environment"))?;
    let accelerator = accelerator_observation(&nvml, device.properties())?;
    let environment = probe_primary_environment(&accelerator)
        .map_err(|error| NativeCalibrationError::context(error, "environment preflight"))?;
    Ok((device, environment))
}

fn capture_sidecar(
    path: &Path,
    specs: &[TensorSpec],
    device: &CudaDevice,
    model: &LoadedModel,
    cases: &mut [CaseState],
) -> ProducerResult<()> {
    let mut sidecar = SafeTensorWriter::create(path, specs)
        .map_err(|error| NativeCalibrationError::context(error, "create sidecar"))?;
    let context = device
        .create_context()
        .map_err(|error| NativeCalibrationError::context(error, "create CUDA context"))?;
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(error) => {
            let primary = NativeCalibrationError::context(error, "create CUDA stream");
            let cleanup = context.close().map_err(|cleanup| {
                NativeCalibrationError::context(cleanup, "close CUDA context after stream failure")
            });
            return finish_with_cleanup(Err(primary), [cleanup]);
        }
    };
    let capture = capture_all_variants(model, &context, &mut stream, &mut sidecar, cases);
    let stream_close = stream
        .close()
        .map_err(|error| NativeCalibrationError::context(error, "close CUDA stream"));
    let context_close = context
        .close()
        .map_err(|error| NativeCalibrationError::context(error, "close CUDA context"));
    finish_with_cleanup(capture, [stream_close, context_close])?;
    sidecar
        .finish()
        .map_err(|error| NativeCalibrationError::context(error, "finish sidecar"))
}

fn accelerator_observation(
    nvml: &rustinfer_cuda::NvidiaEnvironmentSnapshot,
    properties: &rustinfer_cuda::DeviceProperties,
) -> ProducerResult<AcceleratorObservation> {
    let cuda_driver_api_version = cuda_version_text(properties.driver_version())?;
    if properties.driver_version() != nvml.cuda_driver_api_version_raw()
        || cuda_driver_api_version != nvml.cuda_driver_api_version()
    {
        return Err(NativeCalibrationError::new(
            "CUDA and NVML driver API version observations differ",
        ));
    }
    let gpu = nvml
        .devices()
        .first()
        .ok_or_else(|| NativeCalibrationError::new("NVML reported no GPUs"))?;
    let persistence_mode = match gpu.persistence_mode() {
        rustinfer_cuda::NvidiaPersistenceMode::Disabled => "Disabled",
        rustinfer_cuda::NvidiaPersistenceMode::Enabled => "Enabled",
    };
    Ok(AcceleratorObservation {
        gpu_count: usize::try_from(nvml.gpu_count())
            .map_err(|_| NativeCalibrationError::new("NVML GPU count does not fit usize"))?,
        index: gpu.index(),
        name: gpu.name().to_owned(),
        compute_capability: properties.compute_capability(),
        memory_total_mib: gpu.total_memory_bytes() / (1024 * 1024),
        driver_version: nvml.driver_version().to_owned(),
        driver_cuda_api_version: cuda_driver_api_version,
        persistence_mode: persistence_mode.to_owned(),
        compute_process_count: usize::try_from(nvml.compute_process_count()).map_err(|_| {
            NativeCalibrationError::new("NVML compute process count does not fit usize")
        })?,
        memory_used_mib: gpu.used_memory_bytes() / (1024 * 1024),
        temperature_c: gpu.temperature_c(),
        power_limit_w: f64::from(gpu.power_limit_milliwatts()) / 1_000.0,
        application_graphics_clock_mhz: gpu.application_graphics_clock_mhz(),
        application_memory_clock_mhz: gpu.application_memory_clock_mhz(),
    })
}

fn cuda_version_text(encoded: i32) -> ProducerResult<String> {
    if encoded <= 0 {
        return Err(NativeCalibrationError::new(
            "CUDA driver version encoding must be positive",
        ));
    }
    let major = encoded / 1_000;
    let minor = encoded % 1_000 / 10;
    Ok(format!("{major}.{minor}"))
}

fn validate_gate(path: &Path) -> ProducerResult<()> {
    let bytes = fs::read(path)
        .map_err(|error| NativeCalibrationError::context(error, "read correctness gate"))?;
    if sha256_bytes(&bytes) != GATE_MANIFEST_SHA256 {
        return Err(NativeCalibrationError::new(
            "correctness gate bytes differ from the reviewed contract",
        ));
    }
    Ok(())
}

fn hash_sources(root: &Path) -> ProducerResult<BTreeMap<&'static str, String>> {
    NATIVE_SOURCE_PATHS
        .into_iter()
        .map(|(name, relative)| {
            sha256_file(&root.join(relative))
                .map(|digest| (name, digest))
                .map_err(|error| {
                    NativeCalibrationError::context(error, &format!("hash source {relative}"))
                })
        })
        .collect()
}

fn validate_model(model: &LoadedModel) -> ProducerResult<()> {
    if model.provenance().source_model() != MODEL_ID
        || model.provenance().source_revision() != MODEL_REVISION
        || model.spec().embedding().hidden_size() != HIDDEN_SIZE
        || model.spec().embedding().vocabulary_size() != VOCABULARY_SIZE
        || model.spec().max_sequence_length() != MAX_CONTEXT_TOKENS
        || model.spec().special_tokens().bos() != Some(EOS_TOKEN_ID)
        || model.spec().special_tokens().eos() != [EOS_TOKEN_ID]
    {
        return Err(NativeCalibrationError::new(
            "loaded checkpoint differs from the immutable SmolLM2 contract",
        ));
    }
    validate_model_files(model)
}

fn validate_model_files(model: &LoadedModel) -> ProducerResult<()> {
    let expected = [
        ("config.json", MODEL_CONFIG_SHA256),
        ("model.safetensors", MODEL_WEIGHTS_SHA256),
        ("tokenizer.json", TOKENIZER_FILES_SHA256[2].1),
    ];
    for (path, digest) in expected {
        let observed = model
            .provenance()
            .files()
            .get(Path::new(path))
            .ok_or_else(|| {
                NativeCalibrationError::new(format!("checkpoint provenance omits {path}"))
            })?;
        if observed.sha256() != digest {
            return Err(NativeCalibrationError::new(format!(
                "checkpoint {path} SHA-256 differs"
            )));
        }
    }
    Ok(())
}

fn validate_model_root_files(model_root: &Path) -> ProducerResult<()> {
    for (relative, expected) in [
        ("config.json", MODEL_CONFIG_SHA256),
        ("model.safetensors", MODEL_WEIGHTS_SHA256),
    ] {
        let observed = sha256_file(&model_root.join(relative)).map_err(|error| {
            NativeCalibrationError::context(error, &format!("rehash checkpoint {relative}"))
        })?;
        if observed != expected {
            return Err(NativeCalibrationError::new(format!(
                "checkpoint {relative} SHA-256 differs"
            )));
        }
    }
    let mut tokenizer_hashes = BTreeMap::new();
    for (relative, expected) in TOKENIZER_FILES_SHA256 {
        let observed = sha256_file(&model_root.join(relative)).map_err(|error| {
            NativeCalibrationError::context(error, &format!("rehash tokenizer {relative}"))
        })?;
        if observed != expected {
            return Err(NativeCalibrationError::new(format!(
                "tokenizer {relative} SHA-256 differs"
            )));
        }
        tokenizer_hashes.insert(relative, observed);
    }
    let canonical = serde_json::to_vec(&tokenizer_hashes).map_err(|error| {
        NativeCalibrationError::context(error, "serialize tokenizer hash aggregate")
    })?;
    if sha256_bytes(&canonical) != TOKENIZER_SHA256 {
        return Err(NativeCalibrationError::new(
            "tokenizer aggregate SHA-256 differs",
        ));
    }
    Ok(())
}

fn materialize_cases(
    model: &LoadedModel,
    prompts: Vec<PromptRecord>,
) -> ProducerResult<Vec<CaseState>> {
    prompts
        .into_iter()
        .map(|prompt| {
            let mut token_ids = model
                .tokenizer()
                .encode(&prompt.text, EncodeOptions::default())
                .map_err(|error| {
                    NativeCalibrationError::context(
                        error,
                        &format!("tokenize {}", prompt.prompt_id),
                    )
                })?;
            if token_ids.is_empty() {
                let fallback = model
                    .spec()
                    .special_tokens()
                    .bos()
                    .or_else(|| model.spec().special_tokens().eos().first().copied())
                    .ok_or_else(|| {
                        NativeCalibrationError::new("empty tokenization has no BOS/EOS fallback")
                    })?;
                token_ids.push(fallback);
            }
            if let Some(target) = prompt.target_prompt_tokens {
                if token_ids.len() >= target {
                    token_ids.truncate(target);
                } else {
                    let original = token_ids.clone();
                    token_ids
                        .try_reserve_exact(target - token_ids.len())
                        .map_err(|_| {
                            NativeCalibrationError::new("prompt token allocation failed")
                        })?;
                    while token_ids.len() < target {
                        let remaining = target - token_ids.len();
                        token_ids.extend_from_slice(&original[..remaining.min(original.len())]);
                    }
                }
            }
            if token_ids
                .len()
                .checked_add(SEMANTIC_GENERATION_STEPS)
                .is_none_or(|total| total > MAX_CONTEXT_TOKENS)
            {
                return Err(NativeCalibrationError::new(format!(
                    "{} lacks 32-token semantic headroom",
                    prompt.prompt_id
                )));
            }
            Ok(CaseState {
                prompt,
                input_token_ids: token_ids,
                variants: BTreeMap::new(),
            })
        })
        .collect()
}

fn tensor_specs(cases: &[CaseState]) -> ProducerResult<Vec<TensorSpec>> {
    let mut specs = Vec::new();
    specs
        .try_reserve_exact(cases.len() * crate::REQUIRED_REDUCTION_VARIANTS.len() * 3)
        .map_err(|_| NativeCalibrationError::new("tensor inventory allocation failed"))?;
    for case in cases {
        for variant in crate::REQUIRED_REDUCTION_VARIANTS {
            let prefix = format!("cases/{}/{}", case.prompt.prompt_id, variant.id());
            specs.extend([
                TensorSpec {
                    key: format!("{prefix}/first_layer_hidden"),
                    shape: vec![case.input_token_ids.len(), HIDDEN_SIZE],
                    dtype: TensorDType::Bf16,
                },
                TensorSpec {
                    key: format!("{prefix}/final_logits"),
                    shape: vec![VOCABULARY_SIZE],
                    dtype: TensorDType::Bf16,
                },
                TensorSpec {
                    key: format!("{prefix}/final_log_probs"),
                    shape: vec![VOCABULARY_SIZE],
                    dtype: TensorDType::F32,
                },
            ]);
        }
    }
    Ok(specs)
}

fn groups_by_length(cases: &[CaseState]) -> BTreeMap<usize, Vec<usize>> {
    let mut groups = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        groups
            .entry(case.input_token_ids.len())
            .or_insert_with(Vec::new)
            .push(index);
    }
    groups
}

fn capture_all_variants(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    sidecar: &mut SafeTensorWriter,
    cases: &mut [CaseState],
) -> ProducerResult<()> {
    let operation = (|| {
        for variant in crate::REQUIRED_REDUCTION_VARIANTS {
            capture_phase(variant, "numeric", || {
                capture_numeric_variant(model, context, stream, sidecar, cases, variant)
            })?;
            capture_phase(variant, "cache-on", || {
                capture_cache_on_variant(model, context, stream, cases, variant)
            })?;
            capture_phase(variant, "cache-off", || {
                capture_cache_off_variant(model, context, stream, cases, variant)
            })?;
            capture_phase(variant, "semantic-validation", || {
                validate_variant_semantics(cases, variant)
            })?;
        }
        Ok(())
    })();
    let stats = context
        .allocation_stats()
        .map_err(|error| NativeCalibrationError::context(error, "read CUDA allocation stats"));
    let zero = stats.and_then(|stats| {
        if stats.is_zero() {
            Ok(())
        } else {
            Err(NativeCalibrationError::new(format!(
                "CUDA allocations remain live after capture: {stats:?}"
            )))
        }
    });
    finish_with_cleanup(operation, [zero])
}

fn capture_numeric_variant(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    sidecar: &mut SafeTensorWriter,
    cases: &mut [CaseState],
    variant: ReductionVariant,
) -> ProducerResult<()> {
    let profile = reduction_profile(variant);
    let mut log_probs = vec![0_u8; VOCABULARY_SIZE * 4];
    for (sequence_length, indexes) in groups_by_length(cases) {
        let config = PreparedLlamaForwardConfig::default().with_reduction_profile(profile);
        let mut forward =
            PreparedLlamaForward::prepare(model, context, stream, sequence_length, config)
                .map_err(|error| {
                    NativeCalibrationError::context(error, "prepare numeric forward")
                })?;
        let mut trace = match forward
            .prepare_trace_points(&[LlamaTracePoint::Layer0Output, LlamaTracePoint::LastLogits])
        {
            Ok(trace) => trace,
            Err(error) => {
                let primary = NativeCalibrationError::context(error, "prepare numeric trace");
                let cleanup = forward.close().map_err(|cleanup| {
                    NativeCalibrationError::context(
                        cleanup,
                        "close numeric forward after trace failure",
                    )
                });
                return finish_with_cleanup(Err(primary), [cleanup]);
            }
        };
        let operation = (|| {
            for index in indexes {
                forward
                    .upload_tokens(&cases[index].input_token_ids, stream)
                    .and_then(|()| forward.execute_traced(stream, &mut trace))
                    .map_err(|error| {
                        NativeCalibrationError::context(
                            error,
                            &format!("capture {}/{}", cases[index].prompt.prompt_id, variant.id()),
                        )
                    })?;
                if trace.captured_count() != 2 {
                    return Err(NativeCalibrationError::new(
                        "numeric trace did not capture exactly two tensors",
                    ));
                }
                let hidden = trace.tensor(LlamaTracePoint::Layer0Output).ok_or_else(|| {
                    NativeCalibrationError::new("first-layer tensor is absent from trace")
                })?;
                let logits = trace.tensor(LlamaTracePoint::LastLogits).ok_or_else(|| {
                    NativeCalibrationError::new("last-logits tensor is absent from trace")
                })?;
                let prefix = format!("cases/{}/{}", cases[index].prompt.prompt_id, variant.id());
                sidecar
                    .write_tensor(&format!("{prefix}/first_layer_hidden"), hidden)
                    .map_err(|error| {
                        NativeCalibrationError::context(error, "write first-layer tensor")
                    })?;
                sidecar
                    .write_tensor(&format!("{prefix}/final_logits"), logits)
                    .map_err(|error| {
                        NativeCalibrationError::context(error, "write final-logits tensor")
                    })?;
                canonical_log_softmax_bf16(logits, &mut log_probs).map_err(|error| {
                    NativeCalibrationError::context(error, "compute final log-softmax")
                })?;
                sidecar
                    .write_tensor(&format!("{prefix}/final_log_probs"), &log_probs)
                    .map_err(|error| {
                        NativeCalibrationError::context(error, "write final-log-prob tensor")
                    })?;
                let (top_1, top_k) = ranked_top_k_bf16(logits).map_err(|error| {
                    NativeCalibrationError::context(error, "derive top-k from logits")
                })?;
                cases[index].variants.insert(
                    variant.id(),
                    VariantState {
                        top_1_token_id: Some(top_1),
                        top_k_token_id_set: Some(top_k),
                        ..VariantState::default()
                    },
                );
            }
            Ok(())
        })();
        let close = forward
            .close()
            .map_err(|error| NativeCalibrationError::context(error, "close numeric forward"));
        finish_with_cleanup(operation, [close])?;
    }
    Ok(())
}

fn capture_cache_on_variant(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    cases: &mut [CaseState],
    variant: ReductionVariant,
) -> ProducerResult<()> {
    let profile = reduction_profile(variant);
    for (prompt_length, indexes) in groups_by_length(cases) {
        let config = rustinfer_runtime::llama::PreparedLlamaDecodeConfig::default()
            .with_reduction_profile(profile);
        let mut generation = PreparedLlamaGeneration::prepare(
            model,
            context,
            stream,
            prompt_length,
            SEMANTIC_GENERATION_STEPS,
            config,
        )
        .map_err(|error| NativeCalibrationError::context(error, "prepare cache-on generation"))?;
        let operation = (|| {
            for index in indexes {
                let request = GenerationRequest {
                    request_id: format!(
                        "native-calibration/{}/{}",
                        cases[index].prompt.prompt_id,
                        variant.id()
                    )
                    .into_bytes(),
                    seed: 0,
                    prompt_token_ids: cases[index].input_token_ids.clone(),
                    sampling_params: SamplingParams {
                        temperature: 0.0,
                        top_k: None,
                        top_p: None,
                        repetition_penalty: 1.0,
                    },
                    min_new_tokens: 0,
                    max_new_tokens: SEMANTIC_GENERATION_STEPS,
                    eos_token_ids: vec![EOS_TOKEN_ID],
                    stop_token_ids: Vec::new(),
                    stop_strings: Vec::new(),
                };
                let mut state = GenerationState::new(
                    request,
                    VOCABULARY_SIZE,
                    model.tokenizer().maximum_decoded_token_bytes(),
                )
                .map_err(|error| {
                    NativeCalibrationError::context(error, "construct cache-on generation state")
                })?;
                generation
                    .generate(&mut state, stream, || false, |_| Ok::<(), Infallible>(()))
                    .map_err(|error| {
                        NativeCalibrationError::context(
                            error,
                            &format!(
                                "cache-on generation {}/{}",
                                cases[index].prompt.prompt_id,
                                variant.id()
                            ),
                        )
                    })?;
                let stop_reason = semantic_stop_reason(state.finish_reason())?;
                let generated = state.generated_token_ids().to_vec();
                cases[index]
                    .variants
                    .get_mut(variant.id())
                    .ok_or_else(|| NativeCalibrationError::new("numeric variant is absent"))?
                    .cache_on = Some(SemanticPath {
                    generated_token_ids: generated,
                    stop_reason,
                });
            }
            Ok(())
        })();
        let close = generation
            .close()
            .map_err(|error| NativeCalibrationError::context(error, "close cache-on generation"));
        finish_with_cleanup(operation, [close])?;
    }
    Ok(())
}

fn capture_cache_off_variant(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    cases: &mut [CaseState],
    variant: ReductionVariant,
) -> ProducerResult<()> {
    initialize_cache_off_paths(cases, variant)?;

    let profile = reduction_profile(variant);
    let groups = groups_by_length(cases);
    let mut logits = vec![0_u8; VOCABULARY_SIZE * 2];
    for (base_length, group_indexes) in groups {
        for generated_count in 1..SEMANTIC_GENERATION_STEPS {
            let active = group_indexes
                .iter()
                .copied()
                .filter(|&index| {
                    cases[index]
                        .variants
                        .get(variant.id())
                        .and_then(|state| state.cache_off.as_ref())
                        .is_some_and(|path| {
                            path.stop_reason == "pending"
                                && path.generated_token_ids.len() == generated_count
                        })
                })
                .collect::<Vec<_>>();
            if active.is_empty() {
                break;
            }
            capture_cache_off_step(
                model,
                context,
                stream,
                cases,
                CacheOffStep {
                    variant,
                    profile,
                    base_length,
                    generated_count,
                    active,
                },
                &mut logits,
            )?;
        }
    }
    finalize_cache_off_paths(cases, variant)
}

fn capture_cache_off_step(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    cases: &mut [CaseState],
    step: CacheOffStep,
    logits: &mut [u8],
) -> ProducerResult<()> {
    let CacheOffStep {
        variant,
        profile,
        base_length,
        generated_count,
        active,
    } = step;
    let sequence_length = base_length
        .checked_add(generated_count)
        .ok_or_else(|| NativeCalibrationError::new("cache-off length overflow"))?;
    let position = CacheOffPosition {
        variant,
        sequence_length,
        generated_count,
    };
    let active_count = active.len();
    eprintln!(
        "rustinfer-native-calibration event=start variant={} phase=cache-off-step \
base_length={base_length} sequence_length={sequence_length} step={generated_count} \
active_cases={active_count}",
        variant.id()
    );
    let config = PreparedLlamaForwardConfig::default().with_reduction_profile(profile);
    let mut forward =
        PreparedLlamaForward::prepare(model, context, stream, sequence_length, config).map_err(
            |error| {
                NativeCalibrationError::context(
                    error,
                    &format!(
                        "prepare growing-prefix forward variant={} base_length={base_length} \
sequence_length={sequence_length} step={generated_count}",
                        variant.id()
                    ),
                )
            },
        )?;
    let operation = (|| {
        for index in active {
            capture_cache_off_case(&mut forward, stream, cases, index, position, logits)?;
        }
        Ok(())
    })();
    let close = forward.close().map_err(|error| {
        NativeCalibrationError::context(
            error,
            &format!(
                "close growing-prefix forward variant={} base_length={base_length} \
sequence_length={sequence_length} step={generated_count}",
                variant.id()
            ),
        )
    });
    finish_with_cleanup(operation, [close])?;
    eprintln!(
        "rustinfer-native-calibration event=complete variant={} phase=cache-off-step \
base_length={base_length} sequence_length={sequence_length} step={generated_count} \
active_cases={active_count}",
        variant.id()
    );
    Ok(())
}

fn capture_cache_off_case(
    forward: &mut PreparedLlamaForward,
    stream: &mut CudaStream,
    cases: &mut [CaseState],
    index: usize,
    position: CacheOffPosition,
    logits: &mut [u8],
) -> ProducerResult<()> {
    let CacheOffPosition {
        variant,
        sequence_length,
        generated_count,
    } = position;
    let prompt_id = &cases[index].prompt.prompt_id;
    let generated = &cases[index]
        .variants
        .get(variant.id())
        .and_then(|state| state.cache_off.as_ref())
        .ok_or_else(|| NativeCalibrationError::new("cache-off state is absent"))?
        .generated_token_ids;
    let mut prefix = Vec::new();
    prefix
        .try_reserve_exact(sequence_length)
        .map_err(|_| NativeCalibrationError::new("prefix allocation failed"))?;
    prefix.extend_from_slice(&cases[index].input_token_ids);
    prefix.extend_from_slice(generated);
    forward
        .forward(&prefix, stream)
        .and_then(|()| forward.download_last_logits(logits, stream))
        .map_err(|error| {
            NativeCalibrationError::context(
                error,
                &format!(
                    "execute cache-off forward prompt_id={prompt_id} variant={} \
sequence_length={sequence_length} step={generated_count}",
                    variant.id()
                ),
            )
        })?;
    let (next, _) = ranked_top_k_bf16(logits).map_err(|error| {
        NativeCalibrationError::context(
            error,
            &format!(
                "rank cache-off logits prompt_id={prompt_id} variant={} \
sequence_length={sequence_length} step={generated_count}",
                variant.id()
            ),
        )
    })?;
    let cache_on_token_ids = &cases[index]
        .variants
        .get(variant.id())
        .and_then(|state| state.cache_on.as_ref())
        .ok_or_else(|| NativeCalibrationError::new("cache-on state is absent"))?
        .generated_token_ids;
    require_cache_off_exact_token(
        prompt_id,
        variant,
        generated_count,
        cache_on_token_ids,
        next,
    )?;
    let path = cases[index]
        .variants
        .get_mut(variant.id())
        .and_then(|state| state.cache_off.as_mut())
        .ok_or_else(|| NativeCalibrationError::new("cache-off state is absent"))?;
    path.generated_token_ids.push(next);
    if next == EOS_TOKEN_ID {
        path.stop_reason = "eos";
    }
    Ok(())
}

fn initialize_cache_off_paths(
    cases: &mut [CaseState],
    variant: ReductionVariant,
) -> ProducerResult<()> {
    for case in cases.iter_mut() {
        let state = case
            .variants
            .get(variant.id())
            .ok_or_else(|| NativeCalibrationError::new("numeric variant is absent"))?;
        let top_1 = state
            .top_1_token_id
            .ok_or_else(|| NativeCalibrationError::new("numeric top-1 is absent"))?;
        let cache_on_token_ids = &state
            .cache_on
            .as_ref()
            .ok_or_else(|| NativeCalibrationError::new("cache-on state is absent"))?
            .generated_token_ids;
        require_cache_off_exact_token(
            &case.prompt.prompt_id,
            variant,
            0,
            cache_on_token_ids,
            top_1,
        )?;
        let stop_reason = (top_1 == EOS_TOKEN_ID).then_some("eos");
        case.variants
            .get_mut(variant.id())
            .ok_or_else(|| NativeCalibrationError::new("numeric variant is absent"))?
            .cache_off = Some(SemanticPath {
            generated_token_ids: vec![top_1],
            stop_reason: stop_reason.unwrap_or("pending"),
        });
    }
    Ok(())
}

fn finalize_cache_off_paths(
    cases: &mut [CaseState],
    variant: ReductionVariant,
) -> ProducerResult<()> {
    for case in cases {
        let path = case
            .variants
            .get_mut(variant.id())
            .and_then(|state| state.cache_off.as_mut())
            .ok_or_else(|| NativeCalibrationError::new("cache-off state is absent"))?;
        if path.stop_reason == "pending" {
            if path.generated_token_ids.len() != SEMANTIC_GENERATION_STEPS {
                return Err(NativeCalibrationError::new(
                    "cache-off max-token path has the wrong length",
                ));
            }
            path.stop_reason = "max_new_tokens";
        }
    }
    Ok(())
}

fn validate_variant_semantics(
    cases: &[CaseState],
    variant: ReductionVariant,
) -> ProducerResult<()> {
    for case in cases {
        let state = case
            .variants
            .get(variant.id())
            .ok_or_else(|| NativeCalibrationError::new("variant state is absent"))?;
        let top_1 = state
            .top_1_token_id
            .ok_or_else(|| NativeCalibrationError::new("top-1 is absent"))?;
        let top_k = state
            .top_k_token_id_set
            .as_ref()
            .ok_or_else(|| NativeCalibrationError::new("top-k is absent"))?;
        let cache_on = state
            .cache_on
            .as_ref()
            .ok_or_else(|| NativeCalibrationError::new("cache-on path is absent"))?;
        let cache_off = state
            .cache_off
            .as_ref()
            .ok_or_else(|| NativeCalibrationError::new("cache-off path is absent"))?;
        if top_k.len() != CALIBRATION_TOP_K
            || cache_on.generated_token_ids.first() != Some(&top_1)
            || cache_off.generated_token_ids.first() != Some(&top_1)
        {
            return Err(NativeCalibrationError::new(format!(
                "{}/{} semantic top-1 is inconsistent",
                case.prompt.prompt_id,
                variant.id()
            )));
        }
        if !exact_window_match(
            &cache_on.generated_token_ids,
            &cache_off.generated_token_ids,
        ) {
            return Err(NativeCalibrationError::new(format!(
                "{}/{} cache paths diverge before token {}",
                case.prompt.prompt_id,
                variant.id(),
                CROSS_CACHE_EXACT_WINDOW
            )));
        }
        if case.prompt.category == "early-eos"
            && (cache_on.generated_token_ids != [EOS_TOKEN_ID]
                || cache_off.generated_token_ids != [EOS_TOKEN_ID]
                || cache_on.stop_reason != "eos"
                || cache_off.stop_reason != "eos")
        {
            return Err(NativeCalibrationError::new(
                "early-eos prompt did not stop on output token zero",
            ));
        }
    }
    Ok(())
}

fn semantic_stop_reason(reason: Option<FinishReason>) -> ProducerResult<&'static str> {
    match reason {
        Some(FinishReason::Eos) => Ok("eos"),
        Some(FinishReason::Length) => Ok("max_new_tokens"),
        observed => Err(NativeCalibrationError::new(format!(
            "unexpected semantic finish reason {observed:?}"
        ))),
    }
}

const fn reduction_profile(variant: ReductionVariant) -> LlamaReductionProfile {
    match variant {
        ReductionVariant::Canonical => LlamaReductionProfile::CanonicalV1,
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_manifest(
    args: &CalibrationArgs,
    cases: &[CaseState],
    created_at: &str,
    git_revision: &str,
    source_hashes: &BTreeMap<&'static str, String>,
    observed_environment: &Value,
    executable_sha256: &str,
    sidecar_sha256: &str,
) -> ProducerResult<Value> {
    let source_refs = NATIVE_SOURCE_PATHS
        .into_iter()
        .map(|(name, path)| {
            let digest = source_hashes.get(name).ok_or_else(|| {
                NativeCalibrationError::new(format!("source hash {name} is absent"))
            })?;
            Ok((name.to_owned(), json!({"path": path, "sha256": digest})))
        })
        .collect::<ProducerResult<Map<String, Value>>>()?;
    let tokenizer_files = TOKENIZER_FILES_SHA256
        .into_iter()
        .map(|(name, digest)| (name.to_owned(), json!(digest)))
        .collect::<Map<String, Value>>();
    let required_variants = crate::REQUIRED_REDUCTION_VARIANTS
        .into_iter()
        .map(|variant| candidate_reduction_variant(variant.id()))
        .collect::<Vec<_>>();
    let case_values = cases
        .iter()
        .map(case_manifest_value)
        .collect::<ProducerResult<Vec<_>>>()?;
    let capture_argv = vec![
        NATIVE_EXECUTABLE_FILENAME.to_owned(),
        "calibrate".to_owned(),
        "--repository-root".to_owned(),
        path_text(args.repository_root())?,
        "--model".to_owned(),
        path_text(args.model())?,
        "--gate-manifest".to_owned(),
        path_text(args.gate_manifest())?,
        "--prompts".to_owned(),
        path_text(args.prompts())?,
        "--manifest".to_owned(),
        path_text(args.manifest())?,
        "--sidecar".to_owned(),
        path_text(args.sidecar())?,
        "--reduction-variant".to_owned(),
        crate::REQUIRED_REDUCTION_VARIANTS[0].id().to_owned(),
    ];
    Ok(json!({
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "artifact_kind": CANDIDATE_KIND,
        "created_at": created_at,
        "producer": {
            "implementation_id": "rustinfer-native",
            "engine_revision": NATIVE_ENGINE_REVISION,
            "runtime_dependency_class": "native-production",
            "python_version": null,
            "python_executable_sha256": null,
            "python_platform_system": null,
            "python_platform_machine": null,
            "torch_version": null,
            "transformers_version": null,
            "safetensors_version": null,
        },
        "candidate_execution": {
            "executable": {
                "path": NATIVE_EXECUTABLE_FILENAME,
                "sha256": executable_sha256,
            },
            "build_argv": NATIVE_BUILD_ARGV,
            "capture_argv": capture_argv,
        },
        "contract": {
            "model_id": MODEL_ID,
            "gate_id": CALIBRATION_GATE_ID,
            "model_revision": MODEL_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_files_sha256": tokenizer_files,
            "dtype": "bfloat16",
            "attention_backend": ATTENTION_BACKEND,
            "tensor_capture_cache_path": "off",
            "log_prob_pipeline": LOG_PROB_PIPELINE,
            "trust_remote_code": false,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "eos_token_ids": [EOS_TOKEN_ID],
            "semantic_generation_steps": SEMANTIC_GENERATION_STEPS,
            "cross_cache_exact_window": CROSS_CACHE_EXACT_WINDOW,
            "top_k": CALIBRATION_TOP_K,
            "oracle_reduction_variant": oracle_reduction_variant(),
            "required_candidate_reduction_variants": required_variants,
        },
        "provenance": {
            "sources": source_refs,
            "git_revision": git_revision,
            "git_dirty": false,
            "git_status_sha256": EMPTY_SHA256,
            "environment_id": PRIMARY_ENVIRONMENT_ID,
            "observed_environment": observed_environment,
        },
        "corpus": {"prompt_count": cases.len()},
        "sidecar": {
            "path": path_text(args.sidecar())?,
            "sha256": sidecar_sha256,
            "format": "safetensors",
            "tensor_count": cases.len() * crate::REQUIRED_REDUCTION_VARIANTS.len() * 3,
        },
        "cases": case_values,
    }))
}

fn case_manifest_value(case: &CaseState) -> ProducerResult<Value> {
    let mut variants = Map::new();
    for variant in crate::REQUIRED_REDUCTION_VARIANTS {
        let state = case
            .variants
            .get(variant.id())
            .ok_or_else(|| NativeCalibrationError::new("completed variant is absent"))?;
        let top_1 = state
            .top_1_token_id
            .ok_or_else(|| NativeCalibrationError::new("completed top-1 is absent"))?;
        let top_k = state
            .top_k_token_id_set
            .as_ref()
            .ok_or_else(|| NativeCalibrationError::new("completed top-k is absent"))?;
        let cache_on = state
            .cache_on
            .as_ref()
            .ok_or_else(|| NativeCalibrationError::new("completed cache-on is absent"))?;
        let cache_off = state
            .cache_off
            .as_ref()
            .ok_or_else(|| NativeCalibrationError::new("completed cache-off is absent"))?;
        let prefix = format!("cases/{}/{}", case.prompt.prompt_id, variant.id());
        variants.insert(
            variant.id().to_owned(),
            json!({
                "config": candidate_reduction_variant(variant.id()),
                "tensors": {
                    "first_layer_hidden": tensor_ref(
                        &format!("{prefix}/first_layer_hidden"),
                        &[case.input_token_ids.len(), HIDDEN_SIZE],
                        TensorDType::Bf16,
                    ),
                    "final_logits": tensor_ref(
                        &format!("{prefix}/final_logits"),
                        &[VOCABULARY_SIZE],
                        TensorDType::Bf16,
                    ),
                    "final_log_probs": tensor_ref(
                        &format!("{prefix}/final_log_probs"),
                        &[VOCABULARY_SIZE],
                        TensorDType::F32,
                    ),
                },
                "semantic": {
                    "top_1_token_id": top_1,
                    "top_k_token_id_set": top_k,
                    "cache_on": semantic_path_value(cache_on),
                    "cache_off": semantic_path_value(cache_off),
                    "cross_cache_first_divergence_step": first_divergence(
                        &cache_on.generated_token_ids,
                        &cache_off.generated_token_ids,
                    ),
                    "cross_cache_exact_window_match": exact_window_match(
                        &cache_on.generated_token_ids,
                        &cache_off.generated_token_ids,
                    ),
                },
            }),
        );
    }
    let count = case.input_token_ids.len();
    Ok(json!({
        "prompt_id": case.prompt.prompt_id,
        "prompt_text_sha256": sha256_bytes(case.prompt.text.as_bytes()),
        "prompt_metadata": case.prompt.metadata(),
        "input_token_ids_sha256": token_ids_sha256(&case.input_token_ids),
        "input_first_token_id": case.input_token_ids[0],
        "input_token_count": count,
        "hidden_anchor_positions": {
            "first": 0,
            "middle": (count - 1) / 2,
            "last": count - 1,
        },
        "variants": variants,
    }))
}

fn tensor_ref(key: &str, shape: &[usize], dtype: TensorDType) -> Value {
    json!({
        "key": key,
        "shape": shape,
        "dtype": dtype.manifest_name(),
        "cache_path": "off",
    })
}

fn semantic_path_value(path: &SemanticPath) -> Value {
    json!({
        "generated_token_ids": path.generated_token_ids,
        "stop_reason": path.stop_reason,
    })
}

fn path_text(path: &Path) -> ProducerResult<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| NativeCalibrationError::new("evidence path is not valid UTF-8"))
}

fn bundle_executable(destination: &Path) -> ProducerResult<()> {
    let source = std::env::current_exe()
        .and_then(|path| path.canonicalize())
        .map_err(|error| NativeCalibrationError::context(error, "resolve current executable"))?;
    if source.file_name().and_then(|name| name.to_str()) != Some(NATIVE_EXECUTABLE_FILENAME) {
        return Err(NativeCalibrationError::new(
            "running executable basename is not rustinfer-native",
        ));
    }
    let mut input = File::open(&source)
        .map_err(|error| NativeCalibrationError::context(error, "open current executable"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| NativeCalibrationError::context(error, "create bundled executable"))?;
    let result = (|| {
        io::copy(&mut input, &mut output)
            .map_err(|error| NativeCalibrationError::context(error, "copy bundled executable"))?;
        #[cfg(unix)]
        {
            let permissions = fs::metadata(&source)
                .map_err(|error| NativeCalibrationError::context(error, "read executable mode"))?
                .permissions();
            fs::set_permissions(destination, permissions)
                .map_err(|error| NativeCalibrationError::context(error, "set executable mode"))?;
        }
        output
            .sync_all()
            .map_err(|error| NativeCalibrationError::context(error, "sync bundled executable"))?;
        Ok(())
    })();
    if let Err(error) = result {
        drop(output);
        let _ = fs::remove_file(destination);
        return Err(error);
    }
    Ok(())
}

fn write_json_exclusive(path: &Path, value: &Value) -> ProducerResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| NativeCalibrationError::context(error, "create candidate manifest"))?;
    let mut writer = BufWriter::new(file);
    let result = serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| NativeCalibrationError::context(error, "serialize candidate manifest"))
        .and_then(|()| {
            writer
                .write_all(b"\n")
                .map_err(|error| NativeCalibrationError::context(error, "terminate manifest"))
        })
        .and_then(|()| {
            writer
                .flush()
                .map_err(|error| NativeCalibrationError::context(error, "flush manifest"))
        })
        .and_then(|()| {
            writer
                .get_ref()
                .sync_all()
                .map_err(|error| NativeCalibrationError::context(error, "sync manifest"))
        });
    if let Err(error) = result {
        drop(writer);
        let _ = fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        NativeCalibrationError, capture_phase, cuda_version_text, finish_with_cleanup,
        require_cache_off_exact_token, semantic_stop_reason,
    };
    use crate::ReductionVariant;
    use rustinfer_runtime::generation::FinishReason;

    #[test]
    fn cuda_and_stop_metadata_are_canonical() {
        assert_eq!(cuda_version_text(13_000).expect("version"), "13.0");
        assert_eq!(
            semantic_stop_reason(Some(FinishReason::Eos)).expect("eos"),
            "eos"
        );
        assert_eq!(
            semantic_stop_reason(Some(FinishReason::Length)).expect("length"),
            "max_new_tokens"
        );
        assert!(semantic_stop_reason(Some(FinishReason::Cancelled)).is_err());
    }

    #[test]
    fn cache_off_exact_window_fails_fast_at_the_declared_boundary() {
        let variant = ReductionVariant::Canonical;
        let cache_on = (100_u32..116).collect::<Vec<_>>();
        require_cache_off_exact_token("prompt", variant, 0, &cache_on, 100)
            .expect("step zero matches");
        require_cache_off_exact_token("prompt", variant, 15, &cache_on, 115)
            .expect("last exact-window step matches");

        let mismatch = require_cache_off_exact_token("prompt", variant, 15, &cache_on, 999)
            .expect_err("step fifteen mismatch fails");
        let diagnostic = mismatch.to_string();
        assert!(diagnostic.contains("prompt_id=prompt"));
        assert!(diagnostic.contains("variant=canonical-v1"));
        assert!(diagnostic.contains("step=15"));
        assert!(diagnostic.contains("expected=Some(115) observed=999"));

        require_cache_off_exact_token("prompt", variant, 16, &cache_on, 999)
            .expect("the contract permits divergence after the exact window");
        let missing = require_cache_off_exact_token("prompt", variant, 1, &[100], 101)
            .expect_err("missing cache-on token inside the window fails");
        assert!(missing.to_string().contains("expected=None observed=101"));
    }

    #[test]
    fn phase_context_and_cleanup_preserve_the_primary_failure() {
        let phase = capture_phase(ReductionVariant::Canonical, "test-phase", || {
            Err(NativeCalibrationError::new("phase source"))
        })
        .expect_err("injected phase failure");
        assert_eq!(
            phase.to_string(),
            "capture phase=test-phase variant=canonical-v1: phase source"
        );

        let combined = finish_with_cleanup(
            Err(NativeCalibrationError::new("semantic mismatch")),
            [Err(NativeCalibrationError::new("close failure"))],
        )
        .expect_err("primary and cleanup both fail");
        assert_eq!(
            combined.to_string(),
            "semantic mismatch; cleanup also failed: close failure"
        );
    }
}
