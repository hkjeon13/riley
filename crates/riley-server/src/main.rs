#[cfg(any(feature = "cuda", test))]
use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(all(any(feature = "cuda", test), unix))]
use std::sync::Arc;

#[cfg(any(feature = "cuda", test))]
use sha2::Digest;

#[cfg(all(feature = "server", unix, any(feature = "cuda", test)))]
#[allow(unsafe_code)]
mod signal;

const DEFAULT_MAX_WEIGHT_BYTES: u64 = 2 * 1_024 * 1_024 * 1_024;
const MAX_BATCH_SHAPE_BUCKETS: usize = 10;
#[cfg(any(feature = "cuda", test))]
const FIXED37_MAX_SEQUENCE_TOKENS: usize = 8_192;
#[cfg(feature = "cuda")]
const _: () =
    assert!(FIXED37_MAX_SEQUENCE_TOKENS == riley_runtime::llama::LLAMA_FIXED37_MAX_SEQUENCE_TOKENS);
#[cfg(all(feature = "server", unix, any(feature = "cuda", test)))]
const SHUTDOWN_METRICS_ENV: &str = "RILEY_SHUTDOWN_METRICS_PATH";

const USAGE: &str = "\
usage:
  riley --version
  riley serve --model PATH [options]

serve options:
  --model-id ID                  public model ID (default: checkpoint source model)
  --bind ADDRESS                 listener address (default: 127.0.0.1:8080)
  --device ORDINAL               CUDA device ordinal (default: 0)
  --max-active-sequences N       active scheduler capacity (default: 8)
  --max-waiting-requests N       waiting scheduler capacity (default: 64)
  --max-sequence-tokens N        context bound (default: checkpoint maximum)
  --max-output-tokens N          API output bound (default: min(1024, context-1))
  --batch-token-budget N         tokens per CUDA iteration (default: 512)
  --prefill-chunk-tokens N       prompt tokens per request/iteration (default: 512)
  --kv-blocks N                  physical 16-token KV blocks (default: full active promise)
  --residual-rmsnorm MODE        fused E0 candidate or separate path (default: separate)
  --execution-completion MODE    per-operation or iteration-batch (default: iteration-batch)
  --batch-shape-policy MODE      fixed-max or power-of-two (default: fixed-max)
  --batch-shape-buckets LIST     custom power-of-two-policy shapes, ending at token budget
  --metadata-transport MODE      synchronous or packed-async (default: synchronous)
  --sampling-backend MODE        cpu or gpu-greedy (default: cpu)
  --reduction-profile ID         canonical-v1 or fixed-contiguous-37-balanced-v1 (default: canonical-v1)
  --max-weight-bytes N           checkpoint resident-byte bound (default: 2147483648)
  --c02-candidate-id ID          release candidate identity for C02 evidence mode
  --c02-configuration-profile ID stable-default or max-performance-exact
  --c02-startup-artifact PATH    absolute create-only C02 startup artifact path
  --c02-audit-dir PATH           absolute C02 generation-audit output directory
  --c02-shutdown-artifact PATH   direct-child C02 shutdown-v2 JSON artifact in audit dir
  --shutdown-on-stdin            gracefully stop after one input line or EOF
";

#[allow(clippy::large_enum_variant)] // Parsed once at process startup; avoid an extra heap allocation.
#[derive(Debug, Eq, PartialEq)]
enum CliCommand {
    Help,
    Version,
    Serve(ServeOptions),
}

#[derive(Debug, Eq, PartialEq)]
struct ServeOptions {
    model_path: PathBuf,
    model_id: Option<String>,
    bind_address: String,
    device_ordinal: u32,
    max_active_sequences: usize,
    max_waiting_requests: usize,
    max_sequence_tokens: Option<usize>,
    max_output_tokens: Option<usize>,
    batch_token_budget: usize,
    prefill_chunk_tokens: usize,
    physical_kv_blocks: Option<usize>,
    residual_rmsnorm: ResidualRmsNormMode,
    execution_completion: ExecutionCompletionMode,
    batch_shape_policy: BatchShapePolicyMode,
    batch_shape_buckets: Option<Vec<usize>>,
    metadata_transport: MetadataTransportMode,
    sampling_backend: SamplingBackendMode,
    reduction_profile: ReductionProfileMode,
    max_weight_bytes: u64,
    shutdown_on_stdin: bool,
    c02_runtime_config: Option<C02RuntimeConfigOptions>,
    c02_audit_dir: Option<PathBuf>,
    c02_shutdown_artifact: Option<C02ShutdownArtifactOptions>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidualRmsNormMode {
    Fused,
    Separate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutionCompletionMode {
    PerOperation,
    IterationBatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchShapePolicyMode {
    FixedMaximum,
    PowerOfTwo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplingBackendMode {
    Cpu,
    GpuGreedy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetadataTransportMode {
    Synchronous,
    PackedAsync,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReductionProfileMode {
    CanonicalV1,
    FixedContiguous37BalancedV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum C02ConfigurationProfile {
    StableDefault,
    MaxPerformanceExact,
}

impl C02ConfigurationProfile {
    #[cfg(any(feature = "cuda", test))]
    const fn as_str(self) -> &'static str {
        match self {
            Self::StableDefault => "stable-default",
            Self::MaxPerformanceExact => "max-performance-exact",
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct C02RuntimeConfigOptions {
    candidate_id: String,
    configuration_profile: C02ConfigurationProfile,
    startup_artifact: PathBuf,
}

/// Parsed C02 shutdown-v2 target. Only the validated direct-child basename is
/// retained: the writer never reopens the user-provided path during shutdown.
#[derive(Debug, Eq, PartialEq)]
struct C02ShutdownArtifactOptions {
    basename: String,
}

fn main() -> ExitCode {
    let launch_arguments: Vec<OsString> = env::args_os().skip(1).collect();
    match parse_arguments(launch_arguments.iter().cloned()) {
        Ok(CliCommand::Help) => {
            print_usage();
            ExitCode::SUCCESS
        }
        Ok(CliCommand::Version) => print_version(),
        Ok(CliCommand::Serve(options)) => {
            // The receipt hashes the exact validated serve launch identity.
            let launch_environment: Vec<(OsString, OsString)> = env::vars_os().collect();
            match run_serve(options, &launch_arguments, &launch_environment) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("riley: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("riley: {error}\n");
            print_usage_to_stderr();
            ExitCode::from(2)
        }
    }
}

fn print_usage() {
    print!("{USAGE}");
}

fn print_usage_to_stderr() {
    eprint!("{USAGE}");
}

fn print_version() -> ExitCode {
    match riley_server::version_line() {
        Ok(version) => {
            println!("{version}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("riley: {error}");
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)]
fn parse_arguments(arguments: impl IntoIterator<Item = OsString>) -> Result<CliCommand, String> {
    let mut arguments = arguments.into_iter();
    let Some(command) = arguments.next() else {
        return Err("a command is required".to_owned());
    };
    if command == "--help" || command == "-h" {
        ensure_finished(&mut arguments)?;
        return Ok(CliCommand::Help);
    }
    if command == "--version" || command == "-V" {
        ensure_finished(&mut arguments)?;
        return Ok(CliCommand::Version);
    }
    if command != "serve" {
        return Err(format!("unknown command {}", display_argument(&command)));
    }

    let mut model_path = None;
    let mut model_id = None;
    let mut bind_address = None;
    let mut device_ordinal = None;
    let mut max_active_sequences = None;
    let mut max_waiting_requests = None;
    let mut max_sequence_tokens = None;
    let mut max_output_tokens = None;
    let mut batch_token_budget = None;
    let mut prefill_chunk_tokens = None;
    let mut physical_kv_blocks = None;
    let mut residual_rmsnorm = None;
    let mut execution_completion = None;
    let mut batch_shape_policy = None;
    let mut batch_shape_buckets = None;
    let mut metadata_transport = None;
    let mut sampling_backend = None;
    let mut reduction_profile = None;
    let mut max_weight_bytes = None;
    let mut shutdown_on_stdin = false;
    let mut c02_candidate_id = None;
    let mut c02_configuration_profile = None;
    let mut c02_startup_artifact = None;
    let mut c02_audit_dir = None;
    let mut c02_shutdown_artifact = None;

    while let Some(flag) = arguments.next() {
        let Some(flag_text) = flag.to_str() else {
            return Err("serve option names must be UTF-8".to_owned());
        };
        match flag_text {
            "--help" | "-h" => {
                ensure_finished(&mut arguments)?;
                return Ok(CliCommand::Help);
            }
            "--model" => set_once(
                &mut model_path,
                PathBuf::from(next_value(&mut arguments, "--model")?),
                "--model",
            )?,
            "--model-id" => set_once(
                &mut model_id,
                parse_utf8(next_value(&mut arguments, "--model-id")?, "--model-id")?,
                "--model-id",
            )?,
            "--bind" => set_once(
                &mut bind_address,
                parse_utf8(next_value(&mut arguments, "--bind")?, "--bind")?,
                "--bind",
            )?,
            "--device" => set_once(
                &mut device_ordinal,
                parse_number(next_value(&mut arguments, "--device")?, "--device")?,
                "--device",
            )?,
            "--max-active-sequences" => set_once(
                &mut max_active_sequences,
                parse_number(
                    next_value(&mut arguments, "--max-active-sequences")?,
                    "--max-active-sequences",
                )?,
                "--max-active-sequences",
            )?,
            "--max-waiting-requests" => set_once(
                &mut max_waiting_requests,
                parse_number(
                    next_value(&mut arguments, "--max-waiting-requests")?,
                    "--max-waiting-requests",
                )?,
                "--max-waiting-requests",
            )?,
            "--max-sequence-tokens" => set_once(
                &mut max_sequence_tokens,
                parse_number(
                    next_value(&mut arguments, "--max-sequence-tokens")?,
                    "--max-sequence-tokens",
                )?,
                "--max-sequence-tokens",
            )?,
            "--max-output-tokens" => set_once(
                &mut max_output_tokens,
                parse_number(
                    next_value(&mut arguments, "--max-output-tokens")?,
                    "--max-output-tokens",
                )?,
                "--max-output-tokens",
            )?,
            "--batch-token-budget" => set_once(
                &mut batch_token_budget,
                parse_number(
                    next_value(&mut arguments, "--batch-token-budget")?,
                    "--batch-token-budget",
                )?,
                "--batch-token-budget",
            )?,
            "--prefill-chunk-tokens" => set_once(
                &mut prefill_chunk_tokens,
                parse_number(
                    next_value(&mut arguments, "--prefill-chunk-tokens")?,
                    "--prefill-chunk-tokens",
                )?,
                "--prefill-chunk-tokens",
            )?,
            "--kv-blocks" => set_once(
                &mut physical_kv_blocks,
                parse_number(next_value(&mut arguments, "--kv-blocks")?, "--kv-blocks")?,
                "--kv-blocks",
            )?,
            "--residual-rmsnorm" => set_once(
                &mut residual_rmsnorm,
                parse_residual_rmsnorm(next_value(&mut arguments, "--residual-rmsnorm")?)?,
                "--residual-rmsnorm",
            )?,
            "--execution-completion" => set_once(
                &mut execution_completion,
                parse_execution_completion(next_value(&mut arguments, "--execution-completion")?)?,
                "--execution-completion",
            )?,
            "--batch-shape-policy" => set_once(
                &mut batch_shape_policy,
                parse_batch_shape_policy(next_value(&mut arguments, "--batch-shape-policy")?)?,
                "--batch-shape-policy",
            )?,
            "--batch-shape-buckets" => set_once(
                &mut batch_shape_buckets,
                parse_batch_shape_buckets(next_value(&mut arguments, "--batch-shape-buckets")?)?,
                "--batch-shape-buckets",
            )?,
            "--metadata-transport" => set_once(
                &mut metadata_transport,
                parse_metadata_transport(next_value(&mut arguments, "--metadata-transport")?)?,
                "--metadata-transport",
            )?,
            "--sampling-backend" => set_once(
                &mut sampling_backend,
                parse_sampling_backend(next_value(&mut arguments, "--sampling-backend")?)?,
                "--sampling-backend",
            )?,
            "--reduction-profile" => set_once(
                &mut reduction_profile,
                parse_reduction_profile(next_value(&mut arguments, "--reduction-profile")?)?,
                "--reduction-profile",
            )?,
            "--max-weight-bytes" => set_once(
                &mut max_weight_bytes,
                parse_number(
                    next_value(&mut arguments, "--max-weight-bytes")?,
                    "--max-weight-bytes",
                )?,
                "--max-weight-bytes",
            )?,
            "--c02-candidate-id" => set_once(
                &mut c02_candidate_id,
                parse_utf8(
                    next_value(&mut arguments, "--c02-candidate-id")?,
                    "--c02-candidate-id",
                )?,
                "--c02-candidate-id",
            )?,
            "--c02-configuration-profile" => set_once(
                &mut c02_configuration_profile,
                parse_c02_configuration_profile(next_value(
                    &mut arguments,
                    "--c02-configuration-profile",
                )?)?,
                "--c02-configuration-profile",
            )?,
            "--c02-startup-artifact" => set_once(
                &mut c02_startup_artifact,
                PathBuf::from(parse_utf8(
                    next_value(&mut arguments, "--c02-startup-artifact")?,
                    "--c02-startup-artifact",
                )?),
                "--c02-startup-artifact",
            )?,
            "--c02-audit-dir" => set_once(
                &mut c02_audit_dir,
                PathBuf::from(parse_utf8(
                    next_value(&mut arguments, "--c02-audit-dir")?,
                    "--c02-audit-dir",
                )?),
                "--c02-audit-dir",
            )?,
            "--c02-shutdown-artifact" => set_once(
                &mut c02_shutdown_artifact,
                PathBuf::from(parse_utf8(
                    next_value(&mut arguments, "--c02-shutdown-artifact")?,
                    "--c02-shutdown-artifact",
                )?),
                "--c02-shutdown-artifact",
            )?,
            "--shutdown-on-stdin" => {
                if shutdown_on_stdin {
                    return Err("--shutdown-on-stdin may occur only once".to_owned());
                }
                shutdown_on_stdin = true;
            }
            _ => return Err(format!("unknown serve option {flag_text}")),
        }
    }

    let residual_rmsnorm = residual_rmsnorm.unwrap_or(ResidualRmsNormMode::Separate);
    let execution_completion =
        execution_completion.unwrap_or(ExecutionCompletionMode::IterationBatch);
    let batch_shape_policy = batch_shape_policy.unwrap_or(BatchShapePolicyMode::FixedMaximum);
    let batch_token_budget = batch_token_budget.unwrap_or(512);
    let metadata_transport = metadata_transport.unwrap_or(MetadataTransportMode::Synchronous);
    if residual_rmsnorm == ResidualRmsNormMode::Fused
        && execution_completion == ExecutionCompletionMode::IterationBatch
    {
        return Err(
            "fused residual RMSNorm may only be used with per-operation completion".to_owned(),
        );
    }
    if metadata_transport == MetadataTransportMode::PackedAsync
        && execution_completion != ExecutionCompletionMode::IterationBatch
    {
        return Err(
            "packed-async metadata transport requires iteration-batch completion".to_owned(),
        );
    }
    if batch_shape_buckets.is_some() && batch_shape_policy != BatchShapePolicyMode::PowerOfTwo {
        return Err("--batch-shape-buckets requires --batch-shape-policy power-of-two".to_owned());
    }
    if let Some(buckets) = batch_shape_buckets.as_deref() {
        validate_batch_shape_buckets(buckets, batch_token_budget)?;
    }
    let c02_runtime_config = parse_c02_runtime_config_options(
        c02_candidate_id,
        c02_configuration_profile,
        c02_startup_artifact,
    )?;
    let c02_audit_dir = parse_c02_audit_dir(c02_runtime_config.as_ref(), c02_audit_dir)?;
    let c02_shutdown_artifact = parse_c02_shutdown_artifact(
        c02_runtime_config.as_ref(),
        c02_audit_dir.as_deref(),
        c02_shutdown_artifact,
    )?;
    let bind_address = bind_address.unwrap_or_else(|| "127.0.0.1:8080".to_owned());

    Ok(CliCommand::Serve(ServeOptions {
        model_path: model_path.ok_or_else(|| "serve requires --model PATH".to_owned())?,
        model_id,
        bind_address,
        device_ordinal: device_ordinal.unwrap_or(0),
        max_active_sequences: max_active_sequences.unwrap_or(8),
        max_waiting_requests: max_waiting_requests.unwrap_or(64),
        max_sequence_tokens,
        max_output_tokens,
        batch_token_budget,
        prefill_chunk_tokens: prefill_chunk_tokens.unwrap_or(512),
        physical_kv_blocks,
        residual_rmsnorm,
        execution_completion,
        batch_shape_policy,
        batch_shape_buckets,
        metadata_transport,
        sampling_backend: sampling_backend.unwrap_or(SamplingBackendMode::Cpu),
        reduction_profile: reduction_profile.unwrap_or(ReductionProfileMode::CanonicalV1),
        max_weight_bytes: max_weight_bytes.unwrap_or(DEFAULT_MAX_WEIGHT_BYTES),
        shutdown_on_stdin,
        c02_runtime_config,
        c02_audit_dir,
        c02_shutdown_artifact,
    }))
}

fn parse_residual_rmsnorm(value: OsString) -> Result<ResidualRmsNormMode, String> {
    match parse_utf8(value, "--residual-rmsnorm")?.as_str() {
        "fused" => Ok(ResidualRmsNormMode::Fused),
        "separate" => Ok(ResidualRmsNormMode::Separate),
        _ => Err("--residual-rmsnorm requires fused or separate".to_owned()),
    }
}

fn parse_execution_completion(value: OsString) -> Result<ExecutionCompletionMode, String> {
    match parse_utf8(value, "--execution-completion")?.as_str() {
        "per-operation" => Ok(ExecutionCompletionMode::PerOperation),
        "iteration-batch" => Ok(ExecutionCompletionMode::IterationBatch),
        _ => Err("--execution-completion requires per-operation or iteration-batch".to_owned()),
    }
}

fn parse_batch_shape_policy(value: OsString) -> Result<BatchShapePolicyMode, String> {
    match parse_utf8(value, "--batch-shape-policy")?.as_str() {
        "fixed-max" => Ok(BatchShapePolicyMode::FixedMaximum),
        "power-of-two" => Ok(BatchShapePolicyMode::PowerOfTwo),
        _ => Err("--batch-shape-policy requires fixed-max or power-of-two".to_owned()),
    }
}

fn parse_batch_shape_buckets(value: OsString) -> Result<Vec<usize>, String> {
    let value = parse_utf8(value, "--batch-shape-buckets")?;
    if value.is_empty() {
        return Err("--batch-shape-buckets requires a non-empty comma-separated list".to_owned());
    }
    let mut buckets = Vec::new();
    for entry in value.split(',') {
        if buckets.len() == MAX_BATCH_SHAPE_BUCKETS {
            return Err(format!(
                "--batch-shape-buckets supports at most {MAX_BATCH_SHAPE_BUCKETS} entries"
            ));
        }
        let bucket = entry.parse::<usize>().map_err(|_| {
            "--batch-shape-buckets requires comma-separated positive decimal integers".to_owned()
        })?;
        buckets.push(bucket);
    }
    Ok(buckets)
}

fn validate_batch_shape_buckets(buckets: &[usize], maximum_rows: usize) -> Result<(), String> {
    if buckets.first().copied() != Some(1) {
        return Err("--batch-shape-buckets must start with 1".to_owned());
    }
    if buckets.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("--batch-shape-buckets must be strictly increasing".to_owned());
    }
    if buckets.last().copied() != Some(maximum_rows) {
        return Err(
            "--batch-shape-buckets must end at the effective --batch-token-budget".to_owned(),
        );
    }
    Ok(())
}

fn parse_metadata_transport(value: OsString) -> Result<MetadataTransportMode, String> {
    match parse_utf8(value, "--metadata-transport")?.as_str() {
        "synchronous" => Ok(MetadataTransportMode::Synchronous),
        "packed-async" => Ok(MetadataTransportMode::PackedAsync),
        _ => Err("--metadata-transport requires synchronous or packed-async".to_owned()),
    }
}

fn parse_sampling_backend(value: OsString) -> Result<SamplingBackendMode, String> {
    match parse_utf8(value, "--sampling-backend")?.as_str() {
        "cpu" => Ok(SamplingBackendMode::Cpu),
        "gpu-greedy" => Ok(SamplingBackendMode::GpuGreedy),
        _ => Err("--sampling-backend requires cpu or gpu-greedy".to_owned()),
    }
}

fn parse_reduction_profile(value: OsString) -> Result<ReductionProfileMode, String> {
    match parse_utf8(value, "--reduction-profile")?.as_str() {
        "canonical-v1" => Ok(ReductionProfileMode::CanonicalV1),
        "fixed-contiguous-37-balanced-v1" => Ok(ReductionProfileMode::FixedContiguous37BalancedV1),
        _ => Err(
            "--reduction-profile requires canonical-v1 or fixed-contiguous-37-balanced-v1"
                .to_owned(),
        ),
    }
}

fn parse_c02_configuration_profile(value: OsString) -> Result<C02ConfigurationProfile, String> {
    match parse_utf8(value, "--c02-configuration-profile")?.as_str() {
        "stable-default" => Ok(C02ConfigurationProfile::StableDefault),
        "max-performance-exact" => Ok(C02ConfigurationProfile::MaxPerformanceExact),
        _ => Err(
            "--c02-configuration-profile requires stable-default or max-performance-exact"
                .to_owned(),
        ),
    }
}

fn parse_c02_runtime_config_options(
    candidate_id: Option<String>,
    configuration_profile: Option<C02ConfigurationProfile>,
    startup_artifact: Option<PathBuf>,
) -> Result<Option<C02RuntimeConfigOptions>, String> {
    if candidate_id.is_none() && configuration_profile.is_none() && startup_artifact.is_none() {
        return Ok(None);
    }
    let (Some(candidate_id), Some(configuration_profile), Some(startup_artifact)) =
        (candidate_id, configuration_profile, startup_artifact)
    else {
        return Err("--c02-candidate-id, --c02-configuration-profile, and \
--c02-startup-artifact must be supplied together"
            .to_owned());
    };
    validate_c02_candidate_id(&candidate_id)?;
    if startup_artifact.as_os_str().is_empty() || !startup_artifact.is_absolute() {
        return Err("--c02-startup-artifact must be an absolute non-empty path".to_owned());
    }
    Ok(Some(C02RuntimeConfigOptions {
        candidate_id,
        configuration_profile,
        startup_artifact,
    }))
}

fn parse_c02_audit_dir(
    runtime_config: Option<&C02RuntimeConfigOptions>,
    audit_dir: Option<PathBuf>,
) -> Result<Option<PathBuf>, String> {
    let Some(audit_dir) = audit_dir else {
        return Ok(None);
    };
    if runtime_config.is_none() {
        return Err("--c02-audit-dir requires complete C02 runtime configuration".to_owned());
    }
    if audit_dir.as_os_str().is_empty() || !audit_dir.is_absolute() {
        return Err("--c02-audit-dir must be an absolute non-empty path".to_owned());
    }
    Ok(Some(audit_dir))
}

fn parse_c02_shutdown_artifact(
    runtime_config: Option<&C02RuntimeConfigOptions>,
    audit_dir: Option<&std::path::Path>,
    artifact: Option<PathBuf>,
) -> Result<Option<C02ShutdownArtifactOptions>, String> {
    let Some(artifact) = artifact else {
        return Ok(None);
    };
    if runtime_config.is_none() {
        return Err(
            "--c02-shutdown-artifact requires complete C02 runtime configuration".to_owned(),
        );
    }
    let audit_dir = audit_dir.ok_or_else(|| {
        "--c02-shutdown-artifact requires --c02-audit-dir as its held evidence root".to_owned()
    })?;
    if artifact.as_os_str().is_empty() || !artifact.is_absolute() {
        return Err("--c02-shutdown-artifact must be an absolute non-empty path".to_owned());
    }
    let relative = artifact.strip_prefix(audit_dir).map_err(|_| {
        "--c02-shutdown-artifact must be a direct child of --c02-audit-dir".to_owned()
    })?;
    let mut components = relative.components();
    let Some(std::path::Component::Normal(component)) = components.next() else {
        return Err("--c02-shutdown-artifact must name one direct-child file".to_owned());
    };
    if components.next().is_some() {
        return Err("--c02-shutdown-artifact must be a direct child of --c02-audit-dir".to_owned());
    }
    let basename = component.to_str().ok_or_else(|| {
        "--c02-shutdown-artifact requires an ASCII-safe UTF-8 basename".to_owned()
    })?;
    if !c02_shutdown_artifact_basename_is_valid(basename) {
        return Err(
            "--c02-shutdown-artifact must be a nonhidden ASCII-safe .json basename of at most 246 bytes"
                .to_owned(),
        );
    }
    Ok(Some(C02ShutdownArtifactOptions {
        basename: basename.to_owned(),
    }))
}

fn c02_shutdown_artifact_basename_is_valid(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() > b".json".len()
        && bytes.len() <= 246
        // The source contract uses the exact lowercase schema extension,
        // rather than accepting a platform-dependent case variant.
        && bytes.ends_with(b".json")
        && bytes.iter().copied().enumerate().all(|(index, byte)| {
            (index == 0 && byte.is_ascii_alphanumeric())
                || (index != 0
                    && (byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_')))
        })
}

fn validate_c02_candidate_id(candidate_id: &str) -> Result<(), String> {
    let Some(version_and_rc) = candidate_id.strip_prefix("riley-") else {
        return Err(c02_candidate_id_error());
    };
    let Some((version, rc)) = version_and_rc.rsplit_once("-rc") else {
        return Err(c02_candidate_id_error());
    };
    let mut parts = version.split('.');
    let (Some(major), Some(minor), Some(patch)) = (parts.next(), parts.next(), parts.next()) else {
        return Err(c02_candidate_id_error());
    };
    if parts.next().is_some()
        || !is_canonical_non_negative_decimal(major)
        || !is_canonical_non_negative_decimal(minor)
        || !is_canonical_non_negative_decimal(patch)
        || !is_canonical_positive_decimal(rc)
    {
        return Err(c02_candidate_id_error());
    }
    Ok(())
}

fn is_canonical_non_negative_decimal(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && (value == "0" || !value.starts_with('0'))
}

fn is_canonical_positive_decimal(value: &str) -> bool {
    is_canonical_non_negative_decimal(value) && value != "0"
}

fn c02_candidate_id_error() -> String {
    "--c02-candidate-id must match riley-X.Y.Z-rcN with canonical decimal components".to_owned()
}

#[cfg(any(feature = "cuda", test))]
fn validate_reduction_profile_context(
    profile: ReductionProfileMode,
    maximum_sequence_tokens: usize,
) -> Result<(), String> {
    if profile == ReductionProfileMode::FixedContiguous37BalancedV1
        && maximum_sequence_tokens > FIXED37_MAX_SEQUENCE_TOKENS
    {
        return Err(format!(
            "--reduction-profile fixed-contiguous-37-balanced-v1 requires an effective \
--max-sequence-tokens no greater than {FIXED37_MAX_SEQUENCE_TOKENS}"
        ));
    }
    Ok(())
}

fn ensure_finished(arguments: &mut impl Iterator<Item = OsString>) -> Result<(), String> {
    match arguments.next() {
        None => Ok(()),
        Some(extra) => Err(format!("unexpected argument {}", display_argument(&extra))),
    }
}

fn next_value(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> Result<OsString, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_utf8(value: OsString, option: &str) -> Result<String, String> {
    value
        .into_string()
        .map_err(|_| format!("{option} requires a UTF-8 value"))
}

fn parse_number<T>(value: OsString, option: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    let value = parse_utf8(value, option)?;
    value
        .parse::<T>()
        .map_err(|_| format!("{option} requires a non-negative decimal integer"))
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        Err(format!("{option} may occur only once"))
    } else {
        Ok(())
    }
}

fn display_argument(argument: &OsStr) -> String {
    argument.to_string_lossy().into_owned()
}

#[cfg(not(feature = "cuda"))]
fn run_serve(
    options: ServeOptions,
    _launch_arguments: &[OsString],
    _launch_environment: &[(OsString, OsString)],
) -> Result<(), String> {
    let _ = (
        options.model_path,
        options.model_id,
        options.bind_address,
        options.device_ordinal,
        options.max_active_sequences,
        options.max_waiting_requests,
        options.max_sequence_tokens,
        options.max_output_tokens,
        options.batch_token_budget,
        options.prefill_chunk_tokens,
        options.physical_kv_blocks,
        options.residual_rmsnorm,
        options.execution_completion,
        options.batch_shape_policy,
        options.batch_shape_buckets,
        options.metadata_transport,
        options.sampling_backend,
        options.reduction_profile,
        options.max_weight_bytes,
        options.shutdown_on_stdin,
        options.c02_runtime_config,
        options.c02_audit_dir,
        options.c02_shutdown_artifact,
    );
    Err("serve requires a build with --features server,cuda".to_owned())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_lines)]
fn run_serve(
    options: ServeOptions,
    launch_arguments: &[OsString],
    launch_environment: &[(OsString, OsString)],
) -> Result<(), String> {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use riley_model::{LoadLimits, LoadedModel};
    use riley_runtime::llama::{
        LlamaBatchMetadataConfig, LlamaReductionProfile, PreparedLlamaBatchExecutorConfig,
        PreparedLlamaForwardConfig,
    };
    use riley_runtime::paged_kv::KV_BLOCK_SIZE;
    use riley_scheduler::{OverloadPolicy, SchedulerConfig};
    use riley_server::domain::{ModelMetadata, RequestLimits};
    use riley_server::engine::{
        CudaBackendConfig, CudaEngineResources, EngineConfig, InferenceEngine,
    };
    use riley_server::service::{CompletionBackend, ServerConfig, start_server_with_c02_metrics};

    #[cfg(not(unix))]
    return Err("serve currently requires POSIX SIGINT/SIGTERM support".to_owned());
    #[cfg(unix)]
    let shutdown_signals = Arc::new(
        signal::ShutdownSignals::block()
            .map_err(|error| format!("could not block shutdown signals: {error}"))?,
    );
    let shutdown_metrics_path = shutdown_metrics_path_from_env()?;
    let c02_runtime_config = options.c02_runtime_config;
    let c02_audit_dir = options.c02_audit_dir;
    let c02_shutdown_artifact = options.c02_shutdown_artifact;
    if shutdown_metrics_path.is_some() && c02_shutdown_artifact.is_some() {
        return Err(format!(
            "{SHUTDOWN_METRICS_ENV} and --c02-shutdown-artifact are mutually exclusive"
        ));
    }

    validate_positive("--max-active-sequences", options.max_active_sequences)?;
    validate_positive("--max-waiting-requests", options.max_waiting_requests)?;
    validate_positive("--batch-token-budget", options.batch_token_budget)?;
    validate_positive("--prefill-chunk-tokens", options.prefill_chunk_tokens)?;
    if options.prefill_chunk_tokens > options.batch_token_budget {
        return Err("--prefill-chunk-tokens must not exceed --batch-token-budget".to_owned());
    }
    if options.batch_token_budget < options.max_active_sequences {
        return Err(
            "--batch-token-budget must permit at least one token per active sequence".to_owned(),
        );
    }
    if options.max_weight_bytes == 0 {
        return Err("--max-weight-bytes must be greater than zero".to_owned());
    }

    let bind_address = options
        .bind_address
        .parse::<SocketAddr>()
        .map_err(|_| "--bind must be an IP socket address such as 127.0.0.1:8080".to_owned())?;
    if c02_runtime_config.is_some() && bind_address.ip() != IpAddr::V4(Ipv4Addr::LOCALHOST) {
        return Err("C02 evidence mode requires --bind 127.0.0.1:PORT".to_owned());
    }
    // Open the private root and retain its descriptor chain before model/CUDA
    // setup. Generation audit and shutdown-v2 writers clone this one held
    // root, so later pathname rebinding cannot redirect either evidence leaf.
    let c02_audit_root = c02_audit_dir
        .as_deref()
        .map(c02_open_audit_root)
        .transpose()?;
    let c02_process_identity = c02_audit_root
        .as_ref()
        .map(|_| c02_capture_process_identity())
        .transpose()?;
    let load_limits = LoadLimits::default()
        .with_weight_byte_limits(options.max_weight_bytes, options.max_weight_bytes)
        .map_err(|error| format!("invalid model load limit: {error}"))?;
    let model = LoadedModel::load(&options.model_path, load_limits)
        .map_err(|error| format!("model load failed: {error}"))?;
    let model_context = model.spec().max_sequence_length();
    let max_sequence_tokens = options.max_sequence_tokens.unwrap_or(model_context);
    if max_sequence_tokens < 2 || max_sequence_tokens > model_context {
        return Err(format!(
            "--max-sequence-tokens must be between 2 and the model bound {model_context}"
        ));
    }
    validate_reduction_profile_context(options.reduction_profile, max_sequence_tokens)?;
    let default_output_tokens = 1_024_usize.min(max_sequence_tokens - 1);
    let max_output_tokens = options.max_output_tokens.unwrap_or(default_output_tokens);
    if max_output_tokens == 0 || max_output_tokens >= max_sequence_tokens {
        return Err("--max-output-tokens must be positive and smaller than the context".to_owned());
    }

    let blocks_per_sequence = max_sequence_tokens.div_ceil(KV_BLOCK_SIZE);
    let full_active_promise = options
        .max_active_sequences
        .checked_mul(blocks_per_sequence)
        .ok_or_else(|| "KV block promise overflowed".to_owned())?;
    let physical_kv_blocks = options.physical_kv_blocks.unwrap_or(full_active_promise);
    validate_positive("--kv-blocks", physical_kv_blocks)?;
    let maximum_live_requests = options
        .max_active_sequences
        .checked_add(options.max_waiting_requests)
        .ok_or_else(|| "request capacity overflowed".to_owned())?;
    let maximum_waiting_prompt_tokens = options
        .max_waiting_requests
        .checked_mul(max_sequence_tokens)
        .ok_or_else(|| "waiting prompt-token capacity overflowed".to_owned())?;

    let scheduler = SchedulerConfig {
        max_waiting_requests: options.max_waiting_requests,
        max_waiting_prompt_tokens: maximum_waiting_prompt_tokens,
        max_active_sequences: options.max_active_sequences,
        max_sequence_tokens,
        iteration_token_budget: options.batch_token_budget,
        max_prefill_chunk_tokens: options.prefill_chunk_tokens,
        aging_threshold_ns: 100_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: Some(30_000_000_000),
        max_promised_kv_blocks: physical_kv_blocks,
        metrics_window_samples: 1_024,
    };
    let batch_metadata = LlamaBatchMetadataConfig::new(
        options.max_active_sequences,
        options.batch_token_budget,
        physical_kv_blocks,
        options.max_active_sequences,
        physical_kv_blocks,
    )
    .map_err(|error| format!("invalid batch configuration: {error}"))?;
    let executor = PreparedLlamaBatchExecutorConfig::new(
        batch_metadata,
        PreparedLlamaForwardConfig::default(),
    );
    let executor = match options.residual_rmsnorm {
        ResidualRmsNormMode::Fused => executor.with_fused_residual_norm(),
        ResidualRmsNormMode::Separate => executor.with_separate_residual_norm(),
    };
    let executor = match options.execution_completion {
        ExecutionCompletionMode::PerOperation => executor.with_per_operation_completion(),
        ExecutionCompletionMode::IterationBatch => executor.with_iteration_batch_completion(),
    };
    let executor = match options.metadata_transport {
        MetadataTransportMode::Synchronous => executor.with_synchronous_metadata(),
        MetadataTransportMode::PackedAsync => executor.with_packed_async_metadata(),
    };
    let executor = match options.batch_shape_policy {
        BatchShapePolicyMode::FixedMaximum => executor.with_fixed_maximum_shape(),
        BatchShapePolicyMode::PowerOfTwo => match options.batch_shape_buckets.as_deref() {
            Some(buckets) => executor
                .with_custom_active_row_buckets(buckets)
                .map_err(|error| format!("invalid batch shape buckets: {error}"))?,
            None => executor.with_active_row_buckets(),
        },
    };
    let executor = match options.reduction_profile {
        ReductionProfileMode::CanonicalV1 => {
            executor.with_reduction_profile(LlamaReductionProfile::CanonicalV1)
        }
        ReductionProfileMode::FixedContiguous37BalancedV1 => {
            executor.with_reduction_profile(LlamaReductionProfile::FixedContiguous37BalancedV1)
        }
    };
    let model_id = options
        .model_id
        .unwrap_or_else(|| model.provenance().source_model().to_owned());
    let created_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let metadata = ModelMetadata {
        model_id,
        created_unix_seconds,
        owned_by: "riley".to_owned(),
        context_window_tokens: max_sequence_tokens,
        max_output_tokens,
    };
    let resources = CudaEngineResources::prepare(
        metadata,
        model,
        CudaBackendConfig {
            device_ordinal: options.device_ordinal,
            scheduler,
            executor,
            gpu_greedy: options.sampling_backend == SamplingBackendMode::GpuGreedy,
        },
    )
    .map_err(|error| format!("CUDA backend preparation failed: {error}"))?;
    let (c02_receipt, c02_generation_audit) = match c02_runtime_config.as_ref() {
        Some(c02) => {
            let facts = resources.effective_runtime_facts();
            let effective_config = c02_effective_config_from_facts(&facts)?;
            let runtime_identity = c02_runtime_identity(
                c02.configuration_profile,
                launch_arguments,
                launch_environment,
            )?;
            let receipt = c02_endpoint_receipt(c02, &runtime_identity, &effective_config)?;
            let generation_audit = match (c02_audit_root.as_ref(), c02_process_identity) {
                (Some(directory), Some(process_identity)) => {
                    Some(Arc::new(C02GenerationAuditWriter::new(
                        Arc::clone(directory),
                        c02,
                        &runtime_identity,
                        process_identity,
                    )?))
                }
                (Some(_), None) => {
                    return Err("C02 audit root has no captured process identity".to_owned());
                }
                (None, _) => None,
            };
            (Some(receipt), generation_audit)
        }
        None => (None, None),
    };
    let resources = match c02_generation_audit {
        Some(generation_audit) => resources.with_c02_generation_audit(generation_audit),
        None => resources,
    };
    let engine = Arc::new(
        InferenceEngine::start_cuda(
            resources,
            EngineConfig {
                command_queue_capacity: options.max_waiting_requests,
                event_channel_capacity: 32,
                max_inflight_requests: maximum_live_requests,
                admission_timeout: Duration::from_secs(30),
                idle_poll_interval: Duration::from_millis(1),
            },
        )
        .map_err(|error| format!("inference engine startup failed: {error}"))?,
    );
    let runtime_config_body = match (c02_runtime_config.as_ref(), c02_receipt) {
        (Some(c02), Some(receipt)) => {
            write_c02_startup_artifact(&c02.startup_artifact, c02, &receipt)?;
            Some(Arc::<[u8]>::from(receipt.bytes))
        }
        (None, None) => None,
        _ => return Err("internal C02 runtime configuration state mismatch".to_owned()),
    };
    let c02_shutdown_writer = match (
        c02_shutdown_artifact,
        c02_audit_root.as_ref(),
        c02_process_identity,
    ) {
        (Some(artifact), Some(directory), Some(process_identity)) => Some(
            C02ShutdownArtifactWriter::new(Arc::clone(directory), artifact, process_identity)?,
        ),
        (Some(_), _, _) => {
            return Err(
                "C02 shutdown artifact has no held audit root or process identity".to_owned(),
            );
        }
        (None, _, _) => None,
    };

    let request_limits = RequestLimits {
        max_output_tokens,
        ..RequestLimits::default()
    };
    let server_config = ServerConfig {
        bind_address,
        request_limits,
        ..ServerConfig::default()
    };
    let backend: Arc<dyn CompletionBackend> = engine;
    let server = start_server_with_c02_metrics(
        server_config,
        backend,
        runtime_config_body,
        c02_audit_root.is_some(),
    )
    .map_err(|error| format!("HTTP server startup failed: {error}"))?;
    println!(
        "riley listening on http://{} (graceful_signals=SIGINT,SIGTERM graceful_stdin_shutdown={})",
        server.local_address(),
        options.shutdown_on_stdin
    );

    let trigger = wait_for_shutdown(Arc::clone(&shutdown_signals), options.shutdown_on_stdin)?;
    println!("riley graceful shutdown requested by {trigger}");
    match (shutdown_metrics_path, c02_shutdown_writer) {
        (Some(path), None) => {
            let snapshot = server
                .shutdown_with_metrics()
                .map_err(|error| format!("graceful shutdown failed: {error}"))?;
            write_shutdown_metrics(&path, &snapshot)?;
            println!(
                "riley wrote verified shutdown metrics to {}",
                path.display()
            );
            Ok(())
        }
        (None, Some(writer)) => {
            let evidence = server
                .shutdown_with_c02_evidence()
                .map_err(|error| format!("graceful C02 shutdown failed: {error}"))?;
            writer.write(evidence.final_metrics)?;
            println!(
                "riley wrote verified C02 shutdown evidence to {}",
                writer.artifact_basename
            );
            Ok(())
        }
        (None, None) => server
            .shutdown()
            .map_err(|error| format!("graceful shutdown failed: {error}")),
        (Some(_), Some(_)) => Err("internal shutdown artifact configuration conflict".to_owned()),
    }
}

#[cfg(all(feature = "server", feature = "cuda", unix))]
fn shutdown_metrics_path_from_env() -> Result<Option<PathBuf>, String> {
    validate_shutdown_metrics_path(env::var_os(SHUTDOWN_METRICS_ENV))
}

#[cfg(all(feature = "server", unix, any(feature = "cuda", test)))]
fn validate_shutdown_metrics_path(value: Option<OsString>) -> Result<Option<PathBuf>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let path = PathBuf::from(value);
    if path.as_os_str().is_empty() {
        return Err(format!("{SHUTDOWN_METRICS_ENV} must not be empty"));
    }
    if !path.is_absolute() {
        return Err(format!("{SHUTDOWN_METRICS_ENV} must be an absolute path"));
    }
    Ok(Some(path))
}

#[cfg(all(feature = "server", unix, any(feature = "cuda", test)))]
fn write_shutdown_metrics(
    path: &std::path::Path,
    snapshot: &riley_server::service::OperationalMetricsSnapshot,
) -> Result<(), String> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut document = serde_json::to_vec(&snapshot)
        .map_err(|error| format!("could not serialize shutdown metrics: {error}"))?;
    document.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "could not create shutdown metrics {} without replacement: {error}",
                path.display()
            )
        })?;
    file.write_all(&document).map_err(|error| {
        format!(
            "could not write shutdown metrics {}: {error}",
            path.display()
        )
    })?;
    file.sync_all().map_err(|error| {
        format!(
            "could not sync shutdown metrics {}: {error}",
            path.display()
        )
    })
}

#[cfg(any(feature = "cuda", test))]
const C02_RUNTIME_CONFIG_SCHEMA_VERSION: &str = "riley.effective-runtime-config.v1";
#[cfg(any(feature = "cuda", test))]
const C02_STARTUP_ARTIFACT_SCHEMA_VERSION: &str =
    "riley.effective-runtime-config-startup-artifact.v1";
#[cfg(any(feature = "cuda", test))]
const C02_FORBIDDEN_ENVIRONMENT_KEYS: [&str; 4] = [
    "RILEY_FREEZE_SHA",
    "RILEY_GATE_E_REPORT_SHA",
    "RILEY_CONFIGURATION_SHA",
    "RILEY_BASE_RELEASE_CANDIDATE_REPORT_SHA",
];

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Eq, PartialEq)]
struct C02RuntimeIdentity {
    configuration_profile: C02ConfigurationProfile,
    configuration_sha256: String,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug)]
struct C02EndpointReceipt {
    document: BTreeMap<String, serde_json::Value>,
    bytes: Vec<u8>,
}

#[cfg(any(feature = "cuda", test))]
fn c02_runtime_identity(
    configuration_profile: C02ConfigurationProfile,
    launch_arguments: &[OsString],
    launch_environment: &[(OsString, OsString)],
) -> Result<C02RuntimeIdentity, String> {
    let mut argv = Vec::with_capacity(launch_arguments.len());
    for argument in launch_arguments {
        let argument = c02_utf8(argument, "launch argv")?;
        if argument.is_empty() || argument.trim() != argument || argument.contains(['\r', '\n']) {
            return Err(
                "C02 launch argv must contain non-empty trimmed single-line UTF-8 values"
                    .to_owned(),
            );
        }
        argv.push(argument);
    }
    if argv.first().is_none_or(|argument| argument != "serve") {
        return Err("C02 launch argv must exclude the executable and begin with serve".to_owned());
    }

    let mut environment = BTreeMap::new();
    for (key, value) in launch_environment {
        let key = c02_utf8(key, "launch environment key")?;
        let value = c02_utf8(value, "launch environment value")?;
        if !c02_environment_key_is_valid(&key) {
            return Err(
                "C02 launch environment keys must be uppercase ASCII names beginning with A-Z or _"
                    .to_owned(),
            );
        }
        if value.contains(['\r', '\n']) {
            return Err("C02 launch environment values must be single-line UTF-8".to_owned());
        }
        if C02_FORBIDDEN_ENVIRONMENT_KEYS.contains(&key.as_str()) {
            return Err(format!(
                "C02 launch environment must not contain self-referential attestation key {key}"
            ));
        }
        if environment.insert(key.clone(), value).is_some() {
            return Err(format!(
                "C02 launch environment contains duplicate key {key}"
            ));
        }
    }

    let mut launch_document = BTreeMap::new();
    launch_document.insert(
        "argv".to_owned(),
        serde_json::Value::Array(argv.into_iter().map(serde_json::Value::String).collect()),
    );
    launch_document.insert("environment".to_owned(), c02_json_value(&environment)?);
    let encoded = c02_canonical_json_bytes(&launch_document)?;
    Ok(C02RuntimeIdentity {
        configuration_profile,
        configuration_sha256: c02_sha256_hex(&encoded),
    })
}

#[cfg(any(feature = "cuda", test))]
fn c02_endpoint_receipt(
    c02: &C02RuntimeConfigOptions,
    runtime_identity: &C02RuntimeIdentity,
    effective_config: &BTreeMap<String, serde_json::Value>,
) -> Result<C02EndpointReceipt, String> {
    if c02.configuration_profile != runtime_identity.configuration_profile {
        return Err("C02 runtime identity profile drifted from parsed launch profile".to_owned());
    }
    let effective_config_sha256 = c02_sha256_hex(&c02_canonical_json_bytes(effective_config)?);
    let runtime_identity = c02_json_object([
        (
            "configuration_profile".to_owned(),
            serde_json::Value::String(c02.configuration_profile.as_str().to_owned()),
        ),
        (
            "configuration_sha256".to_owned(),
            serde_json::Value::String(runtime_identity.configuration_sha256.clone()),
        ),
    ])?;
    let mut document = BTreeMap::new();
    document.insert(
        "schema_version".to_owned(),
        serde_json::Value::String(C02_RUNTIME_CONFIG_SCHEMA_VERSION.to_owned()),
    );
    document.insert(
        "candidate_id".to_owned(),
        serde_json::Value::String(c02.candidate_id.clone()),
    );
    document.insert("runtime_identity".to_owned(), runtime_identity);
    document.insert(
        "effective_config".to_owned(),
        c02_json_value(effective_config)?,
    );
    document.insert(
        "effective_config_sha256".to_owned(),
        serde_json::Value::String(effective_config_sha256),
    );
    let bytes = c02_canonical_json_bytes(&document)?;
    Ok(C02EndpointReceipt { document, bytes })
}

#[cfg(any(feature = "cuda", test))]
const C02_GENERATION_AUDIT_SCHEMA_VERSION: &str = "riley.c02-generation-audit.v2";
#[cfg(any(feature = "cuda", test))]
const C02_GENERATION_AUDIT_COMPLETION_SCHEMA_VERSION: &str =
    "riley.c02-generation-audit-completion.v2";
// Keep these producer limits synchronized with
// `benchmarks/release/candidates/c02-generation-audit-v2.schema.json`.
//
// `emitted_text_delta` is capped in Unicode scalar values, not UTF-8 bytes:
// serde_json serializes Rust `String` values as valid JSON Unicode strings and
// JSON Schema's `maxLength` is defined over Unicode characters. Rust's
// `str::chars` counts exactly those scalar values for a valid Rust string.
#[cfg(any(feature = "cuda", test))]
const C02_GENERATION_AUDIT_MAX_PROMPT_TOKEN_IDS: usize = 131_072;
#[cfg(any(feature = "cuda", test))]
const C02_GENERATION_AUDIT_MAX_COMMITTED_OUTPUT_TOKENS: usize = 65_536;
#[cfg(any(feature = "cuda", test))]
const C02_GENERATION_AUDIT_MAX_SAMPLING_SELECTIONS: usize = 65_536;
#[cfg(any(feature = "cuda", test))]
const C02_GENERATION_AUDIT_MAX_EMITTED_TEXT_DELTA_CHARS: usize = 1_048_576;

/// Linux process identity retained in each raw C02 audit leaf so a PID cannot
/// be replayed after it has been reused by a later process.
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct C02AuditProcessIdentity {
    pid: u32,
    start_ticks: u64,
}

#[cfg(any(feature = "cuda", test))]
fn c02_process_identity_from_linux_proc_stat(
    stat: &str,
) -> Result<C02AuditProcessIdentity, String> {
    // Field 2 (`comm`) may contain spaces and parentheses, so splitting the
    // whole line is unsound. After its final ')' field 3 is index 0 and
    // starttime (field 22) is index 19.
    let (before_comm_end, after_comm_end) = stat
        .rsplit_once(')')
        .ok_or_else(|| "C02 process stat has no terminating comm field".to_owned())?;
    let pid = before_comm_end
        .trim_start()
        .split_ascii_whitespace()
        .next()
        .ok_or_else(|| "C02 process stat has no PID field".to_owned())?
        .parse::<u32>()
        .map_err(|_| "C02 process stat PID is not an unsigned integer".to_owned())?;
    let start_ticks = after_comm_end
        .split_ascii_whitespace()
        .nth(19)
        .ok_or_else(|| "C02 process stat has no start-tick field".to_owned())?
        .parse::<u64>()
        .map_err(|_| "C02 process stat start ticks are not an unsigned integer".to_owned())?;
    if start_ticks == 0 {
        return Err("C02 process stat start ticks must be greater than zero".to_owned());
    }
    Ok(C02AuditProcessIdentity { pid, start_ticks })
}

#[cfg(any(feature = "cuda", test))]
fn c02_generation_audit_record_basename(server_request_id: &str) -> Result<String, String> {
    if !server_request_id.starts_with("cmpl-")
        || server_request_id.len() == "cmpl-".len()
        || server_request_id.len() > 128
        || !server_request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err("C02 audit record has an invalid source-issued request ID".to_owned());
    }
    Ok(format!("{server_request_id}.json"))
}

#[cfg(any(feature = "cuda", test))]
fn c02_generation_audit_completion_basename(record_basename: &str) -> Result<String, String> {
    let completion_basename = format!("{record_basename}.complete");
    if completion_basename.len() > 255 {
        return Err("C02 audit completion marker basename is too long".to_owned());
    }
    Ok(completion_basename)
}

#[cfg(any(feature = "cuda", test))]
fn c02_generation_audit_completion_marker_bytes(
    record_basename: &str,
    raw_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let _ = c02_generation_audit_completion_basename(record_basename)?;
    let completion = c02_json_object([
        (
            "schema_version".to_owned(),
            serde_json::Value::String(C02_GENERATION_AUDIT_COMPLETION_SCHEMA_VERSION.to_owned()),
        ),
        (
            "artifact_filename".to_owned(),
            serde_json::Value::String(record_basename.to_owned()),
        ),
        (
            "artifact_sha256".to_owned(),
            serde_json::Value::String(c02_sha256_hex(raw_bytes)),
        ),
    ])?;
    c02_canonical_json_bytes(&completion)
}

#[cfg(feature = "cuda")]
fn c02_capture_process_identity() -> Result<C02AuditProcessIdentity, String> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string("/proc/self/stat")
            .map_err(|error| format!("could not read /proc/self/stat for C02 audit: {error}"))?;
        let identity = c02_process_identity_from_linux_proc_stat(&stat)?;
        if identity.pid != std::process::id() {
            return Err("C02 process stat PID differs from the running process".to_owned());
        }
        Ok(identity)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("--c02-audit-dir requires Linux /proc PID start-tick support".to_owned())
    }
}

/// Strict descriptor-relative C02 audit I/O.
///
/// The source has no safe standard-library replacement for `openat` with
/// `O_NOFOLLOW`; the narrowly scoped unsafe boundary below retains every
/// opened directory descriptor for the writer lifetime and never falls back
/// when a required kernel flag is unavailable.
#[cfg(all(any(feature = "cuda", test), unix))]
#[allow(unsafe_code)]
mod c02_strict_audit_io {
    use std::ffi::{CStr, CString};
    use std::fs::File;
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd, RawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Component, Path};

    #[derive(Debug)]
    pub(super) struct C02HeldPrivateDirectory {
        // The final item is the private evidence root. Keeping the complete
        // chain open prevents a later path rebinding from changing the root
        // used by `openat` and retains the validated parent topology.
        chain: Vec<File>,
    }

    impl C02HeldPrivateDirectory {
        pub(super) fn open(path: &Path) -> Result<Self, String> {
            if path.as_os_str().is_empty() || !path.is_absolute() {
                return Err("--c02-audit-dir must be an absolute non-empty path".to_owned());
            }
            let raw_path = path.as_os_str().as_bytes();
            if raw_path.len() < 2
                || raw_path[0] != b'/'
                || raw_path[1..].split(|byte| *byte == b'/').any(|component| {
                    component.is_empty() || component == b"." || component == b".."
                })
            {
                return Err("--c02-audit-dir must contain only normal path components".to_owned());
            }

            let flags = directory_open_flags()?;
            let slash = CString::new("/").expect("literal path has no NUL");
            let root = open_directory_at(libc::AT_FDCWD, &slash, flags, "C02 audit path root")?;
            validate_ancestor_directory(&root, "C02 audit path root")?;
            let mut chain = vec![root];
            let mut normal_components = 0_usize;
            for component in path.components() {
                let Component::Normal(component) = component else {
                    if matches!(component, Component::RootDir) {
                        continue;
                    }
                    return Err(
                        "--c02-audit-dir must contain only normal path components".to_owned()
                    );
                };
                normal_components = normal_components
                    .checked_add(1)
                    .ok_or_else(|| "C02 audit path component count overflowed".to_owned())?;
                let name = CString::new(component.as_bytes())
                    .map_err(|_| "C02 audit path component contains a NUL byte".to_owned())?;
                let parent_fd = chain
                    .last()
                    .expect("C02 audit path always retains /")
                    .as_raw_fd();
                let next = open_directory_at(parent_fd, &name, flags, "C02 audit path ancestor")?;
                validate_ancestor_directory(&next, "C02 audit path ancestor")?;
                chain.push(next);
            }
            if normal_components == 0 {
                return Err("--c02-audit-dir must not be the filesystem root".to_owned());
            }
            validate_private_root(
                chain
                    .last()
                    .expect("C02 audit path with a component retains a root"),
            )?;
            Ok(Self { chain })
        }

        pub(super) fn create_new_regular_leaf(
            &self,
            basename: &str,
            bytes: &[u8],
        ) -> Result<(), String> {
            if !safe_basename(basename) {
                return Err(
                    "C02 audit output must be an ASCII-safe direct-child basename".to_owned(),
                );
            }
            let name = CString::new(basename)
                .map_err(|_| "C02 audit output basename contains a NUL byte".to_owned())?;
            let flags = leaf_open_flags()?;
            let fd = unsafe { libc::openat(self.root().as_raw_fd(), name.as_ptr(), flags, 0o600) };
            if fd < 0 {
                return Err(format!(
                    "could not create C02 audit leaf {basename} without replacement: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let mut file = unsafe { File::from_raw_fd(fd) };
            validate_new_leaf(&file, basename)?;
            file.write_all(bytes)
                .map_err(|error| format!("could not write C02 audit leaf {basename}: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("could not fsync C02 audit leaf {basename}: {error}"))?;
            self.root().sync_all().map_err(|error| {
                format!("could not fsync C02 audit directory after {basename}: {error}")
            })
        }

        fn root(&self) -> &File {
            self.chain
                .last()
                .expect("C02 held directory chain is never empty")
        }
    }

    fn directory_open_flags() -> Result<i32, String> {
        if libc::O_NOFOLLOW == 0
            || libc::O_DIRECTORY == 0
            || libc::O_CLOEXEC == 0
            || libc::O_NONBLOCK == 0
        {
            return Err(
                "C02 audit requires O_NOFOLLOW, O_DIRECTORY, O_CLOEXEC, and O_NONBLOCK; no fallback is allowed"
                    .to_owned(),
            );
        }
        Ok(libc::O_RDONLY
            | libc::O_NOFOLLOW
            | libc::O_DIRECTORY
            | libc::O_CLOEXEC
            | libc::O_NONBLOCK)
    }

    fn leaf_open_flags() -> Result<i32, String> {
        if libc::O_NOFOLLOW == 0 || libc::O_CLOEXEC == 0 {
            return Err(
                "C02 audit requires O_NOFOLLOW and O_CLOEXEC; no fallback is allowed".to_owned(),
            );
        }
        Ok(libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC)
    }

    fn open_directory_at(
        parent_fd: RawFd,
        name: &CStr,
        flags: i32,
        label: &str,
    ) -> Result<File, String> {
        let fd = unsafe { libc::openat(parent_fd, name.as_ptr(), flags) };
        if fd < 0 {
            return Err(format!(
                "could not securely open {label}: {}",
                std::io::Error::last_os_error()
            ));
        }
        let file = unsafe { File::from_raw_fd(fd) };
        let stat = fstat(&file, label)?;
        if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
            return Err(format!("{label} is not a directory"));
        }
        Ok(file)
    }

    fn validate_ancestor_directory(file: &File, label: &str) -> Result<(), String> {
        let stat = fstat(file, label)?;
        let effective_uid = unsafe { libc::geteuid() };
        if stat.st_uid != 0 && stat.st_uid != effective_uid {
            return Err(format!(
                "{label} is not owned by root or the effective user"
            ));
        }
        let writable_by_group_or_other = stat.st_mode & 0o022 != 0;
        if writable_by_group_or_other && stat.st_mode & libc::S_ISVTX == 0 {
            return Err(format!(
                "{label} is group/other-writable without a sticky boundary"
            ));
        }
        Ok(())
    }

    fn validate_private_root(file: &File) -> Result<(), String> {
        let stat = fstat(file, "C02 audit directory")?;
        let effective_uid = unsafe { libc::geteuid() };
        if stat.st_uid != effective_uid || stat.st_mode & 0o7777 != 0o700 {
            return Err(
                "--c02-audit-dir must be owned by the effective user with exact mode 0700"
                    .to_owned(),
            );
        }
        Ok(())
    }

    fn validate_new_leaf(file: &File, basename: &str) -> Result<(), String> {
        let stat = fstat(file, "C02 audit output")?;
        let effective_uid = unsafe { libc::geteuid() };
        if stat.st_mode & libc::S_IFMT != libc::S_IFREG
            || stat.st_nlink != 1
            || stat.st_uid != effective_uid
            || stat.st_mode & 0o7777 != 0o600
        {
            return Err(format!(
                "new C02 audit leaf {basename} is not an euid-owned 0600 single-link regular file"
            ));
        }
        Ok(())
    }

    fn fstat(file: &File, label: &str) -> Result<libc::stat, String> {
        let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
        if unsafe { libc::fstat(file.as_raw_fd(), &mut stat) } != 0 {
            return Err(format!(
                "could not inspect {label}: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(stat)
    }

    fn safe_basename(value: &str) -> bool {
        !value.is_empty()
            && value.len() <= 255
            && value != "."
            && value != ".."
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    }
}

#[cfg(all(any(feature = "cuda", test), unix))]
use c02_strict_audit_io::C02HeldPrivateDirectory;

// The audit root is opened and topology-validated once at startup. Every C02
// writer receives a clone of this held descriptor root; no shutdown path is
// opened again after launch-time validation.
#[cfg(all(any(feature = "cuda", test), unix))]
type C02SharedAuditRoot = Arc<C02HeldPrivateDirectory>;

#[cfg(all(any(feature = "cuda", test), unix))]
fn c02_open_audit_root(path: &std::path::Path) -> Result<C02SharedAuditRoot, String> {
    C02HeldPrivateDirectory::open(path).map(Arc::new)
}

#[cfg(all(feature = "cuda", not(unix)))]
type C02SharedAuditRoot = std::sync::Arc<()>;

#[cfg(all(feature = "cuda", not(unix)))]
fn c02_open_audit_root(_path: &std::path::Path) -> Result<C02SharedAuditRoot, String> {
    Err("--c02-audit-dir requires Unix no-follow descriptor APIs".to_owned())
}

/// Raw source-owned C02 generation-audit writer.
///
/// The writer owns a held, private directory descriptor rather than a path.
/// It creates a request JSON leaf and its hash-bound completion marker with
/// no-replace descriptor-relative calls. It records sampling-path selection
/// only; it does not imply executor, attention, or GEMM fallback.
#[cfg(all(feature = "cuda", unix))]
struct C02GenerationAuditWriter {
    directory: C02SharedAuditRoot,
    candidate_id: String,
    configuration_profile: C02ConfigurationProfile,
    configuration_sha256: String,
    process_identity: C02AuditProcessIdentity,
}

#[cfg(all(feature = "cuda", unix))]
impl C02GenerationAuditWriter {
    fn new(
        directory: C02SharedAuditRoot,
        c02: &C02RuntimeConfigOptions,
        identity: &C02RuntimeIdentity,
        process_identity: C02AuditProcessIdentity,
    ) -> Result<Self, String> {
        if c02.configuration_profile != identity.configuration_profile {
            return Err("C02 audit identity profile drifted from parsed launch profile".to_owned());
        }
        if process_identity.pid == 0 || process_identity.start_ticks == 0 {
            return Err(
                "C02 audit process identity must have nonzero PID and start ticks".to_owned(),
            );
        }
        Ok(Self {
            directory,
            candidate_id: c02.candidate_id.clone(),
            configuration_profile: identity.configuration_profile,
            configuration_sha256: identity.configuration_sha256.clone(),
            process_identity,
        })
    }
}

#[cfg(feature = "cuda")]
fn validate_c02_generation_audit_schema_bounds(
    record: &riley_server::engine::C02GenerationAuditRecord,
) -> Result<(), String> {
    if record.prompt_token_ids.len() > C02_GENERATION_AUDIT_MAX_PROMPT_TOKEN_IDS {
        return Err("C02 generation audit prompt token count exceeds its schema limit".to_owned());
    }
    if record.committed_output_tokens.len() > C02_GENERATION_AUDIT_MAX_COMMITTED_OUTPUT_TOKENS {
        return Err("C02 generation audit output token count exceeds its schema limit".to_owned());
    }
    if record.sampling_selections.len() > C02_GENERATION_AUDIT_MAX_SAMPLING_SELECTIONS {
        return Err(
            "C02 generation audit sampling selection count exceeds its schema limit".to_owned(),
        );
    }
    if record.committed_output_tokens.iter().any(|token| {
        token.emitted_text_delta.chars().count() > C02_GENERATION_AUDIT_MAX_EMITTED_TEXT_DELTA_CHARS
    }) {
        return Err("C02 generation audit emitted text delta exceeds its schema limit".to_owned());
    }
    Ok(())
}

#[cfg(all(feature = "cuda", unix))]
impl riley_server::engine::C02GenerationAuditSink for C02GenerationAuditWriter {
    fn write_record(
        &self,
        record: riley_server::engine::C02GenerationAuditRecord,
    ) -> Result<(), String> {
        // Validate every schema capacity before creating the raw leaf. An
        // oversize record must leave neither an evidence leaf nor a marker.
        validate_c02_generation_audit_schema_bounds(&record)?;
        let record_basename = c02_generation_audit_record_basename(&record.server_request_id)?;
        let finish_reason = match record.finish_reason {
            riley_server::domain::FinishReason::Length => "length",
            riley_server::domain::FinishReason::Stop => "stop",
            riley_server::domain::FinishReason::Cancelled
            | riley_server::domain::FinishReason::Error => {
                return Err("C02 generation audit refuses a non-success terminal".to_owned());
            }
            _ => {
                return Err("C02 generation audit refuses an unsupported terminal".to_owned());
            }
        };
        if record.committed_output_tokens.len() != record.sampling_selections.len()
            || record.prompt_token_ids.is_empty()
            || record.committed_output_tokens.is_empty()
            || record
                .sampling_selections
                .iter()
                .any(|selection| !selection.committed)
            || record.sampling_selections.iter().any(|selection| {
                matches!(
                    (selection.selected_backend, selection.ineligibility_reason),
                    (riley_server::engine::C02SamplingBackend::GpuGreedy, Some(_))
                        | (riley_server::engine::C02SamplingBackend::CpuNormative, None)
                )
            })
        {
            return Err(
                "C02 generation audit record is not an exact committed selection projection"
                    .to_owned(),
            );
        }
        let output_tokens = record
            .committed_output_tokens
            .into_iter()
            .map(|token| {
                c02_json_object([
                    (
                        "token_id".to_owned(),
                        serde_json::Value::Number(serde_json::Number::from(token.token_id)),
                    ),
                    (
                        "emitted_text_delta".to_owned(),
                        serde_json::Value::String(token.emitted_text_delta),
                    ),
                ])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let selections = record
            .sampling_selections
            .into_iter()
            .map(|selection| {
                c02_json_object([
                    (
                        "iteration_id".to_owned(),
                        serde_json::Value::Number(serde_json::Number::from(selection.iteration_id)),
                    ),
                    (
                        "configured_backend".to_owned(),
                        serde_json::Value::String(selection.configured_backend.as_str().to_owned()),
                    ),
                    (
                        "selected_backend".to_owned(),
                        serde_json::Value::String(selection.selected_backend.as_str().to_owned()),
                    ),
                    (
                        "ineligibility_reason".to_owned(),
                        selection
                            .ineligibility_reason
                            .map(|reason| serde_json::Value::String(reason.as_str().to_owned()))
                            .unwrap_or(serde_json::Value::Null),
                    ),
                    (
                        "committed".to_owned(),
                        serde_json::Value::Bool(selection.committed),
                    ),
                ])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let runtime_identity = c02_json_object([
            (
                "configuration_profile".to_owned(),
                serde_json::Value::String(self.configuration_profile.as_str().to_owned()),
            ),
            (
                "configuration_sha256".to_owned(),
                serde_json::Value::String(self.configuration_sha256.clone()),
            ),
        ])?;
        let process_identity = c02_json_object([
            (
                "pid".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(self.process_identity.pid)),
            ),
            (
                "start_ticks".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.process_identity.start_ticks,
                )),
            ),
        ])?;
        let usage = c02_json_object([
            (
                "prompt_tokens".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(record.usage.prompt_tokens())),
            ),
            (
                "completion_tokens".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    record.usage.completion_tokens(),
                )),
            ),
            (
                "total_tokens".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(record.usage.total_tokens())),
            ),
        ])?;
        let mut document = BTreeMap::new();
        document.insert(
            "schema_version".to_owned(),
            serde_json::Value::String(C02_GENERATION_AUDIT_SCHEMA_VERSION.to_owned()),
        );
        document.insert(
            "candidate_id".to_owned(),
            serde_json::Value::String(self.candidate_id.clone()),
        );
        document.insert("runtime_identity".to_owned(), runtime_identity);
        document.insert("process_identity".to_owned(), process_identity);
        document.insert(
            "server_request_id".to_owned(),
            serde_json::Value::String(record.server_request_id),
        );
        document.insert(
            "delivery_mode".to_owned(),
            serde_json::Value::String(record.delivery_mode.as_str().to_owned()),
        );
        document.insert(
            "prompt_token_ids".to_owned(),
            serde_json::Value::Array(
                record
                    .prompt_token_ids
                    .into_iter()
                    .map(serde_json::Number::from)
                    .map(serde_json::Value::Number)
                    .collect(),
            ),
        );
        document.insert(
            "committed_output_tokens".to_owned(),
            serde_json::Value::Array(output_tokens),
        );
        document.insert(
            "sampling_selections".to_owned(),
            serde_json::Value::Array(selections),
        );
        document.insert(
            "finish_reason".to_owned(),
            serde_json::Value::String(finish_reason.to_owned()),
        );
        document.insert("usage".to_owned(), usage);
        let bytes = c02_canonical_json_bytes(&document)?;
        self.directory
            .create_new_regular_leaf(&record_basename, &bytes)?;

        // A raw leaf without this independently create-only hash-bound marker
        // is deliberately incomplete evidence. Do not clean up a failed leaf:
        // a retry could overwrite the causal failure boundary.
        let completion_bytes =
            c02_generation_audit_completion_marker_bytes(&record_basename, &bytes)?;
        self.directory.create_new_regular_leaf(
            &c02_generation_audit_completion_basename(&record_basename)?,
            &completion_bytes,
        )
    }
}

#[cfg(any(feature = "cuda", test))]
const C02_SHUTDOWN_SCHEMA_VERSION: &str = "riley.c02-shutdown-quiescence.v2";
#[cfg(any(feature = "cuda", test))]
const C02_SHUTDOWN_COMPLETION_SCHEMA_VERSION: &str = "riley.c02-shutdown-quiescence-complete.v2";

#[cfg(any(feature = "cuda", test))]
fn c02_shutdown_completion_basename(artifact_basename: &str) -> Result<String, String> {
    if !c02_shutdown_artifact_basename_is_valid(artifact_basename) {
        return Err("C02 shutdown artifact basename is not a safe nonhidden JSON leaf".to_owned());
    }
    let completion_basename = format!("{artifact_basename}.complete");
    if completion_basename.len() > 255 {
        return Err("C02 shutdown completion marker basename is too long".to_owned());
    }
    Ok(completion_basename)
}

#[cfg(any(feature = "cuda", test))]
fn c02_shutdown_artifact_bytes(
    artifact_basename: &str,
    process_identity: C02AuditProcessIdentity,
    final_metrics: riley_server::engine::C02CaptureMetrics,
) -> Result<Vec<u8>, String> {
    let _ = c02_shutdown_completion_basename(artifact_basename)?;
    if process_identity.pid == 0 || process_identity.start_ticks == 0 {
        return Err(
            "C02 shutdown process identity must have nonzero PID and start ticks".to_owned(),
        );
    }
    if !final_metrics.is_quiescent() {
        return Err("C02 shutdown refuses non-quiescent final source metrics".to_owned());
    }
    let mut document = BTreeMap::new();
    document.insert(
        "schema_version".to_owned(),
        serde_json::Value::String(C02_SHUTDOWN_SCHEMA_VERSION.to_owned()),
    );
    document.insert(
        "capture_status".to_owned(),
        serde_json::Value::String("captured".to_owned()),
    );
    document.insert(
        "qualification_status".to_owned(),
        serde_json::Value::String("not-run".to_owned()),
    );
    document.insert(
        "server_pid".to_owned(),
        serde_json::Value::Number(serde_json::Number::from(process_identity.pid)),
    );
    document.insert(
        "server_start_ticks".to_owned(),
        serde_json::Value::Number(serde_json::Number::from(process_identity.start_ticks)),
    );
    document.insert("worker_ready".to_owned(), serde_json::Value::Bool(false));
    document.insert(
        "final_metrics".to_owned(),
        riley_server::service::c02_capture_metrics_json_value(final_metrics)?,
    );
    c02_canonical_json_bytes(&document)
}

#[cfg(any(feature = "cuda", test))]
fn c02_shutdown_completion_marker_bytes(
    artifact_basename: &str,
    artifact_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let _ = c02_shutdown_completion_basename(artifact_basename)?;
    let completion = c02_json_object([
        (
            "schema_version".to_owned(),
            serde_json::Value::String(C02_SHUTDOWN_COMPLETION_SCHEMA_VERSION.to_owned()),
        ),
        (
            "artifact_filename".to_owned(),
            serde_json::Value::String(artifact_basename.to_owned()),
        ),
        (
            "artifact_sha256".to_owned(),
            serde_json::Value::String(c02_sha256_hex(artifact_bytes)),
        ),
    ])?;
    c02_canonical_json_bytes(&completion)
}

/// Create-only writer for C02 shutdown-v2 evidence. Its root descriptor was
/// opened at startup and is shared with generation-audit writing. The final
/// allocation values are sampled from the still-owned context before its
/// close, but this writer is reachable only after the context close succeeds
/// and the engine publishes its final zero-ownership snapshot.
#[cfg(all(any(feature = "cuda", test), unix))]
struct C02ShutdownArtifactWriter {
    directory: C02SharedAuditRoot,
    artifact_basename: String,
    process_identity: C02AuditProcessIdentity,
}

#[cfg(all(any(feature = "cuda", test), unix))]
impl C02ShutdownArtifactWriter {
    fn new(
        directory: C02SharedAuditRoot,
        artifact: C02ShutdownArtifactOptions,
        process_identity: C02AuditProcessIdentity,
    ) -> Result<Self, String> {
        let _ = c02_shutdown_completion_basename(&artifact.basename)?;
        if process_identity.pid == 0 || process_identity.start_ticks == 0 {
            return Err(
                "C02 shutdown process identity must have nonzero PID and start ticks".to_owned(),
            );
        }
        Ok(Self {
            directory,
            artifact_basename: artifact.basename,
            process_identity,
        })
    }

    fn write(&self, final_metrics: riley_server::engine::C02CaptureMetrics) -> Result<(), String> {
        let artifact_bytes = c02_shutdown_artifact_bytes(
            &self.artifact_basename,
            self.process_identity,
            final_metrics,
        )?;
        self.directory
            .create_new_regular_leaf(&self.artifact_basename, &artifact_bytes)?;
        let completion_bytes =
            c02_shutdown_completion_marker_bytes(&self.artifact_basename, &artifact_bytes)?;
        self.directory.create_new_regular_leaf(
            &c02_shutdown_completion_basename(&self.artifact_basename)?,
            &completion_bytes,
        )
    }
}

#[cfg(all(feature = "cuda", not(unix)))]
struct C02ShutdownArtifactWriter;

#[cfg(all(feature = "cuda", not(unix)))]
impl C02ShutdownArtifactWriter {
    fn new(
        _directory: C02SharedAuditRoot,
        _artifact: C02ShutdownArtifactOptions,
        _process_identity: C02AuditProcessIdentity,
    ) -> Result<Self, String> {
        Err("--c02-shutdown-artifact requires Unix no-follow descriptor APIs".to_owned())
    }

    fn write(&self, _final_metrics: riley_server::engine::C02CaptureMetrics) -> Result<(), String> {
        Err("--c02-shutdown-artifact requires Unix no-follow descriptor APIs".to_owned())
    }
}

#[cfg(all(feature = "cuda", not(unix)))]
struct C02GenerationAuditWriter;

#[cfg(all(feature = "cuda", not(unix)))]
impl C02GenerationAuditWriter {
    fn new(
        _directory: C02SharedAuditRoot,
        _c02: &C02RuntimeConfigOptions,
        _identity: &C02RuntimeIdentity,
        _process_identity: C02AuditProcessIdentity,
    ) -> Result<Self, String> {
        Err("--c02-audit-dir requires Unix no-follow descriptor APIs".to_owned())
    }
}

#[cfg(all(feature = "cuda", not(unix)))]
impl riley_server::engine::C02GenerationAuditSink for C02GenerationAuditWriter {
    fn write_record(
        &self,
        _record: riley_server::engine::C02GenerationAuditRecord,
    ) -> Result<(), String> {
        Err("--c02-audit-dir requires Unix no-follow descriptor APIs".to_owned())
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_lines)] // Closed ten-dimension receipt mapping is easier to audit as one unit.
fn c02_effective_config_from_facts(
    facts: &riley_server::engine::CudaEffectiveRuntimeFacts,
) -> Result<BTreeMap<String, serde_json::Value>, String> {
    let batch_token_budget = facts.batch_token_budget();
    let batch_shape_buckets = facts.batch_shape_buckets();
    if batch_token_budget == 0
        || batch_shape_buckets.is_empty()
        || batch_shape_buckets.last().copied() != Some(batch_token_budget)
        || batch_shape_buckets.iter().any(|bucket| *bucket == 0)
        || batch_shape_buckets
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err("prepared C02 batch-shape facts are invalid".to_owned());
    }
    match facts.batch_shape_policy() {
        "fixed-max" if batch_shape_buckets == [batch_token_budget] => {}
        "power-of-two" if batch_shape_buckets.first().copied() == Some(1) => {}
        _ => return Err("prepared C02 batch-shape policy and buckets disagree".to_owned()),
    }
    if !matches!(
        facts.execution_completion_mode(),
        "per-operation" | "iteration-batch"
    ) || !matches!(facts.metadata_transport(), "synchronous" | "packed-async")
        || !matches!(facts.sampling_backend(), "cpu" | "gpu-greedy")
        || facts.metadata_transport() == "packed-async"
            && facts.execution_completion_mode() != "iteration-batch"
        || facts.cross_profile_fallback() != "forbidden"
        || !matches!(
            facts.runtime_selection(),
            "exact-fallback-allowed" | "fail-closed"
        )
        || facts.kv_layout() != "paged"
        || facts.kv_block_tokens() == 0
        || facts.kv_physical_blocks() == 0
        || !c02_implementation_id(facts.attention_prefill())
        || !c02_implementation_id(facts.attention_decode())
        || !c02_implementation_id(facts.gemm_reduction_policy())
        || !matches!(facts.residual_rmsnorm(), "fused" | "separate")
    {
        return Err(
            "prepared C02 effective runtime facts violate the closed receipt contract".to_owned(),
        );
    }

    let bucket_values = batch_shape_buckets
        .iter()
        .copied()
        .map(serde_json::Number::from)
        .map(serde_json::Value::Number)
        .collect();
    let experimental_flags = c02_json_object([(
        "residual_rmsnorm".to_owned(),
        serde_json::Value::String(facts.residual_rmsnorm().to_owned()),
    )])?;
    let mut effective_config = BTreeMap::new();
    effective_config.insert(
        "execution_completion_mode".to_owned(),
        serde_json::Value::String(facts.execution_completion_mode().to_owned()),
    );
    effective_config.insert(
        "batch_shape".to_owned(),
        c02_json_object([
            (
                "policy".to_owned(),
                serde_json::Value::String(facts.batch_shape_policy().to_owned()),
            ),
            (
                "buckets".to_owned(),
                serde_json::Value::Array(bucket_values),
            ),
        ])?,
    );
    effective_config.insert(
        "metadata_transport".to_owned(),
        serde_json::Value::String(facts.metadata_transport().to_owned()),
    );
    effective_config.insert(
        "sampling_backend".to_owned(),
        serde_json::Value::String(facts.sampling_backend().to_owned()),
    );
    effective_config.insert(
        "attention_backend".to_owned(),
        c02_json_object([
            (
                "prefill".to_owned(),
                serde_json::Value::String(facts.attention_prefill().to_owned()),
            ),
            (
                "decode".to_owned(),
                serde_json::Value::String(facts.attention_decode().to_owned()),
            ),
        ])?,
    );
    effective_config.insert(
        "gemm_reduction_policy".to_owned(),
        serde_json::Value::String(facts.gemm_reduction_policy().to_owned()),
    );
    effective_config.insert("experimental_flags".to_owned(), experimental_flags);
    effective_config.insert(
        "fallback_policy".to_owned(),
        c02_json_object([
            (
                "cross_profile_fallback".to_owned(),
                serde_json::Value::String(facts.cross_profile_fallback().to_owned()),
            ),
            (
                "runtime_selection".to_owned(),
                serde_json::Value::String(facts.runtime_selection().to_owned()),
            ),
        ])?,
    );
    effective_config.insert(
        "batch_token_budget".to_owned(),
        serde_json::Value::Number(serde_json::Number::from(batch_token_budget)),
    );
    effective_config.insert(
        "kv_geometry".to_owned(),
        c02_json_object([
            (
                "layout".to_owned(),
                serde_json::Value::String(facts.kv_layout().to_owned()),
            ),
            (
                "block_tokens".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(facts.kv_block_tokens())),
            ),
            (
                "physical_blocks".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(facts.kv_physical_blocks())),
            ),
        ])?,
    );
    Ok(effective_config)
}

#[cfg(feature = "cuda")]
fn c02_implementation_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    let Some(first) = bytes.first() else {
        return false;
    };
    (first.is_ascii_lowercase() || first.is_ascii_digit())
        && bytes.len() <= 256
        && bytes[1..].iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
}

#[cfg(any(feature = "cuda", test))]
fn c02_environment_key_is_valid(value: &str) -> bool {
    let bytes = value.as_bytes();
    let Some(first) = bytes.first() else {
        return false;
    };
    (first.is_ascii_uppercase() || *first == b'_')
        && bytes[1..]
            .iter()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || *byte == b'_')
}

#[cfg(any(feature = "cuda", test))]
fn write_c02_startup_artifact(
    path: &std::path::Path,
    c02: &C02RuntimeConfigOptions,
    endpoint: &C02EndpointReceipt,
) -> Result<(), String> {
    use std::io::Write;

    let mut document = BTreeMap::new();
    document.insert(
        "schema_version".to_owned(),
        serde_json::Value::String(C02_STARTUP_ARTIFACT_SCHEMA_VERSION.to_owned()),
    );
    document.insert(
        "created_at_utc".to_owned(),
        serde_json::Value::String(c02_utc_timestamp(std::time::SystemTime::now())?),
    );
    document.insert(
        "candidate_id".to_owned(),
        serde_json::Value::String(c02.candidate_id.clone()),
    );
    document.insert(
        "endpoint_path".to_owned(),
        serde_json::Value::String("/v1/config".to_owned()),
    );
    document.insert(
        "runtime_identity".to_owned(),
        endpoint
            .document
            .get("runtime_identity")
            .cloned()
            .ok_or_else(|| "C02 endpoint receipt omitted runtime identity".to_owned())?,
    );
    document.insert(
        "endpoint_payload_sha256".to_owned(),
        serde_json::Value::String(c02_sha256_hex(&endpoint.bytes)),
    );
    document.insert(
        "endpoint_payload".to_owned(),
        c02_json_value(&endpoint.document)?,
    );
    let encoded = c02_canonical_json_bytes(&document)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "could not create C02 startup artifact {} without replacement: {error}",
                path.display()
            )
        })?;
    file.write_all(&encoded).map_err(|error| {
        format!(
            "could not write C02 startup artifact {}: {error}",
            path.display()
        )
    })?;
    file.sync_all().map_err(|error| {
        format!(
            "could not sync C02 startup artifact {}: {error}",
            path.display()
        )
    })
}

#[cfg(any(feature = "cuda", test))]
fn c02_json_object(
    values: impl IntoIterator<Item = (String, serde_json::Value)>,
) -> Result<serde_json::Value, String> {
    c02_json_value(&values.into_iter().collect::<BTreeMap<_, _>>())
}

#[cfg(any(feature = "cuda", test))]
fn c02_json_value<T: serde::Serialize>(value: &T) -> Result<serde_json::Value, String> {
    serde_json::to_value(value)
        .map_err(|error| format!("could not construct canonical C02 JSON value: {error}"))
}

#[cfg(any(feature = "cuda", test))]
fn c02_canonical_json_bytes<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value)
        .map_err(|error| format!("could not serialize canonical C02 JSON: {error}"))
}

#[cfg(any(feature = "cuda", test))]
fn c02_sha256_hex(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let digest = sha2::Sha256::digest(value);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(any(feature = "cuda", test))]
fn c02_utf8(value: &OsStr, label: &str) -> Result<String, String> {
    value
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("C02 {label} must be UTF-8"))
}

#[cfg(any(feature = "cuda", test))]
fn c02_utc_timestamp(now: std::time::SystemTime) -> Result<String, String> {
    let seconds = now
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| "C02 startup artifact timestamp predates Unix epoch".to_owned())?
        .as_secs();
    c02_utc_timestamp_from_unix_seconds(seconds)
}

#[cfg(any(feature = "cuda", test))]
fn c02_utc_timestamp_from_unix_seconds(seconds: u64) -> Result<String, String> {
    let seconds = i64::try_from(seconds)
        .map_err(|_| "C02 startup artifact timestamp is out of range".to_owned())?;
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = c02_civil_from_unix_days(days);
    if !(0..=9_999).contains(&year) {
        return Err("C02 startup artifact timestamp year is out of range".to_owned());
    }
    let hour = seconds_of_day / 3_600;
    let minute = seconds_of_day.rem_euclid(3_600) / 60;
    let second = seconds_of_day.rem_euclid(60);
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

#[cfg(any(feature = "cuda", test))]
fn c02_civil_from_unix_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = (if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    }) / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

#[cfg(all(feature = "cuda", unix))]
fn wait_for_shutdown(
    signals: std::sync::Arc<signal::ShutdownSignals>,
    include_stdin: bool,
) -> Result<&'static str, String> {
    use std::io;
    use std::sync::mpsc;
    use std::thread;

    fn signal_name(signal: signal::ShutdownSignal) -> &'static str {
        match signal {
            signal::ShutdownSignal::Interrupt => "SIGINT",
            signal::ShutdownSignal::Terminate => "SIGTERM",
        }
    }

    if !include_stdin {
        return signals
            .wait()
            .map(signal_name)
            .map_err(|error| format!("shutdown signal wait failed: {error}"));
    }

    let (sender, receiver) = mpsc::sync_channel(2);
    let signal_sender = sender.clone();
    thread::Builder::new()
        .name("riley-signal-wait".to_owned())
        .spawn(move || {
            let result = signals
                .wait()
                .map(signal_name)
                .map_err(|error| format!("shutdown signal wait failed: {error}"));
            let _ = signal_sender.send(result);
        })
        .map_err(|error| format!("could not start shutdown signal waiter: {error}"))?;
    thread::Builder::new()
        .name("riley-stdin-wait".to_owned())
        .spawn(move || {
            let mut line = String::new();
            let result = io::stdin()
                .read_line(&mut line)
                .map(|_| "stdin")
                .map_err(|error| format!("shutdown input failed: {error}"));
            let _ = sender.send(result);
        })
        .map_err(|error| format!("could not start shutdown stdin waiter: {error}"))?;
    receiver
        .recv()
        .map_err(|_| "all shutdown waiters exited without a trigger".to_owned())?
}

#[cfg(feature = "cuda")]
fn validate_positive(option: &str, value: usize) -> Result<(), String> {
    if value == 0 {
        Err(format!("{option} must be greater than zero"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::PathBuf;

    use super::{
        BatchShapePolicyMode, C02_GENERATION_AUDIT_COMPLETION_SCHEMA_VERSION,
        C02_GENERATION_AUDIT_MAX_COMMITTED_OUTPUT_TOKENS,
        C02_GENERATION_AUDIT_MAX_EMITTED_TEXT_DELTA_CHARS,
        C02_GENERATION_AUDIT_MAX_PROMPT_TOKEN_IDS, C02_GENERATION_AUDIT_MAX_SAMPLING_SELECTIONS,
        C02_GENERATION_AUDIT_SCHEMA_VERSION, C02_SHUTDOWN_COMPLETION_SCHEMA_VERSION,
        C02_SHUTDOWN_SCHEMA_VERSION, C02AuditProcessIdentity, C02ConfigurationProfile,
        C02RuntimeConfigOptions, C02ShutdownArtifactOptions, CliCommand, DEFAULT_MAX_WEIGHT_BYTES,
        ExecutionCompletionMode, FIXED37_MAX_SEQUENCE_TOKENS, MetadataTransportMode,
        ReductionProfileMode, ResidualRmsNormMode, SamplingBackendMode, ServeOptions, USAGE,
        c02_canonical_json_bytes, c02_endpoint_receipt, c02_generation_audit_completion_basename,
        c02_generation_audit_completion_marker_bytes, c02_generation_audit_record_basename,
        c02_process_identity_from_linux_proc_stat, c02_runtime_identity, c02_sha256_hex,
        c02_shutdown_artifact_bytes, c02_shutdown_completion_basename,
        c02_shutdown_completion_marker_bytes, c02_utc_timestamp_from_unix_seconds, parse_arguments,
        validate_reduction_profile_context, validate_shutdown_metrics_path,
        write_c02_startup_artifact, write_shutdown_metrics,
    };

    fn args<'a>(values: &'a [&'a str]) -> impl Iterator<Item = OsString> + 'a {
        values.iter().map(OsString::from)
    }

    #[test]
    fn c02_generation_audit_v2_schema_is_narrow_and_canonical() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../benchmarks/release/candidates/c02-generation-audit-v2.schema.json"
        ))
        .expect("decode C02 generation-audit schema");
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            C02_GENERATION_AUDIT_SCHEMA_VERSION
        );
        let required = schema["required"].as_array().expect("required fields");
        for field in [
            "runtime_identity",
            "process_identity",
            "server_request_id",
            "committed_output_tokens",
            "sampling_selections",
            "finish_reason",
            "usage",
        ] {
            assert!(
                required.iter().any(|entry| entry == field),
                "missing required audit field {field}"
            );
        }
        assert_eq!(
            schema["$defs"]["samplingSelection"]["properties"]["committed"]["const"],
            true
        );
        assert_eq!(
            schema["$defs"]["outputToken"]["required"],
            serde_json::json!(["token_id", "emitted_text_delta"])
        );
        assert_eq!(
            schema["properties"]["prompt_token_ids"]["maxItems"],
            serde_json::json!(C02_GENERATION_AUDIT_MAX_PROMPT_TOKEN_IDS)
        );
        assert_eq!(
            schema["properties"]["committed_output_tokens"]["maxItems"],
            serde_json::json!(C02_GENERATION_AUDIT_MAX_COMMITTED_OUTPUT_TOKENS)
        );
        assert_eq!(
            schema["properties"]["sampling_selections"]["maxItems"],
            serde_json::json!(C02_GENERATION_AUDIT_MAX_SAMPLING_SELECTIONS)
        );
        assert_eq!(
            schema["$defs"]["outputToken"]["properties"]["emitted_text_delta"]["maxLength"],
            serde_json::json!(C02_GENERATION_AUDIT_MAX_EMITTED_TEXT_DELTA_CHARS)
        );
        assert_eq!(
            schema["properties"]["process_identity"]["properties"]["start_ticks"]["minimum"],
            1
        );
        assert!(
            schema["description"]
                .as_str()
                .expect("schema description")
                .contains("sampling-path selection")
        );

        let completion: serde_json::Value = serde_json::from_str(include_str!(
            "../../../benchmarks/release/candidates/c02-generation-audit-completion-v2.schema.json"
        ))
        .expect("decode C02 audit completion schema");
        assert_eq!(
            completion["properties"]["schema_version"]["const"],
            C02_GENERATION_AUDIT_COMPLETION_SCHEMA_VERSION
        );
        assert_eq!(
            completion["properties"]["artifact_filename"]["pattern"],
            "^cmpl-[A-Za-z0-9_-]{1,123}\\.json$"
        );
        assert!(
            completion["description"]
                .as_str()
                .expect("completion description")
                .contains(".complete")
        );
    }

    #[test]
    fn c02_linux_process_identity_parser_handles_a_parenthesized_comm() {
        assert_eq!(
            c02_process_identity_from_linux_proc_stat(
                "321 (riley worker) R 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 4242 20"
            ),
            Ok(C02AuditProcessIdentity {
                pid: 321,
                start_ticks: 4242,
            })
        );
        assert!(c02_process_identity_from_linux_proc_stat("321 no-parens").is_err());
        assert!(
            c02_process_identity_from_linux_proc_stat(
                "321 (riley worker) R 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 0 20"
            )
            .is_err()
        );
    }

    #[test]
    fn c02_generation_audit_marker_binds_only_a_safe_create_only_leaf() {
        let record = c02_generation_audit_record_basename("cmpl-safe_123")
            .expect("source-issued completion ID becomes a safe leaf");
        assert_eq!(record, "cmpl-safe_123.json");
        assert_eq!(
            c02_generation_audit_completion_basename(&record),
            Ok("cmpl-safe_123.json.complete".to_owned())
        );
        assert!(c02_generation_audit_record_basename("cmpl-").is_err());
        assert!(c02_generation_audit_record_basename("cmpl-../escape").is_err());

        let raw = br#"{\"schema_version\":\"riley.c02-generation-audit.v2\"}"#;
        let marker: serde_json::Value = serde_json::from_slice(
            &c02_generation_audit_completion_marker_bytes(&record, raw)
                .expect("canonical completion marker"),
        )
        .expect("decode completion marker");
        assert_eq!(
            marker["schema_version"],
            C02_GENERATION_AUDIT_COMPLETION_SCHEMA_VERSION
        );
        assert_eq!(marker["artifact_filename"], record);
        assert_eq!(marker["artifact_sha256"], c02_sha256_hex(raw));
    }

    #[cfg(unix)]
    fn c02_test_path(label: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
        std::fs::canonicalize(std::env::temp_dir())
            .expect("canonical test temporary directory")
            .join(format!(
                "riley-c02-{label}-{}-{}",
                std::process::id(),
                NEXT_PATH.fetch_add(1, Ordering::AcqRel),
            ))
    }

    #[cfg(unix)]
    fn create_private_c02_test_directory(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;

        std::fs::create_dir(path).expect("create C02 test directory");
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .expect("set C02 test directory mode");
    }

    #[cfg(unix)]
    #[test]
    fn c02_audit_directory_is_private_held_and_descriptor_relative() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        use super::c02_strict_audit_io::C02HeldPrivateDirectory;

        let outer = c02_test_path("audit-root");
        std::fs::create_dir(&outer).expect("create C02 test parent");
        let evidence = outer.join("evidence");
        create_private_c02_test_directory(&evidence);

        {
            let held = C02HeldPrivateDirectory::open(&evidence)
                .expect("open exact private audit directory");
            held.create_new_regular_leaf("cmpl-held.json", b"{}")
                .expect("create first leaf");
            assert!(
                held.create_new_regular_leaf("cmpl-held.json", b"replacement")
                    .is_err(),
                "a record leaf must not be replaced"
            );

            let rebound = outer.join("rebound");
            std::fs::rename(&evidence, &rebound).expect("rebind visible audit path");
            create_private_c02_test_directory(&evidence);
            held.create_new_regular_leaf("cmpl-after-rebind.json", b"{}")
                .expect("write through held root descriptor");
            assert!(rebound.join("cmpl-after-rebind.json").is_file());
            assert!(!evidence.join("cmpl-after-rebind.json").exists());

            let insecure = outer.join("insecure");
            create_private_c02_test_directory(&insecure);
            std::fs::set_permissions(&insecure, std::fs::Permissions::from_mode(0o750))
                .expect("make test directory non-private");
            assert!(C02HeldPrivateDirectory::open(&insecure).is_err());

            let special_bits = outer.join("special-bits");
            create_private_c02_test_directory(&special_bits);
            std::fs::set_permissions(&special_bits, std::fs::Permissions::from_mode(0o1700))
                .expect("set C02 test directory special bit");
            assert!(C02HeldPrivateDirectory::open(&special_bits).is_err());

            let target = outer.join("target");
            create_private_c02_test_directory(&target);
            let alias = outer.join("alias");
            symlink(&target, &alias).expect("create audit symlink");
            assert!(C02HeldPrivateDirectory::open(&alias).is_err());
        }

        std::fs::remove_dir_all(&outer).expect("remove exact C02 test directory");
    }

    #[cfg(unix)]
    #[test]
    fn c02_shutdown_writer_binds_raw_final_metrics_to_a_nonhidden_marker() {
        use riley_server::engine::C02CaptureMetrics;

        let outer = c02_test_path("shutdown-writer");
        std::fs::create_dir(&outer).expect("create C02 writer parent");
        let evidence = outer.join("evidence");
        create_private_c02_test_directory(&evidence);
        let writer = super::C02ShutdownArtifactWriter::new(
            super::c02_open_audit_root(&evidence).expect("hold private audit root"),
            C02ShutdownArtifactOptions {
                basename: "shutdown.json".to_owned(),
            },
            C02AuditProcessIdentity {
                pid: 321,
                start_ticks: 4_242,
            },
        )
        .expect("construct C02 shutdown writer");

        // After launch-time validation, rebinding the visible path must not
        // redirect either the artifact or its marker.
        let rebound = outer.join("rebound");
        std::fs::rename(&evidence, &rebound).expect("rebind visible audit path");
        create_private_c02_test_directory(&evidence);
        writer
            .write(C02CaptureMetrics::default())
            .expect("write quiescent C02 shutdown evidence");
        assert!(
            writer.write(C02CaptureMetrics::default()).is_err(),
            "shutdown evidence is create-only"
        );
        assert!(!evidence.join("shutdown.json").exists());

        let artifact_path = rebound.join("shutdown.json");
        let raw = std::fs::read(&artifact_path).expect("read shutdown artifact");
        assert!(!raw.ends_with(b"\n"));
        assert_eq!(
            raw,
            c02_shutdown_artifact_bytes(
                "shutdown.json",
                C02AuditProcessIdentity {
                    pid: 321,
                    start_ticks: 4_242,
                },
                C02CaptureMetrics::default(),
            )
            .expect("canonical shutdown bytes")
        );
        let artifact: serde_json::Value =
            serde_json::from_slice(&raw).expect("decode shutdown artifact");
        assert_eq!(artifact.as_object().expect("artifact object").len(), 7);
        assert_eq!(artifact["schema_version"], C02_SHUTDOWN_SCHEMA_VERSION);
        assert_eq!(artifact["capture_status"], "captured");
        assert_eq!(artifact["qualification_status"], "not-run");
        assert_eq!(artifact["server_pid"], 321);
        assert_eq!(artifact["server_start_ticks"], 4_242);
        assert_eq!(artifact["worker_ready"], false);
        assert_eq!(
            artifact["final_metrics"]["schema_version"],
            "riley.c02-capture-metrics.v2"
        );

        let marker_name =
            c02_shutdown_completion_basename("shutdown.json").expect("safe completion basename");
        assert_eq!(marker_name, "shutdown.json.complete");
        let marker_raw = std::fs::read(rebound.join(&marker_name)).expect("read completion marker");
        assert!(!marker_raw.ends_with(b"\n"));
        assert_eq!(
            marker_raw,
            c02_shutdown_completion_marker_bytes("shutdown.json", &raw)
                .expect("canonical completion marker")
        );
        let marker: serde_json::Value =
            serde_json::from_slice(&marker_raw).expect("decode completion marker");
        assert_eq!(marker.as_object().expect("marker object").len(), 3);
        assert_eq!(
            marker["schema_version"],
            C02_SHUTDOWN_COMPLETION_SCHEMA_VERSION
        );
        assert_eq!(marker["artifact_filename"], "shutdown.json");
        assert_eq!(marker["artifact_sha256"], c02_sha256_hex(&raw));
        std::fs::remove_dir_all(&outer).expect("remove exact C02 test directory");
    }

    #[cfg(all(feature = "cuda", unix))]
    #[test]
    fn c02_generation_audit_writer_creates_a_bound_leaf_and_marker_once() {
        use riley_server::domain::{FinishReason, TokenUsage};
        use riley_server::engine::{
            C02CommittedSamplingSelection, C02GenerationAuditDeliveryMode,
            C02GenerationAuditRecord, C02GenerationAuditSink, C02GenerationAuditToken,
            C02GpuGreedyIneligibility, C02SamplingBackend,
        };

        let outer = c02_test_path("writer");
        std::fs::create_dir(&outer).expect("create C02 writer test parent");
        let evidence = outer.join("evidence");
        create_private_c02_test_directory(&evidence);
        let c02 = C02RuntimeConfigOptions {
            candidate_id: "riley-1.2.3-rc4".to_owned(),
            configuration_profile: C02ConfigurationProfile::MaxPerformanceExact,
            startup_artifact: outer.join("startup.json"),
        };
        let identity = super::C02RuntimeIdentity {
            configuration_profile: C02ConfigurationProfile::MaxPerformanceExact,
            configuration_sha256: "a".repeat(64),
        };
        {
            let writer = super::C02GenerationAuditWriter::new(
                super::c02_open_audit_root(&evidence).expect("hold C02 audit root"),
                &c02,
                &identity,
                C02AuditProcessIdentity {
                    pid: 321,
                    start_ticks: 4242,
                },
            )
            .expect("construct strict C02 writer");
            let record = C02GenerationAuditRecord {
                server_request_id: "cmpl-writer-test".to_owned(),
                delivery_mode: C02GenerationAuditDeliveryMode::NonStream,
                prompt_token_ids: vec![1],
                committed_output_tokens: vec![C02GenerationAuditToken {
                    token_id: 2,
                    emitted_text_delta: "x".to_owned(),
                }],
                sampling_selections: vec![C02CommittedSamplingSelection {
                    iteration_id: 7,
                    configured_backend: C02SamplingBackend::GpuGreedy,
                    selected_backend: C02SamplingBackend::CpuNormative,
                    ineligibility_reason: Some(C02GpuGreedyIneligibility::NonZeroTemperature),
                    committed: true,
                }],
                finish_reason: FinishReason::Length,
                usage: TokenUsage::new(1, 1).expect("small usage"),
            };
            let mut empty = record.clone();
            empty.server_request_id = "cmpl-empty-output".to_owned();
            empty.committed_output_tokens.clear();
            empty.sampling_selections.clear();
            assert!(writer.write_record(empty).is_err());
            assert!(!evidence.join("cmpl-empty-output.json").exists());
            writer
                .write_record(record.clone())
                .expect("create audit record");
            assert!(
                writer.write_record(record).is_err(),
                "a source-issued request ID cannot overwrite audit evidence"
            );
        }
        let raw_path = evidence.join("cmpl-writer-test.json");
        let raw = std::fs::read(&raw_path).expect("read created audit leaf");
        let decoded: serde_json::Value = serde_json::from_slice(&raw).expect("decode audit leaf");
        assert_eq!(decoded["process_identity"]["pid"], 321);
        assert_eq!(
            decoded["sampling_selections"][0]["selected_backend"],
            "cpu-normative"
        );
        assert_eq!(
            decoded["committed_output_tokens"][0]["emitted_text_delta"],
            "x"
        );
        let marker: serde_json::Value = serde_json::from_slice(
            &std::fs::read(evidence.join("cmpl-writer-test.json.complete"))
                .expect("read completion marker"),
        )
        .expect("decode completion marker");
        assert_eq!(marker["artifact_filename"], "cmpl-writer-test.json");
        assert_eq!(marker["artifact_sha256"], c02_sha256_hex(&raw));
        std::fs::remove_dir_all(&outer).expect("remove exact C02 writer test directory");
    }

    #[cfg(all(feature = "cuda", unix))]
    #[test]
    fn c02_generation_audit_writer_refuses_schema_oversize_without_evidence() {
        use riley_server::domain::{FinishReason, TokenUsage};
        use riley_server::engine::{
            C02CommittedSamplingSelection, C02GenerationAuditDeliveryMode,
            C02GenerationAuditRecord, C02GenerationAuditSink, C02GenerationAuditToken,
            C02GpuGreedyIneligibility, C02SamplingBackend,
        };

        fn selection() -> C02CommittedSamplingSelection {
            C02CommittedSamplingSelection {
                iteration_id: 7,
                configured_backend: C02SamplingBackend::GpuGreedy,
                selected_backend: C02SamplingBackend::CpuNormative,
                ineligibility_reason: Some(C02GpuGreedyIneligibility::NonZeroTemperature),
                committed: true,
            }
        }

        fn record(server_request_id: &str) -> C02GenerationAuditRecord {
            C02GenerationAuditRecord {
                server_request_id: server_request_id.to_owned(),
                delivery_mode: C02GenerationAuditDeliveryMode::NonStream,
                prompt_token_ids: vec![1],
                committed_output_tokens: vec![C02GenerationAuditToken {
                    token_id: 2,
                    emitted_text_delta: "x".to_owned(),
                }],
                sampling_selections: vec![selection()],
                finish_reason: FinishReason::Length,
                usage: TokenUsage::new(1, 1).expect("small usage"),
            }
        }

        let outer = c02_test_path("writer-oversize");
        std::fs::create_dir(&outer).expect("create C02 writer test parent");
        let evidence = outer.join("evidence");
        create_private_c02_test_directory(&evidence);
        let c02 = C02RuntimeConfigOptions {
            candidate_id: "riley-1.2.3-rc4".to_owned(),
            configuration_profile: C02ConfigurationProfile::MaxPerformanceExact,
            startup_artifact: outer.join("startup.json"),
        };
        let identity = super::C02RuntimeIdentity {
            configuration_profile: C02ConfigurationProfile::MaxPerformanceExact,
            configuration_sha256: "a".repeat(64),
        };
        let writer = super::C02GenerationAuditWriter::new(
            super::c02_open_audit_root(&evidence).expect("hold C02 audit root"),
            &c02,
            &identity,
            C02AuditProcessIdentity {
                pid: 321,
                start_ticks: 4242,
            },
        )
        .expect("construct strict C02 writer");

        let mut oversized_prompt = record("cmpl-oversized-prompt");
        oversized_prompt.prompt_token_ids = vec![1; C02_GENERATION_AUDIT_MAX_PROMPT_TOKEN_IDS + 1];

        let mut oversized_output = record("cmpl-oversized-output");
        oversized_output.committed_output_tokens = vec![
            C02GenerationAuditToken {
                token_id: 2,
                emitted_text_delta: "x".to_owned(),
            };
            C02_GENERATION_AUDIT_MAX_COMMITTED_OUTPUT_TOKENS
                + 1
        ];
        oversized_output.sampling_selections =
            vec![selection(); C02_GENERATION_AUDIT_MAX_COMMITTED_OUTPUT_TOKENS + 1];

        let mut oversized_selections = record("cmpl-oversized-selections");
        oversized_selections.sampling_selections =
            vec![selection(); C02_GENERATION_AUDIT_MAX_SAMPLING_SELECTIONS + 1];

        let mut oversized_delta = record("cmpl-oversized-delta");
        oversized_delta.committed_output_tokens[0].emitted_text_delta =
            "x".repeat(C02_GENERATION_AUDIT_MAX_EMITTED_TEXT_DELTA_CHARS + 1);

        for oversized in [
            oversized_prompt,
            oversized_output,
            oversized_selections,
            oversized_delta,
        ] {
            let record_basename =
                c02_generation_audit_record_basename(&oversized.server_request_id)
                    .expect("oversized test request ID is still a safe basename");
            assert!(writer.write_record(oversized).is_err());
            assert!(
                !evidence.join(&record_basename).exists(),
                "oversized record must not create its raw audit leaf"
            );
            assert!(
                !evidence
                    .join(
                        c02_generation_audit_completion_basename(&record_basename)
                            .expect("marker basename")
                    )
                    .exists(),
                "oversized record must not create its completion marker"
            );
        }

        std::fs::remove_dir_all(&outer).expect("remove exact C02 writer test directory");
    }

    #[test]
    fn version_and_help_are_exact_commands() {
        assert_eq!(
            parse_arguments(args(&["--version"])),
            Ok(CliCommand::Version)
        );
        assert_eq!(parse_arguments(args(&["-h"])), Ok(CliCommand::Help));
        assert!(USAGE.contains("--execution-completion MODE"));
        assert!(USAGE.contains("(default: iteration-batch)"));
        assert!(USAGE.contains("--reduction-profile ID"));
        assert!(USAGE.contains("(default: canonical-v1)"));
        assert!(USAGE.contains("--batch-shape-policy MODE"));
        assert!(USAGE.contains("(default: fixed-max)"));
        assert!(USAGE.contains("--batch-shape-buckets LIST"));
        assert!(USAGE.contains("--metadata-transport MODE"));
        assert!(USAGE.contains("(default: synchronous)"));
        assert!(USAGE.contains("--sampling-backend MODE"));
        assert!(USAGE.contains("(default: cpu)"));
        assert!(USAGE.contains("--c02-candidate-id ID"));
        assert!(USAGE.contains("--c02-configuration-profile ID"));
        assert!(USAGE.contains("--c02-startup-artifact PATH"));
        assert!(USAGE.contains("--c02-audit-dir PATH"));
        assert!(USAGE.contains("--c02-shutdown-artifact PATH"));
        assert!(parse_arguments(args(&["--version", "extra"])).is_err());
        assert!(parse_arguments(args(&[])).is_err());
    }

    #[test]
    fn serve_defaults_are_bounded_and_model_is_required() {
        assert_eq!(
            parse_arguments(args(&["serve", "--model", "/models/fixture"])),
            Ok(CliCommand::Serve(ServeOptions {
                model_path: PathBuf::from("/models/fixture"),
                model_id: None,
                bind_address: "127.0.0.1:8080".to_owned(),
                device_ordinal: 0,
                max_active_sequences: 8,
                max_waiting_requests: 64,
                max_sequence_tokens: None,
                max_output_tokens: None,
                batch_token_budget: 512,
                prefill_chunk_tokens: 512,
                physical_kv_blocks: None,
                residual_rmsnorm: ResidualRmsNormMode::Separate,
                execution_completion: ExecutionCompletionMode::IterationBatch,
                batch_shape_policy: BatchShapePolicyMode::FixedMaximum,
                batch_shape_buckets: None,
                metadata_transport: MetadataTransportMode::Synchronous,
                sampling_backend: SamplingBackendMode::Cpu,
                reduction_profile: ReductionProfileMode::CanonicalV1,
                max_weight_bytes: DEFAULT_MAX_WEIGHT_BYTES,
                shutdown_on_stdin: false,
                c02_runtime_config: None,
                c02_audit_dir: None,
                c02_shutdown_artifact: None,
            }))
        );
        assert!(parse_arguments(args(&["serve"])).is_err());
    }

    #[test]
    fn c02_runtime_identity_is_all_or_none_and_fail_closed() {
        let parsed = parse_arguments(args(&[
            "serve",
            "--model",
            "/models/fixture",
            "--c02-candidate-id",
            "riley-1.2.3-rc4",
            "--c02-configuration-profile",
            "stable-default",
            "--c02-startup-artifact",
            "/tmp/riley-c02-startup.json",
            "--c02-audit-dir",
            "/tmp/riley-c02-audit",
        ]));
        let Ok(CliCommand::Serve(options)) = parsed else {
            panic!("valid C02 identity must parse");
        };
        assert_eq!(
            options.c02_audit_dir,
            Some(PathBuf::from("/tmp/riley-c02-audit"))
        );
        let c02 = options
            .c02_runtime_config
            .expect("all C02 inputs must enable evidence mode");
        assert_eq!(c02.candidate_id, "riley-1.2.3-rc4");
        assert_eq!(
            c02.configuration_profile,
            C02ConfigurationProfile::StableDefault
        );
        assert_eq!(c02.configuration_profile.as_str(), "stable-default");
        assert_eq!(
            c02.startup_artifact,
            PathBuf::from("/tmp/riley-c02-startup.json")
        );

        for invalid in [
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc4",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-01.2.3-rc4",
                "--c02-configuration-profile",
                "stable-default",
                "--c02-startup-artifact",
                "/tmp/riley-c02-startup.json",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc0",
                "--c02-configuration-profile",
                "max-performance-exact",
                "--c02-startup-artifact",
                "relative.json",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-audit-dir",
                "/tmp/riley-c02-audit",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc4",
                "--c02-configuration-profile",
                "stable-default",
                "--c02-startup-artifact",
                "/tmp/riley-c02-startup.json",
                "--c02-audit-dir",
                "relative-audit",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc4",
                "--c02-configuration-profile",
                "unexpected",
                "--c02-startup-artifact",
                "/tmp/riley-c02-startup.json",
            ],
        ] {
            assert!(
                parse_arguments(args(&invalid)).is_err(),
                "invalid C02 inputs: {invalid:?}"
            );
        }
    }

    #[test]
    fn c02_shutdown_artifact_requires_one_safe_nonhidden_audit_child() {
        let parsed = parse_arguments(args(&[
            "serve",
            "--model",
            "/models/fixture",
            "--c02-candidate-id",
            "riley-1.2.3-rc4",
            "--c02-configuration-profile",
            "stable-default",
            "--c02-startup-artifact",
            "/tmp/riley-c02-startup.json",
            "--c02-audit-dir",
            "/tmp/riley-c02-audit",
            "--c02-shutdown-artifact",
            "/tmp/riley-c02-audit/shutdown.json",
        ]));
        let Ok(CliCommand::Serve(options)) = parsed else {
            panic!("valid C02 shutdown artifact must parse");
        };
        assert_eq!(
            options.c02_shutdown_artifact,
            Some(C02ShutdownArtifactOptions {
                basename: "shutdown.json".to_owned(),
            })
        );

        for invalid in [
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-shutdown-artifact",
                "/tmp/shutdown.json",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc4",
                "--c02-configuration-profile",
                "stable-default",
                "--c02-startup-artifact",
                "/tmp/riley-c02-startup.json",
                "--c02-shutdown-artifact",
                "/tmp/riley-c02-audit/shutdown.json",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc4",
                "--c02-configuration-profile",
                "stable-default",
                "--c02-startup-artifact",
                "/tmp/riley-c02-startup.json",
                "--c02-audit-dir",
                "/tmp/riley-c02-audit",
                "--c02-shutdown-artifact",
                "/tmp/riley-c02-audit/nested/shutdown.json",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc4",
                "--c02-configuration-profile",
                "stable-default",
                "--c02-startup-artifact",
                "/tmp/riley-c02-startup.json",
                "--c02-audit-dir",
                "/tmp/riley-c02-audit",
                "--c02-shutdown-artifact",
                "/tmp/riley-c02-audit/.shutdown.json",
            ],
            vec![
                "serve",
                "--model",
                "/models/fixture",
                "--c02-candidate-id",
                "riley-1.2.3-rc4",
                "--c02-configuration-profile",
                "stable-default",
                "--c02-startup-artifact",
                "/tmp/riley-c02-startup.json",
                "--c02-audit-dir",
                "/tmp/riley-c02-audit",
                "--c02-shutdown-artifact",
                "/tmp/riley-c02-audit/shutdown.txt",
            ],
        ] {
            assert!(
                parse_arguments(args(&invalid)).is_err(),
                "invalid C02 shutdown artifact arguments: {invalid:?}"
            );
        }
    }

    #[test]
    fn c02_receipt_bytes_and_startup_artifact_are_canonical_and_create_only() {
        use std::collections::BTreeMap;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
        let c02 = C02RuntimeConfigOptions {
            candidate_id: "riley-1.2.3-rc4".to_owned(),
            configuration_profile: C02ConfigurationProfile::StableDefault,
            startup_artifact: std::env::temp_dir().join(format!(
                "riley-c02-runtime-config-{}-{}",
                std::process::id(),
                NEXT_PATH.fetch_add(1, Ordering::AcqRel),
            )),
        };
        let launch_arguments = vec![
            OsString::from("serve"),
            OsString::from("--model"),
            OsString::from("/model"),
        ];
        let launch_environment = vec![
            (OsString::from("BETA"), OsString::from("two")),
            (OsString::from("ALPHA"), OsString::from("one")),
        ];
        let identity = c02_runtime_identity(
            c02.configuration_profile,
            &launch_arguments,
            &launch_environment,
        )
        .expect("canonical launch identity");
        assert_eq!(
            identity.configuration_sha256,
            c02_sha256_hex(
                b"{\"argv\":[\"serve\",\"--model\",\"/model\"],\"environment\":{\"ALPHA\":\"one\",\"BETA\":\"two\"}}"
            )
        );
        assert!(
            c02_runtime_identity(
                c02.configuration_profile,
                &launch_arguments,
                &[(
                    OsString::from("RILEY_FREEZE_SHA"),
                    OsString::from("forbidden"),
                )],
            )
            .is_err()
        );
        assert!(
            c02_runtime_identity(
                c02.configuration_profile,
                &launch_arguments,
                &[(OsString::from("lowercase"), OsString::from("rejected"))],
            )
            .is_err()
        );

        let effective_config = BTreeMap::from([(
            "batch_token_budget".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(8_usize)),
        )]);
        let receipt = c02_endpoint_receipt(&c02, &identity, &effective_config)
            .expect("canonical endpoint receipt");
        assert!(!receipt.bytes.ends_with(b"\n"));
        assert_eq!(
            receipt.bytes,
            c02_canonical_json_bytes(&receipt.document).expect("canonical endpoint bytes")
        );
        let endpoint: serde_json::Value =
            serde_json::from_slice(&receipt.bytes).expect("decode endpoint receipt");
        assert_eq!(endpoint["candidate_id"], c02.candidate_id);
        assert_eq!(
            endpoint["runtime_identity"]["configuration_sha256"],
            identity.configuration_sha256
        );

        write_c02_startup_artifact(&c02.startup_artifact, &c02, &receipt)
            .expect("write create-only C02 artifact");
        assert!(write_c02_startup_artifact(&c02.startup_artifact, &c02, &receipt).is_err());
        let artifact_raw = std::fs::read(&c02.startup_artifact).expect("read C02 artifact");
        assert!(!artifact_raw.ends_with(b"\n"));
        let artifact: serde_json::Value =
            serde_json::from_slice(&artifact_raw).expect("decode C02 artifact");
        assert_eq!(artifact["endpoint_path"], "/v1/config");
        assert_eq!(
            artifact["endpoint_payload_sha256"],
            c02_sha256_hex(&receipt.bytes)
        );
        assert_eq!(
            artifact["endpoint_payload"],
            serde_json::to_value(&receipt.document).expect("encode endpoint document")
        );
        std::fs::remove_file(&c02.startup_artifact).expect("remove exact test artifact");

        assert_eq!(
            c02_utc_timestamp_from_unix_seconds(0).expect("epoch timestamp"),
            "1970-01-01T00:00:00Z"
        );
        assert_eq!(
            c02_utc_timestamp_from_unix_seconds(1_709_164_800).expect("leap timestamp"),
            "2024-02-29T00:00:00Z"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn serve_explicit_options_and_duplicate_rejection_are_deterministic() {
        let parsed = parse_arguments(args(&[
            "serve",
            "--model",
            "/m",
            "--model-id",
            "fixture",
            "--bind",
            "0.0.0.0:9000",
            "--device",
            "2",
            "--max-active-sequences",
            "4",
            "--max-waiting-requests",
            "9",
            "--max-sequence-tokens",
            "2048",
            "--max-output-tokens",
            "128",
            "--batch-token-budget",
            "64",
            "--prefill-chunk-tokens",
            "32",
            "--kv-blocks",
            "512",
            "--residual-rmsnorm",
            "separate",
            "--execution-completion",
            "iteration-batch",
            "--batch-shape-policy",
            "power-of-two",
            "--batch-shape-buckets",
            "1,2,4,8,16,32,64",
            "--metadata-transport",
            "packed-async",
            "--sampling-backend",
            "gpu-greedy",
            "--reduction-profile",
            "fixed-contiguous-37-balanced-v1",
            "--max-weight-bytes",
            "4096",
            "--shutdown-on-stdin",
        ]));
        assert_eq!(
            parsed,
            Ok(CliCommand::Serve(ServeOptions {
                model_path: PathBuf::from("/m"),
                model_id: Some("fixture".to_owned()),
                bind_address: "0.0.0.0:9000".to_owned(),
                device_ordinal: 2,
                max_active_sequences: 4,
                max_waiting_requests: 9,
                max_sequence_tokens: Some(2048),
                max_output_tokens: Some(128),
                batch_token_budget: 64,
                prefill_chunk_tokens: 32,
                physical_kv_blocks: Some(512),
                residual_rmsnorm: ResidualRmsNormMode::Separate,
                execution_completion: ExecutionCompletionMode::IterationBatch,
                batch_shape_policy: BatchShapePolicyMode::PowerOfTwo,
                batch_shape_buckets: Some(vec![1, 2, 4, 8, 16, 32, 64]),
                metadata_transport: MetadataTransportMode::PackedAsync,
                sampling_backend: SamplingBackendMode::GpuGreedy,
                reduction_profile: ReductionProfileMode::FixedContiguous37BalancedV1,
                max_weight_bytes: 4096,
                shutdown_on_stdin: true,
                c02_runtime_config: None,
                c02_audit_dir: None,
                c02_shutdown_artifact: None,
            }))
        );
        assert!(parse_arguments(args(&["serve", "--model", "/a", "--model", "/b"])).is_err());
        assert!(parse_arguments(args(&["serve", "--model", "/a", "--bogus"])).is_err());
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--metadata-transport",
                "unknown",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--metadata-transport",
                "synchronous",
                "--metadata-transport",
                "packed-async",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--sampling-backend",
                "unknown",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--sampling-backend",
                "cpu",
                "--sampling-backend",
                "gpu-greedy",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--residual-rmsnorm",
                "unknown",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--residual-rmsnorm",
                "fused",
                "--execution-completion",
                "iteration-batch",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--execution-completion",
                "unknown",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--batch-shape-policy",
                "unknown",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--batch-shape-policy",
                "fixed-max",
                "--batch-shape-policy",
                "fixed-max",
            ]))
            .is_err()
        );
    }

    #[test]
    fn custom_batch_shape_buckets_require_active_policy_and_exact_budget_bound() {
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--batch-shape-buckets",
                "1,2,4,512",
            ]))
            .is_err()
        );
        assert_eq!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--batch-token-budget",
                "7",
                "--batch-shape-policy",
                "power-of-two",
                "--batch-shape-buckets",
                "1,3,7",
            ])),
            Ok(CliCommand::Serve(ServeOptions {
                model_path: PathBuf::from("/a"),
                model_id: None,
                bind_address: "127.0.0.1:8080".to_owned(),
                device_ordinal: 0,
                max_active_sequences: 8,
                max_waiting_requests: 64,
                max_sequence_tokens: None,
                max_output_tokens: None,
                batch_token_budget: 7,
                prefill_chunk_tokens: 512,
                physical_kv_blocks: None,
                residual_rmsnorm: ResidualRmsNormMode::Separate,
                execution_completion: ExecutionCompletionMode::IterationBatch,
                batch_shape_policy: BatchShapePolicyMode::PowerOfTwo,
                batch_shape_buckets: Some(vec![1, 3, 7]),
                metadata_transport: MetadataTransportMode::Synchronous,
                sampling_backend: SamplingBackendMode::Cpu,
                reduction_profile: ReductionProfileMode::CanonicalV1,
                max_weight_bytes: DEFAULT_MAX_WEIGHT_BYTES,
                shutdown_on_stdin: false,
                c02_runtime_config: None,
                c02_audit_dir: None,
                c02_shutdown_artifact: None,
            }))
        );

        for invalid in [
            "",
            "0,7",
            "2,7",
            "1,2,2,7",
            "1,4,2,7",
            "1,2,4",
            "1,2,3,4,5,6,7,8,9,10,11",
            "1,,7",
        ] {
            assert!(
                parse_arguments(args(&[
                    "serve",
                    "--model",
                    "/a",
                    "--batch-token-budget",
                    "7",
                    "--batch-shape-policy",
                    "power-of-two",
                    "--batch-shape-buckets",
                    invalid,
                ]))
                .is_err(),
                "invalid bucket list {invalid:?}"
            );
        }
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--batch-token-budget",
                "7",
                "--batch-shape-policy",
                "power-of-two",
                "--batch-shape-buckets",
                "1,7",
                "--batch-shape-buckets",
                "1,7",
            ]))
            .is_err()
        );
    }

    #[test]
    fn reduction_profile_values_and_duplicates_fail_closed() {
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--reduction-profile",
                "unknown",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--reduction-profile",
                "canonical-v1",
                "--reduction-profile",
                "canonical-v1",
            ]))
            .is_err()
        );
    }

    #[test]
    fn fused_residual_norm_requires_explicit_per_operation_completion() {
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--residual-rmsnorm",
                "fused",
            ]))
            .is_err()
        );
        assert_eq!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--residual-rmsnorm",
                "fused",
                "--execution-completion",
                "per-operation",
            ])),
            Ok(CliCommand::Serve(ServeOptions {
                model_path: PathBuf::from("/a"),
                model_id: None,
                bind_address: "127.0.0.1:8080".to_owned(),
                device_ordinal: 0,
                max_active_sequences: 8,
                max_waiting_requests: 64,
                max_sequence_tokens: None,
                max_output_tokens: None,
                batch_token_budget: 512,
                prefill_chunk_tokens: 512,
                physical_kv_blocks: None,
                residual_rmsnorm: ResidualRmsNormMode::Fused,
                execution_completion: ExecutionCompletionMode::PerOperation,
                batch_shape_policy: BatchShapePolicyMode::FixedMaximum,
                batch_shape_buckets: None,
                metadata_transport: MetadataTransportMode::Synchronous,
                sampling_backend: SamplingBackendMode::Cpu,
                reduction_profile: ReductionProfileMode::CanonicalV1,
                max_weight_bytes: DEFAULT_MAX_WEIGHT_BYTES,
                shutdown_on_stdin: false,
                c02_runtime_config: None,
                c02_audit_dir: None,
                c02_shutdown_artifact: None,
            }))
        );
        assert!(
            parse_arguments(args(&[
                "serve",
                "--model",
                "/a",
                "--execution-completion",
                "per-operation",
                "--metadata-transport",
                "packed-async",
            ]))
            .is_err()
        );
    }

    #[test]
    fn fixed37_profile_rejects_an_advertised_context_above_its_ragged_limit() {
        validate_reduction_profile_context(
            ReductionProfileMode::FixedContiguous37BalancedV1,
            FIXED37_MAX_SEQUENCE_TOKENS,
        )
        .expect("the exact fixed37 boundary is supported");
        assert!(
            validate_reduction_profile_context(
                ReductionProfileMode::FixedContiguous37BalancedV1,
                FIXED37_MAX_SEQUENCE_TOKENS + 1,
            )
            .is_err()
        );
        validate_reduction_profile_context(
            ReductionProfileMode::CanonicalV1,
            FIXED37_MAX_SEQUENCE_TOKENS + 1,
        )
        .expect("canonical profile retains the model context bound");
    }

    #[cfg(unix)]
    #[test]
    fn shutdown_metrics_path_is_absolute_and_create_only() {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
        assert_eq!(validate_shutdown_metrics_path(None), Ok(None));
        assert!(validate_shutdown_metrics_path(Some(OsString::new())).is_err());
        assert!(validate_shutdown_metrics_path(Some(OsString::from("relative.json"))).is_err());

        let path = std::env::temp_dir().join(format!(
            "riley-shutdown-metrics-{}-{}",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::AcqRel)
        ));
        assert_eq!(
            validate_shutdown_metrics_path(Some(path.clone().into_os_string())),
            Ok(Some(path.clone()))
        );
        let snapshot = riley_server::service::OperationalMetricsSnapshot::default();
        write_shutdown_metrics(&path, &snapshot).expect("create shutdown metrics");
        assert!(write_shutdown_metrics(&path, &snapshot).is_err());
        let encoded = std::fs::read(&path).expect("read shutdown metrics");
        let decoded: serde_json::Value =
            serde_json::from_slice(&encoded).expect("decode shutdown metrics");
        assert_eq!(decoded["active_requests"], 0);
        assert_eq!(decoded["counters"]["dropped_observations"], 0);
        assert!(decoded["counters"].get("dropped_samples").is_none());
        std::fs::remove_file(&path).expect("remove exact test artifact");
    }
}
