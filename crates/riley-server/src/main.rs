#[cfg(any(feature = "cuda", test))]
use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;

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
    use std::net::SocketAddr;
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
    use riley_server::service::{
        CompletionBackend, ServerConfig, start_server_with_runtime_config,
    };

    #[cfg(not(unix))]
    return Err("serve currently requires POSIX SIGINT/SIGTERM support".to_owned());
    #[cfg(unix)]
    let shutdown_signals = Arc::new(
        signal::ShutdownSignals::block()
            .map_err(|error| format!("could not block shutdown signals: {error}"))?,
    );
    let shutdown_metrics_path = shutdown_metrics_path_from_env()?;
    let c02_runtime_config = options.c02_runtime_config;

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
    let c02_receipt = match c02_runtime_config.as_ref() {
        Some(c02) => {
            let facts = resources.effective_runtime_facts();
            let effective_config = c02_effective_config_from_facts(&facts)?;
            let runtime_identity = c02_runtime_identity(
                c02.configuration_profile,
                launch_arguments,
                launch_environment,
            )?;
            let receipt = c02_endpoint_receipt(c02, &runtime_identity, &effective_config)?;
            Some(receipt)
        }
        None => None,
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
    let server = start_server_with_runtime_config(server_config, backend, runtime_config_body)
        .map_err(|error| format!("HTTP server startup failed: {error}"))?;
    println!(
        "riley listening on http://{} (graceful_signals=SIGINT,SIGTERM graceful_stdin_shutdown={})",
        server.local_address(),
        options.shutdown_on_stdin
    );

    let trigger = wait_for_shutdown(Arc::clone(&shutdown_signals), options.shutdown_on_stdin)?;
    println!("riley graceful shutdown requested by {trigger}");
    match shutdown_metrics_path {
        Some(path) => {
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
        None => server
            .shutdown()
            .map_err(|error| format!("graceful shutdown failed: {error}")),
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
        BatchShapePolicyMode, C02ConfigurationProfile, C02RuntimeConfigOptions, CliCommand,
        DEFAULT_MAX_WEIGHT_BYTES, ExecutionCompletionMode, FIXED37_MAX_SEQUENCE_TOKENS,
        MetadataTransportMode, ReductionProfileMode, ResidualRmsNormMode, SamplingBackendMode,
        ServeOptions, USAGE, c02_canonical_json_bytes, c02_endpoint_receipt, c02_runtime_identity,
        c02_sha256_hex, c02_utc_timestamp_from_unix_seconds, parse_arguments,
        validate_reduction_profile_context, validate_shutdown_metrics_path,
        write_c02_startup_artifact, write_shutdown_metrics,
    };

    fn args<'a>(values: &'a [&'a str]) -> impl Iterator<Item = OsString> + 'a {
        values.iter().map(OsString::from)
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
        ]));
        let Ok(CliCommand::Serve(options)) = parsed else {
            panic!("valid C02 identity must parse");
        };
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
