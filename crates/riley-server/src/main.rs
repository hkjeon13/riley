use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;

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
  --shutdown-on-stdin            gracefully stop after one input line or EOF
";

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

fn main() -> ExitCode {
    match parse_arguments(env::args_os().skip(1)) {
        Ok(CliCommand::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(CliCommand::Version) => print_version(),
        Ok(CliCommand::Serve(options)) => match run_serve(options) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("riley: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("riley: {error}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
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

    Ok(CliCommand::Serve(ServeOptions {
        model_path: model_path.ok_or_else(|| "serve requires --model PATH".to_owned())?,
        model_id,
        bind_address: bind_address.unwrap_or_else(|| "127.0.0.1:8080".to_owned()),
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
fn run_serve(options: ServeOptions) -> Result<(), String> {
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
    );
    Err("serve requires a build with --features server,cuda".to_owned())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_lines)]
fn run_serve(options: ServeOptions) -> Result<(), String> {
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
    use riley_server::service::{CompletionBackend, ServerConfig, start_server};

    #[cfg(not(unix))]
    return Err("serve currently requires POSIX SIGINT/SIGTERM support".to_owned());
    #[cfg(unix)]
    let shutdown_signals = Arc::new(
        signal::ShutdownSignals::block()
            .map_err(|error| format!("could not block shutdown signals: {error}"))?,
    );
    let shutdown_metrics_path = shutdown_metrics_path_from_env()?;

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
    let server = start_server(server_config, backend)
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
        BatchShapePolicyMode, CliCommand, DEFAULT_MAX_WEIGHT_BYTES, ExecutionCompletionMode,
        FIXED37_MAX_SEQUENCE_TOKENS, MetadataTransportMode, ReductionProfileMode,
        ResidualRmsNormMode, SamplingBackendMode, ServeOptions, USAGE, parse_arguments,
        validate_reduction_profile_context, validate_shutdown_metrics_path, write_shutdown_metrics,
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
            }))
        );
        assert!(parse_arguments(args(&["serve"])).is_err());
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
