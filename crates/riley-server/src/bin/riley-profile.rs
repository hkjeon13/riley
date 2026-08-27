//! Python-free native paired-profile evidence runner.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use riley_model::{EncodeOptions, LoadLimits, LoadedModel, TOKENIZER_FILENAME};
use riley_runtime::llama::{
    LlamaBatchMetadataConfig, PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use riley_runtime::paged_kv::KV_BLOCK_SIZE;
use riley_scheduler::{OverloadPolicy, SchedulerConfig};
use riley_server::benchmark::{
    NativeBenchmarkConfig, NativeBenchmarkExecutor, NativeBenchmarkIterationTrace,
    NativeBenchmarkRequestTrace, NativeBenchmarkTrialTrace, NativeGpuTimingValidity,
    PretokenizedBenchmarkRequest,
};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_WEIGHT_BYTES: u64 = 2 * 1_024 * 1_024 * 1_024;
const CANONICAL_BATCH_TOKEN_BUDGET: usize = 512;
const MAX_PROMPT_CORPUS_BYTES: u64 = 16 * 1_024 * 1_024;
const MAX_CONCURRENCY: usize = 8;
const MAX_PROMPT_TOKENS: usize = 8_192;
const MIN_OUTPUT_TOKENS: usize = 2;
const MAX_OUTPUT_TOKENS: usize = 512;
const MAX_TRIAL_COUNT: usize = 100;
const SCHEMA_VERSION: &str = "riley.native-profile-run.v1";
const PRIMARY_METRIC: &str = "aggregate.host.execute_ns";
const RESIDUAL_RMSNORM_CORRECTNESS_GATE: &str = "pr15-fused-residual-rmsnorm-exact-v1";
const EXECUTION_COMPLETION_CORRECTNESS_GATE: &str = "pr15-iteration-command-batch-exact-v1";

const USAGE: &str = "\
usage: riley-profile [options]

native execution:
  --model PATH
  --prompts PATH
  --output PATH|-                    stdout when omitted or '-'

run identity:
  --role baseline|candidate
  --pair-index 1..5
  --run-id ID
  --recorded-at-utc RFC3339-UTC

source provenance (all required):
  --git-commit SHA                   40 or 64 lowercase hex characters
  --git-dirty false
  --executable-sha256 SHA256
  --implementation-id ID
  --runtime-flag-name residual_rmsnorm|execution_completion
  --runtime-flag-value separate|fused|per-operation|iteration-batch
  --semantic-class E0
  --correctness-gate-id ID
  --correctness-report-sha256 SHA256

GPU/host/software environment (all required):
  --gpu-model TEXT --gpu-uuid TEXT --device-index N --gpu-pci-bus-id TEXT
  --gpu-compute-capability M.m --gpu-vram-bytes N
  --environment-id ID --cpu-model TEXT --physical-core-count N
  --logical-core-count N --ram-bytes N --os-release TEXT
  --kernel-release TEXT --architecture TEXT
  --nvidia-driver-version TEXT --cuda-runtime-version TEXT
  --cuda-toolkit-version TEXT --cublas-version TEXT
  --container-image-sha256 SHA256

workload (all required):
  --workload-id ID --model-id TEXT --model-revision TEXT
  --weights-sha256 SHA256 --tokenizer-sha256 SHA256 --dtype TEXT
  --concurrency N --prompt-tokens N --output-tokens N
  --warmups N --measured-iterations N --sampling-id greedy --seed none
";

const KNOWN_FLAGS: &[&str] = &[
    "--model",
    "--prompts",
    "--output",
    "--role",
    "--pair-index",
    "--run-id",
    "--recorded-at-utc",
    "--git-commit",
    "--git-dirty",
    "--executable-sha256",
    "--implementation-id",
    "--runtime-flag-name",
    "--runtime-flag-value",
    "--semantic-class",
    "--correctness-gate-id",
    "--correctness-report-sha256",
    "--gpu-model",
    "--gpu-uuid",
    "--device-index",
    "--gpu-pci-bus-id",
    "--gpu-compute-capability",
    "--gpu-vram-bytes",
    "--environment-id",
    "--cpu-model",
    "--physical-core-count",
    "--logical-core-count",
    "--ram-bytes",
    "--os-release",
    "--kernel-release",
    "--architecture",
    "--nvidia-driver-version",
    "--cuda-runtime-version",
    "--cuda-toolkit-version",
    "--cublas-version",
    "--container-image-sha256",
    "--workload-id",
    "--model-id",
    "--model-revision",
    "--weights-sha256",
    "--tokenizer-sha256",
    "--dtype",
    "--concurrency",
    "--prompt-tokens",
    "--output-tokens",
    "--warmups",
    "--measured-iterations",
    "--sampling-id",
    "--seed",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    Baseline,
    Candidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidualRmsNormMode {
    Separate,
    Fused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutionCompletionMode {
    PerOperation,
    IterationBatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeSelection {
    ResidualRmsNorm(ResidualRmsNormMode),
    ExecutionCompletion(ExecutionCompletionMode),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RuntimeFlag {
    name: String,
    value: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceProvenance {
    git_commit: String,
    git_dirty: bool,
    executable_sha256: String,
    implementation_id: String,
    runtime_flag: RuntimeFlag,
    semantic_class: String,
    correctness_gate_id: String,
    correctness_report_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct GpuEnvironment {
    model: String,
    uuid: String,
    device_index: u32,
    pci_bus_id: String,
    compute_capability: String,
    vram_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct HostEnvironment {
    environment_id: String,
    cpu_model: String,
    physical_core_count: usize,
    logical_core_count: usize,
    ram_bytes: u64,
    os_release: String,
    kernel_release: String,
    architecture: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SoftwareEnvironment {
    nvidia_driver_version: String,
    cuda_runtime_version: String,
    cuda_toolkit_version: String,
    cublas_version: String,
    container_image_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Environment {
    gpu: GpuEnvironment,
    host: HostEnvironment,
    software: SoftwareEnvironment,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
// These names are part of the versioned native-profile JSON contract.
#[allow(clippy::struct_field_names)]
struct Workload {
    workload_id: String,
    model_id: String,
    model_revision: String,
    weights_sha256: String,
    tokenizer_sha256: String,
    dtype: String,
    concurrency: usize,
    prompt_tokens: usize,
    output_tokens: usize,
    warmups: usize,
    measured_iterations: usize,
    sampling_id: String,
    seed: Option<u64>,
}

#[derive(Debug, Eq, PartialEq)]
struct Options {
    model_path: PathBuf,
    prompts_path: PathBuf,
    output_path: Option<PathBuf>,
    role: Role,
    pair_index: u8,
    run_id: String,
    recorded_at_utc: String,
    source: SourceProvenance,
    environment: Environment,
    workload: Workload,
}

impl Options {
    fn runtime_selection(&self) -> Result<RuntimeSelection, String> {
        match (
            self.source.runtime_flag.name.as_str(),
            self.source.runtime_flag.value.as_str(),
        ) {
            ("residual_rmsnorm", "separate") => Ok(RuntimeSelection::ResidualRmsNorm(
                ResidualRmsNormMode::Separate,
            )),
            ("residual_rmsnorm", "fused") => Ok(RuntimeSelection::ResidualRmsNorm(
                ResidualRmsNormMode::Fused,
            )),
            ("execution_completion", "per-operation") => Ok(RuntimeSelection::ExecutionCompletion(
                ExecutionCompletionMode::PerOperation,
            )),
            ("execution_completion", "iteration-batch") => Ok(
                RuntimeSelection::ExecutionCompletion(ExecutionCompletionMode::IterationBatch),
            ),
            _ => Err("runtime flag must be residual_rmsnorm=separate|fused or \
                 execution_completion=per-operation|iteration-batch"
                .to_owned()),
        }
    }

    fn validate_runtime_binding(&self) -> Result<(), String> {
        let runtime_selection = self.runtime_selection()?;
        let expected_correctness_gate = match runtime_selection {
            RuntimeSelection::ResidualRmsNorm(_) => RESIDUAL_RMSNORM_CORRECTNESS_GATE,
            RuntimeSelection::ExecutionCompletion(_) => EXECUTION_COMPLETION_CORRECTNESS_GATE,
        };
        if self.source.correctness_gate_id != expected_correctness_gate {
            return Err(format!(
                "--correctness-gate-id must be {expected_correctness_gate} for the selected runtime flag"
            ));
        }
        match (self.role, runtime_selection) {
            (
                Role::Baseline,
                RuntimeSelection::ResidualRmsNorm(ResidualRmsNormMode::Separate)
                | RuntimeSelection::ExecutionCompletion(ExecutionCompletionMode::PerOperation),
            )
            | (
                Role::Candidate,
                RuntimeSelection::ResidualRmsNorm(ResidualRmsNormMode::Fused)
                | RuntimeSelection::ExecutionCompletion(ExecutionCompletionMode::IterationBatch),
            ) => Ok(()),
            _ => Err(
                "runtime flag must bind baseline/candidate to separate/fused or \
                 per-operation/iteration-batch"
                    .to_owned(),
            ),
        }
    }

    fn validate(&self) -> Result<(), String> {
        validate_id("--run-id", &self.run_id)?;
        validate_rfc3339_utc(&self.recorded_at_utc)?;
        validate_git_commit(&self.source.git_commit)?;
        if self.source.git_dirty {
            return Err("--git-dirty must be false for source-bound evidence".to_owned());
        }
        validate_sha256("--executable-sha256", &self.source.executable_sha256)?;
        validate_id("--implementation-id", &self.source.implementation_id)?;
        validate_id("--runtime-flag-name", &self.source.runtime_flag.name)?;
        if self.source.semantic_class != "E0" {
            return Err("--semantic-class must be E0".to_owned());
        }
        validate_id("--correctness-gate-id", &self.source.correctness_gate_id)?;
        validate_sha256(
            "--correctness-report-sha256",
            &self.source.correctness_report_sha256,
        )?;
        self.validate_runtime_binding()?;

        validate_nonempty_environment(&self.environment)?;
        validate_id("--environment-id", &self.environment.host.environment_id)?;
        validate_compute_capability(&self.environment.gpu.compute_capability)?;
        if self.environment.host.logical_core_count < self.environment.host.physical_core_count {
            return Err("--logical-core-count must be >= --physical-core-count".to_owned());
        }
        validate_sha256(
            "--container-image-sha256",
            &self.environment.software.container_image_sha256,
        )?;

        validate_id("--workload-id", &self.workload.workload_id)?;
        validate_sha256("--weights-sha256", &self.workload.weights_sha256)?;
        validate_sha256("--tokenizer-sha256", &self.workload.tokenizer_sha256)?;
        for (name, value) in [
            ("--model-id", self.workload.model_id.as_str()),
            ("--model-revision", self.workload.model_revision.as_str()),
            ("--dtype", self.workload.dtype.as_str()),
        ] {
            validate_nonempty(name, value)?;
        }
        if self.workload.concurrency == 0 || self.workload.prompt_tokens == 0 {
            return Err("concurrency and prompt tokens must be positive".to_owned());
        }
        if self.workload.output_tokens < MIN_OUTPUT_TOKENS {
            return Err(
                "native profile output tokens must include at least one decode step".to_owned(),
            );
        }
        if self.workload.warmups < 5
            || self.workload.measured_iterations < 30
            || self.workload.warmups > MAX_TRIAL_COUNT
            || self.workload.measured_iterations > MAX_TRIAL_COUNT
        {
            return Err(
                "native profile requires 5..=100 warmups and 30..=100 measured iterations"
                    .to_owned(),
            );
        }
        if self.workload.concurrency > MAX_CONCURRENCY
            || self.workload.prompt_tokens > MAX_PROMPT_TOKENS
            || self.workload.output_tokens > MAX_OUTPUT_TOKENS
        {
            return Err("native profile workload exceeds its v1 resource bounds".to_owned());
        }
        let sequence_tokens = self
            .workload
            .prompt_tokens
            .checked_add(self.workload.output_tokens)
            .ok_or_else(|| "prompt plus output token count overflowed".to_owned())?;
        if sequence_tokens > u32::MAX as usize {
            return Err("prompt plus output tokens must fit u32".to_owned());
        }
        if self.workload.sampling_id != "greedy" || self.workload.seed.is_some() {
            return Err("native profile requires --sampling-id greedy --seed none".to_owned());
        }
        Ok(())
    }
}

enum Command {
    Help,
    Run(Box<Options>),
}

fn main() -> ExitCode {
    match parse_arguments(env::args_os().skip(1)) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Run(options)) => match run_profile(*options) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("riley-profile: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("riley-profile: {error}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[allow(clippy::too_many_lines)]
fn parse_arguments(arguments: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut arguments = arguments.into_iter();
    let Some(first) = arguments.next() else {
        return Err("options are required".to_owned());
    };
    if first == "--help" || first == "-h" {
        if let Some(extra) = arguments.next() {
            return Err(format!(
                "unexpected argument {} after --help",
                display_argument(&extra)
            ));
        }
        return Ok(Command::Help);
    }

    let mut values = BTreeMap::new();
    let mut pending = Some(first);
    loop {
        let Some(flag) = pending.take().or_else(|| arguments.next()) else {
            break;
        };
        let flag = flag
            .into_string()
            .map_err(|_| "option names must be UTF-8".to_owned())?;
        if !KNOWN_FLAGS.contains(&flag.as_str()) {
            return Err(format!("unknown option {flag}"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?
            .into_string()
            .map_err(|_| format!("{flag} requires a UTF-8 value"))?;
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("{flag} may occur only once"));
        }
    }

    let output = values.remove("--output").unwrap_or_else(|| "-".to_owned());
    let role = parse_role(&take_required(&mut values, "--role")?)?;
    let git_dirty = match take_required(&mut values, "--git-dirty")?.as_str() {
        "false" => false,
        "true" => true,
        _ => return Err("--git-dirty requires true or false".to_owned()),
    };
    let runtime_flag_value = take_required(&mut values, "--runtime-flag-value")?;
    if !matches!(
        runtime_flag_value.as_str(),
        "separate" | "fused" | "per-operation" | "iteration-batch"
    ) {
        return Err(
            "--runtime-flag-value requires separate, fused, per-operation, or iteration-batch"
                .to_owned(),
        );
    }
    let seed = match take_required(&mut values, "--seed")?.as_str() {
        "none" => None,
        _ => return Err("--seed must be none for canonical greedy sampling".to_owned()),
    };

    let options = Options {
        model_path: PathBuf::from(take_required(&mut values, "--model")?),
        prompts_path: PathBuf::from(take_required(&mut values, "--prompts")?),
        output_path: (output != "-").then(|| PathBuf::from(output)),
        role,
        pair_index: parse_number(&take_required(&mut values, "--pair-index")?, "--pair-index")?,
        run_id: take_required(&mut values, "--run-id")?,
        recorded_at_utc: take_required(&mut values, "--recorded-at-utc")?,
        source: SourceProvenance {
            git_commit: take_required(&mut values, "--git-commit")?,
            git_dirty,
            executable_sha256: take_required(&mut values, "--executable-sha256")?,
            implementation_id: take_required(&mut values, "--implementation-id")?,
            runtime_flag: RuntimeFlag {
                name: take_required(&mut values, "--runtime-flag-name")?,
                value: runtime_flag_value,
            },
            semantic_class: take_required(&mut values, "--semantic-class")?,
            correctness_gate_id: take_required(&mut values, "--correctness-gate-id")?,
            correctness_report_sha256: take_required(&mut values, "--correctness-report-sha256")?,
        },
        environment: Environment {
            gpu: GpuEnvironment {
                model: take_required(&mut values, "--gpu-model")?,
                uuid: take_required(&mut values, "--gpu-uuid")?,
                device_index: parse_number(
                    &take_required(&mut values, "--device-index")?,
                    "--device-index",
                )?,
                pci_bus_id: take_required(&mut values, "--gpu-pci-bus-id")?,
                compute_capability: take_required(&mut values, "--gpu-compute-capability")?,
                vram_bytes: parse_number(
                    &take_required(&mut values, "--gpu-vram-bytes")?,
                    "--gpu-vram-bytes",
                )?,
            },
            host: HostEnvironment {
                environment_id: take_required(&mut values, "--environment-id")?,
                cpu_model: take_required(&mut values, "--cpu-model")?,
                physical_core_count: parse_number(
                    &take_required(&mut values, "--physical-core-count")?,
                    "--physical-core-count",
                )?,
                logical_core_count: parse_number(
                    &take_required(&mut values, "--logical-core-count")?,
                    "--logical-core-count",
                )?,
                ram_bytes: parse_number(
                    &take_required(&mut values, "--ram-bytes")?,
                    "--ram-bytes",
                )?,
                os_release: take_required(&mut values, "--os-release")?,
                kernel_release: take_required(&mut values, "--kernel-release")?,
                architecture: take_required(&mut values, "--architecture")?,
            },
            software: SoftwareEnvironment {
                nvidia_driver_version: take_required(&mut values, "--nvidia-driver-version")?,
                cuda_runtime_version: take_required(&mut values, "--cuda-runtime-version")?,
                cuda_toolkit_version: take_required(&mut values, "--cuda-toolkit-version")?,
                cublas_version: take_required(&mut values, "--cublas-version")?,
                container_image_sha256: take_required(&mut values, "--container-image-sha256")?,
            },
        },
        workload: Workload {
            workload_id: take_required(&mut values, "--workload-id")?,
            model_id: take_required(&mut values, "--model-id")?,
            model_revision: take_required(&mut values, "--model-revision")?,
            weights_sha256: take_required(&mut values, "--weights-sha256")?,
            tokenizer_sha256: take_required(&mut values, "--tokenizer-sha256")?,
            dtype: take_required(&mut values, "--dtype")?,
            concurrency: parse_number(
                &take_required(&mut values, "--concurrency")?,
                "--concurrency",
            )?,
            prompt_tokens: parse_number(
                &take_required(&mut values, "--prompt-tokens")?,
                "--prompt-tokens",
            )?,
            output_tokens: parse_number(
                &take_required(&mut values, "--output-tokens")?,
                "--output-tokens",
            )?,
            warmups: parse_number(&take_required(&mut values, "--warmups")?, "--warmups")?,
            measured_iterations: parse_number(
                &take_required(&mut values, "--measured-iterations")?,
                "--measured-iterations",
            )?,
            sampling_id: take_required(&mut values, "--sampling-id")?,
            seed,
        },
    };
    if !values.is_empty() {
        return Err("internal option parser did not consume every value".to_owned());
    }
    if !(1..=5).contains(&options.pair_index) {
        return Err("--pair-index must be in 1..=5".to_owned());
    }
    options.validate()?;
    Ok(Command::Run(Box::new(options)))
}

fn take_required(values: &mut BTreeMap<String, String>, flag: &str) -> Result<String, String> {
    values
        .remove(flag)
        .ok_or_else(|| format!("missing required option {flag}"))
}

fn parse_role(value: &str) -> Result<Role, String> {
    match value {
        "baseline" => Ok(Role::Baseline),
        "candidate" => Ok(Role::Candidate),
        _ => Err("--role requires baseline or candidate".to_owned()),
    }
}

fn parse_number<T>(value: &str, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("{flag} requires a non-negative decimal integer"))
}

fn display_argument(argument: &OsStr) -> String {
    argument.to_string_lossy().into_owned()
}

fn validate_id(flag: &str, value: &str) -> Result<(), String> {
    let mut bytes = value.bytes();
    if !bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphanumeric())
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(format!("{flag} has an invalid evidence ID"));
    }
    Ok(())
}

fn validate_nonempty(flag: &str, value: &str) -> Result<(), String> {
    if value.is_empty() {
        Err(format!("{flag} must not be empty"))
    } else {
        Ok(())
    }
}

fn validate_sha256(flag: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{flag} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_git_commit(value: &str) -> Result<(), String> {
    if !matches!(value.len(), 40 | 64)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("--git-commit must be 40 or 64 lowercase hexadecimal characters".to_owned());
    }
    Ok(())
}

fn validate_compute_capability(value: &str) -> Result<(), String> {
    let Some((major, minor)) = value.split_once('.') else {
        return Err("--gpu-compute-capability must have major.minor form".to_owned());
    };
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err("--gpu-compute-capability must have major.minor form".to_owned());
    }
    Ok(())
}

fn validate_nonempty_environment(environment: &Environment) -> Result<(), String> {
    for (flag, value) in [
        ("--gpu-model", environment.gpu.model.as_str()),
        ("--gpu-uuid", environment.gpu.uuid.as_str()),
        ("--gpu-pci-bus-id", environment.gpu.pci_bus_id.as_str()),
        ("--cpu-model", environment.host.cpu_model.as_str()),
        ("--os-release", environment.host.os_release.as_str()),
        ("--kernel-release", environment.host.kernel_release.as_str()),
        ("--architecture", environment.host.architecture.as_str()),
        (
            "--nvidia-driver-version",
            environment.software.nvidia_driver_version.as_str(),
        ),
        (
            "--cuda-runtime-version",
            environment.software.cuda_runtime_version.as_str(),
        ),
        (
            "--cuda-toolkit-version",
            environment.software.cuda_toolkit_version.as_str(),
        ),
        (
            "--cublas-version",
            environment.software.cublas_version.as_str(),
        ),
    ] {
        validate_nonempty(flag, value)?;
    }
    if environment.gpu.vram_bytes == 0
        || environment.host.physical_core_count == 0
        || environment.host.logical_core_count == 0
        || environment.host.ram_bytes == 0
    {
        return Err("GPU/host capacity fields must be positive".to_owned());
    }
    Ok(())
}

fn validate_rfc3339_utc(value: &str) -> Result<(), String> {
    let invalid = || "--recorded-at-utc must be an RFC 3339 UTC timestamp ending in Z".to_owned();
    let value = value.strip_suffix('Z').ok_or_else(invalid)?;
    let (date, time) = value.split_once('T').ok_or_else(invalid)?;
    let mut date_parts = date.split('-');
    let year = parse_timestamp_part(date_parts.next(), 4).ok_or_else(invalid)?;
    let month = parse_timestamp_part(date_parts.next(), 2).ok_or_else(invalid)?;
    let day = parse_timestamp_part(date_parts.next(), 2).ok_or_else(invalid)?;
    if date_parts.next().is_some() || year == 0 || !(1..=12).contains(&month) {
        return Err(invalid());
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let maximum_day = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if day == 0 || day > maximum_day {
        return Err(invalid());
    }

    let (whole_time, fraction) = time
        .split_once('.')
        .map_or((time, None), |(whole, fraction)| (whole, Some(fraction)));
    let mut time_parts = whole_time.split(':');
    let hour = parse_timestamp_part(time_parts.next(), 2).ok_or_else(invalid)?;
    let minute = parse_timestamp_part(time_parts.next(), 2).ok_or_else(invalid)?;
    let second = parse_timestamp_part(time_parts.next(), 2).ok_or_else(invalid)?;
    if time_parts.next().is_some() || hour > 23 || minute > 59 || second > 59 {
        return Err(invalid());
    }
    if fraction.is_some_and(|digits| {
        digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit())
    }) {
        return Err(invalid());
    }
    Ok(())
}

fn parse_timestamp_part(value: Option<&str>, width: usize) -> Option<u32> {
    let value = value?;
    if value.len() != width || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptRecord {
    contract_version: String,
    prompt_id: String,
    category: String,
    language: String,
    text: String,
    target_prompt_tokens: Option<usize>,
    boundary_kind: String,
    expected_behavior: String,
    contains_sensitive_data: bool,
}

struct PromptSeed {
    text: String,
}

fn read_prompt_seeds(
    path: &Path,
    prompt_tokens: usize,
    concurrency: usize,
) -> Result<Vec<PromptSeed>, String> {
    let metadata = fs::metadata(path).map_err(|_| "cannot inspect prompt corpus".to_owned())?;
    if metadata.len() > MAX_PROMPT_CORPUS_BYTES {
        return Err("prompt corpus exceeds the native profiling byte bound".to_owned());
    }
    let corpus =
        fs::read_to_string(path).map_err(|_| "cannot read prompt corpus as UTF-8".to_owned())?;
    let mut prompt_ids = BTreeSet::new();
    let mut selected_texts = BTreeSet::new();
    let mut selected = Vec::new();
    for (line_index, line) in corpus.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!(
                "prompt corpus line {} must not be blank",
                line_index + 1
            ));
        }
        let record: PromptRecord = serde_json::from_str(line).map_err(|_| {
            format!(
                "prompt corpus line {} is not a strict prompt record",
                line_index + 1
            )
        })?;
        if record.contract_version != "1.0.0"
            || record.prompt_id.is_empty()
            || record.category.is_empty()
            || record.language.is_empty()
            || record.boundary_kind.is_empty()
            || record.expected_behavior.is_empty()
            || record.contains_sensitive_data
            || record.target_prompt_tokens == Some(0)
        {
            return Err(format!(
                "prompt corpus line {} violates the prompt contract",
                line_index + 1
            ));
        }
        if !prompt_ids.insert(record.prompt_id) {
            return Err(format!(
                "prompt corpus line {} has a duplicate prompt identity",
                line_index + 1
            ));
        }
        if record.target_prompt_tokens == Some(prompt_tokens) && selected.len() < concurrency {
            if !selected_texts.insert(record.text.clone()) {
                return Err(format!(
                    "prompt corpus line {} duplicates a selected prompt text",
                    line_index + 1
                ));
            }
            selected.push(PromptSeed { text: record.text });
        }
    }
    if selected.len() != concurrency {
        return Err(format!(
            "prompt corpus must provide {concurrency} distinct seeds for {prompt_tokens} tokens"
        ));
    }
    Ok(selected)
}

fn resize_token_ids(token_ids: &[u32], target: usize) -> Result<Vec<u32>, String> {
    if token_ids.is_empty() || target == 0 {
        return Err("cannot canonically resize an empty token sequence".to_owned());
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(target)
        .map_err(|_| "cannot reserve canonical prompt token storage".to_owned())?;
    output.extend(token_ids.iter().copied().cycle().take(target));
    Ok(output)
}

fn validate_loaded_model(model: &LoadedModel, workload: &Workload) -> Result<(), String> {
    let provenance = model.provenance();
    if provenance.source_model() != workload.model_id {
        return Err("--model-id differs from loaded checkpoint provenance".to_owned());
    }
    if provenance.source_revision() != workload.model_revision {
        return Err("--model-revision differs from loaded checkpoint provenance".to_owned());
    }
    if provenance.dtype().name() != workload.dtype {
        return Err("--dtype differs from loaded checkpoint provenance".to_owned());
    }
    let tokenizer = provenance
        .files()
        .get(Path::new(TOKENIZER_FILENAME))
        .ok_or_else(|| "loaded checkpoint has no tokenizer checksum".to_owned())?;
    if tokenizer.sha256() != workload.tokenizer_sha256 {
        return Err("--tokenizer-sha256 differs from loaded checkpoint provenance".to_owned());
    }
    let weights = provenance
        .files()
        .get(Path::new("model.safetensors"))
        .ok_or_else(|| {
            "native profiling requires a single-shard model.safetensors checkpoint".to_owned()
        })?;
    if weights.sha256() != workload.weights_sha256 {
        return Err("--weights-sha256 differs from loaded checkpoint provenance".to_owned());
    }
    Ok(())
}

fn materialize_token_rows(
    model: &LoadedModel,
    seeds: Vec<PromptSeed>,
    prompt_tokens: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(seeds.len())
        .map_err(|_| "cannot reserve prompt row storage".to_owned())?;
    for seed in seeds {
        let encoded = model
            .tokenizer()
            .encode(
                &seed.text,
                EncodeOptions {
                    add_special_tokens: true,
                },
            )
            .map_err(|_| "native tokenizer failed for a selected prompt".to_owned())?;
        rows.push(resize_token_ids(&encoded, prompt_tokens)?);
    }
    Ok(rows)
}

fn benchmark_config(options: &Options) -> Result<NativeBenchmarkConfig, String> {
    let workload = &options.workload;
    let max_sequence_tokens = workload
        .prompt_tokens
        .checked_add(workload.output_tokens)
        .ok_or_else(|| "sequence token bound overflowed".to_owned())?;
    let blocks_per_sequence = max_sequence_tokens.div_ceil(KV_BLOCK_SIZE);
    let physical_kv_blocks = workload
        .concurrency
        .checked_mul(blocks_per_sequence)
        .ok_or_else(|| "physical KV block bound overflowed".to_owned())?;
    let maximum_waiting_prompt_tokens = workload
        .concurrency
        .checked_mul(workload.prompt_tokens)
        .ok_or_else(|| "waiting prompt-token bound overflowed".to_owned())?;
    let prefill_chunk_tokens = CANONICAL_BATCH_TOKEN_BUDGET.min(max_sequence_tokens);
    let scheduler = SchedulerConfig {
        max_waiting_requests: workload.concurrency,
        max_waiting_prompt_tokens: maximum_waiting_prompt_tokens,
        max_active_sequences: workload.concurrency,
        max_sequence_tokens,
        iteration_token_budget: CANONICAL_BATCH_TOKEN_BUDGET,
        max_prefill_chunk_tokens: prefill_chunk_tokens,
        aging_threshold_ns: 100_000_000,
        overload_policy: OverloadPolicy::RejectImmediately,
        admission_timeout_ns: None,
        max_promised_kv_blocks: physical_kv_blocks,
        metrics_window_samples: 1_024,
    };
    let batch_metadata = LlamaBatchMetadataConfig::new(
        workload.concurrency,
        CANONICAL_BATCH_TOKEN_BUDGET,
        physical_kv_blocks,
        workload.concurrency,
        physical_kv_blocks,
    )
    .map_err(|_| "invalid fixed native batch metadata bounds".to_owned())?;
    let executor = PreparedLlamaBatchExecutorConfig::new(
        batch_metadata,
        PreparedLlamaForwardConfig::default(),
    );
    let executor = match options.runtime_selection()? {
        RuntimeSelection::ResidualRmsNorm(ResidualRmsNormMode::Separate)
        | RuntimeSelection::ExecutionCompletion(ExecutionCompletionMode::PerOperation) => executor
            .with_separate_residual_norm()
            .with_per_operation_completion(),
        RuntimeSelection::ResidualRmsNorm(ResidualRmsNormMode::Fused) => executor
            .with_fused_residual_norm()
            .with_per_operation_completion(),
        RuntimeSelection::ExecutionCompletion(ExecutionCompletionMode::IterationBatch) => executor
            .with_separate_residual_norm()
            .with_iteration_batch_completion(),
    };
    Ok(NativeBenchmarkConfig {
        device_ordinal: options.environment.gpu.device_index,
        scheduler,
        executor,
    })
}

fn execute_trial(
    executor: &mut NativeBenchmarkExecutor,
    token_rows: &[Vec<u32>],
    output_tokens: usize,
) -> Result<NativeBenchmarkTrialTrace, String> {
    let requests = token_rows
        .iter()
        .cloned()
        .map(PretokenizedBenchmarkRequest::new)
        .collect();
    let trial = executor
        .prepare_trial(requests, output_tokens)
        .map_err(|error| format!("trial preparation failed: {error}"))?;
    executor
        .run_trial(trial)
        .map_err(|error| format!("trial execution failed: {error}"))
}

#[derive(Serialize)]
struct Measurement<T> {
    validity: &'static str,
    value: Option<T>,
}

impl<T> Measurement<T> {
    fn measured(value: T) -> Self {
        Self {
            validity: "measured",
            value: Some(value),
        }
    }

    fn unmeasured() -> Self {
        Self {
            validity: "unmeasured",
            value: None,
        }
    }
}

#[derive(Serialize)]
// These names are part of the versioned native-profile JSON contract.
#[allow(clippy::struct_field_names)]
struct HostAggregate {
    plan_ns: Measurement<u64>,
    execute_ns: Measurement<u64>,
    sampling_ns: Measurement<u64>,
    commit_ns: Measurement<u64>,
}

#[derive(Serialize)]
struct CudaAggregate {
    stream_span_ns: Measurement<u64>,
    idle_ns: Measurement<u64>,
}

#[derive(Serialize)]
struct CopyCounters {
    h2d_calls: Measurement<u64>,
    h2d_bytes: Measurement<u64>,
    d2h_calls: Measurement<u64>,
    d2h_bytes: Measurement<u64>,
}

#[derive(Serialize)]
struct AllocationCounters {
    device_allocations: Measurement<u64>,
    device_frees: Measurement<u64>,
    pinned_allocations: Measurement<u64>,
    pinned_frees: Measurement<u64>,
    peak_device_bytes: Measurement<u64>,
}

#[derive(Serialize)]
struct Counters {
    iterations: Measurement<u64>,
    kernel_launches: Measurement<u64>,
    copies: CopyCounters,
    allocations: AllocationCounters,
}

#[derive(Serialize)]
struct Aggregate {
    host: HostAggregate,
    cuda: CudaAggregate,
    counters: Counters,
    throughput_output_tokens_per_second: f64,
}

#[derive(Serialize)]
struct TraceInventory {
    capacity: usize,
    retained_records: usize,
    dropped_records: usize,
}

#[derive(Serialize)]
struct RequestEvidence {
    input_index: usize,
    prompt_u32le_sha256: String,
    generated_u32le_sha256: String,
    prompt_token_count: usize,
    requested_output_token_count: usize,
    generated_token_count: usize,
    ttft_ms: f64,
    tpot_ms: f64,
    e2e_ms: f64,
}

#[derive(Serialize)]
struct Evidence {
    schema_version: &'static str,
    role: Role,
    pair_index: u8,
    run_id: String,
    recorded_at_utc: String,
    status: &'static str,
    failure_count: usize,
    source: SourceProvenance,
    environment: Environment,
    workload: Workload,
    trace: TraceInventory,
    primary_metric: &'static str,
    aggregate: Aggregate,
    requests: Vec<RequestEvidence>,
}

struct AggregateState {
    capacity: usize,
    retained_records: usize,
    plan_ns: u64,
    execute_ns: u64,
    sampling_ns: u64,
    commit_ns: u64,
    cuda_stream_span_ns: Option<u64>,
    cuda_idle_ns: Option<u64>,
    iterations: u64,
    measured_batch_ns: u64,
    requests: Vec<RequestEvidence>,
}

impl AggregateState {
    fn new(options: &Options) -> Result<Self, String> {
        let requests = options
            .workload
            .concurrency
            .checked_mul(options.workload.measured_iterations)
            .ok_or_else(|| "measured request count overflowed".to_owned())?;
        let maximum_iterations_per_request = options
            .workload
            .prompt_tokens
            .checked_add(options.workload.output_tokens - 1)
            .ok_or_else(|| "trace iteration capacity overflowed".to_owned())?;
        let maximum_iterations = requests
            .checked_mul(maximum_iterations_per_request)
            .ok_or_else(|| "trace iteration capacity overflowed".to_owned())?;
        let capacity = requests
            .checked_add(maximum_iterations)
            .ok_or_else(|| "trace capacity overflowed".to_owned())?;
        let mut request_rows = Vec::new();
        request_rows
            .try_reserve_exact(requests)
            .map_err(|_| "cannot reserve measured request evidence".to_owned())?;
        Ok(Self {
            capacity,
            retained_records: 0,
            plan_ns: 0,
            execute_ns: 0,
            sampling_ns: 0,
            commit_ns: 0,
            cuda_stream_span_ns: Some(0),
            cuda_idle_ns: Some(0),
            iterations: 0,
            measured_batch_ns: 0,
            requests: request_rows,
        })
    }

    fn observe_trial(
        &mut self,
        trace: &NativeBenchmarkTrialTrace,
        trial_index: usize,
        concurrency: usize,
    ) -> Result<(), String> {
        let batch_ns = duration_ns(trace.batch_started_ns(), trace.batch_finished_ns())?;
        self.measured_batch_ns = checked_add(self.measured_batch_ns, batch_ns)?;
        for request in trace.requests() {
            let flattened_index = trial_index
                .checked_mul(concurrency)
                .and_then(|base| base.checked_add(request.input_index()))
                .ok_or_else(|| "flattened request index overflowed".to_owned())?;
            self.requests
                .push(summarize_request(request, flattened_index)?);
            self.retained_records = self
                .retained_records
                .checked_add(1)
                .ok_or_else(|| "retained trace count overflowed".to_owned())?;
        }
        for &iteration in trace.iterations() {
            self.observe_iteration(iteration)?;
        }
        if self.retained_records > self.capacity {
            return Err("native trace exceeded its declared fixed capacity".to_owned());
        }
        Ok(())
    }

    fn observe_iteration(
        &mut self,
        iteration: NativeBenchmarkIterationTrace,
    ) -> Result<(), String> {
        self.plan_ns = checked_add(
            self.plan_ns,
            duration_ns(iteration.plan_started_ns(), iteration.plan_finished_ns())?,
        )?;
        self.execute_ns = checked_add(
            self.execute_ns,
            duration_ns(
                iteration.execution_started_ns(),
                iteration.execution_finished_ns(),
            )?,
        )?;
        self.sampling_ns = checked_add(
            self.sampling_ns,
            duration_ns(
                iteration.execution_finished_ns(),
                iteration.commit_started_ns(),
            )?,
        )?;
        self.commit_ns = checked_add(
            self.commit_ns,
            duration_ns(
                iteration.commit_started_ns(),
                iteration.commit_observed_ns(),
            )?,
        )?;
        let metric = iteration.metric();
        match iteration.gpu_timing_validity() {
            NativeGpuTimingValidity::MeasuredCudaEvents => {
                if let Some(value) = self.cuda_stream_span_ns {
                    self.cuda_stream_span_ns = Some(checked_add(value, metric.gpu_execution_ns)?);
                }
                if let Some(value) = self.cuda_idle_ns {
                    self.cuda_idle_ns = Some(checked_add(value, metric.gpu_idle_gap_ns)?);
                }
            }
            NativeGpuTimingValidity::UnavailableExecutionBridgeBoundary => {
                self.cuda_stream_span_ns = None;
                self.cuda_idle_ns = None;
            }
        }
        self.iterations = self
            .iterations
            .checked_add(1)
            .ok_or_else(|| "iteration count overflowed".to_owned())?;
        self.retained_records = self
            .retained_records
            .checked_add(1)
            .ok_or_else(|| "retained trace count overflowed".to_owned())?;
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    fn finish(self) -> Result<(TraceInventory, Aggregate, Vec<RequestEvidence>), String> {
        if self.measured_batch_ns == 0 || self.requests.is_empty() {
            return Err("measured batch duration and request set must be positive".to_owned());
        }
        let generated_tokens = self.requests.iter().try_fold(0_u64, |total, request| {
            let count = u64::try_from(request.generated_token_count)
                .map_err(|_| "generated token count does not fit u64".to_owned())?;
            checked_add(total, count)
        })?;
        let throughput = generated_tokens as f64 * 1_000_000_000.0 / self.measured_batch_ns as f64;
        if !throughput.is_finite() || throughput <= 0.0 {
            return Err("computed throughput is not a positive finite number".to_owned());
        }
        let cuda = CudaAggregate {
            stream_span_ns: self
                .cuda_stream_span_ns
                .map_or_else(Measurement::unmeasured, Measurement::measured),
            idle_ns: self
                .cuda_idle_ns
                .map_or_else(Measurement::unmeasured, Measurement::measured),
        };
        let unmeasured = Measurement::<u64>::unmeasured;
        let aggregate = Aggregate {
            host: HostAggregate {
                plan_ns: Measurement::measured(self.plan_ns),
                execute_ns: Measurement::measured(self.execute_ns),
                sampling_ns: Measurement::measured(self.sampling_ns),
                commit_ns: Measurement::measured(self.commit_ns),
            },
            cuda,
            counters: Counters {
                iterations: Measurement::measured(self.iterations),
                kernel_launches: unmeasured(),
                copies: CopyCounters {
                    h2d_calls: unmeasured(),
                    h2d_bytes: unmeasured(),
                    d2h_calls: unmeasured(),
                    d2h_bytes: unmeasured(),
                },
                allocations: AllocationCounters {
                    device_allocations: unmeasured(),
                    device_frees: unmeasured(),
                    pinned_allocations: unmeasured(),
                    pinned_frees: unmeasured(),
                    peak_device_bytes: unmeasured(),
                },
            },
            throughput_output_tokens_per_second: throughput,
        };
        Ok((
            TraceInventory {
                capacity: self.capacity,
                retained_records: self.retained_records,
                dropped_records: 0,
            },
            aggregate,
            self.requests,
        ))
    }
}

#[allow(clippy::cast_precision_loss)]
fn summarize_request(
    request: &NativeBenchmarkRequestTrace,
    input_index: usize,
) -> Result<RequestEvidence, String> {
    let committed = request.committed_token_ns();
    if request.generated_tokens() != request.requested_output_tokens() {
        return Err("measured request did not commit the exact output length".to_owned());
    }
    let (ttft_ms, tpot_ms, e2e_ms) = request_latency_ms(
        request.arrival_ns(),
        committed,
        request.terminal_observed_ns(),
    )?;
    let request = RequestEvidence {
        input_index,
        prompt_u32le_sha256: request.prompt_token_ids_sha256().to_hex(),
        generated_u32le_sha256: request.generated_token_ids_sha256().to_hex(),
        prompt_token_count: request.prompt_tokens(),
        requested_output_token_count: request.requested_output_tokens(),
        generated_token_count: request.generated_tokens(),
        ttft_ms,
        tpot_ms,
        e2e_ms,
    };
    if request.ttft_ms <= 0.0 || request.tpot_ms <= 0.0 || request.e2e_ms <= 0.0 {
        return Err("request millisecond metrics are not strictly positive".to_owned());
    }
    Ok(request)
}

#[allow(clippy::cast_precision_loss)]
fn request_latency_ms(
    arrival_ns: u64,
    committed_ns: &[u64],
    terminal_ns: u64,
) -> Result<(f64, f64, f64), String> {
    if committed_ns.len() < MIN_OUTPUT_TOKENS {
        return Err("measured request must include at least one decode token".to_owned());
    }
    let (&first, &last) = committed_ns
        .first()
        .zip(committed_ns.last())
        .ok_or_else(|| "measured request has no committed token".to_owned())?;
    let ttft_ns = duration_ns(arrival_ns, first)?;
    let e2e_ns = duration_ns(arrival_ns, terminal_ns)?;
    if ttft_ns == 0 || e2e_ns == 0 || e2e_ns < ttft_ns {
        return Err("request latency boundaries are not strictly positive".to_owned());
    }
    let span = duration_ns(first, last)?;
    if span == 0 {
        return Err("multi-token request has a zero inter-token span".to_owned());
    }
    let tpot_ns = span as f64 / (committed_ns.len() - 1) as f64;
    Ok((
        ttft_ns as f64 / 1_000_000.0,
        tpot_ns / 1_000_000.0,
        e2e_ns as f64 / 1_000_000.0,
    ))
}

fn duration_ns(start: u64, finish: u64) -> Result<u64, String> {
    finish
        .checked_sub(start)
        .ok_or_else(|| "monotonic trace boundary regressed".to_owned())
}

fn checked_add(left: u64, right: u64) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| "native trace aggregate overflowed u64".to_owned())
}

fn run_trials(
    executor: &mut NativeBenchmarkExecutor,
    options: &Options,
    token_rows: &[Vec<u32>],
) -> Result<AggregateState, String> {
    for _ in 0..options.workload.warmups {
        drop(execute_trial(
            executor,
            token_rows,
            options.workload.output_tokens,
        )?);
    }
    let mut aggregate = AggregateState::new(options)?;
    for trial_index in 0..options.workload.measured_iterations {
        let trace = execute_trial(executor, token_rows, options.workload.output_tokens)?;
        aggregate.observe_trial(&trace, trial_index, options.workload.concurrency)?;
    }
    Ok(aggregate)
}

fn run_profile(options: Options) -> Result<(), String> {
    let seeds = read_prompt_seeds(
        &options.prompts_path,
        options.workload.prompt_tokens,
        options.workload.concurrency,
    )?;
    let load_limits = LoadLimits::default()
        .with_weight_byte_limits(DEFAULT_MAX_WEIGHT_BYTES, DEFAULT_MAX_WEIGHT_BYTES)
        .map_err(|_| "invalid fixed model weight limit".to_owned())?;
    let model = LoadedModel::load(&options.model_path, load_limits)
        .map_err(|_| "model loading failed".to_owned())?;
    validate_loaded_model(&model, &options.workload)?;
    let token_rows = materialize_token_rows(&model, seeds, options.workload.prompt_tokens)?;
    let config = benchmark_config(&options)?;
    let mut executor = NativeBenchmarkExecutor::prepare(model, config)
        .map_err(|error| format!("native executor preparation failed: {error}"))?;

    let measured = run_trials(&mut executor, &options, &token_rows);
    let cleanup = executor.close();
    let aggregate = match (measured, cleanup) {
        (Ok(aggregate), Ok(report)) => {
            if report.scheduler_completions() != 0 || !report.allocations_are_zero() {
                return Err(
                    "explicit cleanup did not prove an idle zero-allocation state".to_owned(),
                );
            }
            aggregate
        }
        (Err(error), Ok(_)) => return Err(error),
        (Ok(_), Err(error)) => return Err(format!("explicit cleanup failed: {error}")),
        (Err(execution), Err(cleanup)) => {
            return Err(format!(
                "{execution}; explicit cleanup also failed: {cleanup}"
            ));
        }
    };
    let (trace, aggregate, requests) = aggregate.finish()?;
    let output_path = options.output_path;
    let evidence = Evidence {
        schema_version: SCHEMA_VERSION,
        role: options.role,
        pair_index: options.pair_index,
        run_id: options.run_id,
        recorded_at_utc: options.recorded_at_utc,
        status: "success",
        failure_count: 0,
        source: options.source,
        environment: options.environment,
        workload: options.workload,
        trace,
        primary_metric: PRIMARY_METRIC,
        aggregate,
        requests,
    };
    let mut output = serde_json::to_vec_pretty(&evidence)
        .map_err(|_| "cannot serialize native profile evidence".to_owned())?;
    output.push(b'\n');
    write_output(output_path.as_deref(), &output)
}

fn write_output(path: Option<&Path>, output: &[u8]) -> Result<(), String> {
    match path {
        None => {
            let mut stdout = io::stdout().lock();
            stdout
                .write_all(output)
                .and_then(|()| stdout.flush())
                .map_err(|_| "cannot write evidence to stdout".to_owned())
        }
        Some(path) => {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|_| "cannot exclusively create evidence output".to_owned())?;
            if file
                .write_all(output)
                .and_then(|()| file.sync_all())
                .is_err()
            {
                drop(file);
                let _ = fs::remove_file(path);
                return Err("cannot durably write evidence output".to_owned());
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use serde_json::json;

    use super::{
        Command, ExecutionCompletionMode, Measurement, ResidualRmsNormMode, Role, RuntimeSelection,
        parse_arguments, request_latency_ms, resize_token_ids, validate_rfc3339_utc,
    };

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const GIT_COMMIT: &str = "cccccccccccccccccccccccccccccccccccccccc";

    fn valid_arguments(role: &str, flag_name: &str, flag_value: &str) -> Vec<OsString> {
        let correctness_gate = match flag_name {
            "execution_completion" => "pr15-iteration-command-batch-exact-v1",
            _ => "pr15-fused-residual-rmsnorm-exact-v1",
        };
        let pairs = [
            ("--model", "/models/smollm2"),
            ("--prompts", "/repo/benchmarks/prompts.jsonl"),
            ("--role", role),
            ("--pair-index", "3"),
            ("--run-id", "profile.run:003"),
            ("--recorded-at-utc", "2026-08-26T12:34:56.123Z"),
            ("--git-commit", GIT_COMMIT),
            ("--git-dirty", "false"),
            ("--executable-sha256", SHA_A),
            ("--implementation-id", "riley.fused"),
            ("--runtime-flag-name", flag_name),
            ("--runtime-flag-value", flag_value),
            ("--semantic-class", "E0"),
            ("--correctness-gate-id", correctness_gate),
            ("--correctness-report-sha256", SHA_B),
            ("--gpu-model", "NVIDIA GeForce RTX 4090"),
            ("--gpu-uuid", "GPU-fixture"),
            ("--device-index", "0"),
            ("--gpu-pci-bus-id", "00000000:01:00.0"),
            ("--gpu-compute-capability", "8.9"),
            ("--gpu-vram-bytes", "25757220864"),
            ("--environment-id", "server-4096"),
            ("--cpu-model", "fixture cpu"),
            ("--physical-core-count", "8"),
            ("--logical-core-count", "16"),
            ("--ram-bytes", "68719476736"),
            ("--os-release", "Ubuntu 24.04"),
            ("--kernel-release", "6.8.0"),
            ("--architecture", "x86_64"),
            ("--nvidia-driver-version", "570.00"),
            ("--cuda-runtime-version", "12.8"),
            ("--cuda-toolkit-version", "12.8"),
            ("--cublas-version", "12.8"),
            ("--container-image-sha256", SHA_A),
            ("--workload-id", "c1.p128.o32.warm"),
            ("--model-id", "HuggingFaceTB/SmolLM2-135M"),
            ("--model-revision", GIT_COMMIT),
            ("--weights-sha256", SHA_A),
            ("--tokenizer-sha256", SHA_B),
            ("--dtype", "bf16"),
            ("--concurrency", "1"),
            ("--prompt-tokens", "128"),
            ("--output-tokens", "32"),
            ("--warmups", "5"),
            ("--measured-iterations", "30"),
            ("--sampling-id", "greedy"),
            ("--seed", "none"),
        ];
        let mut arguments = Vec::with_capacity(pairs.len() * 2);
        for (flag, value) in pairs {
            arguments.push(OsString::from(flag));
            arguments.push(OsString::from(value));
        }
        arguments
    }

    #[test]
    fn parser_binds_candidate_to_fused_and_preserves_explicit_metadata() {
        let command = parse_arguments(valid_arguments("candidate", "residual_rmsnorm", "fused"))
            .expect("strict candidate arguments");
        let Command::Run(options) = command else {
            panic!("expected run command");
        };
        assert_eq!(options.role, Role::Candidate);
        assert_eq!(
            options.runtime_selection().expect("validated runtime flag"),
            RuntimeSelection::ResidualRmsNorm(ResidualRmsNormMode::Fused)
        );
        assert_eq!(options.pair_index, 3);
        assert_eq!(options.workload.measured_iterations, 30);
        assert_eq!(options.environment.gpu.device_index, 0);
        assert!(options.output_path.is_none());
    }

    #[test]
    fn parser_rejects_swapped_roles_duplicates_and_non_greedy_metadata() {
        assert!(parse_arguments(valid_arguments("baseline", "residual_rmsnorm", "fused")).is_err());
        let mut duplicate = valid_arguments("candidate", "residual_rmsnorm", "fused");
        duplicate.extend([OsString::from("--role"), OsString::from("candidate")]);
        assert!(parse_arguments(duplicate).is_err());
        let mut non_greedy = valid_arguments("candidate", "residual_rmsnorm", "fused");
        let position = non_greedy
            .iter()
            .position(|value| value.to_str() == Some("greedy"))
            .expect("sampling value");
        non_greedy[position] = OsString::from("random");
        assert!(parse_arguments(non_greedy).is_err());
    }

    #[test]
    fn parser_binds_iteration_completion_candidate_independently_from_fusion() {
        let command = parse_arguments(valid_arguments(
            "candidate",
            "execution_completion",
            "iteration-batch",
        ))
        .expect("strict iteration-batch candidate arguments");
        let Command::Run(options) = command else {
            panic!("expected run command");
        };
        assert_eq!(
            options.runtime_selection().expect("validated runtime flag"),
            RuntimeSelection::ExecutionCompletion(ExecutionCompletionMode::IterationBatch)
        );
        assert!(
            parse_arguments(valid_arguments(
                "baseline",
                "execution_completion",
                "iteration-batch",
            ))
            .is_err()
        );
        assert!(
            parse_arguments(valid_arguments(
                "candidate",
                "residual_rmsnorm",
                "iteration-batch",
            ))
            .is_err()
        );
    }

    #[test]
    fn canonical_token_resize_truncates_or_repeats_exact_u32_ids() {
        assert_eq!(
            resize_token_ids(&[7, 11, 13], 8).expect("repeat"),
            [7, 11, 13, 7, 11, 13, 7, 11]
        );
        assert_eq!(
            resize_token_ids(&[7, 11, 13], 2).expect("truncate"),
            [7, 11]
        );
        assert!(resize_token_ids(&[], 8).is_err());
    }

    #[test]
    fn latency_math_requires_a_real_decode_interval() {
        assert!(request_latency_ms(1_000, &[2_001_000], 3_001_000).is_err());
        assert_eq!(
            request_latency_ms(1_000, &[1_001_000, 3_001_000, 5_001_000], 6_001_000)
                .expect("three tokens"),
            (1.0, 2.0, 6.0)
        );
        assert!(request_latency_ms(2_000, &[1_000], 3_000).is_err());
        assert!(request_latency_ms(0, &[], 1).is_err());
    }

    #[test]
    fn timestamp_and_measurement_serialization_match_closed_schema_shapes() {
        assert!(validate_rfc3339_utc("2024-02-29T23:59:59Z").is_ok());
        assert!(validate_rfc3339_utc("2023-02-29T23:59:59Z").is_err());
        assert!(validate_rfc3339_utc("2026-08-26T12:00:00+00:00").is_err());
        assert_eq!(
            serde_json::to_value(Measurement::measured(17_u64)).expect("measured JSON"),
            json!({"validity": "measured", "value": 17})
        );
        assert_eq!(
            serde_json::to_value(Measurement::<u64>::unmeasured()).expect("unmeasured JSON"),
            json!({"validity": "unmeasured", "value": null})
        );
    }
}
