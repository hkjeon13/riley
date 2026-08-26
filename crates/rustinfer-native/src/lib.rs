//! Development-only contract and implementation for native calibration.
//!
//! This crate remains a non-default workspace member. Its feature-off library
//! owns the side-effect-free contract parser; the CUDA-gated binary owns the
//! real Python-free candidate producer.

#[cfg(any(feature = "cuda", test))]
mod contract;
#[cfg(any(feature = "cuda", test))]
mod environment;
#[cfg(any(feature = "cuda", test))]
mod git;
#[cfg(any(feature = "cuda", test))]
mod numeric;
#[cfg(feature = "cuda")]
mod producer;
#[cfg(any(feature = "cuda", test))]
mod sidecar;

#[cfg(feature = "cuda")]
pub use producer::{NativeCalibrationError, run_calibration};

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

/// Versioned native candidate engine contract owned by this crate.
pub const NATIVE_ENGINE_REVISION: &str = "rustinfer-native-contract-v3";

/// Sibling executable name reserved by the candidate evidence manifest.
pub const NATIVE_EXECUTABLE_FILENAME: &str = "rustinfer-native";

/// Exact build command bound into native candidate evidence.
pub const NATIVE_BUILD_ARGV: [&str; 11] = [
    "cargo",
    "build",
    "--locked",
    "--release",
    "--package",
    "rustinfer-native",
    "--no-default-features",
    "--features",
    "cuda",
    "--bin",
    NATIVE_EXECUTABLE_FILENAME,
];

/// Correctness gate path required by the native capture ABI.
pub const GATE_MANIFEST_PATH: &str = "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v3.json";

/// Ordered prompt corpus path required by the native capture ABI.
pub const PROMPTS_PATH: &str = "benchmarks/prompts.jsonl";

/// One reduction execution required from every native candidate case.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionVariant {
    /// Existing production-default operator reductions.
    Canonical,
}

impl ReductionVariant {
    /// Stable manifest and CLI identifier.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Canonical => "canonical-v1",
        }
    }

    fn parse(value: &str) -> Result<Self, CalibrationCliError> {
        match value {
            "canonical-v1" => Ok(Self::Canonical),
            _ => Err(CalibrationCliError::UnsupportedReductionVariant(
                value.to_owned(),
            )),
        }
    }
}

/// Exact reduction variant required by candidate contract v3.
pub const REQUIRED_REDUCTION_VARIANTS: [ReductionVariant; 1] = [ReductionVariant::Canonical];

const REQUIRED_CAPTURE_FLAGS: [&str; 7] = [
    "--repository-root",
    "--model",
    "--gate-manifest",
    "--prompts",
    "--manifest",
    "--sidecar",
    "--reduction-variant",
];

/// Parsed, side-effect-free portion of the future `calibrate` command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationArgs {
    repository_root: PathBuf,
    model: PathBuf,
    gate_manifest: PathBuf,
    prompts: PathBuf,
    manifest: PathBuf,
    sidecar: PathBuf,
    reduction_variants: [ReductionVariant; 1],
}

impl CalibrationArgs {
    /// Absolute repository root used to resolve contract-owned inputs.
    #[must_use]
    pub fn repository_root(&self) -> &Path {
        &self.repository_root
    }

    /// Absolute checkpoint directory loaded by the future native producer.
    #[must_use]
    pub fn model(&self) -> &Path {
        &self.model
    }

    /// Correctness gate manifest bound by this invocation.
    #[must_use]
    pub fn gate_manifest(&self) -> &Path {
        &self.gate_manifest
    }

    /// Ordered prompt corpus bound by this invocation.
    #[must_use]
    pub fn prompts(&self) -> &Path {
        &self.prompts
    }

    /// Sibling JSON manifest output name.
    #[must_use]
    pub fn manifest(&self) -> &Path {
        &self.manifest
    }

    /// Sibling safetensors output name.
    #[must_use]
    pub fn sidecar(&self) -> &Path {
        &self.sidecar
    }

    /// The sole release-qualified reduction variant.
    #[must_use]
    pub const fn reduction_variants(&self) -> &[ReductionVariant; 1] {
        &self.reduction_variants
    }
}

/// Fail-closed parse error for the side-effect-free calibration ABI skeleton.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CalibrationCliError {
    /// No subcommand was provided.
    MissingSubcommand,
    /// The only reserved subcommand is `calibrate`.
    UnsupportedSubcommand(String),
    /// An argument was not valid UTF-8 and cannot be recorded in JSON evidence.
    NonUtf8Argument,
    /// A flag is not part of contract v3.
    UnknownFlag(String),
    /// A required flag was absent.
    MissingFlag(&'static str),
    /// A single-valued flag appeared more than once.
    DuplicateFlag(&'static str),
    /// A flag did not have a following value.
    MissingValue(&'static str),
    /// A reduction variant identifier is not reviewed.
    UnsupportedReductionVariant(String),
    /// The sole reviewed variant was absent or duplicated.
    ReductionVariantOrder,
    /// Reviewed flags did not appear in the canonical contract-v3 order.
    ArgumentOrder,
    /// A contract-owned input path differed from its canonical repository path.
    InputPathMismatch(&'static str),
    /// A runtime-owned input was not a normalized absolute POSIX path.
    InvalidAbsoluteInput(&'static str),
    /// An output was not an explicit sibling filename with the required suffix.
    InvalidOutput(&'static str),
}

impl fmt::Display for CalibrationCliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSubcommand => formatter.write_str("missing `calibrate` subcommand"),
            Self::UnsupportedSubcommand(command) => {
                write!(formatter, "unsupported native subcommand {command:?}")
            }
            Self::NonUtf8Argument => {
                formatter.write_str("native calibration argv must be valid UTF-8")
            }
            Self::UnknownFlag(flag) => write!(formatter, "unknown calibration flag {flag:?}"),
            Self::MissingFlag(flag) => write!(formatter, "required flag {flag} is missing"),
            Self::DuplicateFlag(flag) => write!(formatter, "flag {flag} may appear only once"),
            Self::MissingValue(flag) => write!(formatter, "flag {flag} lacks a value"),
            Self::UnsupportedReductionVariant(variant) => {
                write!(formatter, "unsupported reduction variant {variant:?}")
            }
            Self::ReductionVariantOrder => {
                formatter.write_str("reduction variants must be exactly canonical-v1")
            }
            Self::ArgumentOrder => {
                formatter.write_str("calibration flags must use canonical contract-v3 order")
            }
            Self::InputPathMismatch(flag) => {
                write!(
                    formatter,
                    "{flag} differs from its canonical repository path"
                )
            }
            Self::InvalidAbsoluteInput(flag) => {
                write!(formatter, "{flag} must be a normalized absolute POSIX path")
            }
            Self::InvalidOutput(flag) => write!(
                formatter,
                "{flag} must be an explicit sibling filename with the required suffix"
            ),
        }
    }
}

impl Error for CalibrationCliError {}

/// Parses the side-effect-free contract portion of the future native command.
///
/// The iterator starts with `calibrate`, as it would after a future binary
/// removes `argv[0]`. Parsing performs no filesystem, CUDA, or model access.
/// Unknown flags fail closed so later producer PRs must review every ABI
/// extension explicitly.
///
/// # Errors
///
/// Returns [`CalibrationCliError`] for malformed, incomplete, non-canonical,
/// or reordered arguments.
pub fn parse_calibration_command<I, S>(arguments: I) -> Result<CalibrationArgs, CalibrationCliError>
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString>,
{
    let mut arguments = arguments
        .into_iter()
        .map(Into::into)
        .map(|argument| {
            argument
                .into_string()
                .map_err(|_| CalibrationCliError::NonUtf8Argument)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter();

    let command = arguments
        .next()
        .ok_or(CalibrationCliError::MissingSubcommand)?;
    if command != "calibrate" {
        return Err(CalibrationCliError::UnsupportedSubcommand(command));
    }

    let mut repository_root = None;
    let mut model = None;
    let mut gate_manifest = None;
    let mut prompts = None;
    let mut manifest = None;
    let mut sidecar = None;
    let mut variants = Vec::with_capacity(REQUIRED_REDUCTION_VARIANTS.len());
    let mut observed_flags = Vec::with_capacity(REQUIRED_CAPTURE_FLAGS.len());

    while let Some(flag) = arguments.next() {
        observed_flags.push(flag.clone());
        match flag.as_str() {
            "--repository-root" => set_once(
                &mut repository_root,
                take_value(&mut arguments, "--repository-root")?,
                "--repository-root",
            )?,
            "--model" => set_once(
                &mut model,
                take_value(&mut arguments, "--model")?,
                "--model",
            )?,
            "--gate-manifest" => set_once(
                &mut gate_manifest,
                take_value(&mut arguments, "--gate-manifest")?,
                "--gate-manifest",
            )?,
            "--prompts" => set_once(
                &mut prompts,
                take_value(&mut arguments, "--prompts")?,
                "--prompts",
            )?,
            "--manifest" => set_once(
                &mut manifest,
                take_value(&mut arguments, "--manifest")?,
                "--manifest",
            )?,
            "--sidecar" => set_once(
                &mut sidecar,
                take_value(&mut arguments, "--sidecar")?,
                "--sidecar",
            )?,
            "--reduction-variant" => {
                let value = take_value(&mut arguments, "--reduction-variant")?;
                variants.push(ReductionVariant::parse(&value)?);
            }
            _ => return Err(CalibrationCliError::UnknownFlag(flag)),
        }
    }

    let repository_root = required(repository_root, "--repository-root")?;
    let model = required(model, "--model")?;
    let gate_manifest = required(gate_manifest, "--gate-manifest")?;
    let prompts = required(prompts, "--prompts")?;
    let manifest = required(manifest, "--manifest")?;
    let sidecar = required(sidecar, "--sidecar")?;
    if observed_flags.len() != REQUIRED_CAPTURE_FLAGS.len()
        || observed_flags
            .iter()
            .zip(REQUIRED_CAPTURE_FLAGS)
            .any(|(observed, expected)| observed != expected)
    {
        return Err(CalibrationCliError::ArgumentOrder);
    }
    if gate_manifest != GATE_MANIFEST_PATH {
        return Err(CalibrationCliError::InputPathMismatch("--gate-manifest"));
    }
    if prompts != PROMPTS_PATH {
        return Err(CalibrationCliError::InputPathMismatch("--prompts"));
    }
    validate_absolute_input(&repository_root, "--repository-root")?;
    validate_absolute_input(&model, "--model")?;
    if variants.as_slice() != REQUIRED_REDUCTION_VARIANTS {
        return Err(CalibrationCliError::ReductionVariantOrder);
    }
    validate_output(&manifest, "--manifest", ".json")?;
    validate_output(&sidecar, "--sidecar", ".safetensors")?;

    Ok(CalibrationArgs {
        repository_root: PathBuf::from(repository_root),
        model: PathBuf::from(model),
        gate_manifest: PathBuf::from(gate_manifest),
        prompts: PathBuf::from(prompts),
        manifest: PathBuf::from(manifest),
        sidecar: PathBuf::from(sidecar),
        reduction_variants: REQUIRED_REDUCTION_VARIANTS,
    })
}

fn take_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, CalibrationCliError> {
    let value = arguments
        .next()
        .ok_or(CalibrationCliError::MissingValue(flag))?;
    if value.starts_with("--") {
        return Err(CalibrationCliError::MissingValue(flag));
    }
    Ok(value)
}

fn set_once(
    slot: &mut Option<String>,
    value: String,
    flag: &'static str,
) -> Result<(), CalibrationCliError> {
    if slot.replace(value).is_some() {
        return Err(CalibrationCliError::DuplicateFlag(flag));
    }
    Ok(())
}

fn required(value: Option<String>, flag: &'static str) -> Result<String, CalibrationCliError> {
    value.ok_or(CalibrationCliError::MissingFlag(flag))
}

fn validate_output(
    value: &str,
    flag: &'static str,
    suffix: &str,
) -> Result<(), CalibrationCliError> {
    let path = Path::new(value);
    if value.contains('/')
        || value.contains('\\')
        || value.contains('\0')
        || value.len() <= suffix.len()
        || path.file_name().and_then(|name| name.to_str()) != Some(value)
        || !value.ends_with(suffix)
    {
        return Err(CalibrationCliError::InvalidOutput(flag));
    }
    Ok(())
}

fn validate_absolute_input(value: &str, flag: &'static str) -> Result<(), CalibrationCliError> {
    let path = Path::new(value);
    let invalid_component = value
        .split('/')
        .skip(1)
        .any(|component| component.is_empty() || component == "." || component == "..");
    if !path.is_absolute()
        || value == "/"
        || value.contains('\\')
        || value.contains('\0')
        || invalid_component
    {
        return Err(CalibrationCliError::InvalidAbsoluteInput(flag));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CalibrationCliError, GATE_MANIFEST_PATH, NATIVE_BUILD_ARGV, NATIVE_ENGINE_REVISION,
        NATIVE_EXECUTABLE_FILENAME, PROMPTS_PATH, REQUIRED_REDUCTION_VARIANTS, ReductionVariant,
        parse_calibration_command,
    };

    fn canonical_arguments() -> Vec<&'static str> {
        vec![
            "calibrate",
            "--repository-root",
            "/workspace/rustinfer",
            "--model",
            "/models/smollm2",
            "--gate-manifest",
            GATE_MANIFEST_PATH,
            "--prompts",
            PROMPTS_PATH,
            "--manifest",
            "candidate-manifest.json",
            "--sidecar",
            "candidate-sidecar.safetensors",
            "--reduction-variant",
            "canonical-v1",
        ]
    }

    #[test]
    fn contract_v3_constants_are_exact() {
        assert_eq!(NATIVE_ENGINE_REVISION, "rustinfer-native-contract-v3");
        assert_eq!(NATIVE_EXECUTABLE_FILENAME, "rustinfer-native");
        assert_eq!(
            GATE_MANIFEST_PATH,
            "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v3.json"
        );
        assert_eq!(REQUIRED_REDUCTION_VARIANTS, [ReductionVariant::Canonical]);
        assert_eq!(canonical_arguments().len(), 15);
        assert_eq!(
            NATIVE_BUILD_ARGV,
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--package",
                "rustinfer-native",
                "--no-default-features",
                "--features",
                "cuda",
                "--bin",
                "rustinfer-native",
            ]
        );
    }

    #[test]
    fn parses_only_the_canonical_side_effect_free_contract() {
        let parsed = parse_calibration_command(canonical_arguments()).expect("canonical ABI");
        assert_eq!(
            parsed.repository_root().to_str(),
            Some("/workspace/rustinfer")
        );
        assert_eq!(parsed.model().to_str(), Some("/models/smollm2"));
        assert_eq!(parsed.gate_manifest().to_str(), Some(GATE_MANIFEST_PATH));
        assert_eq!(parsed.prompts().to_str(), Some(PROMPTS_PATH));
        assert_eq!(parsed.manifest().to_str(), Some("candidate-manifest.json"));
        assert_eq!(
            parsed.sidecar().to_str(),
            Some("candidate-sidecar.safetensors")
        );
        assert_eq!(parsed.reduction_variants(), &REQUIRED_REDUCTION_VARIANTS);
    }

    #[test]
    fn rejects_missing_duplicate_and_unknown_flags() {
        let mut missing = canonical_arguments();
        missing.drain(5..=6);
        assert_eq!(
            parse_calibration_command(missing),
            Err(CalibrationCliError::MissingFlag("--gate-manifest"))
        );

        let mut duplicate = canonical_arguments();
        duplicate.extend(["--manifest", "other.json"]);
        assert_eq!(
            parse_calibration_command(duplicate),
            Err(CalibrationCliError::DuplicateFlag("--manifest"))
        );

        let mut unknown = canonical_arguments();
        unknown.extend(["--unreviewed", "value"]);
        assert_eq!(
            parse_calibration_command(unknown),
            Err(CalibrationCliError::UnknownFlag("--unreviewed".to_owned()))
        );
    }

    #[test]
    fn rejects_flag_reordering_or_reduction_substitution() {
        let mut flag_order = canonical_arguments();
        flag_order.swap(1, 3);
        flag_order.swap(2, 4);
        assert_eq!(
            parse_calibration_command(flag_order),
            Err(CalibrationCliError::ArgumentOrder)
        );

        let mut unsupported = canonical_arguments();
        unsupported[14] = "fixed-contiguous-37-balanced-v1";
        assert_eq!(
            parse_calibration_command(unsupported),
            Err(CalibrationCliError::UnsupportedReductionVariant(
                "fixed-contiguous-37-balanced-v1".to_owned()
            ))
        );

        let mut duplicated = canonical_arguments();
        duplicated.extend(["--reduction-variant", "canonical-v1"]);
        assert_eq!(
            parse_calibration_command(duplicated),
            Err(CalibrationCliError::ArgumentOrder)
        );
    }

    #[test]
    fn rejects_unbound_inputs_and_non_sibling_outputs() {
        let mut gate = canonical_arguments();
        gate[6] = "other-gate.json";
        assert_eq!(
            parse_calibration_command(gate),
            Err(CalibrationCliError::InputPathMismatch("--gate-manifest"))
        );

        let mut output = canonical_arguments();
        output[10] = "artifacts/candidate-manifest.json";
        assert_eq!(
            parse_calibration_command(output),
            Err(CalibrationCliError::InvalidOutput("--manifest"))
        );

        for invalid in ["candidate\\manifest.json", ".json"] {
            let mut output = canonical_arguments();
            output[10] = invalid;
            assert_eq!(
                parse_calibration_command(output),
                Err(CalibrationCliError::InvalidOutput("--manifest"))
            );
        }

        for invalid in ["models/smollm2", "/models//smollm2", "/models/../smollm2"] {
            let mut model = canonical_arguments();
            model[4] = invalid;
            assert_eq!(
                parse_calibration_command(model),
                Err(CalibrationCliError::InvalidAbsoluteInput("--model"))
            );
        }
    }
}
