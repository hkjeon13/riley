use std::error;
use std::fmt;
use std::path::PathBuf;

use rustinfer_tensor::DType;

/// Result type for model artifact validation.
pub type ModelResult<T> = Result<T, ModelError>;

/// Stable artifact categories used in diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArtifactKind {
    /// Hugging Face `config.json`.
    Config,
    /// Hugging Face `tokenizer.json`.
    Tokenizer,
    /// A safetensors file or header.
    Safetensors,
    /// A safetensors shard index.
    ShardIndex,
    /// A rustinfer checkpoint provenance manifest.
    Manifest,
}

impl ArtifactKind {
    /// Returns the stable lowercase diagnostic name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::Tokenizer => "tokenizer",
            Self::Safetensors => "safetensors",
            Self::ShardIndex => "shard-index",
            Self::Manifest => "manifest",
        }
    }
}

/// An explicit model configuration, checkpoint, or tokenizer failure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ModelError {
    /// Reading a local artifact failed.
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Artifact path.
        path: PathBuf,
        /// Platform error rendered without an OS-specific error object.
        reason: String,
    },
    /// JSON syntax or structure was invalid, including duplicate keys.
    InvalidJson {
        /// Artifact category.
        artifact: ArtifactKind,
        /// Parser detail.
        reason: String,
    },
    /// A supported field contained an internally inconsistent value.
    InvalidConfig {
        /// Configuration field.
        field: String,
        /// Validation detail.
        reason: String,
    },
    /// A well-formed value changes semantics outside the initial support set.
    UnsupportedConfig {
        /// Configuration field.
        field: String,
        /// Stable rendered value.
        value: String,
    },
    /// A bounded parser input exceeded its declared maximum.
    LimitExceeded {
        /// Resource whose bound was exceeded.
        resource: &'static str,
        /// Inclusive maximum.
        limit: u64,
        /// Observed size when it is known.
        actual: Option<u64>,
    },
    /// Integer conversion or checked size arithmetic overflowed.
    NumericOverflow {
        /// Field or resource being computed.
        field: String,
    },
    /// A serialized artifact violated its format contract.
    InvalidArtifact {
        /// Artifact-relative diagnostic name.
        artifact: String,
        /// Validation detail.
        reason: String,
    },
    /// An artifact attempted to escape its checkpoint root.
    UnsafePath {
        /// Rejected path.
        path: PathBuf,
    },
    /// A file digest differed from its provenance manifest.
    ChecksumMismatch {
        /// Manifest-relative file path.
        path: PathBuf,
        /// Expected lowercase SHA-256.
        expected: String,
        /// Computed lowercase SHA-256.
        actual: String,
    },
    /// One or more canonical tensors were absent.
    MissingTensors {
        /// Sorted canonical source names.
        names: Vec<String>,
    },
    /// One or more checkpoint tensors are not part of the canonical model.
    ExtraTensors {
        /// Sorted source names.
        names: Vec<String>,
    },
    /// A tensor appeared more than once.
    DuplicateTensor {
        /// Duplicate source name.
        name: String,
    },
    /// A tensor dtype differs from its canonical requirement.
    TensorDTypeMismatch {
        /// Source tensor name.
        name: String,
        /// Required dtype.
        expected: DType,
        /// Serialized dtype.
        actual: DType,
    },
    /// A tensor shape differs from its canonical requirement.
    TensorShapeMismatch {
        /// Source tensor name.
        name: String,
        /// Required dimensions.
        expected: Vec<usize>,
        /// Serialized dimensions.
        actual: Vec<usize>,
    },
    /// A physical tied LM head differed from the token embedding.
    TiedWeightMismatch,
    /// A converter transform is recorded but not implemented by this loader.
    UnsupportedTransform {
        /// Stable transform name.
        transform: String,
    },
    /// Tokenization input or metadata was invalid.
    InvalidTokenizer {
        /// Validation detail.
        reason: String,
    },
    /// Decode referenced an ID absent from the vocabulary.
    InvalidTokenId {
        /// Unknown ID.
        id: u32,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                reason,
            } => write!(formatter, "cannot {operation} {}: {reason}", path.display()),
            Self::InvalidJson { artifact, reason } => {
                write!(formatter, "invalid {} JSON: {reason}", artifact.name())
            }
            Self::InvalidConfig { field, reason } => {
                write!(formatter, "invalid config field {field}: {reason}")
            }
            Self::UnsupportedConfig { field, value } => {
                write!(formatter, "unsupported config field {field}={value}")
            }
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => match actual {
                Some(actual) => write!(
                    formatter,
                    "{resource} exceeds limit {limit} bytes/items (observed {actual})"
                ),
                None => write!(formatter, "{resource} exceeds limit {limit} bytes/items"),
            },
            Self::NumericOverflow { field } => {
                write!(formatter, "numeric overflow while validating {field}")
            }
            Self::InvalidArtifact { artifact, reason } => {
                write!(formatter, "invalid artifact {artifact}: {reason}")
            }
            Self::UnsafePath { path } => write!(
                formatter,
                "artifact path is not a safe relative path: {}",
                path.display()
            ),
            Self::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "checksum mismatch for {}: expected {expected}, got {actual}",
                path.display()
            ),
            Self::MissingTensors { names } => {
                write!(formatter, "missing tensors: {}", names.join(", "))
            }
            Self::ExtraTensors { names } => {
                write!(formatter, "extra tensors: {}", names.join(", "))
            }
            Self::DuplicateTensor { name } => write!(formatter, "duplicate tensor: {name}"),
            Self::TensorDTypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor {name} dtype mismatch: expected {expected}, got {actual}"
            ),
            Self::TensorShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor {name} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::TiedWeightMismatch => {
                formatter.write_str("physical LM head differs from tied token embedding")
            }
            Self::UnsupportedTransform { transform } => {
                write!(formatter, "unsupported checkpoint transform: {transform}")
            }
            Self::InvalidTokenizer { reason } => write!(formatter, "invalid tokenizer: {reason}"),
            Self::InvalidTokenId { id } => write!(formatter, "invalid token id: {id}"),
        }
    }
}

impl error::Error for ModelError {}
