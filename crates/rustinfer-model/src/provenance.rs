use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rustinfer_tensor::DType;
use serde::Deserialize;
use serde_json::Value;

use crate::artifact::{ArtifactBytes, read_bounded_file, validate_relative_file};
use crate::{ArtifactKind, LoadLimits, ModelError, ModelResult, strict_json};

/// Required checkpoint provenance filename.
pub const PROVENANCE_FILENAME: &str = "rustinfer-checkpoint.json";

/// One immutable file assertion from a checkpoint manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceFile {
    path: PathBuf,
    byte_len: u64,
    sha256: String,
}

impl ProvenanceFile {
    /// Returns the checkpoint-root-relative path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the exact serialized byte length.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Returns the exact lowercase SHA-256 digest.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Validated source identity and file checksums for a checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointProvenance {
    source_model: String,
    source_revision: String,
    converter_revision: Option<String>,
    dtype: DType,
    files: BTreeMap<PathBuf, ProvenanceFile>,
}

impl CheckpointProvenance {
    /// Loads the mandatory manifest from a local checkpoint directory.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe/missing/oversized file or an invalid
    /// manifest contract.
    pub fn load(root: &Path, limits: LoadLimits) -> ModelResult<Self> {
        let artifact = read_bounded_file(
            root,
            Path::new(PROVENANCE_FILENAME),
            limits.manifest_bytes(),
            "checkpoint provenance manifest",
        )?;
        Self::from_json_slice_with_limits(artifact.bytes(), limits)
    }

    /// Parses a manifest using production limits.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed JSON, duplicate/unknown fields,
    /// mutable-looking revisions, unsupported transforms, or invalid files.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::production())
    }

    /// Parses a manifest with explicit resource limits.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::LimitExceeded`] before parsing oversized input.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        let input_len = u64::try_from(input.len()).map_err(|_| ModelError::NumericOverflow {
            field: "manifest byte length".to_owned(),
        })?;
        if input_len > limits.manifest_bytes() {
            return Err(ModelError::LimitExceeded {
                resource: "checkpoint provenance manifest",
                limit: limits.manifest_bytes(),
                actual: Some(input_len),
            });
        }
        let raw: RawManifest = strict_json::from_slice(input, ArtifactKind::Manifest)?;
        Self::from_raw(raw, limits)
    }

    fn from_raw(raw: RawManifest, limits: LoadLimits) -> ModelResult<Self> {
        if let Some((field, value)) = raw.unknown.iter().next() {
            return invalid_manifest(&format!("unknown field {field}={}", stable_value(value)));
        }
        if raw.format != "rustinfer-checkpoint-v1" {
            return invalid_manifest("format must be rustinfer-checkpoint-v1");
        }
        validate_source_model(&raw.source_model)?;
        validate_revision("source_revision", &raw.source_revision, true)?;
        if let Some(revision) = &raw.converter_revision {
            validate_revision("converter_revision", revision, false)?;
        }
        if let Some(transform) = first_duplicate(&raw.transforms) {
            return invalid_manifest(&format!("duplicate transform {transform:?}"));
        }
        if let Some(transform) = raw.transforms.first() {
            return Err(ModelError::UnsupportedTransform {
                transform: transform.clone(),
            });
        }
        let dtype = parse_dtype(&raw.dtype)?;
        let maximum_files =
            limits
                .shards()
                .checked_add(4)
                .ok_or_else(|| ModelError::NumericOverflow {
                    field: "maximum provenance files".to_owned(),
                })?;
        if raw.files.len() > maximum_files {
            return Err(ModelError::LimitExceeded {
                resource: "checkpoint provenance files",
                limit: u64::try_from(maximum_files).unwrap_or(u64::MAX),
                actual: u64::try_from(raw.files.len()).ok(),
            });
        }
        if raw.files.is_empty() {
            return invalid_manifest("files must not be empty");
        }

        let mut files = BTreeMap::new();
        for raw_file in raw.files {
            let path = PathBuf::from(raw_file.path);
            validate_relative_file(&path)?;
            validate_sha256(&raw_file.sha256)?;
            let file = ProvenanceFile {
                path: path.clone(),
                byte_len: raw_file.bytes,
                sha256: raw_file.sha256,
            };
            if files.insert(path.clone(), file).is_some() {
                return invalid_manifest(&format!("duplicate file path {}", path.display()));
            }
        }

        Ok(Self {
            source_model: raw.source_model,
            source_revision: raw.source_revision,
            converter_revision: raw.converter_revision,
            dtype,
            files,
        })
    }

    /// Returns the immutable source model identifier.
    #[must_use]
    pub fn source_model(&self) -> &str {
        &self.source_model
    }

    /// Returns the immutable source revision.
    #[must_use]
    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    /// Returns the converter revision when an offline converter produced it.
    #[must_use]
    pub fn converter_revision(&self) -> Option<&str> {
        self.converter_revision.as_deref()
    }

    /// Returns the declared checkpoint dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the exact, path-sorted file assertions.
    #[must_use]
    pub const fn files(&self) -> &BTreeMap<PathBuf, ProvenanceFile> {
        &self.files
    }

    pub(crate) fn verify_artifact(&self, artifact: &ArtifactBytes) -> ModelResult<()> {
        self.verify_observed(
            artifact.relative_path(),
            artifact.byte_len(),
            artifact.sha256(),
        )
    }

    pub(crate) fn verify_observed(
        &self,
        path: &Path,
        byte_len: u64,
        sha256: &str,
    ) -> ModelResult<()> {
        let expected = self
            .files
            .get(path)
            .ok_or_else(|| ModelError::InvalidArtifact {
                artifact: path.display().to_string(),
                reason: "file is absent from provenance manifest".to_owned(),
            })?;
        if byte_len != expected.byte_len {
            return Err(ModelError::InvalidArtifact {
                artifact: path.display().to_string(),
                reason: format!(
                    "byte length mismatch: expected {}, got {byte_len}",
                    expected.byte_len
                ),
            });
        }
        if sha256 != expected.sha256 {
            return Err(ModelError::ChecksumMismatch {
                path: path.to_owned(),
                expected: expected.sha256.clone(),
                actual: sha256.to_owned(),
            });
        }
        Ok(())
    }

    /// Verifies that a loader consumed exactly the files declared by the manifest.
    ///
    /// # Errors
    ///
    /// Returns a deterministic missing/extra file-set diagnostic.
    pub fn require_exact_file_set(&self, observed: &BTreeSet<PathBuf>) -> ModelResult<()> {
        let expected: BTreeSet<_> = self.files.keys().cloned().collect();
        if expected == *observed {
            return Ok(());
        }
        let missing: Vec<_> = expected
            .difference(observed)
            .map(|path| path.display().to_string())
            .collect();
        let extra: Vec<_> = observed
            .difference(&expected)
            .map(|path| path.display().to_string())
            .collect();
        Err(ModelError::InvalidArtifact {
            artifact: PROVENANCE_FILENAME.to_owned(),
            reason: format!("file set mismatch: missing={missing:?}, extra={extra:?}"),
        })
    }
}

#[derive(Deserialize)]
struct RawManifest {
    format: String,
    source_model: String,
    source_revision: String,
    converter_revision: Option<String>,
    transforms: Vec<String>,
    dtype: String,
    files: Vec<RawManifestFile>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

fn validate_source_model(value: &str) -> ModelResult<()> {
    if value.is_empty()
        || value.len() > 512
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return invalid_manifest("source_model must be a bounded non-empty identifier");
    }
    Ok(())
}

fn validate_revision(field: &str, value: &str, immutable_only: bool) -> ModelResult<()> {
    let valid_length = if immutable_only {
        matches!(value.len(), 40 | 64)
    } else {
        (7..=64).contains(&value.len())
    };
    if !valid_length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid_manifest(&format!(
            "{field} must be a lowercase immutable hexadecimal revision"
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> ModelResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid_manifest("file sha256 must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

fn parse_dtype(value: &str) -> ModelResult<DType> {
    match value {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        other => Err(ModelError::UnsupportedConfig {
            field: "manifest.dtype".to_owned(),
            value: other.to_owned(),
        }),
    }
}

fn first_duplicate(values: &[String]) -> Option<&str> {
    let mut seen = BTreeSet::new();
    values
        .iter()
        .find(|value| !seen.insert(value.as_str()))
        .map(String::as_str)
}

fn stable_value(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unrenderable-json>".to_owned())
}

fn invalid_manifest<T>(reason: &str) -> ModelResult<T> {
    Err(ModelError::InvalidArtifact {
        artifact: PROVENANCE_FILENAME.to_owned(),
        reason: reason.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::CheckpointProvenance;
    use crate::{ModelError, PROVENANCE_FILENAME};

    const VALID: &str = r#"{
      "format":"rustinfer-checkpoint-v1",
      "source_model":"HuggingFaceTB/SmolLM2-135M",
      "source_revision":"93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
      "converter_revision":null,
      "transforms":[],
      "dtype":"bf16",
      "files":[{"path":"model.safetensors","bytes":16,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]
    }"#;

    #[test]
    fn valid_manifest_preserves_immutable_identity() {
        let manifest = CheckpointProvenance::from_json_slice(VALID.as_bytes()).unwrap();
        assert_eq!(manifest.source_model(), "HuggingFaceTB/SmolLM2-135M");
        assert_eq!(
            manifest.source_revision(),
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
        );
        assert_eq!(manifest.files().len(), 1);
    }

    #[test]
    fn mutable_revision_and_transform_fail_closed() {
        let mutable = VALID.replace("93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "main");
        assert!(CheckpointProvenance::from_json_slice(mutable.as_bytes()).is_err());
        let transformed = VALID.replace("\"transforms\":[]", "\"transforms\":[\"packed_qkv\"]");
        assert!(matches!(
            CheckpointProvenance::from_json_slice(transformed.as_bytes()),
            Err(ModelError::UnsupportedTransform { transform }) if transform == "packed_qkv"
        ));
    }

    #[test]
    fn unsafe_or_duplicate_file_paths_are_rejected() {
        let unsafe_path = VALID.replace("model.safetensors", "../model.safetensors");
        assert!(matches!(
            CheckpointProvenance::from_json_slice(unsafe_path.as_bytes()),
            Err(ModelError::UnsafePath { .. })
        ));
        let duplicate = VALID.replace(
            "]\n    }",
            ", {\"path\":\"model.safetensors\",\"bytes\":16,\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}]\n    }",
        );
        let error = CheckpointProvenance::from_json_slice(duplicate.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("duplicate file path"));
        assert!(error.to_string().contains(PROVENANCE_FILENAME));
    }
}
