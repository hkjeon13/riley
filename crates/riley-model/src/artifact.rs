use std::collections::BTreeSet;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{CheckpointProvenance, LoadLimits, ModelError, ModelResult};

const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";

pub(crate) struct ArtifactBytes {
    relative_path: PathBuf,
    bytes: Box<[u8]>,
    sha256: String,
}

pub(crate) struct VerifiedArtifactSession {
    root: PathBuf,
    provenance: CheckpointProvenance,
    observed: BTreeSet<PathBuf>,
}

impl VerifiedArtifactSession {
    pub(crate) fn open(root: &Path, limits: LoadLimits) -> ModelResult<Self> {
        let root = canonicalize_artifact_root(root)?;
        let provenance = CheckpointProvenance::load(&root, limits)?;
        Ok(Self {
            root,
            provenance,
            observed: BTreeSet::new(),
        })
    }

    pub(crate) fn with_provenance(
        root: &Path,
        provenance: CheckpointProvenance,
    ) -> ModelResult<Self> {
        Ok(Self {
            root: canonicalize_artifact_root(root)?,
            provenance,
            observed: BTreeSet::new(),
        })
    }

    pub(crate) const fn provenance(&self) -> &CheckpointProvenance {
        &self.provenance
    }

    pub(crate) fn read_once(
        &mut self,
        relative_path: &Path,
        limit: u64,
        resource: &'static str,
    ) -> ModelResult<ArtifactBytes> {
        if self.observed.contains(relative_path) {
            return Err(ModelError::InvalidArtifact {
                artifact: relative_path.display().to_string(),
                reason: "artifact was consumed more than once".to_owned(),
            });
        }
        if !self.provenance.files().contains_key(relative_path) {
            return Err(ModelError::InvalidArtifact {
                artifact: relative_path.display().to_string(),
                reason: "file is absent from provenance manifest".to_owned(),
            });
        }
        let artifact = read_bounded_file(&self.root, relative_path, limit, resource)?;
        self.provenance.verify_artifact(&artifact)?;
        self.observed.insert(relative_path.to_owned());
        Ok(artifact)
    }

    pub(crate) fn finish_exact(self) -> ModelResult<BTreeSet<PathBuf>> {
        self.provenance.require_exact_file_set(&self.observed)?;
        Ok(self.observed)
    }
}

impl ArtifactBytes {
    pub(crate) fn relative_path(&self) -> &Path {
        &self.relative_path
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn into_bytes(self) -> Box<[u8]> {
        self.bytes
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(crate) fn byte_len(&self) -> u64 {
        u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
    }
}

pub(crate) fn validate_relative_file(path: &Path) -> ModelResult<()> {
    let Some(text) = path.to_str() else {
        return Err(ModelError::UnsafePath {
            path: path.to_owned(),
        });
    };
    if text.is_empty() || text.contains(['\\', ':']) || text.chars().any(char::is_control) {
        return Err(ModelError::UnsafePath {
            path: path.to_owned(),
        });
    }
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(ModelError::UnsafePath {
            path: path.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn read_bounded_file(
    root: &Path,
    relative_path: &Path,
    limit: u64,
    resource: &'static str,
) -> ModelResult<ArtifactBytes> {
    validate_relative_file(relative_path)?;
    let canonical_root = canonicalize_artifact_root(root)?;
    let requested = canonical_root.join(relative_path);
    let requested_metadata = requested
        .symlink_metadata()
        .map_err(|error| ModelError::Io {
            operation: "inspect artifact path",
            path: relative_path.to_owned(),
            reason: error.to_string(),
        })?;
    if requested_metadata.file_type().is_symlink() {
        return Err(ModelError::UnsafePath {
            path: relative_path.to_owned(),
        });
    }
    let canonical_file = requested.canonicalize().map_err(|error| ModelError::Io {
        operation: "resolve artifact",
        path: relative_path.to_owned(),
        reason: error.to_string(),
    })?;
    if canonical_file.parent() != Some(canonical_root.as_path()) {
        return Err(ModelError::UnsafePath {
            path: relative_path.to_owned(),
        });
    }

    let mut file = File::open(&canonical_file).map_err(|error| ModelError::Io {
        operation: "open artifact",
        path: relative_path.to_owned(),
        reason: error.to_string(),
    })?;
    let metadata = file.metadata().map_err(|error| ModelError::Io {
        operation: "inspect artifact",
        path: relative_path.to_owned(),
        reason: error.to_string(),
    })?;
    if !metadata.is_file() {
        return Err(ModelError::InvalidArtifact {
            artifact: relative_path.display().to_string(),
            reason: "artifact is not a regular file".to_owned(),
        });
    }
    let byte_len = metadata.len();
    if byte_len > limit {
        return Err(ModelError::LimitExceeded {
            resource,
            limit,
            actual: Some(byte_len),
        });
    }
    let allocation_len = usize::try_from(byte_len).map_err(|_| ModelError::NumericOverflow {
        field: format!("{} byte length", relative_path.display()),
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(allocation_len)
        .map_err(|error| ModelError::InvalidArtifact {
            artifact: relative_path.display().to_string(),
            reason: format!("cannot reserve {allocation_len} bytes: {error}"),
        })?;
    bytes.resize(allocation_len, 0);
    file.read_exact(&mut bytes)
        .map_err(|error| ModelError::Io {
            operation: "read artifact",
            path: relative_path.to_owned(),
            reason: if error.kind() == ErrorKind::UnexpectedEof {
                "artifact shrank while being read".to_owned()
            } else {
                error.to_string()
            },
        })?;
    let mut extra = [0_u8; 1];
    if file.read(&mut extra).map_err(|error| ModelError::Io {
        operation: "verify artifact end",
        path: relative_path.to_owned(),
        reason: error.to_string(),
    })? != 0
    {
        return Err(ModelError::InvalidArtifact {
            artifact: relative_path.display().to_string(),
            reason: "artifact grew while being read".to_owned(),
        });
    }

    let sha256 = sha256_hex(&bytes);
    Ok(ArtifactBytes {
        relative_path: relative_path.to_owned(),
        bytes: bytes.into_boxed_slice(),
        sha256,
    })
}

fn canonicalize_artifact_root(root: &Path) -> ModelResult<PathBuf> {
    let canonical_root = root.canonicalize().map_err(|error| ModelError::Io {
        operation: "canonicalize checkpoint root",
        path: root.to_owned(),
        reason: error.to_string(),
    })?;
    if !canonical_root.is_dir() {
        return Err(ModelError::InvalidArtifact {
            artifact: root.display().to_string(),
            reason: "checkpoint root is not a directory".to_owned(),
        });
    }
    Ok(canonical_root)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(LOWER_HEX[usize::from(byte >> 4)]));
        output.push(char::from(LOWER_HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{sha256_hex, validate_relative_file};

    #[test]
    fn relative_artifact_names_are_single_safe_components() {
        assert!(validate_relative_file(Path::new("model.safetensors")).is_ok());
        for rejected in [
            "",
            ".",
            "..",
            "../escape",
            "nested/file",
            "nested\\file",
            "C:model",
            "line\nfeed",
        ] {
            assert!(
                validate_relative_file(Path::new(rejected)).is_err(),
                "{rejected}"
            );
        }
    }

    #[test]
    fn sha256_is_lowercase_and_stable() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
