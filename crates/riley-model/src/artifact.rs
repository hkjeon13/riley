use std::collections::BTreeSet;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Component, Path, PathBuf};
use std::thread;

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
        let bytes = self.read_unverified(relative_path, limit, resource)?;
        let sha256 = sha256_hex(&bytes);
        self.record_verified(relative_path, bytes, sha256)
    }

    /// Reads the shards in path order, overlapping two checksums at most.
    /// Both reads spend the same remaining byte budget before allocation.
    pub(crate) fn read_shards<'a>(
        &mut self,
        mut paths: impl Iterator<Item = &'a PathBuf>,
        limits: LoadLimits,
        mut consume: impl FnMut(ArtifactBytes) -> ModelResult<()>,
    ) -> ModelResult<()> {
        let mut remaining = limits.total_weight_bytes();
        while let Some(first) = paths.next() {
            let bytes = self.read_shard_bytes(first, limits, &mut remaining)?;
            let Some(second) = paths.next() else {
                let sha256 = sha256_hex(&bytes);
                return consume(self.record_verified(first, bytes, sha256)?);
            };
            let (sha256, next) = thread::scope(|scope| {
                let first_checksum =
                    thread::Builder::new().spawn_scoped(scope, || sha256_hex(&bytes));
                let next = self
                    .read_shard_bytes(second, limits, &mut remaining)
                    .map(|bytes| {
                        let sha256 = sha256_hex(&bytes);
                        (bytes, sha256)
                    });
                let sha256 = first_checksum.map_or_else(
                    |_| sha256_hex(&bytes),
                    |job| {
                        job.join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                    },
                );
                (sha256, next)
            });
            // Preserve path-ordered verification and parse errors if both
            // of the in-flight shards fail.
            consume(self.record_verified(first, bytes, sha256)?)?;
            let (bytes, sha256) = next?;
            consume(self.record_verified(second, bytes, sha256)?)?;
        }
        Ok(())
    }

    fn read_shard_bytes(
        &self,
        relative_path: &Path,
        limits: LoadLimits,
        remaining: &mut u64,
    ) -> ModelResult<Box<[u8]>> {
        let bytes = self.read_unverified(
            relative_path,
            limits.shard_bytes().min(*remaining),
            "safetensors shard",
        )?;
        let byte_len = u64::try_from(bytes.len()).map_err(|_| ModelError::NumericOverflow {
            field: "checkpoint shard byte length".to_owned(),
        })?;
        *remaining -= byte_len;
        Ok(bytes)
    }

    fn read_unverified(
        &self,
        relative_path: &Path,
        limit: u64,
        resource: &'static str,
    ) -> ModelResult<Box<[u8]>> {
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
        read_file_bytes(&self.root, relative_path, limit, resource)
    }

    fn record_verified(
        &mut self,
        relative_path: &Path,
        bytes: Box<[u8]>,
        sha256: String,
    ) -> ModelResult<ArtifactBytes> {
        let artifact = ArtifactBytes {
            relative_path: relative_path.to_owned(),
            bytes,
            sha256,
        };
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
    let bytes = read_file_bytes(root, relative_path, limit, resource)?;
    let sha256 = sha256_hex(&bytes);
    Ok(ArtifactBytes {
        relative_path: relative_path.to_owned(),
        bytes,
        sha256,
    })
}

fn read_file_bytes(
    root: &Path,
    relative_path: &Path,
    limit: u64,
    resource: &'static str,
) -> ModelResult<Box<[u8]>> {
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

    Ok(bytes.into_boxed_slice())
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
