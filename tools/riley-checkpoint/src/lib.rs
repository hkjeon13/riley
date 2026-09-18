//! Python-free checkpoint preparation, not an inference runtime.
//!
//! Inputs, the destination parent, and published checkpoints must be trusted and
//! immutable while used. Observed symlinks are rejected, but portable std file
//! operations do not protect against hostile concurrent path replacement.

use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use riley_model::{CheckpointProvenance, LoadLimits, LoadedModel, PROVENANCE_FILENAME, ProvenanceFile};
use sha2::{Digest, Sha256};

#[cfg(feature = "hub")]
pub mod hub;

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const MAX_PLAN_BYTES: u64 = 16 * 1024 * 1024;
const STAGING: &str = ".riley-staging";
const TRANSPORT: &str = ".riley-transport";

pub(crate) fn invalid(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

/// An independently supplied manifest is both the download allowlist and the
/// expected-byte contract. It is never derived from downloaded, untrusted bytes.
pub struct Plan {
    bytes: Vec<u8>,
    provenance: CheckpointProvenance,
}

impl Plan {
    pub fn read(path: &Path) -> Result<Self> {
        let input = regular_file(path)?;
        if input.metadata()?.len() > MAX_PLAN_BYTES {
            return Err(invalid("manifest exceeds preparation size limit"));
        }
        let mut bytes = Vec::new();
        input.take(MAX_PLAN_BYTES + 1).read_to_end(&mut bytes)?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() as u64 > MAX_PLAN_BYTES {
            return Err(invalid("manifest exceeds preparation size limit"));
        }
        // Reuse the production duplicate-key, unknown-field, digest and path
        // checks instead of introducing a more permissive manifest parser.
        let provenance = CheckpointProvenance::from_json_slice(&bytes)?;
        let revision = provenance.source_revision();
        if !lower_hex(revision, 40) {
            return Err(invalid("source_revision must be a full lowercase 40-hex commit"));
        }
        let parts: Vec<_> = provenance.source_model().split('/').collect();
        if parts.len() != 2 || parts.iter().any(|part| !safe_segment(part)) {
            return Err(invalid("source_model must be a safe owner/model identifier"));
        }
        for entry in provenance.files().values() {
            let name = entry.path().to_str().ok_or_else(|| invalid("non-UTF-8 filename"))?;
            if !allowed_payload(name) {
                return Err(invalid(format!("unsupported checkpoint filename: {name}")));
            }
            if entry.byte_len() == 0 || entry.byte_len() == u64::MAX {
                return Err(invalid("payload length must be positive and bounded"));
            }
        }
        let files = provenance.files();
        for required in ["config.json", "tokenizer.json"] {
            if !files.contains_key(Path::new(required)) {
                return Err(invalid(format!("missing required allowlist entry: {required}")));
            }
        }
        let single = files.contains_key(Path::new("model.safetensors"));
        let sharded = files.contains_key(Path::new("model.safetensors.index.json"));
        if single == sharded {
            return Err(invalid("select exactly one single-file or indexed weight layout"));
        }
        Ok(Self { bytes, provenance })
    }

    pub fn provenance(&self) -> &CheckpointProvenance {
        &self.provenance
    }
}

fn safe_segment(value: &str) -> bool {
    !value.is_empty()
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && !value.contains("..")
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

pub(crate) fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn allowed_payload(name: &str) -> bool {
    matches!(name, "config.json" | "tokenizer.json" | "tokenizer_config.json"
        | "model.safetensors" | "model.safetensors.index.json")
        || (safe_segment(name) && name.starts_with("model-") && name.ends_with(".safetensors"))
}

/// Verify an existing current-format checkpoint without network or subprocesses.
pub fn verify(root: &Path) -> Result<()> {
    checked_directory(root)?;
    let plan = Plan::read(&root.join(PROVENANCE_FILENAME))?;
    for entry in plan.provenance.files().values() {
        // The loader also checks these paths; this additionally enforces the
        // preparation tool's no-observed-symlink policy on root ancestors.
        regular_file(&root.join(entry.path()))?;
    }
    drop(LoadedModel::load(root, LoadLimits::production())?);
    Ok(())
}

/// Copy a caller-owned regular-file tree. This function never falls back to Hub.
pub fn prepare(plan: &Plan, source: &Path, destination: &Path) -> Result<()> {
    checked_directory(source)?;
    publish_with(plan, destination, |_, entry| Ok(source.join(entry.path())))
}

pub(crate) fn publish_with(
    plan: &Plan,
    destination: &Path,
    mut fetch: impl FnMut(&Path, &ProvenanceFile) -> Result<PathBuf>,
) -> Result<()> {
    let mut transaction = Transaction::reserve(destination)?;
    for entry in plan.provenance.files().values() {
        let source = fetch(&transaction.root, entry)?;
        copy_verified(&source, &transaction.stage.join(entry.path()), entry)?;
    }
    let manifest = transaction.stage.join(PROVENANCE_FILENAME);
    let mut output = create_file(&manifest)?;
    output.write_all(&plan.bytes)?;
    output.sync_all()?;
    drop(output);

    // Only the actual production loader decides whether config, tokenizer,
    // weights, dtype and the exact consumed file set form a usable checkpoint.
    verify(&transaction.stage)?;
    let transport = transaction.root.join(TRANSPORT);
    if transport.exists() {
        fs::remove_dir_all(transport)?;
    }
    for entry in plan.provenance.files().values() {
        // Both paths belong to our private same-filesystem transaction.
        // hard_link is create-only; rename would overwrite an existing target.
        fs::hard_link(transaction.stage.join(entry.path()), transaction.root.join(entry.path()))?;
    }
    sync_directory(&transaction.root)?;
    fs::hard_link(&manifest, transaction.root.join(PROVENANCE_FILENAME))?;
    // The manifest is the commit marker. Never roll back a visible, valid
    // checkpoint after a subsequent directory sync or cleanup error.
    transaction.published = true;
    sync_directory(&transaction.root)?;
    fs::remove_dir_all(&transaction.stage)?;
    sync_directory(&transaction.root)?;
    Ok(())
}

fn copy_verified(source: &Path, destination: &Path, assertion: &ProvenanceFile) -> Result<()> {
    let input = regular_file(source)?;
    if input.metadata()?.len() != assertion.byte_len() {
        return Err(invalid(format!("size mismatch: {}", assertion.path().display())));
    }
    let mut input = input.take(assertion.byte_len() + 1);
    let mut output = create_file(destination)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut count = 0_u64;
    loop {
        let length = input.read(&mut buffer)?;
        if length == 0 { break; }
        count += length as u64;
        if count > assertion.byte_len() {
            return Err(invalid("payload grew during materialization"));
        }
        hasher.update(&buffer[..length]);
        output.write_all(&buffer[..length])?;
    }
    let digest = format!("{:x}", hasher.finalize());
    if count != assertion.byte_len() || digest != assertion.sha256() {
        return Err(invalid(format!("checksum/length mismatch: {}", assertion.path().display())));
    }
    output.sync_all()?;
    Ok(())
}

/// Inspect every component without canonicalizing away a symlink. The parent
/// directories remain a trusted input: this is not an openat/O_NOFOLLOW sandbox.
pub(crate) fn checked_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() { path.to_owned() } else { std::env::current_dir()?.join(path) };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => return Err(invalid("parent traversal is not allowed")),
            Component::CurDir => continue,
            _ => current.push(component.as_os_str()),
        }
        if fs::symlink_metadata(&current)?.file_type().is_symlink() {
            return Err(invalid("symlinks are not allowed in checkpoint paths"));
        }
    }
    Ok(current)
}

pub(crate) fn checked_directory(path: &Path) -> Result<PathBuf> {
    let path = checked_path(path)?;
    if !fs::symlink_metadata(&path)?.is_dir() { return Err(invalid("expected a directory")); }
    Ok(path)
}

pub(crate) fn regular_file(path: &Path) -> Result<File> {
    let path = checked_path(path)?;
    if !fs::symlink_metadata(&path)?.is_file() { return Err(invalid("expected a regular file")); }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() { return Err(invalid("opened object is not a regular file")); }
    Ok(file)
}

fn create_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

pub(crate) fn create_directory(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)] {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)] { File::open(path)?.sync_all()?; }
    #[cfg(not(unix))] let _ = path;
    Ok(())
}

struct Transaction {
    root: PathBuf,
    stage: PathBuf,
    published: bool,
}

impl Transaction {
    fn reserve(destination: &Path) -> Result<Self> {
        // file_name normalizes trailing '/.'; inspect the original components
        // as well and require an actual final normal component.
        let name = destination.file_name().ok_or_else(|| invalid("destination needs a new directory name"))?;
        if destination.components().any(|part| part == Component::ParentDir) {
            return Err(invalid("destination parent traversal is not allowed"));
        }
        let parent = destination.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let parent = checked_directory(parent)?;
        let root = parent.join(name);
        // Ownership begins ONLY after this atomic create succeeds. Existing,
        // partial and concurrent destinations are never modified or removed.
        create_directory(&root)?;
        let transaction = Self { stage: root.join(STAGING), root, published: false };
        create_directory(&transaction.stage)?;
        Ok(transaction)
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if !self.published { let _ = fs::remove_dir_all(&self.root); }
    }
}

#[cfg(test)]
mod tests;
