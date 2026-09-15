//! Immutable input and artifact boundary for the Candle Qwen2.5-3B diagnostic.
//!
//! This crate deliberately owns no Riley serving dependency. It binds one
//! existing local checkpoint and one existing P2048 workload, then gives the
//! CUDA-only binary a validated set of files and deterministic JSON helpers.
//! Candle's numerical contract differs from the pinned Transformers eager
//! contract, so these records are for triangulation only, never a byte-exact
//! serving correctness gate.

#![deny(unsafe_code)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const MODEL_ID: &str = "Qwen/Qwen2.5-3B-Instruct";
pub const MODEL_REVISION: &str = "aa8e72537993ba99e69dfaafa59ed015b17504d1";
pub const WORKLOAD_SCHEMA: &str = "riley.n06a-d128-serving-workload.v1";
pub const WORKLOAD_CASE: &str = "qwen3b-c8-p2048-o128";
pub const WORKLOAD_SHA256: &str =
    "7a0a8fec31d45e397e1ec57335fa1c9de2d3da7daa9a32e9002a62763c05261e";
pub const PROMPT_TOKENS_SHA256: &str =
    "56619bc156fb385345c12e523c71604fa9ef8ad3c52d9c913d0f6ff09f1c1fd9";
pub const OUTPUT_TOKENS_SHA256: &str =
    "83ffd904307abe16d04e2ad022d5113498337532245c170d3a729ffbb5a82df1";
pub const PROMPT_TOKENS: usize = 2_048;
pub const OUTPUT_TOKENS: usize = 128;
pub const PROMPT_TOKEN_ID: u32 = 3_409;
pub const VOCABULARY_SIZE: usize = 151_936;
pub const ADDRESSABLE_TOKENS: usize = 151_665;
pub const TOP_K: usize = 32;
pub const CHECKPOINT_RECEIPT_SHA256: &str =
    "f85648630f4aef16ddc62b04848ce082b577d08d6e14915c2ccd402fdbd377f8";

const MAX_WORKLOAD_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CHECKPOINT_METADATA_BYTES: u64 = 16 * 1024 * 1024;
const EXPECTED_OUTPUT_PREFIX: [u32; 8] = [374, 264, 198, 750, 198, 16, 17, 17];
const PROBE_IDS: [usize; 15] = [
    0, 8, 16, 17, 198, 264, 304, 374, 750, 3_409, 151_643, 151_645, 151_664, 151_665, 151_935,
];

#[derive(Debug)]
pub struct OracleError(String);

impl OracleError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for OracleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for OracleError {}

pub type OracleResult<T> = Result<T, OracleError>;

#[derive(Clone, Debug, Serialize)]
pub struct FileRecord {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct ServingWorkload {
    pub source_bytes: u64,
    pub source_sha256: String,
    pub prompt_token_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct CheckpointBinding {
    pub config_bytes: Vec<u8>,
    pub receipt: FileRecord,
    pub files: Vec<FileRecord>,
    weight_paths: [PathBuf; 2],
}

impl CheckpointBinding {
    #[must_use]
    pub fn weight_paths(&self) -> &[PathBuf; 2] {
        &self.weight_paths
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceProvenance {
    pub git_revision: String,
    pub tracked_checkout_dirty: bool,
    pub tracked_checkout_status_sha256: String,
    pub sources: BTreeMap<String, FileRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LogitCapture {
    pub dtype: &'static str,
    pub element_count: usize,
    pub canonical_byte_order: &'static str,
    pub raw_bf16_le_sha256: String,
    pub addressable_bf16_le_sha256: String,
    pub non_addressable_bf16_le_sha256: String,
    pub argmax_token_id: u32,
    pub argmax_value_bf16_as_f32: f32,
    pub top_token_ids: Vec<u32>,
    pub top_values_bf16_as_f32: Vec<f32>,
    pub probe_values_bf16_as_f32: BTreeMap<String, f32>,
}

#[derive(Clone, Copy)]
struct ExpectedFile {
    path: &'static str,
    size_bytes: u64,
    sha256: &'static str,
    verify_content: bool,
}

const EXPECTED_FILES: [ExpectedFile; 6] = [
    ExpectedFile {
        path: "config.json",
        size_bytes: 661,
        sha256: "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b",
        verify_content: true,
    },
    ExpectedFile {
        path: "tokenizer.json",
        size_bytes: 7_031_645,
        sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        verify_content: true,
    },
    ExpectedFile {
        path: "tokenizer_config.json",
        size_bytes: 7_305,
        sha256: "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        verify_content: true,
    },
    ExpectedFile {
        path: "model.safetensors.index.json",
        size_bytes: 35_581,
        sha256: "bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56",
        verify_content: true,
    },
    ExpectedFile {
        path: "model-00001-of-00002.safetensors",
        size_bytes: 3_968_658_944,
        sha256: "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
        verify_content: false,
    },
    ExpectedFile {
        path: "model-00002-of-00002.safetensors",
        size_bytes: 2_203_268_048,
        sha256: "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
        verify_content: false,
    },
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkloadDocument {
    schema_version: String,
    case: String,
    model_id: String,
    model_revision: String,
    prompt: String,
    prompt_token_ids: Vec<u64>,
    output_token_ids: Vec<u64>,
    output_text: String,
    finish_reason: String,
    sampling: SamplingDocument,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingDocument {
    temperature: f64,
    top_p: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptDocument {
    format: String,
    source_model: String,
    source_revision: String,
    converter_revision: Option<String>,
    transforms: Vec<Value>,
    dtype: String,
    files: Vec<ReceiptFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    let digest = digest.finalize();
    let mut output = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        use std::fmt::Write as _;

        write!(&mut output, "{byte:02x}").expect("writing a String is infallible");
    }
    output
}

pub fn load_workload(path: &Path) -> OracleResult<ServingWorkload> {
    let bytes = read_regular_file(path, MAX_WORKLOAD_BYTES, "workload")?;
    let source_sha256 = sha256_hex(&bytes);
    if source_sha256 != WORKLOAD_SHA256 {
        return Err(OracleError::new(
            "workload SHA-256 differs from the immutable N06-A input",
        ));
    }
    let document: WorkloadDocument = serde_json::from_slice(&bytes)
        .map_err(|error| OracleError::new(format!("parse workload JSON: {error}")))?;
    if document.schema_version != WORKLOAD_SCHEMA
        || document.case != WORKLOAD_CASE
        || document.model_id != MODEL_ID
        || document.model_revision != MODEL_REVISION
    {
        return Err(OracleError::new(
            "workload model identity or case differs from the fixed diagnostic",
        ));
    }
    if document.prompt.is_empty()
        || document.output_text.is_empty()
        || document.finish_reason != "length"
        || document.sampling.temperature != 0.0
        || document.sampling.top_p != 1.0
    {
        return Err(OracleError::new(
            "workload generation fields differ from the fixed diagnostic",
        ));
    }
    let prompt_token_ids = u32_token_ids(&document.prompt_token_ids, "prompt_token_ids")?;
    let output_token_ids = u32_token_ids(&document.output_token_ids, "output_token_ids")?;
    if prompt_token_ids.len() != PROMPT_TOKENS
        || prompt_token_ids
            .iter()
            .any(|&token_id| token_id != PROMPT_TOKEN_ID)
        || token_ids_sha256(&prompt_token_ids) != PROMPT_TOKENS_SHA256
    {
        return Err(OracleError::new(
            "workload prompt IDs differ from the immutable P2048 binding",
        ));
    }
    if output_token_ids.len() != OUTPUT_TOKENS
        || output_token_ids[..EXPECTED_OUTPUT_PREFIX.len()] != EXPECTED_OUTPUT_PREFIX
        || token_ids_sha256(&output_token_ids) != OUTPUT_TOKENS_SHA256
    {
        return Err(OracleError::new(
            "workload output receipt differs from the immutable P2048 binding",
        ));
    }
    Ok(ServingWorkload {
        source_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        source_sha256,
        prompt_token_ids,
    })
}

pub fn verify_checkpoint(root: &Path) -> OracleResult<CheckpointBinding> {
    let root = canonical_checkpoint_root(root)?;
    let receipt_path = root.join("riley-checkpoint.json");
    let receipt_bytes = read_regular_file(
        &receipt_path,
        MAX_CHECKPOINT_METADATA_BYTES,
        "checkpoint receipt",
    )?;
    let receipt_sha256 = sha256_hex(&receipt_bytes);
    if receipt_sha256 != CHECKPOINT_RECEIPT_SHA256 {
        return Err(OracleError::new(
            "riley-checkpoint.json SHA-256 differs from the pinned receipt",
        ));
    }
    let receipt: ReceiptDocument = serde_json::from_slice(&receipt_bytes)
        .map_err(|error| OracleError::new(format!("parse riley-checkpoint.json: {error}")))?;
    if receipt.format != "riley-checkpoint-v1"
        || receipt.source_model != MODEL_ID
        || receipt.source_revision != MODEL_REVISION
        || receipt.converter_revision.is_some()
        || !receipt.transforms.is_empty()
        || receipt.dtype != "bf16"
    {
        return Err(OracleError::new(
            "checkpoint receipt identity differs from the fixed diagnostic",
        ));
    }
    let expected_receipt = expected_file_map();
    let mut actual_receipt = BTreeMap::new();
    for file in receipt.files {
        if actual_receipt
            .insert(file.path, (file.bytes, file.sha256))
            .is_some()
        {
            return Err(OracleError::new(
                "checkpoint receipt contains a duplicate file path",
            ));
        }
    }
    if actual_receipt != expected_receipt {
        return Err(OracleError::new(
            "checkpoint receipt files differ from the pinned Qwen2.5-3B receipt",
        ));
    }

    let mut files = Vec::with_capacity(EXPECTED_FILES.len());
    let mut config_bytes = None;
    for expected in EXPECTED_FILES {
        let path = root.join(expected.path);
        let metadata = regular_file_metadata(&path, expected.path)?;
        if metadata.len() != expected.size_bytes {
            return Err(OracleError::new(format!(
                "{} has {} bytes, expected {}",
                expected.path,
                metadata.len(),
                expected.size_bytes
            )));
        }
        if expected.verify_content {
            let bytes = read_regular_file(&path, MAX_CHECKPOINT_METADATA_BYTES, expected.path)?;
            let observed = sha256_hex(&bytes);
            if observed != expected.sha256 {
                return Err(OracleError::new(format!(
                    "{} SHA-256 differs",
                    expected.path
                )));
            }
            if expected.path == "config.json" {
                config_bytes = Some(bytes);
            }
        }
        files.push(FileRecord {
            path: expected.path.to_owned(),
            size_bytes: expected.size_bytes,
            sha256: expected.sha256.to_owned(),
        });
    }
    let config_bytes =
        config_bytes.ok_or_else(|| OracleError::new("config.json was not loaded"))?;
    Ok(CheckpointBinding {
        config_bytes,
        receipt: FileRecord {
            path: "riley-checkpoint.json".to_owned(),
            size_bytes: u64::try_from(receipt_bytes.len()).unwrap_or(u64::MAX),
            sha256: receipt_sha256,
        },
        files,
        weight_paths: [
            root.join("model-00001-of-00002.safetensors"),
            root.join("model-00002-of-00002.safetensors"),
        ],
    })
}

pub fn source_provenance(repo_root: &Path) -> OracleResult<SourceProvenance> {
    let root = repo_root
        .canonicalize()
        .map_err(|error| OracleError::new(format!("canonicalize repo root: {error}")))?;
    if !root.is_dir() {
        return Err(OracleError::new("repo root is not a directory"));
    }
    let git_revision = git_output(&root, ["rev-parse", "HEAD"])?;
    if git_revision.len() != 40 || !git_revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(OracleError::new("git HEAD is not a full revision"));
    }
    // The serving host retains large, unrelated untracked benchmark evidence.
    // Relevant source files are hashed below, so avoid recursively scanning
    // those external result trees just to construct a diagnostic receipt.
    let status = git_output_bytes(&root, ["status", "--porcelain=v1", "--untracked-files=no"])?;
    let mut sources = BTreeMap::new();
    for relative in [
        "tools/candle-qwen3b-oracle/Cargo.toml",
        "tools/candle-qwen3b-oracle/Cargo.lock",
        "tools/candle-qwen3b-oracle/src/lib.rs",
        "tools/candle-qwen3b-oracle/src/main.rs",
    ] {
        let path = root.join(relative);
        let bytes = read_regular_file(&path, MAX_CHECKPOINT_METADATA_BYTES, relative)?;
        sources.insert(
            relative.to_owned(),
            FileRecord {
                path: relative.to_owned(),
                size_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                sha256: sha256_hex(&bytes),
            },
        );
    }
    Ok(SourceProvenance {
        git_revision,
        tracked_checkout_dirty: !status.is_empty(),
        tracked_checkout_status_sha256: sha256_hex(&status),
        sources,
    })
}

pub fn capture_logits(bits: &[u16], values: &[f32]) -> OracleResult<LogitCapture> {
    if bits.len() != VOCABULARY_SIZE || values.len() != VOCABULARY_SIZE {
        return Err(OracleError::new(
            "Candle logits must contain the full Qwen2.5-3B vocabulary row",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OracleError::new(
            "Candle logits contain a non-finite BF16 value",
        ));
    }
    let canonical_bytes = bf16_le_bytes(bits);
    let addressable_bytes = bf16_le_bytes(&bits[..ADDRESSABLE_TOKENS]);
    let non_addressable_bytes = bf16_le_bytes(&bits[ADDRESSABLE_TOKENS..]);
    let mut ranked: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    let (argmax, argmax_value) = ranked
        .first()
        .copied()
        .ok_or_else(|| OracleError::new("Candle logits are empty"))?;
    let mut probes = BTreeMap::new();
    for id in PROBE_IDS {
        probes.insert(id.to_string(), values[id]);
    }
    Ok(LogitCapture {
        dtype: "bfloat16",
        element_count: bits.len(),
        canonical_byte_order: "little-endian-u16",
        raw_bf16_le_sha256: sha256_hex(&canonical_bytes),
        addressable_bf16_le_sha256: sha256_hex(&addressable_bytes),
        non_addressable_bf16_le_sha256: sha256_hex(&non_addressable_bytes),
        argmax_token_id: u32::try_from(argmax)
            .map_err(|_| OracleError::new("argmax does not fit U32"))?,
        argmax_value_bf16_as_f32: argmax_value,
        top_token_ids: ranked
            .iter()
            .take(TOP_K)
            .map(|(id, _)| {
                u32::try_from(*id).map_err(|_| OracleError::new("top-k ID does not fit U32"))
            })
            .collect::<OracleResult<Vec<_>>>()?,
        top_values_bf16_as_f32: ranked.iter().take(TOP_K).map(|(_, value)| *value).collect(),
        probe_values_bf16_as_f32: probes,
    })
}

#[must_use]
pub fn bf16_le_bytes(bits: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len().saturating_mul(2));
    for value in bits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn publish_create_only(output: &Path, repo_root: &Path, bytes: &[u8]) -> OracleResult<PathBuf> {
    if !output.is_absolute() {
        return Err(OracleError::new(
            "--output must be an absolute path outside the repository",
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| OracleError::new("--output has no parent directory"))?
        .canonicalize()
        .map_err(|error| OracleError::new(format!("canonicalize output parent: {error}")))?;
    if !parent.is_dir() {
        return Err(OracleError::new("--output parent is not a directory"));
    }
    let filename = output
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| OracleError::new("--output has no filename"))?;
    let target = parent.join(filename);
    let canonical_repo = repo_root
        .canonicalize()
        .map_err(|error| OracleError::new(format!("canonicalize repo root: {error}")))?;
    if target.starts_with(&canonical_repo) {
        return Err(OracleError::new(
            "--output must be outside the repository checkout",
        ));
    }
    match fs::symlink_metadata(&target) {
        Ok(_) => {
            return Err(OracleError::new(
                "--output already exists; artifacts are create-only",
            ));
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(OracleError::new(format!("inspect --output: {error}"))),
    }
    let temporary = temporary_output_path(&parent, filename)?;
    let result = (|| -> OracleResult<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| OracleError::new(format!("create temporary artifact: {error}")))?;
        file.write_all(bytes)
            .map_err(|error| OracleError::new(format!("write temporary artifact: {error}")))?;
        file.sync_all()
            .map_err(|error| OracleError::new(format!("sync temporary artifact: {error}")))?;
        fs::hard_link(&temporary, &target).map_err(|error| {
            OracleError::new(format!(
                "publish create-only artifact (target may already exist): {error}"
            ))
        })?;
        Ok(())
    })();
    let cleanup = fs::remove_file(&temporary);
    if let Err(error) = result {
        let _ = cleanup;
        return Err(error);
    }
    cleanup.map_err(|error| OracleError::new(format!("remove temporary artifact: {error}")))?;
    Ok(target)
}

fn u32_token_ids(values: &[u64], field: &str) -> OracleResult<Vec<u32>> {
    values
        .iter()
        .map(|&value| {
            u32::try_from(value)
                .map_err(|_| OracleError::new(format!("{field} contains a value outside U32")))
        })
        .collect()
}

fn token_ids_sha256(ids: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(ids.len().saturating_mul(4));
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn canonical_checkpoint_root(root: &Path) -> OracleResult<PathBuf> {
    let metadata = fs::symlink_metadata(root)
        .map_err(|error| OracleError::new(format!("inspect checkpoint root: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(OracleError::new(
            "checkpoint root must be a non-symlink directory",
        ));
    }
    root.canonicalize()
        .map_err(|error| OracleError::new(format!("canonicalize checkpoint root: {error}")))
}

fn regular_file_metadata(path: &Path, label: &str) -> OracleResult<fs::Metadata> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| OracleError::new(format!("inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(OracleError::new(format!(
            "{label} must be a regular non-symlink file"
        )));
    }
    Ok(metadata)
}

fn read_regular_file(path: &Path, limit: u64, label: &str) -> OracleResult<Vec<u8>> {
    let metadata = regular_file_metadata(path, label)?;
    if metadata.len() > limit {
        return Err(OracleError::new(format!(
            "{label} exceeds the {limit}-byte limit"
        )));
    }
    fs::read(path).map_err(|error| OracleError::new(format!("read {label}: {error}")))
}

fn expected_file_map() -> BTreeMap<String, (u64, String)> {
    EXPECTED_FILES
        .iter()
        .map(|file| {
            (
                file.path.to_owned(),
                (file.size_bytes, file.sha256.to_owned()),
            )
        })
        .collect()
}

fn git_output<const N: usize>(root: &Path, args: [&str; N]) -> OracleResult<String> {
    let bytes = git_output_bytes(root, args)?;
    let text = String::from_utf8(bytes)
        .map_err(|_| OracleError::new("git returned non-UTF-8 source provenance"))?;
    Ok(text.trim_end_matches(['\n', '\r']).to_owned())
}

fn git_output_bytes<const N: usize>(root: &Path, args: [&str; N]) -> OracleResult<Vec<u8>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|error| OracleError::new(format!("run git for source provenance: {error}")))?;
    if !output.status.success() {
        return Err(OracleError::new(format!(
            "git source provenance failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

fn temporary_output_path(parent: &Path, filename: &std::ffi::OsStr) -> OracleResult<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| OracleError::new(format!("read system clock: {error}")))?
        .as_nanos();
    let prefix = format!(
        ".{}.{}.{}",
        filename.to_string_lossy(),
        std::process::id(),
        timestamp
    );
    for attempt in 0..128_u32 {
        let candidate = parent.join(format!("{prefix}.{attempt}.tmp"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(OracleError::new(
        "could not allocate a temporary create-only artifact path",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_bf16_bytes_are_little_endian() {
        assert_eq!(bf16_le_bytes(&[0x3f80, 0xbf80]), [0x80, 0x3f, 0x80, 0xbf]);
    }

    #[test]
    fn logits_have_deterministic_tie_ordering_and_partition_hashes() {
        let bits = vec![0x0000_u16; VOCABULARY_SIZE];
        let mut values = vec![0.0_f32; VOCABULARY_SIZE];
        values[304] = 2.0;
        values[374] = 2.0;
        let capture = capture_logits(&bits, &values).expect("capture must succeed");
        assert_eq!(capture.argmax_token_id, 304);
        assert_eq!(&capture.top_token_ids[..2], &[304, 374]);
        assert_ne!(
            capture.addressable_bf16_le_sha256, capture.non_addressable_bf16_le_sha256,
            "partitions have different byte lengths"
        );
    }

    #[test]
    fn workload_hash_is_checked_before_json_is_accepted() {
        let path = std::env::temp_dir().join(format!(
            "riley-candle-workload-invalid-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::write(&path, b"{}\n").expect("write invalid workload");
        let error = load_workload(&path).expect_err("incorrect workload must fail");
        assert!(error.to_string().contains("SHA-256"));
        fs::remove_file(path).expect("remove test workload");
    }

    #[test]
    fn publication_is_create_only_and_rejects_the_checkout() {
        let root = std::env::temp_dir().join(format!(
            "riley-candle-publication-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let repo = root.join("repo");
        let output_directory = root.join("outputs");
        fs::create_dir_all(&repo).expect("create test repo");
        fs::create_dir_all(&output_directory).expect("create test output directory");
        let output = output_directory.join("artifact.json");
        let published = publish_create_only(&output, &repo, b"{}\n")
            .expect("external create-only publication must succeed");
        assert_eq!(fs::read(&published).expect("read artifact"), b"{}\n");
        assert!(publish_create_only(&output, &repo, b"different\n").is_err());
        assert!(publish_create_only(&repo.join("inside.json"), &repo, b"{}\n").is_err());
        fs::remove_dir_all(root).expect("remove test directory");
    }
}
