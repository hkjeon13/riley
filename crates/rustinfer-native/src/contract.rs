#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(crate) const CALIBRATION_SCHEMA_VERSION: &str = "1.0.0";
pub(crate) const CALIBRATION_GATE_ID: &str = "smollm2-fp32-bf16-native-e0-v2";
pub(crate) const GATE_MANIFEST_SHA256: &str =
    "eb97b2011bd77e6b2bfdb039c846484e281b35108ba6b357cdd1aba7033479e9";
pub(crate) const CANDIDATE_KIND: &str = "candidate";
pub(crate) const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-135M";
pub(crate) const MODEL_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
pub(crate) const MODEL_WEIGHTS_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";
pub(crate) const MODEL_CONFIG_SHA256: &str =
    "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843";
pub(crate) const TOKENIZER_SHA256: &str =
    "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db";
pub(crate) const TOKENIZER_FILES_SHA256: [(&str, &str); 5] = [
    (
        "merges.txt",
        "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    ),
    (
        "special_tokens_map.json",
        "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    ),
    (
        "tokenizer.json",
        "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    ),
    (
        "tokenizer_config.json",
        "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    ),
    (
        "vocab.json",
        "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    ),
];
pub(crate) const MAX_CONTEXT_TOKENS: usize = 8_192;
pub(crate) const SEMANTIC_GENERATION_STEPS: usize = 32;
pub(crate) const CROSS_CACHE_EXACT_WINDOW: usize = 16;
pub(crate) const CALIBRATION_TOP_K: usize = 10;
pub(crate) const CALIBRATION_PROMPT_COUNT: usize = 31;
pub(crate) const EOS_TOKEN_ID: u32 = 0;
pub(crate) const ATTENTION_BACKEND: &str = "eager";
pub(crate) const LOG_PROB_PIPELINE: &str = "log-softmax-fp32-v1";
pub(crate) const PRIMARY_ENVIRONMENT_ID: &str = "rtx4090-ubuntu22-driver580-v1";

pub(crate) const NATIVE_SOURCE_PATHS: [(&str, &str); 8] = [
    ("matrix", "benchmarks/matrix.yaml"),
    ("prompts", "benchmarks/prompts.jsonl"),
    (
        "gate_manifest",
        "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
    ),
    ("dependency_lock", "Cargo.lock"),
    (
        "python_version_file",
        "tools/python/reference/.python-version",
    ),
    ("lane_manifest", "benchmarks/lanes/rustinfer-native.json"),
    ("environment", "benchmarks/environment.md"),
    (
        "environment_probe",
        "tools/python/reference/rustinfer_reference/environment.py",
    ),
];

#[derive(Debug)]
pub(crate) enum ContractError {
    Io(io::Error),
    Json(serde_json::Error),
    InvalidPrompt { line: usize, reason: String },
    InvalidCorpus(String),
    Clock,
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => source.fmt(formatter),
            Self::Json(source) => source.fmt(formatter),
            Self::InvalidPrompt { line, reason } => {
                write!(formatter, "invalid prompt row {line}: {reason}")
            }
            Self::InvalidCorpus(reason) => write!(formatter, "invalid prompt corpus: {reason}"),
            Self::Clock => formatter.write_str("system clock predates the Unix epoch"),
        }
    }
}

impl Error for ContractError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Json(source) => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for ContractError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for ContractError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromptRecord {
    contract_version: String,
    pub(crate) prompt_id: String,
    pub(crate) category: String,
    pub(crate) language: String,
    pub(crate) text: String,
    pub(crate) target_prompt_tokens: Option<usize>,
    pub(crate) boundary_kind: String,
    pub(crate) expected_behavior: String,
    contains_sensitive_data: bool,
}

impl PromptRecord {
    pub(crate) fn metadata(&self) -> Value {
        json!({
            "category": self.category,
            "language": self.language,
            "target_prompt_tokens": self.target_prompt_tokens,
            "boundary_kind": self.boundary_kind,
            "expected_behavior": self.expected_behavior,
        })
    }

    fn validate(&self) -> Result<(), String> {
        if self.contract_version != CALIBRATION_SCHEMA_VERSION {
            return Err("contract_version must be 1.0.0".to_owned());
        }
        if !valid_prompt_id(&self.prompt_id) {
            return Err("prompt_id has invalid characters".to_owned());
        }
        if !matches!(
            self.category.as_str(),
            "short"
                | "multilingual"
                | "symbols-code"
                | "long-repetition"
                | "context-boundary"
                | "minimal"
                | "early-eos"
        ) {
            return Err("category is unsupported".to_owned());
        }
        if !matches!(
            self.language.as_str(),
            "ko" | "en" | "mixed" | "code" | "none"
        ) {
            return Err("language is unsupported".to_owned());
        }
        if self.expected_behavior.is_empty() || self.contains_sensitive_data {
            return Err(
                "expected_behavior must be present and data must be non-sensitive".to_owned(),
            );
        }
        if self
            .target_prompt_tokens
            .is_some_and(|target| target == 0 || target > MAX_CONTEXT_TOKENS)
        {
            return Err("target_prompt_tokens is outside 1..=8192".to_owned());
        }
        if self.category == "context-boundary" {
            if self.boundary_kind != "near-max-context"
                || self
                    .target_prompt_tokens
                    .is_none_or(|target| target < 7_168)
            {
                return Err("context-boundary metadata is inconsistent".to_owned());
            }
        } else if self.boundary_kind != "none" {
            return Err("only context-boundary may select near-max-context".to_owned());
        }
        if self.category == "early-eos"
            && (self.text.is_empty()
                || self.target_prompt_tokens.is_some()
                || self.language != "en"
                || self.expected_behavior != "greedy-eos-at-first-output-token")
        {
            return Err("early-eos metadata is inconsistent".to_owned());
        }
        Ok(())
    }
}

pub(crate) fn load_prompts(path: &Path) -> Result<Vec<PromptRecord>, ContractError> {
    let file = File::open(path)?;
    let mut prompts = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line?;
        if line.trim().is_empty() {
            return Err(ContractError::InvalidPrompt {
                line: line_number,
                reason: "blank rows are forbidden".to_owned(),
            });
        }
        let prompt: PromptRecord =
            serde_json::from_str(&line).map_err(|error| ContractError::InvalidPrompt {
                line: line_number,
                reason: error.to_string(),
            })?;
        prompt
            .validate()
            .map_err(|reason| ContractError::InvalidPrompt {
                line: line_number,
                reason,
            })?;
        if !ids.insert(prompt.prompt_id.clone()) {
            return Err(ContractError::InvalidPrompt {
                line: line_number,
                reason: "duplicate prompt_id".to_owned(),
            });
        }
        prompts.push(prompt);
    }
    if prompts.len() != CALIBRATION_PROMPT_COUNT {
        return Err(ContractError::InvalidCorpus(format!(
            "expected {CALIBRATION_PROMPT_COUNT} rows, found {}",
            prompts.len()
        )));
    }
    Ok(prompts)
}

pub(crate) fn sha256_file(path: &Path) -> Result<String, ContractError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex_lower(&digest.finalize()))
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

pub(crate) fn token_ids_sha256(token_ids: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token_id in token_ids {
        digest.update(token_id.to_le_bytes());
    }
    hex_lower(&digest.finalize())
}

pub(crate) fn utc_now() -> Result<String, ContractError> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ContractError::Clock)?
        .as_secs();
    Ok(utc_from_unix_seconds(seconds))
}

fn utc_from_unix_seconds(seconds: u64) -> String {
    let days = seconds / 86_400;
    let day_seconds = seconds % 86_400;
    let hour = day_seconds / 3_600;
    let minute = day_seconds % 3_600 / 60;
    let second = day_seconds % 60;
    let (year, month, day) = civil_from_days(i64::try_from(days).unwrap_or(i64::MAX));
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// Howard Hinnant's public-domain civil calendar conversion, with day zero at
// 1970-01-01 after the epoch adjustment.
fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let days = days_since_epoch + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = days - era * 146_097;
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

fn valid_prompt_id(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_lowercase() || first.is_ascii_digit())
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

pub(crate) fn oracle_reduction_variant() -> Value {
    json!({
        "variant_id": "hf-eager-default-v1",
        "partition_kind": "hf-eager-runtime-default",
        "chunk_elements": null,
        "remainder_policy": "runtime-default",
        "merge_order": "runtime-default",
    })
}

pub(crate) fn candidate_reduction_variant(id: &str) -> Value {
    match id {
        "canonical-v1" => json!({
            "variant_id": "canonical-v1",
            "partition_kind": "production-default",
            "chunk_elements": null,
            "remainder_policy": "implementation-default",
            "merge_order": "implementation-default",
        }),
        "fixed-contiguous-37-balanced-v1" => json!({
            "variant_id": "fixed-contiguous-37-balanced-v1",
            "partition_kind": "fixed-contiguous",
            "chunk_elements": 37,
            "remainder_policy": "last-short-chunk",
            "merge_order": "deterministic-balanced-binary-tree-by-chunk-index",
        }),
        _ => unreachable!("parser admits only reviewed reduction variants"),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        CALIBRATION_PROMPT_COUNT, GATE_MANIFEST_SHA256, load_prompts, sha256_file,
        token_ids_sha256, utc_from_unix_seconds,
    };

    #[test]
    fn calendar_and_token_hash_contract_are_stable() {
        assert_eq!(utc_from_unix_seconds(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_from_unix_seconds(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(
            token_ids_sha256(&[0, 1, 0xffff_ffff]),
            "de25d19943926b201c1693709bc5eca70ecf04229c1668e2f276249f9bebe043"
        );
    }

    #[test]
    fn repository_prompt_corpus_is_exact() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let prompts = load_prompts(&root.join("benchmarks/prompts.jsonl")).expect("prompts");
        assert_eq!(prompts.len(), CALIBRATION_PROMPT_COUNT);
        assert_eq!(prompts[0].prompt_id, "perf-0128-00");
        assert_eq!(prompts[30].prompt_id, "correct-early-eos");
        assert_eq!(
            sha256_file(&root.join("benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json"))
                .expect("gate digest"),
            GATE_MANIFEST_SHA256
        );
    }

    #[test]
    fn duplicate_prompt_fields_fail_closed() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rustinfer-prompts-{nonce}.jsonl"));
        fs::write(
            &path,
            concat!(
                "{\"contract_version\":\"1.0.0\",\"prompt_id\":\"a\",",
                "\"prompt_id\":\"b\",\"category\":\"short\",\"language\":\"en\",",
                "\"text\":\"x\",\"target_prompt_tokens\":null,\"boundary_kind\":\"none\",",
                "\"expected_behavior\":\"success\",\"contains_sensitive_data\":false}\n"
            ),
        )
        .expect("write");
        let error = load_prompts(&path).expect_err("duplicate field");
        assert!(error.to_string().contains("duplicate field"));
        fs::remove_file(path).expect("remove");
    }
}
