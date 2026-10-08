use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use crate::artifact::validate_relative_file;
use crate::{ArtifactKind, LoadLimits, ModelError, ModelResult, strict_json};

pub(crate) struct ShardIndex {
    weight_map: BTreeMap<String, PathBuf>,
    shards: BTreeSet<PathBuf>,
    total_size: Option<u64>,
}

impl ShardIndex {
    pub(crate) fn from_json_slice(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        let input_len = u64::try_from(input.len()).map_err(|_| ModelError::NumericOverflow {
            field: "safetensors index byte length".to_owned(),
        })?;
        if input_len > limits.index_bytes() {
            return Err(ModelError::LimitExceeded {
                resource: "safetensors shard index",
                limit: limits.index_bytes(),
                actual: Some(input_len),
            });
        }
        let raw: RawShardIndex = strict_json::from_slice(input, ArtifactKind::ShardIndex)?;
        if let Some((field, value)) = raw.unknown.iter().next() {
            return invalid(format!(
                "unknown index field {field}={}",
                stable_value(value)
            ));
        }
        if raw.weight_map.is_empty() {
            return invalid("weight_map must not be empty");
        }
        if raw.weight_map.len() > limits.tensors() {
            return Err(ModelError::LimitExceeded {
                resource: "safetensors index tensors",
                limit: usize_to_u64(limits.tensors()),
                actual: Some(usize_to_u64(raw.weight_map.len())),
            });
        }
        let total_size = if let Some(metadata) = raw.metadata {
            if let Some((field, value)) = metadata.unknown.iter().next() {
                return invalid(format!(
                    "unknown index metadata field {field}={}",
                    stable_value(value)
                ));
            }
            metadata.total_size
        } else {
            None
        };

        let mut weight_map = BTreeMap::new();
        let mut shards = BTreeSet::new();
        for (name, raw_path) in raw.weight_map {
            if name.is_empty() || name.len() > 1024 || name.chars().any(char::is_control) {
                return invalid(
                    "weight_map tensor names must be bounded, non-empty, and printable",
                );
            }
            let path = PathBuf::from(raw_path);
            validate_relative_file(&path)?;
            if path.extension().and_then(|extension| extension.to_str()) != Some("safetensors") {
                return invalid(format!(
                    "shard {} must have a .safetensors extension",
                    path.display()
                ));
            }
            shards.insert(path.clone());
            weight_map.insert(name, path);
        }
        if shards.len() > limits.shards() {
            return Err(ModelError::LimitExceeded {
                resource: "safetensors shards",
                limit: usize_to_u64(limits.shards()),
                actual: Some(usize_to_u64(shards.len())),
            });
        }

        Ok(Self {
            weight_map,
            shards,
            total_size,
        })
    }

    pub(crate) const fn weight_map(&self) -> &BTreeMap<String, PathBuf> {
        &self.weight_map
    }

    pub(crate) const fn shards(&self) -> &BTreeSet<PathBuf> {
        &self.shards
    }

    pub(crate) const fn total_size(&self) -> Option<u64> {
        self.total_size
    }
}

#[derive(Deserialize)]
struct RawShardIndex {
    metadata: Option<RawIndexMetadata>,
    weight_map: BTreeMap<String, String>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct RawIndexMetadata {
    total_size: Option<u64>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

fn stable_value(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unrenderable-json>".to_owned())
}

fn invalid<T>(reason: impl Into<String>) -> ModelResult<T> {
    Err(ModelError::InvalidArtifact {
        artifact: "model.safetensors.index.json".to_owned(),
        reason: reason.into(),
    })
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::ShardIndex;
    use crate::{LoadLimits, ModelError};

    const VALID: &str = r#"{
      "metadata":{"total_size":12},
      "weight_map":{
        "model.embed_tokens.weight":"model-00001-of-00002.safetensors",
        "model.norm.weight":"model-00002-of-00002.safetensors"
      }
    }"#;

    #[test]
    fn parses_deterministic_weight_and_shard_sets() {
        let index = ShardIndex::from_json_slice(VALID.as_bytes(), LoadLimits::default()).unwrap();
        assert_eq!(index.weight_map().len(), 2);
        assert_eq!(index.shards().len(), 2);
        assert_eq!(index.total_size(), Some(12));
    }

    #[test]
    fn rejects_duplicate_tensor_and_path_traversal() {
        let duplicate = VALID.replace("\"model.norm.weight\"", "\"model.embed_tokens.weight\"");
        assert!(matches!(
            ShardIndex::from_json_slice(duplicate.as_bytes(), LoadLimits::default()),
            Err(ModelError::InvalidJson { .. })
        ));
        let traversal = VALID.replace(
            "model-00001-of-00002.safetensors",
            "../model-00001-of-00002.safetensors",
        );
        assert!(matches!(
            ShardIndex::from_json_slice(traversal.as_bytes(), LoadLimits::default()),
            Err(ModelError::UnsafePath { .. })
        ));
    }

    #[test]
    fn rejects_unknown_metadata_that_could_change_semantics() {
        let changed = VALID.replace("\"total_size\":12", "\"total_size\":12,\"format\":\"pt\"");
        assert!(ShardIndex::from_json_slice(changed.as_bytes(), LoadLimits::default()).is_err());
    }
}
