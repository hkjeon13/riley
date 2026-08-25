use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::artifact::{ArtifactBytes, VerifiedArtifactSession};
use crate::safetensors::{ParsedShard, ParsedTensor};
use crate::shard_index::ShardIndex;
use crate::{LoadLimits, ModelError, ModelResult, PROVENANCE_FILENAME};

pub(crate) const SINGLE_SHARD_FILENAME: &str = "model.safetensors";
pub(crate) const SHARD_INDEX_FILENAME: &str = "model.safetensors.index.json";

pub(crate) struct PhysicalCheckpoint {
    shards: Vec<OwnedShard>,
    inventory: BTreeMap<String, TensorLocator>,
    observed_paths: BTreeSet<PathBuf>,
}

impl PhysicalCheckpoint {
    pub(crate) fn load(
        session: &mut VerifiedArtifactSession,
        limits: LoadLimits,
    ) -> ModelResult<Self> {
        let files = session.provenance().files();
        let single_declared = files.contains_key(Path::new(SINGLE_SHARD_FILENAME));
        let index_declared = files.contains_key(Path::new(SHARD_INDEX_FILENAME));
        match (single_declared, index_declared) {
            (true, true) => Err(ModelError::InvalidArtifact {
                artifact: PROVENANCE_FILENAME.to_owned(),
                reason: "both single-file and sharded checkpoint layouts are declared".to_owned(),
            }),
            (false, false) => Err(ModelError::InvalidArtifact {
                artifact: PROVENANCE_FILENAME.to_owned(),
                reason: "neither model.safetensors nor model.safetensors.index.json is declared"
                    .to_owned(),
            }),
            (true, false) => Self::load_single(session, limits),
            (false, true) => Self::load_sharded(session, limits),
        }
    }

    fn load_single(session: &mut VerifiedArtifactSession, limits: LoadLimits) -> ModelResult<Self> {
        let path = PathBuf::from(SINGLE_SHARD_FILENAME);
        let artifact = session.read_once(
            &path,
            limits.shard_bytes().min(limits.total_weight_bytes()),
            "safetensors shard",
        )?;
        let shard = owned_shard(path.clone(), artifact, limits)?;
        let (inventory, data_bytes) = build_inventory(std::slice::from_ref(&shard), None)?;
        enforce_total_weight_limit(data_bytes, limits)?;
        Ok(Self {
            shards: vec![shard],
            inventory,
            observed_paths: BTreeSet::from([path]),
        })
    }

    fn load_sharded(
        session: &mut VerifiedArtifactSession,
        limits: LoadLimits,
    ) -> ModelResult<Self> {
        let index_path = PathBuf::from(SHARD_INDEX_FILENAME);
        let index_artifact =
            session.read_once(&index_path, limits.index_bytes(), "safetensors shard index")?;
        let index = ShardIndex::from_json_slice(index_artifact.bytes(), limits)?;
        let mut observed_paths = BTreeSet::from([index_path]);
        let mut shards = Vec::with_capacity(index.shards().len());
        let mut total_file_bytes = 0_u64;
        for path in index.shards() {
            let remaining = limits
                .total_weight_bytes()
                .checked_sub(total_file_bytes)
                .ok_or_else(|| ModelError::LimitExceeded {
                    resource: "checkpoint shard bytes",
                    limit: limits.total_weight_bytes(),
                    actual: Some(total_file_bytes),
                })?;
            let artifact = session.read_once(
                path,
                limits.shard_bytes().min(remaining),
                "safetensors shard",
            )?;
            total_file_bytes = total_file_bytes
                .checked_add(artifact.byte_len())
                .ok_or_else(|| ModelError::NumericOverflow {
                    field: "total checkpoint shard bytes".to_owned(),
                })?;
            if total_file_bytes > limits.total_weight_bytes() {
                return Err(ModelError::LimitExceeded {
                    resource: "checkpoint shard bytes",
                    limit: limits.total_weight_bytes(),
                    actual: Some(total_file_bytes),
                });
            }
            observed_paths.insert(path.clone());
            shards.push(owned_shard(path.clone(), artifact, limits)?);
        }
        let (inventory, data_bytes) = build_inventory(&shards, Some(&index))?;
        enforce_total_weight_limit(data_bytes, limits)?;
        if let Some(expected) = index.total_size() {
            if expected != data_bytes {
                return Err(ModelError::InvalidArtifact {
                    artifact: SHARD_INDEX_FILENAME.to_owned(),
                    reason: format!(
                        "metadata.total_size mismatch: expected {expected}, physical tensors total {data_bytes}"
                    ),
                });
            }
        }
        Ok(Self {
            shards,
            inventory,
            observed_paths,
        })
    }

    pub(crate) fn tensor(&self, name: &str) -> Option<(&Path, &ParsedTensor, &[u8])> {
        let locator = self.inventory.get(name)?;
        let shard = self.shards.get(locator.shard_index)?;
        let tensor = shard.parsed.tensor(name)?;
        let bytes = shard.parsed.tensor_bytes(name)?;
        Some((&shard.path, tensor, bytes))
    }

    pub(crate) const fn inventory(&self) -> &BTreeMap<String, TensorLocator> {
        &self.inventory
    }

    pub(crate) const fn observed_paths(&self) -> &BTreeSet<PathBuf> {
        &self.observed_paths
    }
}

struct OwnedShard {
    path: PathBuf,
    parsed: ParsedShard,
}

pub(crate) struct TensorLocator {
    shard_index: usize,
}

fn owned_shard(
    path: PathBuf,
    artifact: ArtifactBytes,
    limits: LoadLimits,
) -> ModelResult<OwnedShard> {
    let display = path.display().to_string();
    let parsed = ParsedShard::from_bytes(&display, artifact.into_bytes(), limits)?;
    Ok(OwnedShard { path, parsed })
}

fn build_inventory(
    shards: &[OwnedShard],
    index: Option<&ShardIndex>,
) -> ModelResult<(BTreeMap<String, TensorLocator>, u64)> {
    let mut inventory = BTreeMap::new();
    let mut data_bytes = 0_u64;
    for (shard_index, shard) in shards.iter().enumerate() {
        for (name, tensor) in shard.parsed.tensors() {
            if let Some(index) = index {
                match index.weight_map().get(name) {
                    Some(path) if path == &shard.path => {}
                    Some(path) => {
                        return Err(ModelError::InvalidArtifact {
                            artifact: SHARD_INDEX_FILENAME.to_owned(),
                            reason: format!(
                                "tensor {name} is in {}, but index maps it to {}",
                                shard.path.display(),
                                path.display()
                            ),
                        });
                    }
                    None => {
                        return Err(ModelError::InvalidArtifact {
                            artifact: shard.path.display().to_string(),
                            reason: format!("physical tensor {name} is absent from shard index"),
                        });
                    }
                }
            }
            if inventory
                .insert(name.clone(), TensorLocator { shard_index })
                .is_some()
            {
                return Err(ModelError::DuplicateTensor { name: name.clone() });
            }
            let tensor_bytes = u64::try_from(tensor.byte_range().len()).map_err(|_| {
                ModelError::NumericOverflow {
                    field: format!("tensor {name} byte length"),
                }
            })?;
            data_bytes = data_bytes.checked_add(tensor_bytes).ok_or_else(|| {
                ModelError::NumericOverflow {
                    field: "total physical tensor bytes".to_owned(),
                }
            })?;
        }
    }

    if let Some(index) = index {
        let physical: BTreeSet<_> = inventory.keys().cloned().collect();
        let declared: BTreeSet<_> = index.weight_map().keys().cloned().collect();
        if physical != declared {
            let missing: Vec<_> = declared.difference(&physical).cloned().collect();
            let extra: Vec<_> = physical.difference(&declared).cloned().collect();
            return Err(ModelError::InvalidArtifact {
                artifact: SHARD_INDEX_FILENAME.to_owned(),
                reason: format!(
                    "index/physical tensor set mismatch: missing={missing:?}, extra={extra:?}"
                ),
            });
        }
    }
    Ok((inventory, data_bytes))
}

fn enforce_total_weight_limit(actual: u64, limits: LoadLimits) -> ModelResult<()> {
    if actual > limits.total_weight_bytes() {
        return Err(ModelError::LimitExceeded {
            resource: "checkpoint tensor bytes",
            limit: limits.total_weight_bytes(),
            actual: Some(actual),
        });
    }
    Ok(())
}
