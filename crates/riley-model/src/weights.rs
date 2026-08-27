use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use riley_tensor::{DType, TensorView};
use serde::Serialize;

use crate::artifact::VerifiedArtifactSession;
use crate::checkpoint::PhysicalCheckpoint;
use crate::{CheckpointProvenance, LoadLimits, ModelError, ModelResult, ModelSpec};

/// Parameter kind within one canonical decoder block.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DecoderWeight {
    /// Pre-attention RMS normalization scale.
    InputNormScale,
    /// Query projection matrix.
    QueryWeight,
    /// Query projection bias.
    QueryBias,
    /// Key projection matrix.
    KeyWeight,
    /// Key projection bias.
    KeyBias,
    /// Value projection matrix.
    ValueWeight,
    /// Value projection bias.
    ValueBias,
    /// Attention output projection matrix.
    OutputWeight,
    /// Attention output projection bias.
    OutputBias,
    /// Post-attention RMS normalization scale.
    PostAttentionNormScale,
    /// Gated MLP gate matrix.
    GateWeight,
    /// Gated MLP gate bias.
    GateBias,
    /// Gated MLP up-projection matrix.
    UpWeight,
    /// Gated MLP up-projection bias.
    UpBias,
    /// Gated MLP down-projection matrix.
    DownWeight,
    /// Gated MLP down-projection bias.
    DownBias,
}

impl DecoderWeight {
    const fn name(self) -> &'static str {
        match self {
            Self::InputNormScale => "input_norm.scale",
            Self::QueryWeight => "attention.query.weight",
            Self::QueryBias => "attention.query.bias",
            Self::KeyWeight => "attention.key.weight",
            Self::KeyBias => "attention.key.bias",
            Self::ValueWeight => "attention.value.weight",
            Self::ValueBias => "attention.value.bias",
            Self::OutputWeight => "attention.output.weight",
            Self::OutputBias => "attention.output.bias",
            Self::PostAttentionNormScale => "post_attention_norm.scale",
            Self::GateWeight => "mlp.gate.weight",
            Self::GateBias => "mlp.gate.bias",
            Self::UpWeight => "mlp.up.weight",
            Self::UpBias => "mlp.up.bias",
            Self::DownWeight => "mlp.down.weight",
            Self::DownBias => "mlp.down.bias",
        }
    }
}

/// Stable identity of one execution-facing model parameter.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WeightSlot {
    /// Token embedding matrix.
    TokenEmbedding,
    /// A decoder-block parameter.
    Decoder {
        /// Zero-based layer index.
        layer: usize,
        /// Parameter within the block.
        parameter: DecoderWeight,
    },
    /// Final RMS normalization scale.
    FinalNormScale,
    /// Language-model output projection.
    LmHead,
}

impl WeightSlot {
    /// Returns the stable canonical diagnostic name.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::TokenEmbedding => "token_embedding.weight".to_owned(),
            Self::Decoder { layer, parameter } => {
                format!("decoder.{layer}.{}", parameter.name())
            }
            Self::FinalNormScale => "final_norm.scale".to_owned(),
            Self::LmHead => "lm_head.weight".to_owned(),
        }
    }
}

/// Serialized source identity for a canonical weight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSource {
    tensor_name: String,
    shard_path: PathBuf,
}

impl TensorSource {
    /// Returns the exact safetensors key.
    #[must_use]
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    /// Returns the checkpoint-root-relative shard path.
    #[must_use]
    pub fn shard_path(&self) -> &Path {
        &self.shard_path
    }
}

/// Physical or tied-alias binding for one canonical slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WeightBinding {
    /// A physical tensor in a safetensors shard.
    Tensor(TensorSource),
    /// An exact alias of another canonical slot.
    Alias(WeightSlot),
}

/// A validated borrowed tensor view and its source identity.
pub struct BoundWeight<'a> {
    slot: WeightSlot,
    source: &'a TensorSource,
    view: TensorView<'a>,
}

impl<'a> BoundWeight<'a> {
    /// Returns the originally requested canonical slot.
    #[must_use]
    pub const fn slot(&self) -> WeightSlot {
        self.slot
    }

    /// Returns the physical source, after resolving a tied alias.
    #[must_use]
    pub const fn source(&self) -> &'a TensorSource {
        self.source
    }

    /// Returns the immutable view anchored to the loaded checkpoint.
    #[must_use]
    pub const fn view(&self) -> &TensorView<'a> {
        &self.view
    }
}

/// Fully validated canonical weight bindings and owned checkpoint shards.
pub struct LoadedWeights {
    physical: PhysicalCheckpoint,
    bindings: BTreeMap<WeightSlot, WeightBinding>,
    provenance: CheckpointProvenance,
}

impl LoadedWeights {
    /// Loads a standalone weight-only manifest and binds its safetensors.
    ///
    /// The manifest file set must contain exactly `model.safetensors`, or the
    /// shard index and every referenced shard. Full model manifests that also
    /// contain config/tokenizer files are consumed by the aggregate model
    /// loader rather than this deliberately narrower entry point.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing/ambiguous layout, failed provenance,
    /// invalid shard/index, or any missing/extra/dtype/shape/tied mismatch.
    pub fn load_weight_only(
        root: &Path,
        spec: &ModelSpec,
        provenance: CheckpointProvenance,
        limits: LoadLimits,
    ) -> ModelResult<Self> {
        let mut session = VerifiedArtifactSession::with_provenance(root, provenance)?;
        let loaded = Self::load_verified_subset(&mut session, spec, limits)?;
        session.finish_exact()?;
        Ok(loaded)
    }

    pub(crate) fn load_verified_subset(
        session: &mut VerifiedArtifactSession,
        spec: &ModelSpec,
        limits: LoadLimits,
    ) -> ModelResult<Self> {
        let provenance = session.provenance().clone();
        if provenance.dtype() != spec.dtype() {
            return Err(ModelError::InvalidArtifact {
                artifact: "riley-checkpoint.json".to_owned(),
                reason: format!(
                    "manifest dtype {} differs from model dtype {}",
                    provenance.dtype(),
                    spec.dtype()
                ),
            });
        }
        let physical = PhysicalCheckpoint::load(session, limits)?;
        let bindings = bind_canonical_weights(spec, &physical)?;
        Ok(Self {
            physical,
            bindings,
            provenance,
        })
    }

    /// Returns one canonical binding.
    #[must_use]
    pub fn binding(&self, slot: WeightSlot) -> Option<&WeightBinding> {
        self.bindings.get(&slot)
    }

    /// Returns all bindings in stable slot order.
    #[must_use]
    pub const fn bindings(&self) -> &BTreeMap<WeightSlot, WeightBinding> {
        &self.bindings
    }

    /// Returns verified checkpoint provenance.
    #[must_use]
    pub const fn provenance(&self) -> &CheckpointProvenance {
        &self.provenance
    }

    /// Creates a borrowed view whose lifetime cannot outlive this owner.
    ///
    /// ```compile_fail
    /// use riley_model::{BoundWeight, LoadedWeights, WeightSlot};
    ///
    /// fn escape(weights: &LoadedWeights) -> BoundWeight<'static> {
    ///     weights.view(WeightSlot::TokenEmbedding).unwrap()
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if internal validated ownership metadata is inconsistent.
    pub fn view(&self, slot: WeightSlot) -> ModelResult<BoundWeight<'_>> {
        let source = self.resolve_source(slot)?;
        let (_, tensor, bytes) = self.physical.tensor(source.tensor_name()).ok_or_else(|| {
            ModelError::InvalidArtifact {
                artifact: source.tensor_name.clone(),
                reason: "bound tensor disappeared from physical inventory".to_owned(),
            }
        })?;
        let view = TensorView::from_contiguous(bytes, tensor.dtype(), tensor.shape().clone())
            .map_err(|error| ModelError::InvalidArtifact {
                artifact: source.tensor_name.clone(),
                reason: format!("cannot construct validated tensor view: {error}"),
            })?;
        Ok(BoundWeight { slot, source, view })
    }

    /// Serializes deterministic canonical slot-to-source bindings.
    ///
    /// # Errors
    ///
    /// Returns an error for non-UTF-8 paths or unexpected JSON serialization failure.
    pub fn binding_snapshot_json(&self) -> ModelResult<String> {
        let mut entries = Vec::with_capacity(self.bindings.len());
        for (slot, binding) in &self.bindings {
            let binding = match binding {
                WeightBinding::Tensor(source) => SnapshotBinding::Tensor {
                    tensor_name: &source.tensor_name,
                    shard_path: source.shard_path.to_str().ok_or_else(|| {
                        ModelError::InvalidArtifact {
                            artifact: source.tensor_name.clone(),
                            reason: "shard path is not UTF-8".to_owned(),
                        }
                    })?,
                },
                WeightBinding::Alias(target) => SnapshotBinding::Alias {
                    slot: target.name(),
                },
            };
            entries.push(BindingSnapshotEntry {
                slot: slot.name(),
                binding,
            });
        }
        serde_json::to_string_pretty(&entries).map_err(|error| ModelError::InvalidArtifact {
            artifact: "canonical-weight-bindings".to_owned(),
            reason: error.to_string(),
        })
    }

    /// Returns the manifest-relative index and shard files that were verified.
    #[must_use]
    pub const fn verified_weight_files(&self) -> &BTreeSet<PathBuf> {
        self.physical.observed_paths()
    }

    fn resolve_source(&self, requested: WeightSlot) -> ModelResult<&TensorSource> {
        let mut current = requested;
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current) {
                return Err(ModelError::InvalidArtifact {
                    artifact: requested.name(),
                    reason: "canonical weight alias cycle".to_owned(),
                });
            }
            match self.bindings.get(&current) {
                Some(WeightBinding::Tensor(source)) => return Ok(source),
                Some(WeightBinding::Alias(target)) => current = *target,
                None => {
                    return Err(ModelError::InvalidArtifact {
                        artifact: requested.name(),
                        reason: "canonical slot has no binding".to_owned(),
                    });
                }
            }
        }
    }
}

#[derive(Serialize)]
struct BindingSnapshotEntry<'a> {
    slot: String,
    binding: SnapshotBinding<'a>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SnapshotBinding<'a> {
    Tensor {
        tensor_name: &'a str,
        shard_path: &'a str,
    },
    Alias {
        slot: String,
    },
}

struct Requirement {
    slot: WeightSlot,
    source_name: String,
    shape: Vec<usize>,
}

fn bind_canonical_weights(
    spec: &ModelSpec,
    physical: &PhysicalCheckpoint,
) -> ModelResult<BTreeMap<WeightSlot, WeightBinding>> {
    let requirements = requirements(spec)?;
    let required_names: BTreeSet<_> = requirements
        .iter()
        .map(|requirement| requirement.source_name.clone())
        .collect();
    let mut allowed_names = required_names.clone();
    if spec.lm_head().tied_to_embedding() {
        allowed_names.insert("lm_head.weight".to_owned());
    }
    let physical_names: BTreeSet<_> = physical.inventory().keys().cloned().collect();
    let missing: Vec<_> = required_names
        .difference(&physical_names)
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(ModelError::MissingTensors { names: missing });
    }
    let extra: Vec<_> = physical_names.difference(&allowed_names).cloned().collect();
    if !extra.is_empty() {
        return Err(ModelError::ExtraTensors { names: extra });
    }

    let mut bindings = BTreeMap::new();
    for requirement in requirements {
        let source = validate_requirement(spec.dtype(), &requirement, physical)?;
        bindings.insert(requirement.slot, WeightBinding::Tensor(source));
    }
    if spec.lm_head().tied_to_embedding() {
        validate_optional_tied_head(spec, physical)?;
        bindings.insert(
            WeightSlot::LmHead,
            WeightBinding::Alias(WeightSlot::TokenEmbedding),
        );
    }
    Ok(bindings)
}

fn validate_requirement(
    dtype: DType,
    requirement: &Requirement,
    physical: &PhysicalCheckpoint,
) -> ModelResult<TensorSource> {
    let (path, tensor, _) =
        physical
            .tensor(&requirement.source_name)
            .ok_or_else(|| ModelError::MissingTensors {
                names: vec![requirement.source_name.clone()],
            })?;
    if tensor.dtype() != dtype {
        return Err(ModelError::TensorDTypeMismatch {
            name: requirement.source_name.clone(),
            expected: dtype,
            actual: tensor.dtype(),
        });
    }
    if tensor.shape().dimensions() != requirement.shape {
        return Err(ModelError::TensorShapeMismatch {
            name: requirement.source_name.clone(),
            expected: requirement.shape.clone(),
            actual: tensor.shape().dimensions().to_vec(),
        });
    }
    Ok(TensorSource {
        tensor_name: requirement.source_name.clone(),
        shard_path: path.to_owned(),
    })
}

fn validate_optional_tied_head(spec: &ModelSpec, physical: &PhysicalCheckpoint) -> ModelResult<()> {
    let Some((_, tensor, lm_head_bytes)) = physical.tensor("lm_head.weight") else {
        return Ok(());
    };
    let expected_shape = vec![
        spec.lm_head().vocabulary_size(),
        spec.lm_head().hidden_size(),
    ];
    if tensor.dtype() != spec.dtype() {
        return Err(ModelError::TensorDTypeMismatch {
            name: "lm_head.weight".to_owned(),
            expected: spec.dtype(),
            actual: tensor.dtype(),
        });
    }
    if tensor.shape().dimensions() != expected_shape {
        return Err(ModelError::TensorShapeMismatch {
            name: "lm_head.weight".to_owned(),
            expected: expected_shape,
            actual: tensor.shape().dimensions().to_vec(),
        });
    }
    let (_, _, embedding_bytes) =
        physical
            .tensor("model.embed_tokens.weight")
            .ok_or_else(|| ModelError::MissingTensors {
                names: vec!["model.embed_tokens.weight".to_owned()],
            })?;
    if lm_head_bytes != embedding_bytes {
        return Err(ModelError::TiedWeightMismatch);
    }
    Ok(())
}

fn requirements(spec: &ModelSpec) -> ModelResult<Vec<Requirement>> {
    let mut result = Vec::new();
    result.push(Requirement {
        slot: WeightSlot::TokenEmbedding,
        source_name: "model.embed_tokens.weight".to_owned(),
        shape: vec![
            spec.embedding().vocabulary_size(),
            spec.embedding().hidden_size(),
        ],
    });
    for block in spec.blocks() {
        extend_block_requirements(&mut result, block)?;
    }
    result.push(Requirement {
        slot: WeightSlot::FinalNormScale,
        source_name: "model.norm.weight".to_owned(),
        shape: vec![spec.final_norm().hidden_size()],
    });
    if !spec.lm_head().tied_to_embedding() {
        result.push(Requirement {
            slot: WeightSlot::LmHead,
            source_name: "lm_head.weight".to_owned(),
            shape: vec![
                spec.lm_head().vocabulary_size(),
                spec.lm_head().hidden_size(),
            ],
        });
    }
    Ok(result)
}

fn extend_block_requirements(
    output: &mut Vec<Requirement>,
    block: &crate::DecoderBlockSpec,
) -> ModelResult<()> {
    let layer = block.index();
    let prefix = format!("model.layers.{layer}");
    let attention = block.attention();
    let query_width = checked_product(
        attention.query_heads(),
        attention.head_dimension(),
        "query projection width",
    )?;
    let key_value_width = checked_product(
        attention.key_value_heads(),
        attention.head_dimension(),
        "key/value projection width",
    )?;
    let hidden = attention.hidden_size();
    push_requirement(
        output,
        layer,
        DecoderWeight::InputNormScale,
        format!("{prefix}.input_layernorm.weight"),
        vec![hidden],
    );
    extend_attention_requirements(
        output,
        layer,
        &prefix,
        hidden,
        query_width,
        key_value_width,
        *attention.bias(),
    );
    push_requirement(
        output,
        layer,
        DecoderWeight::PostAttentionNormScale,
        format!("{prefix}.post_attention_layernorm.weight"),
        vec![hidden],
    );
    extend_mlp_requirements(output, layer, &prefix, block.mlp());
    Ok(())
}

fn extend_attention_requirements(
    output: &mut Vec<Requirement>,
    layer: usize,
    prefix: &str,
    hidden: usize,
    query_width: usize,
    key_value_width: usize,
    bias: crate::AttentionBiasSpec,
) {
    for (parameter, suffix, shape) in [
        (
            DecoderWeight::QueryWeight,
            "self_attn.q_proj.weight",
            vec![query_width, hidden],
        ),
        (
            DecoderWeight::KeyWeight,
            "self_attn.k_proj.weight",
            vec![key_value_width, hidden],
        ),
        (
            DecoderWeight::ValueWeight,
            "self_attn.v_proj.weight",
            vec![key_value_width, hidden],
        ),
        (
            DecoderWeight::OutputWeight,
            "self_attn.o_proj.weight",
            vec![hidden, query_width],
        ),
    ] {
        push_requirement(
            output,
            layer,
            parameter,
            format!("{prefix}.{suffix}"),
            shape,
        );
    }
    for (present, parameter, suffix, width) in [
        (
            bias.query(),
            DecoderWeight::QueryBias,
            "self_attn.q_proj.bias",
            query_width,
        ),
        (
            bias.key(),
            DecoderWeight::KeyBias,
            "self_attn.k_proj.bias",
            key_value_width,
        ),
        (
            bias.value(),
            DecoderWeight::ValueBias,
            "self_attn.v_proj.bias",
            key_value_width,
        ),
        (
            bias.output(),
            DecoderWeight::OutputBias,
            "self_attn.o_proj.bias",
            hidden,
        ),
    ] {
        if present {
            push_requirement(
                output,
                layer,
                parameter,
                format!("{prefix}.{suffix}"),
                vec![width],
            );
        }
    }
}

fn extend_mlp_requirements(
    output: &mut Vec<Requirement>,
    layer: usize,
    prefix: &str,
    mlp: &crate::GatedMlpSpec,
) {
    let hidden = mlp.hidden_size();
    let intermediate = mlp.intermediate_size();
    for (parameter, suffix, shape) in [
        (
            DecoderWeight::GateWeight,
            "mlp.gate_proj.weight",
            vec![intermediate, hidden],
        ),
        (
            DecoderWeight::UpWeight,
            "mlp.up_proj.weight",
            vec![intermediate, hidden],
        ),
        (
            DecoderWeight::DownWeight,
            "mlp.down_proj.weight",
            vec![hidden, intermediate],
        ),
    ] {
        push_requirement(
            output,
            layer,
            parameter,
            format!("{prefix}.{suffix}"),
            shape,
        );
    }
    if mlp.has_bias() {
        for (parameter, suffix, width) in [
            (DecoderWeight::GateBias, "mlp.gate_proj.bias", intermediate),
            (DecoderWeight::UpBias, "mlp.up_proj.bias", intermediate),
            (DecoderWeight::DownBias, "mlp.down_proj.bias", hidden),
        ] {
            push_requirement(
                output,
                layer,
                parameter,
                format!("{prefix}.{suffix}"),
                vec![width],
            );
        }
    }
}

fn push_requirement(
    output: &mut Vec<Requirement>,
    layer: usize,
    parameter: DecoderWeight,
    source_name: String,
    shape: Vec<usize>,
) {
    output.push(Requirement {
        slot: WeightSlot::Decoder { layer, parameter },
        source_name,
        shape,
    });
}

fn checked_product(left: usize, right: usize, field: &str) -> ModelResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| ModelError::NumericOverflow {
            field: field.to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::requirements;
    use crate::{ModelConfig, WeightSlot};

    const TINY_QWEN2_CONFIG: &str = r#"{
      "architectures":["Qwen2ForCausalLM"],
      "bos_token_id":0,
      "eos_token_id":1,
      "hidden_act":"silu",
      "hidden_size":4,
      "intermediate_size":8,
      "max_position_embeddings":16,
      "model_type":"qwen2",
      "num_attention_heads":2,
      "num_hidden_layers":1,
      "num_key_value_heads":1,
      "rms_norm_eps":0.000001,
      "rope_scaling":null,
      "rope_theta":1000000,
      "tie_word_embeddings":true,
      "torch_dtype":"bfloat16",
      "use_sliding_window":false,
      "vocab_size":8
    }"#;

    #[test]
    fn qwen2_weight_spec_requires_qkv_bias_but_not_output_bias() {
        let spec = ModelConfig::from_json_slice(TINY_QWEN2_CONFIG.as_bytes())
            .unwrap()
            .to_model_spec();
        let requirements = requirements(&spec).unwrap();
        let by_name: BTreeMap<_, _> = requirements
            .iter()
            .map(|requirement| {
                (
                    requirement.source_name.as_str(),
                    requirement.shape.as_slice(),
                )
            })
            .collect();

        assert_eq!(
            by_name.get("model.layers.0.self_attn.q_proj.bias").copied(),
            Some(&[4][..])
        );
        assert_eq!(
            by_name.get("model.layers.0.self_attn.k_proj.bias").copied(),
            Some(&[2][..])
        );
        assert_eq!(
            by_name.get("model.layers.0.self_attn.v_proj.bias").copied(),
            Some(&[2][..])
        );
        assert!(!by_name.contains_key("model.layers.0.self_attn.o_proj.bias"));
        assert_eq!(by_name.len(), 14);
        assert!(
            requirements
                .iter()
                .all(|requirement| { !matches!(requirement.slot, WeightSlot::LmHead) })
        );
    }
}
