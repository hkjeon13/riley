use rustinfer_cuda::{AttentionPreference, CudaGemmReductionPolicy};
use rustinfer_model::{ModelConfig, ModelSpec};
use rustinfer_tensor::DType;

use super::LlamaReductionProfile;

const REVIEWED_HIDDEN_SIZE: usize = 576;
const REVIEWED_INTERMEDIATE_SIZE: usize = 1_536;
const REVIEWED_VOCABULARY_SIZE: usize = 49_152;
const REVIEWED_QUERY_HEADS: usize = 9;
const REVIEWED_KEY_VALUE_HEADS: usize = 3;
const REVIEWED_HEAD_DIMENSION: usize = 64;
const REVIEWED_LAYER_COUNT: usize = 30;
const REVIEWED_MAX_SEQUENCE_LENGTH: usize = 8_192;
const REVIEWED_RMS_NORM_EPSILON_BITS: u32 = 0x3727_c5ac;

/// Cold, per-plan cuBLASLt reduction policy resolved from validated source
/// semantics before an execution graph is prepared.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LlamaGemmReductionPolicies {
    hidden: CudaGemmReductionPolicy,
    key_value: CudaGemmReductionPolicy,
    intermediate: CudaGemmReductionPolicy,
    down: CudaGemmReductionPolicy,
    lm_head: CudaGemmReductionPolicy,
}

impl LlamaGemmReductionPolicies {
    const STRICT: Self = Self {
        hidden: CudaGemmReductionPolicy::StrictNoSplitV1,
        key_value: CudaGemmReductionPolicy::StrictNoSplitV1,
        intermediate: CudaGemmReductionPolicy::StrictNoSplitV1,
        down: CudaGemmReductionPolicy::StrictNoSplitV1,
        lm_head: CudaGemmReductionPolicy::StrictNoSplitV1,
    };

    const REVIEWED_LLAMA: Self = Self {
        hidden: CudaGemmReductionPolicy::AllowInPlaceAndOutputTypeSplitKV1,
        key_value: CudaGemmReductionPolicy::AllowInPlaceAndOutputTypeSplitKV1,
        intermediate: CudaGemmReductionPolicy::StrictNoSplitV1,
        down: CudaGemmReductionPolicy::AllowInPlaceAndOutputTypeSplitKV1,
        lm_head: CudaGemmReductionPolicy::StrictNoSplitV1,
    };

    #[must_use]
    pub(super) const fn strict() -> Self {
        Self::STRICT
    }

    #[must_use]
    pub(super) const fn hidden(self) -> CudaGemmReductionPolicy {
        self.hidden
    }

    #[must_use]
    pub(super) const fn key_value(self) -> CudaGemmReductionPolicy {
        self.key_value
    }

    #[must_use]
    pub(super) const fn intermediate(self) -> CudaGemmReductionPolicy {
        self.intermediate
    }

    #[must_use]
    pub(super) const fn down(self) -> CudaGemmReductionPolicy {
        self.down
    }

    #[must_use]
    pub(super) const fn lm_head(self) -> CudaGemmReductionPolicy {
        self.lm_head
    }
}

/// Resolves the reviewed policy once at the validated model boundary.
///
/// The reviewed BF16 dense-576 Llama profile preserves the raw first-heuristic
/// reduction for attention projections and the down projection. Gate/up and the
/// LM head stay no-split, matching the frozen oracle. Other Llama geometries and
/// dense Qwen2 are entirely no-split. Execution code consumes only the prepared
/// plans and never branches on source-model identity.
pub(super) fn canonical_gemm_reduction_policies(
    config: &ModelConfig,
    spec: &ModelSpec,
) -> LlamaGemmReductionPolicies {
    match config {
        ModelConfig::Llama(_) if is_reviewed_llama_geometry(spec) => {
            LlamaGemmReductionPolicies::REVIEWED_LLAMA
        }
        ModelConfig::Llama(_) | ModelConfig::Qwen2(_) => LlamaGemmReductionPolicies::STRICT,
    }
}

/// Resolves the canonical prefill implementation at the validated model
/// boundary. The fixed-sequence forward always has `B=1`; the remaining
/// reviewed Hugging Face eager shape is pinned here so unrelated dense Llama
/// and Qwen2 checkpoints continue to use the established online backend.
pub(super) fn canonical_prefill_attention_preference(
    config: &ModelConfig,
    spec: &ModelSpec,
    sequence_length: usize,
    configured: AttentionPreference,
    profile: LlamaReductionProfile,
) -> AttentionPreference {
    if configured == AttentionPreference::Optimized
        && profile == LlamaReductionProfile::CanonicalV1
        && sequence_length <= REVIEWED_MAX_SEQUENCE_LENGTH
        && matches!(config, ModelConfig::Llama(_))
        && is_reviewed_llama_geometry(spec)
    {
        AttentionPreference::HuggingFaceEager
    } else {
        configured
    }
}

pub(super) fn is_reviewed_llama_geometry(spec: &ModelSpec) -> bool {
    let embedding = spec.embedding();
    let lm_head = spec.lm_head();
    spec.dtype() == DType::BF16
        && embedding.hidden_size() == REVIEWED_HIDDEN_SIZE
        && embedding.vocabulary_size() == REVIEWED_VOCABULARY_SIZE
        && spec.max_sequence_length() == REVIEWED_MAX_SEQUENCE_LENGTH
        && spec.blocks().len() == REVIEWED_LAYER_COUNT
        && lm_head.hidden_size() == REVIEWED_HIDDEN_SIZE
        && lm_head.vocabulary_size() == REVIEWED_VOCABULARY_SIZE
        && lm_head.tied_to_embedding()
        && spec.blocks().iter().all(|block| {
            let attention = block.attention();
            let mlp = block.mlp();
            attention.hidden_size() == REVIEWED_HIDDEN_SIZE
                && attention.query_heads() == REVIEWED_QUERY_HEADS
                && attention.key_value_heads() == REVIEWED_KEY_VALUE_HEADS
                && attention.head_dimension() == REVIEWED_HEAD_DIMENSION
                && !attention.has_bias()
                && mlp.hidden_size() == REVIEWED_HIDDEN_SIZE
                && mlp.intermediate_size() == REVIEWED_INTERMEDIATE_SIZE
                && !mlp.has_bias()
        })
}

pub(super) fn is_reviewed_smollm2_rms_norm_geometry(spec: &ModelSpec) -> bool {
    let reviewed_norm = |norm: &rustinfer_model::NormSpec| {
        norm.hidden_size() == REVIEWED_HIDDEN_SIZE
            && execution_epsilon_bits(norm.epsilon()) == REVIEWED_RMS_NORM_EPSILON_BITS
    };
    is_reviewed_llama_geometry(spec)
        && reviewed_norm(spec.final_norm())
        && spec.blocks().iter().all(|block| {
            reviewed_norm(block.input_norm()) && reviewed_norm(block.post_attention_norm())
        })
}

#[allow(clippy::cast_possible_truncation)]
fn execution_epsilon_bits(epsilon: f64) -> u32 {
    // Llama plan preparation performs this same validated finite-positive
    // narrowing before passing epsilon to CUDA. Gate on the execution value.
    (epsilon as f32).to_bits()
}

#[cfg(test)]
mod tests {
    use rustinfer_cuda::{
        AttentionPreference,
        CudaGemmReductionPolicy::{AllowInPlaceAndOutputTypeSplitKV1, StrictNoSplitV1},
    };
    use rustinfer_model::ModelConfig;

    use super::{
        canonical_gemm_reduction_policies, canonical_prefill_attention_preference,
        is_reviewed_smollm2_rms_norm_geometry,
    };
    use crate::llama::LlamaReductionProfile;

    const LLAMA_CONFIG: &str = r#"{
      "architectures":["LlamaForCausalLM"],
      "attention_bias":false,
      "attention_dropout":0.0,
      "bos_token_id":1,
      "eos_token_id":0,
      "head_dim":64,
      "hidden_act":"silu",
      "hidden_size":576,
      "initializer_range":0.041666666666666664,
      "intermediate_size":1536,
      "is_llama_config":true,
      "max_position_embeddings":8192,
      "mlp_bias":false,
      "model_type":"llama",
      "num_attention_heads":9,
      "num_hidden_layers":30,
      "num_key_value_heads":3,
      "pretraining_tp":1,
      "rms_norm_eps":1e-5,
      "rope_interleaved":false,
      "rope_scaling":null,
      "rope_theta":100000,
      "tie_word_embeddings":true,
      "torch_dtype":"bfloat16",
      "transformers_version":"4.40.0",
      "use_cache":true,
      "vocab_size":49152
    }"#;

    const QWEN2_CONFIG: &str = r#"{
      "architectures":["Qwen2ForCausalLM"],
      "bos_token_id":6,
      "eos_token_id":7,
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
    fn source_family_resolves_one_cold_per_plan_policy_without_hot_dispatch() {
        let llama = ModelConfig::from_json_slice(LLAMA_CONFIG.as_bytes()).unwrap();
        let llama_spec = llama.to_model_spec();
        let llama = canonical_gemm_reduction_policies(&llama, &llama_spec);
        assert_eq!(llama.hidden(), AllowInPlaceAndOutputTypeSplitKV1);
        assert_eq!(llama.key_value(), AllowInPlaceAndOutputTypeSplitKV1);
        assert_eq!(llama.intermediate(), StrictNoSplitV1);
        assert_eq!(llama.down(), AllowInPlaceAndOutputTypeSplitKV1);
        assert_eq!(llama.lm_head(), StrictNoSplitV1);

        let qwen = ModelConfig::from_json_slice(QWEN2_CONFIG.as_bytes()).unwrap();
        let qwen_spec = qwen.to_model_spec();
        let qwen = canonical_gemm_reduction_policies(&qwen, &qwen_spec);
        assert_eq!(qwen.hidden(), StrictNoSplitV1);
        assert_eq!(qwen.key_value(), StrictNoSplitV1);
        assert_eq!(qwen.intermediate(), StrictNoSplitV1);
        assert_eq!(qwen.down(), StrictNoSplitV1);
        assert_eq!(qwen.lm_head(), StrictNoSplitV1);

        let other_llama = LLAMA_CONFIG
            .replace("\"hidden_size\":576", "\"hidden_size\":768")
            .replace("\"num_attention_heads\":9", "\"num_attention_heads\":12");
        let other_llama = ModelConfig::from_json_slice(other_llama.as_bytes()).unwrap();
        let other_spec = other_llama.to_model_spec();
        let other = canonical_gemm_reduction_policies(&other_llama, &other_spec);
        assert_eq!(other.hidden(), StrictNoSplitV1);
        assert_eq!(other.key_value(), StrictNoSplitV1);
        assert_eq!(other.intermediate(), StrictNoSplitV1);
        assert_eq!(other.down(), StrictNoSplitV1);
        assert_eq!(other.lm_head(), StrictNoSplitV1);

        let f16_llama = LLAMA_CONFIG.replace("\"bfloat16\"", "\"float16\"");
        let f16_llama = ModelConfig::from_json_slice(f16_llama.as_bytes()).unwrap();
        let f16_spec = f16_llama.to_model_spec();
        let f16 = canonical_gemm_reduction_policies(&f16_llama, &f16_spec);
        assert_eq!(f16.hidden(), StrictNoSplitV1);
        assert_eq!(f16.key_value(), StrictNoSplitV1);
        assert_eq!(f16.intermediate(), StrictNoSplitV1);
        assert_eq!(f16.down(), StrictNoSplitV1);
        assert_eq!(f16.lm_head(), StrictNoSplitV1);
    }

    #[test]
    fn canonical_attention_promotes_only_the_reviewed_dense_llama_shape() {
        let llama = ModelConfig::from_json_slice(LLAMA_CONFIG.as_bytes()).unwrap();
        let llama_spec = llama.to_model_spec();
        assert_eq!(
            canonical_prefill_attention_preference(
                &llama,
                &llama_spec,
                8_192,
                AttentionPreference::Optimized,
                LlamaReductionProfile::CanonicalV1,
            ),
            AttentionPreference::HuggingFaceEager
        );

        let qwen = ModelConfig::from_json_slice(QWEN2_CONFIG.as_bytes()).unwrap();
        let qwen_spec = qwen.to_model_spec();
        assert_eq!(
            canonical_prefill_attention_preference(
                &qwen,
                &qwen_spec,
                8,
                AttentionPreference::Optimized,
                LlamaReductionProfile::CanonicalV1,
            ),
            AttentionPreference::Optimized
        );

        let other_llama = LLAMA_CONFIG
            .replace("\"hidden_size\":576", "\"hidden_size\":768")
            .replace("\"num_attention_heads\":9", "\"num_attention_heads\":12");
        let other_llama = ModelConfig::from_json_slice(other_llama.as_bytes()).unwrap();
        let other_spec = other_llama.to_model_spec();
        assert_eq!(
            canonical_prefill_attention_preference(
                &other_llama,
                &other_spec,
                128,
                AttentionPreference::Optimized,
                LlamaReductionProfile::CanonicalV1,
            ),
            AttentionPreference::Optimized
        );
    }

    #[test]
    fn attention_promotion_preserves_explicit_and_unreviewed_selections() {
        let llama = ModelConfig::from_json_slice(LLAMA_CONFIG.as_bytes()).unwrap();
        let spec = llama.to_model_spec();

        for (sequence_length, configured, profile) in [
            (
                128,
                AttentionPreference::Reference,
                LlamaReductionProfile::CanonicalV1,
            ),
            (
                128,
                AttentionPreference::Optimized,
                LlamaReductionProfile::FixedContiguous37BalancedV1,
            ),
            (
                8_193,
                AttentionPreference::Optimized,
                LlamaReductionProfile::CanonicalV1,
            ),
        ] {
            assert_eq!(
                canonical_prefill_attention_preference(
                    &llama,
                    &spec,
                    sequence_length,
                    configured,
                    profile,
                ),
                configured
            );
        }
    }

    #[test]
    fn smollm2_rms_norm_geometry_requires_bf16_hidden576_and_exact_epsilon() {
        let llama = ModelConfig::from_json_slice(LLAMA_CONFIG.as_bytes()).unwrap();
        assert!(is_reviewed_smollm2_rms_norm_geometry(
            &llama.to_model_spec()
        ));

        let different_epsilon =
            LLAMA_CONFIG.replace("\"rms_norm_eps\":1e-5", "\"rms_norm_eps\":0.000001");
        let different_epsilon = ModelConfig::from_json_slice(different_epsilon.as_bytes()).unwrap();
        assert!(!is_reviewed_smollm2_rms_norm_geometry(
            &different_epsilon.to_model_spec()
        ));

        let f16 = LLAMA_CONFIG.replace("\"bfloat16\"", "\"float16\"");
        let f16 = ModelConfig::from_json_slice(f16.as_bytes()).unwrap();
        assert!(!is_reviewed_smollm2_rms_norm_geometry(&f16.to_model_spec()));

        let qwen = ModelConfig::from_json_slice(QWEN2_CONFIG.as_bytes()).unwrap();
        assert!(!is_reviewed_smollm2_rms_norm_geometry(
            &qwen.to_model_spec()
        ));
    }
}
