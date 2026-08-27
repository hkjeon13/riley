//! Cold, immutable planning contract for a fixed-length Llama forward.

mod batch;
#[cfg(any(feature = "cuda", test))]
mod batch_executor;
#[cfg(any(feature = "cuda", test))]
mod decode;
mod error;
#[cfg(any(feature = "cuda", test))]
mod forward;
#[cfg(any(feature = "cuda", test))]
mod gemm_policy;
#[cfg(feature = "cuda")]
mod generation;
mod plan;
#[cfg(any(feature = "cuda", test))]
mod reduction_profile;

pub use batch::{
    LLAMA_BATCH_METADATA_V1_VERSION, LLAMA_BATCH_NO_OUTPUT_SLOT, LlamaBatchBlockTable,
    LlamaBatchBufferCapacities, LlamaBatchError, LlamaBatchMetadataConfig, LlamaBatchResult,
    LlamaBatchRow, LlamaBatchRowKind, LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
};

#[cfg(feature = "cuda")]
pub use batch_executor::{
    ExecutionCompletionImplementation, LlamaBatchExecutorError, LlamaBatchExecutorResource,
    LlamaBatchExecutorResult, PreparedLlamaBatchAllocationReport, PreparedLlamaBatchExecutor,
    PreparedLlamaBatchExecutorConfig, ResidualNormImplementation,
};

pub use error::{
    ExecutionSite, LlamaBufferRole, LlamaDimension, LlamaOp, LlamaPlanError, LlamaPlanResult,
    LlamaScalar,
};
pub use plan::{
    HIDDEN_WORKSPACE_BUFFER_COUNT, INTERMEDIATE_WORKSPACE_BUFFER_COUNT,
    KEY_VALUE_WORKSPACE_BUFFER_COUNT, LlamaDimensions, LlamaExecutionPlan, LlamaLayerPlan,
    LlamaWorkspaceSpec,
};

#[cfg(any(feature = "cuda", test))]
pub use reduction_profile::{LLAMA_FIXED37_MAX_SEQUENCE_TOKENS, LlamaReductionProfile};

#[cfg(feature = "cuda")]
pub use forward::{
    LlamaForwardError, LlamaForwardResource, LlamaForwardResult, LlamaTracePoint,
    PreparedLlamaAllocationReport, PreparedLlamaForward, PreparedLlamaForwardConfig,
    PreparedLlamaTrace,
};

#[cfg(feature = "cuda")]
pub use decode::{
    LlamaDecodeError, LlamaDecodePhase, LlamaDecodeResource, LlamaDecodeResult, LlamaKvCacheLayout,
    LlamaKvCachePolicy, LlamaKvCacheStorageLayout, PreparedLlamaDecode,
    PreparedLlamaDecodeAllocationReport, PreparedLlamaDecodeAttention, PreparedLlamaDecodeConfig,
};

#[cfg(feature = "cuda")]
pub use generation::{
    GenerationModelStage, GenerationTokenTiming, LlamaGenerationCleanupError, LlamaGenerationError,
    LlamaGenerationEvent, LlamaGenerationFailure, LlamaGenerationResult,
    LlamaGenerationTimingSummary, PreparedLlamaGeneration,
};

#[cfg(feature = "cuda")]
pub use rustinfer_cuda::{
    AttentionBackend, AttentionPreference, AttentionReductionProfile, AttentionSelectionTrace,
};

#[cfg(any(feature = "cuda", test))]
pub(crate) use plan::{PhysicalWeightId, PhysicalWeightMetadata};

#[cfg(test)]
mod source_contract_tests {
    use super::batch::LlamaBatchMetadataConfig;
    use super::batch_executor::{
        ExecutionCompletionImplementation, PreparedLlamaBatchExecutorConfig,
        ResidualNormImplementation, normalize_prepared_config,
    };
    use super::decode::{LlamaKvCachePolicy, PreparedLlamaDecodeConfig};
    use super::forward::{LlamaTracePoint, PreparedLlamaForwardConfig};
    use super::{LLAMA_FIXED37_MAX_SEQUENCE_TOKENS, LlamaReductionProfile};
    use rustinfer_cuda::{AttentionPreference, AttentionReductionProfile};

    #[test]
    fn llama_reduction_profile_has_stable_ids_and_cuda_mapping() {
        let canonical = LlamaReductionProfile::default();
        assert_eq!(canonical, LlamaReductionProfile::CanonicalV1);
        assert_eq!(canonical.id(), "canonical-v1");
        assert_eq!(
            canonical.attention_profile(),
            AttentionReductionProfile::CanonicalV1
        );

        let fixed = LlamaReductionProfile::FixedContiguous37BalancedV1;
        assert_eq!(fixed.id(), "fixed-contiguous-37-balanced-v1");
        assert_eq!(LLAMA_FIXED37_MAX_SEQUENCE_TOKENS, 8_192);
        assert_eq!(
            fixed.attention_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
    }

    #[test]
    fn optimized_attention_is_default_and_reference_is_explicit() {
        let defaults = PreparedLlamaForwardConfig::default();
        assert_eq!(defaults.attention_budget_bytes(), 1_342_177_280);
        assert_eq!(
            defaults.attention_preference(),
            AttentionPreference::Optimized
        );
        assert_eq!(
            defaults.with_reference_attention().attention_preference(),
            AttentionPreference::Reference
        );
        assert_eq!(
            defaults.with_optimized_attention().attention_preference(),
            AttentionPreference::Optimized
        );

        let explicit_small_budget = PreparedLlamaForwardConfig::new(1, 1, 1, 512 * 1024 * 1024);
        assert_eq!(
            explicit_small_budget.attention_budget_bytes(),
            512 * 1024 * 1024
        );
    }

    #[test]
    fn batch_prepare_normalization_preserves_fused_residual_norm_selection() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid bounds");
        let config = PreparedLlamaBatchExecutorConfig::new(
            metadata,
            PreparedLlamaForwardConfig::default().with_reference_attention(),
        )
        .with_fused_residual_norm();

        let normalized = normalize_prepared_config(config);
        assert_eq!(
            normalized.residual_norm_implementation(),
            ResidualNormImplementation::Fused
        );
        assert_eq!(
            normalized.forward().attention_preference(),
            AttentionPreference::Optimized
        );
    }

    #[test]
    fn batch_prepare_normalization_preserves_iteration_completion_selection() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        assert_eq!(
            defaults.execution_completion_implementation(),
            ExecutionCompletionImplementation::PerOperation
        );
        let config = PreparedLlamaBatchExecutorConfig::new(
            metadata,
            PreparedLlamaForwardConfig::default().with_reference_attention(),
        )
        .with_iteration_batch_completion()
        .with_fused_residual_norm();

        let normalized = normalize_prepared_config(config);
        assert_eq!(
            normalized.execution_completion_implementation(),
            ExecutionCompletionImplementation::IterationBatch
        );
        assert_eq!(
            normalized.residual_norm_implementation(),
            ResidualNormImplementation::Fused
        );
        assert_eq!(
            normalized
                .with_per_operation_completion()
                .execution_completion_implementation(),
            ExecutionCompletionImplementation::PerOperation
        );
        assert_eq!(
            normalized.forward().attention_preference(),
            AttentionPreference::Optimized
        );
    }

    #[test]
    fn batch_ragged_attention_profile_is_explicit_reversible_and_preserved() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        assert_eq!(
            defaults.ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );
        assert_eq!(
            defaults
                .with_fixed37_ragged_attention()
                .ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            defaults
                .with_fixed37_ragged_attention()
                .with_canonical_ragged_attention()
                .ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );

        let normalized = normalize_prepared_config(
            defaults
                .with_ragged_attention_reduction_profile(
                    AttentionReductionProfile::FixedContiguous37BalancedV1,
                )
                .with_iteration_batch_completion(),
        );
        assert_eq!(
            normalized.ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            normalized.execution_completion_implementation(),
            ExecutionCompletionImplementation::IterationBatch
        );
    }

    #[test]
    fn exact_paged_cache_is_default_and_contiguous_is_explicit() {
        let defaults = PreparedLlamaDecodeConfig::default();
        assert_eq!(defaults.kv_cache_policy(), LlamaKvCachePolicy::paged());
        assert_eq!(
            defaults.with_contiguous_kv_cache().kv_cache_policy(),
            LlamaKvCachePolicy::Contiguous
        );
    }

    #[test]
    fn hot_execute_source_uses_only_prebound_direct_index_state() {
        let source = include_str!("forward.rs");
        let begin = source
            .find("// HOT_EXECUTE_BEGIN")
            .expect("hot execute begin marker");
        let end = source
            .find("// HOT_EXECUTE_END")
            .expect("hot execute end marker");
        let hot = &source[begin..end];

        for forbidden in [
            "WeightSlot",
            ".view(",
            "BTreeMap",
            "HashMap",
            "serde_json",
            "json!",
            "Vec::",
            "Box::",
            "vec!",
            ".collect(",
            "String::",
            "format!",
            "allocate_device_buffer",
            "allocate_pinned_host_buffer",
            "PreparedPrefillAttention::select",
            "AttentionBackend::",
            "qk_gqa(",
            "scale_causal_mask_in_place(",
            "causal_softmax_in_place(",
            "av_gqa(",
        ] {
            assert!(
                !hot.contains(forbidden),
                "hot execute source contains forbidden cold-path token {forbidden:?}"
            );
        }
        assert!(
            hot.contains("let attention = &self.attention;"),
            "hot execute must use the backend fixed during cold preparation"
        );
        assert!(
            source.contains("self.execute_inner(stream, None, None)"),
            "public cache-free execute must not attach a PR09 cache sink"
        );
    }

    #[test]
    fn decode_hot_source_contains_no_preparation_or_allocation() {
        let source = include_str!("decode.rs");
        let begin = source
            .find("// HOT_DECODE_BEGIN")
            .expect("hot decode begin marker");
        let end = source
            .find("// HOT_DECODE_END")
            .expect("hot decode end marker");
        let hot = &source[begin..end];

        for forbidden in [
            "Vec::",
            "Box::",
            "vec!",
            ".collect(",
            "String::",
            "format!",
            "BTreeMap",
            "HashMap",
            "allocate_device_buffer",
            "allocate_pinned_host_buffer",
            "PreparedDecodeAttention::select",
            "CudaGemmConfig::new",
            "prepare_gemm",
            "build_decode_rope_tables",
            "try_reserve",
        ] {
            assert!(
                !hot.contains(forbidden),
                "hot decode source contains forbidden cold-path token {forbidden:?}"
            );
        }
        assert!(
            hot.contains(".execute(logical_token_count, &mut params, stream)"),
            "hot decode must use the backend selected during cold preparation"
        );
    }

    #[test]
    fn diagnostic_trace_names_match_the_pinned_hugging_face_artifact() {
        let names: Vec<_> = LlamaTracePoint::ALL
            .into_iter()
            .map(LlamaTracePoint::name)
            .collect();
        assert_eq!(
            names,
            [
                "embedding",
                "layer0.input_norm",
                "layer0.q_proj",
                "layer0.k_proj",
                "layer0.v_proj",
                "layer0.attention_probs",
                "layer0.attention_context",
                "layer0.after_attention_residual",
                "layer0.post_attention_norm",
                "layer0.gate_proj",
                "layer0.up_proj",
                "layer0.gated",
                "layer0.down_proj",
                "layer0.output",
                "layer14.output",
                "final_norm.input",
                "final_norm.output",
                "last_logits",
            ]
        );
    }

    #[test]
    fn continuous_batch_hot_source_is_allocation_free_and_not_serial_dispatch() {
        let source = include_str!("batch_executor.rs");
        let begin = source
            .find("// HOT_BATCH_EXECUTE_BEGIN")
            .expect("batch hot execute begin marker");
        let end = source
            .find("// HOT_BATCH_EXECUTE_END")
            .expect("batch hot execute end marker");
        let hot = &source[begin..end];

        for forbidden in [
            "Vec::",
            "Box::",
            "vec!",
            ".collect(",
            "String::",
            "format!",
            "allocate_device_buffer",
            "allocate_pinned_host_buffer",
            "PreparedLlamaForward::prepare",
            ".execute(stream)",
            "for row in",
        ] {
            assert!(
                !hot.contains(forbidden),
                "batch hot execute source contains forbidden token {forbidden:?}"
            );
        }
        for required in [
            "execute_fixed_graph(",
            "PackedBatchHostV1::new(",
            "PackedBatchV1::new(",
            "output_token_indices",
        ] {
            assert!(
                hot.contains(required),
                "batch hot execute source omits required tensor-batch contract {required:?}"
            );
        }
    }

    #[test]
    fn iteration_completion_guard_wraps_only_the_fixed_graph_and_output_gather() {
        let source = include_str!("batch_executor.rs");
        let begin = source
            .find("// HOT_BATCH_EXECUTE_BEGIN")
            .expect("batch hot execute begin marker");
        let end = source
            .find("// HOT_BATCH_EXECUTE_END")
            .expect("batch hot execute end marker");
        let hot = &source[begin..end];

        let metadata_upload = hot
            .find("let batch = PackedBatchV1::new(")
            .expect("metadata is bound before execution");
        let command_begin = hot
            .find("let mut command_batch = stream")
            .expect("iteration completion begins explicitly");
        let body_result = hot
            .find("let body_result = {")
            .expect("iteration body result is retained through completion");
        let command_proxy = hot
            .find("let mut commands = command_batch.commands();")
            .expect("iteration body uses the non-replaceable command proxy");
        let body_call = hot
            .find("execute_iteration_body(&mut commands)")
            .expect("iteration body dispatches through the command proxy");
        let command_finish = hot
            .find("let completion_result = command_batch")
            .expect("iteration completion finishes explicitly");

        assert!(metadata_upload < command_begin);
        assert!(command_begin < body_result);
        assert!(body_result < command_proxy);
        assert!(command_proxy < body_call);
        assert!(body_result < command_finish);
        assert!(
            hot[body_result..command_finish].contains("command_batch.commands()"),
            "body errors must be retained without skipping command-batch finish"
        );
        assert!(
            !hot.contains("command_batch.stream_mut()"),
            "the guarded CudaStream must not be exposed for replacement"
        );
        assert!(hot.contains("LlamaOp::IterationCompletion"));
    }
}
