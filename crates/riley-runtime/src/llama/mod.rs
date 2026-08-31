//! Cold, immutable planning contract for a fixed-length Llama forward.

mod batch;
#[cfg(any(feature = "cuda", test))]
mod batch_executor;
#[cfg(any(feature = "cuda", test))]
mod decode;
mod error;
#[cfg(any(feature = "cuda", test))]
mod executor;
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
    BatchMetadataTransport, ExecutionCompletionImplementation, LlamaBatchExecutorError,
    LlamaBatchExecutorResource, LlamaBatchExecutorResult, LlamaBatchShapeBucketHit,
    LlamaBatchShapeObservation, LlamaBatchShapePolicy, MAX_LLAMA_BATCH_SHAPE_BUCKETS,
    PreparedLlamaBatchAllocationReport, PreparedLlamaBatchExecutor,
    PreparedLlamaBatchExecutorConfig, RaggedAttentionImplementation, ResidualNormImplementation,
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
pub use riley_cuda::{
    AttentionBackend, AttentionPreference, AttentionReductionProfile, AttentionSelectionTrace,
};

#[cfg(any(feature = "cuda", test))]
pub(crate) use plan::{PhysicalWeightId, PhysicalWeightMetadata};

#[cfg(test)]
mod source_contract_tests {
    use std::error::Error as _;

    use super::batch::LlamaBatchMetadataConfig;
    use super::batch_executor::{
        ExecutionCompletionImplementation, PreparedLlamaBatchExecutorConfig,
        RaggedAttentionImplementation, ResidualNormImplementation, normalize_prepared_config,
    };
    use super::decode::{LlamaKvCachePolicy, PreparedLlamaDecodeConfig};
    use super::forward::{LlamaTracePoint, PreparedLlamaForwardConfig};
    use super::{LLAMA_FIXED37_MAX_SEQUENCE_TOKENS, LlamaReductionProfile};
    use riley_cuda::{AttentionPreference, AttentionReductionProfile};

    #[test]
    fn batch_executor_facade_reexports_the_executor_metric_value_types() {
        let observation: super::batch_executor::LlamaBatchShapeObservation =
            super::executor::metrics::LlamaBatchShapeObservation::new(3, 4, 1);
        assert_eq!(observation.active_rows(), 3);
        assert_eq!(observation.selected_dense_rows(), 4);
        assert_eq!(observation.padding_rows(), 1);

        let hit: super::batch_executor::LlamaBatchShapeBucketHit =
            super::executor::metrics::LlamaBatchShapeBucketHit::new(4);
        assert_eq!(hit.dense_rows(), 4);
        assert_eq!(hit.hit_count(), 0);
    }

    #[test]
    fn batch_executor_facade_reexports_the_executor_error_vocabulary() {
        let resource: super::batch_executor::LlamaBatchExecutorResource =
            super::executor::error::LlamaBatchExecutorResource::HostWorkspace;
        assert_eq!(resource.to_string(), "batch_host_workspace");

        let batch_error = super::batch::LlamaBatchError::InvalidBatch {
            field: "rows",
            reason: "test-only invalid batch",
        };
        let executor_error: super::batch_executor::LlamaBatchExecutorError = batch_error.into();
        assert_eq!(
            executor_error.to_string(),
            "invalid Llama batch rows: test-only invalid batch"
        );
        assert!(executor_error.source().is_some());

        let result: super::batch_executor::LlamaBatchExecutorResult<()> = Err(executor_error);
        assert!(matches!(
            result,
            Err(super::batch_executor::LlamaBatchExecutorError::Metadata(_))
        ));
    }

    #[test]
    fn batch_executor_facade_reexports_the_executor_shape_policy() {
        let policy: super::batch_executor::LlamaBatchShapePolicy =
            super::executor::shape::LlamaBatchShapePolicy::ActiveRowBuckets;
        assert_eq!(
            policy
                .select_dense_rows(3, 512)
                .expect("select active bucket"),
            4
        );
        assert_eq!(
            super::batch_executor::MAX_LLAMA_BATCH_SHAPE_BUCKETS,
            super::executor::shape::MAX_LLAMA_BATCH_SHAPE_BUCKETS
        );
    }

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
    fn batch_ragged_attention_launch_is_explicit_reversible_and_preserved() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        assert_eq!(
            defaults.ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
        );
        assert_eq!(
            defaults
                .with_grouped_ragged_attention_heads()
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::GroupedHeads
        );
        assert_eq!(
            normalize_prepared_config(defaults.with_grouped_ragged_attention_heads())
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::GroupedHeads
        );
        assert_eq!(
            defaults
                .with_grouped_ragged_attention_heads()
                .with_legacy_ragged_attention_heads()
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
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
            "build_decode_rope_angles",
            "build_decode_cpu_rope_tables",
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
            "dispatch_output_primitives(",
            "PackedBatchHostV1::new(",
            "per_operation_device_views(",
            "packed_device_views(",
            "output_token_indices",
        ] {
            assert!(
                hot.contains(required),
                "batch hot execute source omits required tensor-batch contract {required:?}"
            );
        }
        let fixed_graph_offset = hot
            .find("execute_fixed_graph(")
            .expect("fixed graph dispatch remains explicit");
        let output_dispatch_offset = hot
            .find("dispatch_output_primitives(")
            .expect("output primitive dispatch remains explicit");
        assert!(
            fixed_graph_offset < output_dispatch_offset,
            "output primitives must run after the fixed graph produces logits"
        );
    }

    #[test]
    fn batch_shape_gemms_use_the_configured_cap_and_one_cold_shared_workspace() {
        let forward = include_str!("forward.rs");
        let variant_begin = forward
            .find("fn prepare_batch_shape_variant(")
            .expect("batch shape variant preparation remains explicit");
        let variant_end = forward[variant_begin..]
            .find("fn ensure_batch_shape_gemm_workspace(")
            .map(|offset| variant_begin + offset)
            .expect("shared workspace reconciliation remains explicit");
        let variant = &forward[variant_begin..variant_end];
        assert!(variant.contains("self.gemm_workspace_cap_bytes"));
        assert!(variant.contains("Some(&self.gemms)"));
        assert!(!variant.contains("gemm_workspace.as_ref()"));
        let gemm_prepare_begin = forward
            .find("pub(super) fn prepare_gemms(")
            .expect("GEMM preparation remains explicit");
        let gemm_prepare = &forward[gemm_prepare_begin..];
        assert!(gemm_prepare.contains(".prepare_gemm_anchored(config, anchor)"));

        let batch = include_str!("batch_executor.rs");
        let prepare_variants = batch
            .find("let shape_variants = match prepare_shape_variants(")
            .expect("shape variants are cold-prepared");
        let maximum_requirement = batch
            .find("let required_gemm_workspace_bytes = shape_variants.iter().fold(")
            .expect("all prepared shape requirements are reduced to one maximum");
        let reconcile = batch
            .find("forward.ensure_batch_shape_gemm_workspace(")
            .expect("the shared workspace is reconciled once during preparation");
        let hot_begin = batch
            .find("// HOT_BATCH_EXECUTE_BEGIN")
            .expect("batch hot-path marker remains present");
        assert!(prepare_variants < maximum_requirement);
        assert!(maximum_requirement < reconcile);
        assert!(reconcile < hot_begin);
        assert_eq!(
            batch.matches("ensure_batch_shape_gemm_workspace(").count(),
            1
        );
    }

    #[test]
    fn packed_metadata_transport_keeps_the_exact_synchronous_fallback() {
        let source = include_str!("batch_executor.rs");
        let begin = source
            .find("// HOT_BATCH_EXECUTE_BEGIN")
            .expect("batch hot execute begin marker");
        let end = source
            .find("// HOT_BATCH_EXECUTE_END")
            .expect("batch hot execute end marker");
        let hot = &source[begin..end];

        assert_eq!(
            hot.matches("upload_batch_tokens(").count(),
            1,
            "synchronous fallback must retain its established token upload"
        );
        assert_eq!(
            hot.matches("upload_prefix(").count(),
            6,
            "synchronous fallback must retain all six metadata uploads"
        );
        assert_eq!(
            hot.matches(".copy_from_pinned_in_command_batch(").count(),
            1,
            "packed async must enqueue exactly one contiguous H2D call"
        );
        assert!(hot.contains("BatchMetadataTransport::Synchronous"));
        assert!(hot.contains("BatchMetadataTransport::PackedAsync"));
    }

    #[test]
    fn iteration_completion_guards_synchronous_and_packed_async_bodies() {
        let source = include_str!("batch_executor.rs");
        let begin = source
            .find("// HOT_BATCH_EXECUTE_BEGIN")
            .expect("batch hot execute begin marker");
        let end = source
            .find("// HOT_BATCH_EXECUTE_END")
            .expect("batch hot execute end marker");
        let hot = &source[begin..end];
        let pack_input = hot
            .find("pack_iteration_input(")
            .expect("packed host bytes are assembled before their pinned write");
        let pinned_write = hot
            .find("host.pinned")
            .expect("packed host bytes are written to pinned storage explicitly");
        let packed_copy = hot
            .find("copy_from_pinned_in_command_batch(")
            .expect("packed input uses one command-batch H2D");
        assert!(pack_input < pinned_write);
        assert!(pinned_write < packed_copy);

        let execution_match = hot
            .find("match (config.execution_completion, config.metadata_transport)")
            .expect("execution and metadata policies dispatch together");
        let execution = &hot[execution_match..];
        let synchronous_arm = execution
            .find("BatchMetadataTransport::Synchronous,\n        ) => {")
            .expect("synchronous iteration arm remains explicit");
        let packed_arm = execution
            .find("BatchMetadataTransport::PackedAsync,\n        ) => {")
            .expect("packed async iteration arm remains explicit");
        let synchronous = &execution[synchronous_arm..packed_arm];
        assert!(synchronous.contains("per_operation_device_views("));
        assert!(synchronous.contains("let mut command_batch = stream"));
        assert!(synchronous.contains("&mut commands,"));
        assert!(synchronous.contains("let completion_result = command_batch"));
        assert!(!synchronous.contains("copy_from_pinned_in_command_batch"));

        let packed = &execution[packed_arm..];
        let command_begin = packed
            .find("let mut command_batch = stream")
            .expect("packed iteration completion begins explicitly");
        let body_result = packed
            .find("let body_result = {")
            .expect("packed body result is retained through completion");
        let command_proxy = packed
            .find("let mut commands = command_batch.commands();")
            .expect("packed body uses the non-replaceable command proxy");
        let copy = packed
            .find("copy_from_pinned_in_command_batch(")
            .expect("packed input uses one command-batch H2D");
        let bind_views = packed
            .find("match packed_device_views(")
            .expect("packed device spans bind after the H2D enqueue");
        let body_call = packed
            .find("Ok(views) => execute_iteration_body(")
            .expect("packed graph dispatches through the command proxy");
        let command_finish = packed
            .find("let completion_result = command_batch")
            .expect("packed iteration completion finishes explicitly");

        assert!(command_begin < body_result);
        assert!(body_result < command_proxy);
        assert!(command_proxy < copy);
        assert!(copy < bind_views);
        assert!(bind_views < body_call);
        assert!(body_result < command_finish);
        assert!(
            packed[body_result..command_finish].contains("command_batch.commands()"),
            "body errors must be retained without skipping command-batch finish"
        );
        assert!(
            !hot.contains("command_batch.stream_mut()"),
            "the guarded CudaStream must not be exposed for replacement"
        );
        assert!(hot.contains("LlamaOp::IterationCompletion"));
    }
}
