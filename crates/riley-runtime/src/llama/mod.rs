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
#[path = "executor/graph.rs"]
mod graph;
#[allow(dead_code)] // C07-4 deliberately precedes its future metadata packer.
mod graph_decode_binding;
#[allow(dead_code)] // C07-8 deliberately precedes opaque metadata-tail materialization.
mod graph_decode_exact_projection;
#[allow(dead_code)] // C07-9 deliberately precedes fixed source construction and slab writing.
mod graph_decode_exact_sources;
#[allow(dead_code)] // C07-1 deliberately precedes its future executor owner.
mod graph_decode_layout;
#[allow(dead_code)] // C07-5 deliberately precedes its future executor adapter.
mod graph_decode_packer;
#[allow(dead_code)] // C07-3 deliberately precedes its future metadata packer.
mod graph_decode_padding;
#[allow(dead_code)] // C07-6 deliberately precedes its future executor adapter.
mod graph_decode_preflight;
#[allow(dead_code)] // C07-7 deliberately precedes its future V1 field/capacity adapter.
mod graph_decode_preflight_binding;
#[path = "executor/graph_metrics.rs"]
mod graph_metrics;
#[path = "executor/graph_registry.rs"]
mod graph_registry;
#[path = "executor/graph_registry_dispatch.rs"]
mod graph_registry_dispatch;
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
pub use graph::{
    ExecutionGraphPolicy, ExecutionMode, GRAPH_SIGNATURE_SCHEMA_VERSION, GraphCaptureSafety,
    GraphComputeType, GraphDataType, GraphDeviceSignature, GraphDispatchDecision,
    GraphDispatchEligibility, GraphDispatchError, GraphDispatchRequest, GraphFallbackReason,
    GraphGemmPlanSetId, GraphGeometrySignature, GraphImplementationId,
    GraphImplementationSignature, GraphInventoryState, GraphIterationSignature,
    GraphLayoutSignature, GraphMetadataLayoutSignature, GraphModelArchitecture,
    GraphModelSignature, GraphOperatorCapability, GraphReductionPolicyId, GraphRevisionFingerprint,
    GraphSamplingBackend, GraphSignature, GraphSignatureFingerprint, GraphStaticSignature,
    GraphTensorSignature, GraphWorkloadStage, select_execution_graph,
};
pub use graph_metrics::{GraphDispatchMetrics, GraphDispatchMetricsSnapshot};
pub use graph_registry::{
    GraphEntryFootprint, GraphRegistry, GraphRegistryAvailability, GraphRegistryBuildError,
    GraphRegistryEntry, GraphRegistryEntryState, GraphRegistryLimits, GraphRegistryLookup,
    GraphRegistryUsage, GraphReplayMode, GraphReplaySlot,
};
pub use graph_registry_dispatch::{
    GraphRegistryDispatchDecision, select_registered_execution_graph,
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
    fn absolute_rope_position_shape_is_shared_as_value_only_arithmetic() {
        let source = include_str!("batch_executor.rs");
        assert!(
            !source.contains("fn model_max_position("),
            "RoPE position shape must not retain a CUDA-buffer helper in the batch owner"
        );
        assert_eq!(
            source.matches("absolute_rope_position_count(").count(),
            3,
            "public query, metadata preflight, and fixed graph must share one RoPE shape helper"
        );
    }

    #[test]
    fn absolute_rope_builders_share_cold_table_shape_preflight() {
        let source = include_str!("executor/rope.rs");
        assert!(source.contains("fn absolute_rope_table_shape("));
        assert_eq!(
            source
                .matches("absolute_rope_table_shape(position_count, head_dimension)")
                .count(),
            2,
            "both cold absolute RoPE builders must share one shape preflight"
        );
    }

    #[test]
    fn cold_zeroed_host_bytes_share_one_checked_allocator() {
        for (boundary, source, expected_calls) in [
            ("batch owner", include_str!("batch_executor.rs"), 1),
            ("batch buffers", include_str!("executor/buffers.rs"), 7),
            ("RoPE builders", include_str!("executor/rope.rs"), 3),
        ] {
            assert!(
                !source.contains("fn allocate_zeroed_bytes("),
                "{boundary} must not retain a local zeroed-byte allocator"
            );
            assert_eq!(
                source.matches("allocate_zeroed_host_bytes(").count(),
                expected_calls,
                "{boundary} must use the shared zeroed-byte allocator at every existing boundary"
            );
        }
    }

    #[test]
    fn executor_byte_lengths_share_one_typed_overflow_facade() {
        for (boundary, source, expected_calls) in [
            ("batch owner", include_str!("batch_executor.rs"), 7),
            ("batch buffers", include_str!("executor/buffers.rs"), 2),
            ("packed metadata", include_str!("executor/metadata.rs"), 3),
        ] {
            assert!(
                !source.contains("fn checked_host_byte_len(")
                    && !source.contains("fn checked_byte_len("),
                "{boundary} must not retain a local typed byte-length helper"
            );
            assert_eq!(
                source.matches("checked_byte_len(").count(),
                expected_calls,
                "{boundary} must use the shared typed byte-length helper at every existing boundary"
            );
        }
    }

    #[test]
    fn packed_metadata_encoders_share_checked_region_validation() {
        let source = include_str!("executor/metadata.rs");
        let helper_begin = source
            .find("fn checked_region_slice_mut")
            .expect("shared region validation remains explicit");
        let helper_end = source[helper_begin..]
            .find("/// Encodes native-endian `u16`")
            .map(|offset| helper_begin + offset)
            .expect("U16 encoder follows the shared region validator");
        let helper = &source[helper_begin..helper_end];
        assert!(helper.contains("checked_byte_len(source_len, element_bytes, resource)"));
        assert!(helper.contains("region_slice_mut(destination, region, resource)"));

        for function in ["fn encode_u32_region(", "fn encode_u16_region("] {
            let begin = source
                .find(function)
                .expect("typed region encoder remains explicit");
            let end = source[begin..]
                .find("\n}\n")
                .map(|offset| begin + offset + 3)
                .expect("typed region encoder has one function boundary");
            let body = &source[begin..end];
            assert!(body.contains("checked_region_slice_mut("));
            assert!(!body.contains("checked_byte_len("));
            assert!(!body.contains("let bytes = region_slice_mut("));
        }
    }

    #[test]
    fn packed_slab_capacity_validation_is_shared_before_write_or_bind() {
        let metadata = include_str!("executor/metadata.rs");
        let helper_begin = metadata
            .find("pub(crate) fn validate_u64_capacity")
            .expect("packed slab conversion remains in the layout boundary");
        let helper_end = metadata[helper_begin..]
            .find("\n}\n\n/// Validates")
            .map(|offset| helper_begin + offset)
            .expect("layout conversion helper remains before host preflight");
        let helper = &metadata[helper_begin..helper_end];
        assert!(helper.contains("usize::try_from(capacity)"));
        assert!(helper.contains("LlamaBatchExecutorResource::PackedIterationInput"));
        assert!(helper.contains("self.validate_capacity(capacity)"));

        for (boundary, source) in [
            ("batch owner", include_str!("batch_executor.rs")),
            ("device views", include_str!("executor/device_views.rs")),
        ] {
            assert_eq!(
                source
                    .matches("validate_u64_capacity(slab.byte_len())")
                    .count(),
                1,
                "{boundary} must validate the packed CUDA slab through the shared layout helper"
            );
            assert!(
                !source.contains("usize::try_from(slab.byte_len())"),
                "{boundary} must not retain a local packed CUDA slab conversion"
            );
        }
    }

    #[test]
    fn sequence_block_offset_count_is_shared_before_capacity_or_allocation() {
        assert_eq!(
            super::executor::metadata::sequence_block_offset_count(0)
                .expect("zero rows still need one CSR offset"),
            1
        );
        assert!(matches!(
            super::executor::metadata::sequence_block_offset_count(usize::MAX),
            Err(
                super::executor::error::LlamaBatchExecutorError::ArithmeticOverflow {
                    resource:
                        super::executor::error::LlamaBatchExecutorResource::SequenceBlockOffsets,
                }
            )
        ));
        for (boundary, source, expected_calls) in [
            (
                "batch allocation report",
                include_str!("batch_executor.rs"),
                1,
            ),
            ("buffer allocation", include_str!("executor/buffers.rs"), 2),
            ("packed metadata", include_str!("executor/metadata.rs"), 2),
        ] {
            assert_eq!(
                source.matches("sequence_block_offset_count(").count(),
                expected_calls,
                "{boundary} must share the sequence-block-offset count helper"
            );
        }
        for (boundary, source) in [
            ("batch allocation report", include_str!("batch_executor.rs")),
            ("buffer allocation", include_str!("executor/buffers.rs")),
        ] {
            assert!(
                !source.contains("checked_add(1)"),
                "{boundary} must not retain a local sequence-block-offset count"
            );
        }
    }

    #[test]
    fn executor_usize_u64_conversions_share_one_typed_error_facade() {
        for (boundary, source, expected_calls) in [
            ("batch owner", include_str!("batch_executor.rs"), 14),
            ("batch buffers", include_str!("executor/buffers.rs"), 3),
            ("device views", include_str!("executor/device_views.rs"), 3),
            ("output dispatch", include_str!("executor/dispatch.rs"), 5),
            ("output sizing", include_str!("executor/output.rs"), 3),
            ("RoPE scalar", include_str!("executor/rope.rs"), 1),
        ] {
            assert!(
                !source.contains("fn usize_u64("),
                "{boundary} must not retain a local typed usize-to-u64 conversion"
            );
            assert_eq!(
                source.matches("usize_u64(").count(),
                expected_calls,
                "{boundary} must use the shared typed usize-to-u64 conversion at every existing boundary"
            );
        }
    }

    #[test]
    fn cold_output_capacity_reuses_canonical_output_sizing() {
        let source = include_str!("batch_executor.rs");
        let capacity_begin = source
            .find("let gathered_logits_capacity_bytes")
            .expect("cold gathered-logits capacity remains explicit");
        let capacity_end = source[capacity_begin..]
            .find("let host = allocate_host_workspace(")
            .map(|offset| capacity_begin + offset)
            .expect("cold host workspace follows output capacity preparation");
        let capacities = &source[capacity_begin..capacity_end];
        assert_eq!(capacities.matches("output_logits_bytes(").count(), 1);
        assert_eq!(
            capacities.matches("greedy_result_capacity_bytes(").count(),
            1
        );
        assert!(!capacities.contains("checked_product_u64("));
        assert!(!capacities.contains("LlamaBatchExecutorResource::GatheredLogits"));
        assert!(!capacities.contains("LlamaBatchExecutorResource::GreedyResults"));
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
    #[allow(clippy::too_many_lines)]
    fn iteration_completion_guards_synchronous_and_packed_async_bodies() {
        let source = include_str!("batch_executor.rs");
        let dispatch = include_str!("executor/dispatch.rs");
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
        assert!(synchronous.contains("execute_iteration_command_batch("));
        assert!(synchronous.contains("execute_iteration_body("));
        assert!(!synchronous.contains("begin_command_batch()"));
        assert!(!synchronous.contains("copy_from_pinned_in_command_batch"));

        let packed = &execution[packed_arm..];
        let command_guard = packed
            .find("execute_iteration_command_batch(")
            .expect("packed iteration uses the shared command-batch guard");
        let copy = packed
            .find("copy_from_pinned_in_command_batch(")
            .expect("packed input uses one command-batch H2D");
        let bind_views = packed
            .find("match packed_device_views(")
            .expect("packed device spans bind after the H2D enqueue");
        let body_call = packed
            .find("Ok(views) => execute_iteration_body(")
            .expect("packed graph dispatches through the command proxy");
        assert!(command_guard < copy);
        assert!(copy < bind_views);
        assert!(bind_views < body_call);
        assert_eq!(
            hot.matches("execute_iteration_command_batch(").count(),
            2,
            "both iteration-completion arms must share the command-batch guard"
        );
        assert!(!hot.contains(".begin_command_batch()"));
        assert!(!hot.contains("command_batch.commands()"));
        assert!(!hot.contains("let completion_result = command_batch"));

        let guard_start = dispatch
            .find("fn execute_iteration_command_batch")
            .expect("shared command-batch guard remains explicit");
        let guard = &dispatch[guard_start..];
        let command_begin = guard
            .find(".begin_command_batch()")
            .expect("shared guard begins the native command batch");
        let disposition = guard
            .find("CommandSubmissionStarted")
            .expect("shared guard records mutation-unknown disposition after begin");
        let body_result = guard
            .find("let body_result = {")
            .expect("shared guard retains body errors through completion");
        let command_proxy = guard
            .find("let mut commands = command_batch.commands();")
            .expect("shared guard uses the non-replaceable command proxy");
        let guard_body_call = guard
            .find("body(&mut commands)")
            .expect("shared guard invokes the borrowed body through the proxy");
        let command_finish = guard
            .find("let completion_result = command_batch")
            .expect("shared guard finishes every opened command batch");
        let completion_match = guard
            .find("match completion_result")
            .expect("shared guard makes completion error precedence explicit");
        let finish_error = guard
            .find("Err(error) => Err(error)")
            .expect("completion error remains higher priority than body error");
        let body_success = guard
            .find("Ok(()) => body_result")
            .expect("successful completion returns the retained body result");

        assert!(command_begin < disposition);
        assert!(disposition < body_result);
        assert!(body_result < command_proxy);
        assert!(command_proxy < guard_body_call);
        assert!(guard_body_call < command_finish);
        assert!(body_result < command_finish);
        assert!(command_finish < completion_match);
        assert!(completion_match < finish_error);
        assert!(finish_error < body_success);
        assert!(
            guard[body_result..command_finish].contains("command_batch.commands()"),
            "body errors must be retained without skipping command-batch finish"
        );
        assert!(
            !guard.contains("command_batch.stream_mut()"),
            "the guarded CudaStream must not be exposed for replacement"
        );
        assert!(guard.contains("LlamaOp::IterationCompletion"));
    }
}
