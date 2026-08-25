//! Cold, immutable planning contract for a fixed-length Llama forward.

mod batch;
#[cfg(any(feature = "cuda", test))]
mod batch_executor;
#[cfg(any(feature = "cuda", test))]
mod decode;
mod error;
#[cfg(any(feature = "cuda", test))]
mod forward;
#[cfg(feature = "cuda")]
mod generation;
mod plan;

pub use batch::{
    LLAMA_BATCH_METADATA_V1_VERSION, LLAMA_BATCH_NO_OUTPUT_SLOT, LlamaBatchBlockTable,
    LlamaBatchBufferCapacities, LlamaBatchError, LlamaBatchMetadataConfig, LlamaBatchResult,
    LlamaBatchRow, LlamaBatchRowKind, LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
};

#[cfg(feature = "cuda")]
pub use batch_executor::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    PreparedLlamaBatchAllocationReport, PreparedLlamaBatchExecutor,
    PreparedLlamaBatchExecutorConfig,
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
pub use rustinfer_cuda::{AttentionBackend, AttentionPreference, AttentionSelectionTrace};

#[cfg(any(feature = "cuda", test))]
pub(crate) use plan::{PhysicalWeightId, PhysicalWeightMetadata};

#[cfg(test)]
mod source_contract_tests {
    use super::decode::{LlamaKvCachePolicy, PreparedLlamaDecodeConfig};
    use super::forward::{LlamaTracePoint, PreparedLlamaForwardConfig};
    use rustinfer_cuda::AttentionPreference;

    #[test]
    fn optimized_attention_is_default_and_reference_is_explicit() {
        let defaults = PreparedLlamaForwardConfig::default();
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
}
