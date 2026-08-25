//! Cold, immutable planning contract for a fixed-length Llama forward.

mod error;
#[cfg(any(feature = "cuda", test))]
mod forward;
mod plan;

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

#[cfg(any(feature = "cuda", test))]
pub(crate) use plan::{PhysicalWeightId, PhysicalWeightMetadata};

#[cfg(test)]
mod source_contract_tests {
    use super::forward::LlamaTracePoint;

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
            "String::",
            "format!",
            "allocate_device_buffer",
            "allocate_pinned_host_buffer",
        ] {
            assert!(
                !hot.contains(forbidden),
                "hot execute source contains forbidden cold-path token {forbidden:?}"
            );
        }
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
}
