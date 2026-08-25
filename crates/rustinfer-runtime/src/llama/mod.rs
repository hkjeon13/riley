//! Cold, immutable planning contract for a fixed-length Llama forward.

mod error;
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
pub(crate) use plan::{PhysicalWeightId, PhysicalWeightMetadata};
