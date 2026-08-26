/// Maximum logical sequence length supported by the complete fixed37 profile.
///
/// Ragged batch attention addresses positions `0..8192`; callers selecting the
/// complete profile must bound their advertised context to this value.
pub const LLAMA_FIXED37_MAX_SEQUENCE_TOKENS: usize = 8_192;

/// Stable reduction contract selected for Llama execution.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum LlamaReductionProfile {
    /// Preserve the established canonical reduction contract.
    #[default]
    CanonicalV1,
    /// Use contiguous groups of 37 inputs followed by balanced reductions.
    FixedContiguous37BalancedV1,
}

const _: () = assert!(
    LLAMA_FIXED37_MAX_SEQUENCE_TOKENS as u64 == rustinfer_cuda::FIXED37_RAGGED_MAX_LOGICAL_TOKENS
);

impl LlamaReductionProfile {
    /// Returns the stable identifier used by configuration and evidence artifacts.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::CanonicalV1 => "canonical-v1",
            Self::FixedContiguous37BalancedV1 => "fixed-contiguous-37-balanced-v1",
        }
    }

    pub(crate) const fn attention_profile(self) -> rustinfer_cuda::AttentionReductionProfile {
        match self {
            Self::CanonicalV1 => rustinfer_cuda::AttentionReductionProfile::CanonicalV1,
            Self::FixedContiguous37BalancedV1 => {
                rustinfer_cuda::AttentionReductionProfile::FixedContiguous37BalancedV1
            }
        }
    }
}
