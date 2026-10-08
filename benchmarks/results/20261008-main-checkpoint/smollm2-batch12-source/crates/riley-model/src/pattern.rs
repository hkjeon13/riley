//! Closed semantic pattern vocabulary owned by the canonical model layer.
//!
//! This module deliberately describes only model semantics. Runtime
//! implementation IDs, capability predicates, CUDA implementations, and
//! dispatch selection belong to the later executable-registry migration.

/// Stable schema version for the closed semantic pattern vocabulary.
///
/// Version 1 reserves numeric value zero and assigns the eight values listed
/// in [`SemanticPattern::ALL`]. Changing an assignment or adding a semantic
/// pattern requires a schema-version change.
pub const SEMANTIC_PATTERN_SCHEMA_VERSION: u16 = 1;

/// Stable numeric identity for one known [`SemanticPattern`].
///
/// Construction is intentionally constrained to [`Self::from_u16`]: unknown
/// raw values never become untyped pattern identities.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct PatternId(u16);

impl PatternId {
    /// Decodes one schema-stable numeric value.
    ///
    /// Reserved, unknown, and future values return `None` so callers remain
    /// fail-closed.
    #[must_use]
    pub const fn from_u16(value: u16) -> Option<Self> {
        match SemanticPattern::from_numeric(value) {
            Some(pattern) => Some(pattern.id()),
            None => None,
        }
    }

    /// Returns the schema-stable numeric value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self.0
    }
}

/// One closed, model-owned semantic pattern.
///
/// These identifiers describe what a model operation means, not how the
/// runtime executes it. They intentionally carry no dtype, layout, GPU, or
/// implementation-selection detail.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u16)]
pub enum SemanticPattern {
    /// Normalization without an adjacent residual addition.
    Norm = 1,
    /// Attention input preparation before attention execution.
    AttentionPrepare = 2,
    /// Attention over a prefill workload.
    PrefillAttention = 3,
    /// Attention over a pure-decode workload.
    DecodeAttention = 4,
    /// `SwiGLU` gated MLP computation.
    MlpSwiGlu = 5,
    /// Residual addition followed by normalization.
    ResidualNorm = 6,
    /// Final language-model projection.
    LmHead = 7,
    /// Greedy output selection from language-model scores.
    GreedyOutput = 8,
}

impl SemanticPattern {
    /// All semantic patterns defined by schema version 1, in ascending ID order.
    pub const ALL: [Self; 8] = [
        Self::Norm,
        Self::AttentionPrepare,
        Self::PrefillAttention,
        Self::DecodeAttention,
        Self::MlpSwiGlu,
        Self::ResidualNorm,
        Self::LmHead,
        Self::GreedyOutput,
    ];

    /// Returns this pattern's stable numeric identity.
    #[must_use]
    pub const fn id(self) -> PatternId {
        PatternId(self as u16)
    }

    /// Decodes a known [`PatternId`] into its semantic pattern.
    ///
    /// The `Option` preserves fail-closed behavior if a future schema expands
    /// the ID type before this version learns the new semantic value.
    #[must_use]
    pub const fn from_id(id: PatternId) -> Option<Self> {
        Self::from_numeric(id.0)
    }

    const fn from_numeric(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Norm),
            2 => Some(Self::AttentionPrepare),
            3 => Some(Self::PrefillAttention),
            4 => Some(Self::DecodeAttention),
            5 => Some(Self::MlpSwiGlu),
            6 => Some(Self::ResidualNorm),
            7 => Some(Self::LmHead),
            8 => Some(Self::GreedyOutput),
            _ => None,
        }
    }
}
