//! Canonical model IR boundary; model parsing arrives in PR 05.

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "canonical model IR boundary";

/// Returns the lower-layer role this crate is allowed to depend on.
#[must_use]
pub const fn tensor_dependency_role() -> &'static str {
    rustinfer_tensor::CRATE_ROLE
}
