//! Tensor metadata and ownership boundary; allocation arrives in PR 04.

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "tensor metadata and ownership boundary";

/// Whether tensor metadata is compiled with the CUDA backend available.
pub const CUDA_ENABLED: bool = cfg!(feature = "cuda");
