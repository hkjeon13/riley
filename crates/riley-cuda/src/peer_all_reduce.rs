//! Deterministic tensor-parallel all-reduce over CUDA peer access.
//!
//! A [`CudaPeerAllReduceGroup`] spans one context per rank on distinct,
//! peer-accessible devices of this process. Each rank sums its BF16 data with
//! every other rank's in place through
//! [`crate::peer_all_reduce_bf16_in_place`]: element `i` becomes the FP32 sum
//! of the ranks' values in rank order, rounded once to BF16. The term order
//! depends only on the rank count, so the result is the same whatever the
//! element count (batch size) and bit-identical on every rank.

use std::fmt;
use std::sync::Arc;

use crate::error::{CudaError, CudaResult};
#[cfg(feature = "cuda")]
use crate::ffi;
use crate::runtime::{ContextInner, CudaContext};

/// Peer all-reduce resources of a tensor-parallel group.
///
/// The group is `Send + Sync`: each rank's host thread executes on its own
/// stream. Every rank must execute the same sequence of element counts; a rank
/// that stops participating makes its peers' barriers trap after ten seconds.
pub struct CudaPeerAllReduceGroup {
    #[cfg(feature = "cuda")]
    native: ffi::PeerAllReduceGroupHandle,
    // Declared after native so the group closes before the contexts drop.
    contexts: Vec<Arc<ContextInner>>,
    max_elements: u64,
}

impl CudaPeerAllReduceGroup {
    /// Creates the group over `contexts` in rank order and enables peer
    /// access between every pair.
    ///
    /// # Errors
    ///
    /// Fails for fewer than two or more than eight contexts, repeated devices,
    /// a pair without peer access, zero `max_elements`, an active graph
    /// capture, or a CUDA allocation failure.
    pub fn new(contexts: &[&CudaContext], max_elements: u64) -> CudaResult<Self> {
        const OPERATION: &str = "create CUDA peer all-reduce group";
        if !(2..=8).contains(&contexts.len()) {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "two to eight contexts are required",
            ));
        }
        if max_elements == 0 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "max_elements must be non-zero",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let handles: Vec<&ffi::ContextHandle> =
                contexts.iter().map(|context| &context.inner.native).collect();
            let native = ffi::PeerAllReduceGroupHandle::create(&handles, max_elements)?;
            Ok(Self {
                native,
                contexts: contexts.iter().map(|context| Arc::clone(&context.inner)).collect(),
                max_elements,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Number of ranks.
    #[must_use]
    pub fn rank_count(&self) -> u32 {
        u32::try_from(self.contexts.len()).unwrap_or(u32::MAX)
    }

    /// Largest element count one execute may reduce.
    #[must_use]
    pub const fn max_elements(&self) -> u64 {
        self.max_elements
    }

    /// Device ordinal of `rank`.
    #[must_use]
    pub fn device_ordinal(&self, rank: u32) -> Option<u32> {
        self.contexts
            .get(usize::try_from(rank).ok()?)
            .map(|context| context.ordinal)
    }

    pub(crate) fn context(&self, rank: u32) -> Option<&Arc<ContextInner>> {
        self.contexts.get(usize::try_from(rank).ok()?)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn native(&self) -> &ffi::PeerAllReduceGroupHandle {
        &self.native
    }
}

impl fmt::Debug for CudaPeerAllReduceGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaPeerAllReduceGroup")
            .field(
                "devices",
                &self.contexts.iter().map(|context| context.ordinal).collect::<Vec<_>>(),
            )
            .field("max_elements", &self.max_elements)
            .finish()
    }
}
