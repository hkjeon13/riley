//! Internal components of the Llama continuous-batch executor.
//!
//! The public facade remains `crate::llama`.  Components in this directory
//! must not take ownership of CUDA buffers, weights, KV storage, streams, or
//! scheduling policy unless their dedicated ownership boundary says so.

pub(crate) mod error;
pub(crate) mod metrics;
