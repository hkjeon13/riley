//! Internal components of the Llama continuous-batch executor.
//!
//! The public facade remains `crate::llama`.  Components in this directory
//! must not take ownership of CUDA buffers, weights, KV storage, streams, or
//! scheduling policy unless their dedicated ownership boundary says so.

pub(crate) mod allocation;
pub(crate) mod buffers;
pub(in crate::llama) mod config;
pub(crate) mod device_views;
pub(crate) mod dispatch;
pub(crate) mod error;
pub(crate) mod gemm_plan;
pub(crate) mod host;
pub(crate) mod metadata;
pub(crate) mod metrics;
pub(crate) mod output;
pub(crate) mod poison;
pub(crate) mod rope;
pub(crate) mod shape;
