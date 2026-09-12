//! Bounded scheduler configuration and overload policy.

use crate::error::{SchedulerError, SchedulerResult};

/// Behavior when a request cannot be admitted to the active set immediately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverloadPolicy {
    /// Reject the request at the capacity boundary without queueing it.
    RejectImmediately,
    /// Keep the request in the bounded waiting queue.
    Wait,
}

/// Opt-in iteration shapes supported by the executor prepared with a scheduler.
///
/// This is separate from [`SchedulerConfig`] so existing configuration literals
/// and [`crate::Scheduler::new`] keep the general scheduling policy unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionShapePolicy {
    /// Existing decode-first, chunked-prefill scheduling with aging.
    #[default]
    General,
    /// One complete P128 prefill or independent decode rows, never mixed.
    ///
    /// Requires active capacity 2, 4 or 8, token/chunk budget 128, sequence capacity 160
    /// and at least ten promised/physical KV blocks per active request. Inputs
    /// are limited to vocabulary 49152 and output limits 1..=32. This policy only
    /// constrains host plans; it does not qualify a multi-sequence GPU executor.
    CompletePrefill128DecodeN,
    /// One variable-length prefill chunk or up to eight independent decode rows.
    /// Alternates ready classes after dispatch; never mixes them in one plan.
    /// This is a host contract only; GPU support must be checked separately.
    VariablePrefillDecodeN,
    /// Variable prefill and up to sixteen decode rows on a prepared V4 session.
    VariablePrefillDecode16,
    /// Variable prefill and up to thirty-two decode rows on a prepared V5 session.
    VariablePrefillDecode32,
}

/// All host-memory, work, and KV promises enforced by one scheduler.
///
/// The fields are public to make deployment configuration explicit. Call
/// [`Self::validate`] before allocating scheduler state; [`crate::Scheduler`]
/// does this on construction as a second line of defense.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    /// Maximum number of requests retained in the waiting queue.
    pub max_waiting_requests: usize,
    /// Maximum total number of prompt tokens retained by waiting requests.
    pub max_waiting_prompt_tokens: usize,
    /// Maximum number of sequences with an owned KV-cache reservation.
    pub max_active_sequences: usize,
    /// Maximum prompt plus generated-token capacity of one sequence.
    pub max_sequence_tokens: usize,
    /// Maximum input tokens placed in a single immutable iteration plan.
    pub iteration_token_budget: usize,
    /// Maximum tokens taken from one prefill request in one iteration.
    pub max_prefill_chunk_tokens: usize,
    /// Waiting duration after which aging may override decode-first ordering.
    pub aging_threshold_ns: u64,
    /// Capacity-boundary behavior for newly submitted requests.
    pub overload_policy: OverloadPolicy,
    /// Optional bounded wait before a queued request is failed.
    ///
    /// This must be `None` with [`OverloadPolicy::RejectImmediately`].
    pub admission_timeout_ns: Option<u64>,
    /// Maximum total KV blocks promised to active sequences.
    pub max_promised_kv_blocks: usize,
    /// Fixed sample capacity of every rolling latency/size metric.
    pub metrics_window_samples: usize,
}

impl SchedulerConfig {
    /// Validates every non-zero bound and cross-field invariant.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::InvalidConfiguration`] for an unusable bound
    /// or an inconsistent overload/timeout policy.
    pub fn validate(self) -> SchedulerResult<Self> {
        validate_nonzero("max_waiting_requests", self.max_waiting_requests)?;
        validate_nonzero("max_waiting_prompt_tokens", self.max_waiting_prompt_tokens)?;
        validate_nonzero("max_active_sequences", self.max_active_sequences)?;
        validate_nonzero("max_sequence_tokens", self.max_sequence_tokens)?;
        if self.max_sequence_tokens > u32::MAX as usize {
            return Err(SchedulerError::InvalidConfiguration {
                field: "max_sequence_tokens",
                reason: "must fit the paged-KV V1 u32 logical length",
            });
        }
        validate_nonzero("iteration_token_budget", self.iteration_token_budget)?;
        validate_nonzero("max_prefill_chunk_tokens", self.max_prefill_chunk_tokens)?;
        validate_nonzero("max_promised_kv_blocks", self.max_promised_kv_blocks)?;
        validate_nonzero("metrics_window_samples", self.metrics_window_samples)?;
        if self.aging_threshold_ns == 0 {
            return Err(SchedulerError::InvalidConfiguration {
                field: "aging_threshold_ns",
                reason: "must be greater than zero",
            });
        }
        if self.max_prefill_chunk_tokens > self.iteration_token_budget {
            return Err(SchedulerError::InvalidConfiguration {
                field: "max_prefill_chunk_tokens",
                reason: "must not exceed iteration_token_budget",
            });
        }
        if self.max_prefill_chunk_tokens > self.max_sequence_tokens {
            return Err(SchedulerError::InvalidConfiguration {
                field: "max_prefill_chunk_tokens",
                reason: "must not exceed max_sequence_tokens",
            });
        }
        match (self.overload_policy, self.admission_timeout_ns) {
            (OverloadPolicy::RejectImmediately, Some(_)) => {
                return Err(SchedulerError::InvalidConfiguration {
                    field: "admission_timeout_ns",
                    reason: "must be absent when overload_policy rejects immediately",
                });
            }
            (OverloadPolicy::Wait, Some(0)) => {
                return Err(SchedulerError::InvalidConfiguration {
                    field: "admission_timeout_ns",
                    reason: "must be greater than zero when present",
                });
            }
            (OverloadPolicy::RejectImmediately, None) | (OverloadPolicy::Wait, None | Some(_)) => {}
        }
        Ok(self)
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_waiting_requests: 1_024,
            max_waiting_prompt_tokens: 1_048_576,
            max_active_sequences: 128,
            max_sequence_tokens: 32_768,
            iteration_token_budget: 2_048,
            max_prefill_chunk_tokens: 512,
            aging_threshold_ns: 100_000_000,
            overload_policy: OverloadPolicy::Wait,
            admission_timeout_ns: Some(30_000_000_000),
            max_promised_kv_blocks: 262_144,
            metrics_window_samples: 1_024,
        }
    }
}

fn validate_nonzero(field: &'static str, value: usize) -> SchedulerResult<()> {
    if value == 0 {
        Err(SchedulerError::InvalidConfiguration {
            field,
            reason: "must be greater than zero",
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OverloadPolicy, SchedulerConfig};

    #[test]
    fn default_configuration_is_valid() {
        SchedulerConfig::default().validate().unwrap();
    }

    #[test]
    fn timeout_requires_wait_policy() {
        let config = SchedulerConfig {
            overload_policy: OverloadPolicy::RejectImmediately,
            admission_timeout_ns: Some(1),
            ..SchedulerConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn chunk_must_fit_iteration_budget() {
        let config = SchedulerConfig {
            iteration_token_budget: 3,
            max_prefill_chunk_tokens: 4,
            ..SchedulerConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
