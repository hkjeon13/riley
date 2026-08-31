//! Typed failure-to-poison routing for the Llama batch execution boundary.
//!
//! The batch owner retains CUDA resource lifetime and the outer iteration
//! decision for a command submission whose mutation result is unknown. This
//! component only maps typed errors onto borrowed poison flags.

use super::super::forward::{poison_for_cuda_error, poison_for_forward_error};
use super::error::LlamaBatchExecutorError;

/// Tracks whether a command submission could have mutated iteration state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::llama) enum BatchDispatchDisposition {
    /// Validation or setup failed before a command batch began.
    #[default]
    PreDispatch,
    /// A command batch was opened, so partial device-side mutation is possible.
    CommandSubmissionStarted,
}

impl BatchDispatchDisposition {
    /// Returns whether the enclosing iteration may have been partially mutated.
    #[must_use]
    pub(in crate::llama) const fn mutation_may_have_occurred(self) -> bool {
        matches!(self, Self::CommandSubmissionStarted)
    }
}

/// Applies established typed error poison rules to borrowed executor flags.
///
/// The `forward_gemms_poisoned` callback reads state owned by the caller only
/// after nested forward-error routing has run, preserving the existing
/// short-circuit ordering without retaining the forward owner.
pub(in crate::llama) fn poison_for_batch_error<F>(
    poisoned: &mut bool,
    forward_poisoned: &mut bool,
    error: &LlamaBatchExecutorError,
    forward_gemms_poisoned: F,
) where
    F: FnOnce() -> bool,
{
    match error {
        LlamaBatchExecutorError::Cuda { source, .. } => {
            poison_for_cuda_error(poisoned, source);
            poison_for_cuda_error(forward_poisoned, source);
        }
        LlamaBatchExecutorError::Forward(source) => {
            poison_for_forward_error(forward_poisoned, source);
            *poisoned |= *forward_poisoned || forward_gemms_poisoned();
        }
        LlamaBatchExecutorError::InvalidConfiguration { .. }
        | LlamaBatchExecutorError::ArithmeticOverflow { .. } => {
            *poisoned = true;
            *forward_poisoned = true;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::llama::forward::LlamaForwardError;

    #[test]
    fn dispatch_disposition_distinguishes_preflight_from_unknown_mutation() {
        let mut disposition = BatchDispatchDisposition::PreDispatch;
        assert!(!disposition.mutation_may_have_occurred());

        disposition = BatchDispatchDisposition::CommandSubmissionStarted;
        assert!(disposition.mutation_may_have_occurred());
    }

    #[test]
    fn forward_error_folds_owner_gemm_state_only_when_forward_is_not_poisoned() {
        let mut poisoned = false;
        let mut forward_poisoned = false;
        let gemm_state_read = Cell::new(false);
        let error = LlamaBatchExecutorError::Forward(LlamaForwardError::AttentionBudgetExceeded {
            required_bytes: 9,
            maximum_bytes: 8,
        });

        poison_for_batch_error(&mut poisoned, &mut forward_poisoned, &error, || {
            gemm_state_read.set(true);
            true
        });

        assert!(gemm_state_read.get());
        assert!(poisoned);
        assert!(!forward_poisoned);
    }

    #[test]
    fn configuration_and_overflow_fail_close_without_reading_owner_gemm_state() {
        for error in [
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "test",
                reason: "test-only invalid configuration",
            },
            LlamaBatchExecutorError::ArithmeticOverflow {
                resource: super::super::error::LlamaBatchExecutorResource::HostWorkspace,
            },
        ] {
            let mut poisoned = false;
            let mut forward_poisoned = false;
            let gemm_state_read = Cell::new(false);

            poison_for_batch_error(&mut poisoned, &mut forward_poisoned, &error, || {
                gemm_state_read.set(true);
                true
            });

            assert!(poisoned);
            assert!(forward_poisoned);
            assert!(!gemm_state_read.get());
        }
    }

    #[test]
    fn forward_error_categories_preserve_flag_and_gemm_folding_rules() {
        let mut poisoned = false;
        let mut forward_poisoned = false;
        let gemm_state_read = Cell::new(false);
        let invalid_configuration =
            LlamaBatchExecutorError::Forward(LlamaForwardError::InvalidConfiguration {
                field: "test",
                reason: "test-only invalid configuration",
            });

        poison_for_batch_error(
            &mut poisoned,
            &mut forward_poisoned,
            &invalid_configuration,
            || {
                gemm_state_read.set(true);
                true
            },
        );

        assert!(poisoned);
        assert!(forward_poisoned);
        assert!(!gemm_state_read.get());

        poisoned = false;
        forward_poisoned = false;
        let tokens_not_uploaded =
            LlamaBatchExecutorError::Forward(LlamaForwardError::TokensNotUploaded);
        poison_for_batch_error(
            &mut poisoned,
            &mut forward_poisoned,
            &tokens_not_uploaded,
            || {
                gemm_state_read.set(true);
                false
            },
        );

        assert!(gemm_state_read.get());
        assert!(!poisoned);
        assert!(!forward_poisoned);
    }
}
