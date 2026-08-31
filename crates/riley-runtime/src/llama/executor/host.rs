//! Checked cold host-byte allocation for the Llama batch executor.
//!
//! This component owns no CUDA resource or executor state. Callers retain the
//! semantic preflight that determines their element count and consume the
//! returned zeroed bytes in their existing ownership boundary.

use super::error::{LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult};

/// Allocates exact zero-filled host bytes with the stable workspace errors.
pub(in crate::llama) fn allocate_zeroed_host_bytes(
    elements: usize,
    element_bytes: usize,
) -> LlamaBatchExecutorResult<Box<[u8]>> {
    let requested =
        elements
            .checked_mul(element_bytes)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(requested)
        .map_err(|_| LlamaBatchExecutorError::HostAllocation {
            resource: LlamaBatchExecutorResource::HostWorkspace,
            requested_bytes: requested as u64,
        })?;
    bytes.resize(requested, 0);
    Ok(bytes.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_host_bytes_are_exact_and_fail_closed_on_overflow() {
        let bytes = allocate_zeroed_host_bytes(3, 2).expect("representable host bytes");
        assert_eq!(&*bytes, &[0; 6]);
        assert!(matches!(
            allocate_zeroed_host_bytes(usize::MAX, 2),
            Err(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })
        ));
    }
}
