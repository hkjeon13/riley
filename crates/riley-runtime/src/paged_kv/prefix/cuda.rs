//! Local CUDA adapter. Native event completion, not a caller boolean, controls
//! publication and reclamation. Network/peer transport is outside this adapter.
use super::*;
use riley_cuda::{CudaDeviceBuffer, CudaError, CudaPageCopySpec, CudaPendingKvCopy, CudaStream};

#[derive(Debug)]
pub enum CowTransferError {
    Cuda(CudaError),
    Host(PagedKvError),
}

impl fmt::Display for CowTransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => error.fmt(f),
            Self::Host(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for CowTransferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Cuda(error) => error,
            Self::Host(error) => error,
        })
    }
}

/// Owns the host COW transaction and the actual device event's buffer/stream
/// borrows. Target sequence and pool remain available for unrelated scheduling.
/// Explicit completion is required: dropping without it conservatively retains
/// host pages, even though the CUDA token attempts to drain in Drop.
#[must_use = "poll or synchronize to publish/discard the host COW transaction"]
pub struct CudaPendingCow<'a> {
    copy: CudaPendingKvCopy<'a>,
    ticket: CopyOnWrite,
    cancelled: bool,
}

impl CopyOnWrite {
    /// Submits this transaction to idle local K/V buffers. The model owner must
    /// supply the actual buffers of this ticket's pool. Captured graph buffers
    /// require a separate owner-authorized integration, not a raw-pointer bypass.
    ///
    /// # Errors
    /// Returns the intact host ticket on pre-submission failure for cancellation
    /// or retry. Native partial submission errors stay in the pending token.
    pub fn submit_cuda<'a>(
        self,
        keys: &'a mut CudaDeviceBuffer,
        values: &'a mut CudaDeviceBuffer,
        stream: &'a mut CudaStream,
    ) -> Result<CudaPendingCow<'a>, (Self, CowTransferError)> {
        let Some(state) = self.state.as_ref() else {
            return Err((
                self,
                CowTransferError::Host(invalid("copy_ticket", "already completed")),
            ));
        };
        let (source, destination, valid) = self.copy_range().expect("live COW range");
        let layout = state.layout;
        let spec = CudaPageCopySpec {
            layers: layout.layer_count() as u64,
            heads: layout.key_value_head_count() as u64,
            layer_stride: layout.layer_stride_bytes(),
            block_stride: layout.block_stride_bytes(),
            head_stride: layout.head_stride_bytes(),
            valid_bytes: u64::from(valid) * layout.head_dimension() as u64 * BF16_BYTES,
            source_page: source.physical_index(),
            destination_page: destination.physical_index(),
        };
        match keys.copy_kv_page_async(values, spec, stream) {
            Ok(copy) => Ok(CudaPendingCow {
                copy,
                ticket: self,
                cancelled: false,
            }),
            Err(error) => Err((self, CowTransferError::Cuda(error))),
        }
    }
}

impl CudaPendingCow<'_> {
    /// Requests discard; buffers/pages remain held until the event completes.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Queries the real event and publishes/discards only after quiescence.
    ///
    /// # Errors
    /// Native failures remain sticky. Ambiguous completion retains all holds.
    pub fn poll(
        &mut self,
        sequence: &mut SequenceState,
        pool: &mut KvBlockPool,
    ) -> Result<Option<CopyOutcome>, CowTransferError> {
        let observation = self.copy.query();
        self.resolve(Some(sequence), pool, observation)
    }

    /// Waits for device completion and resolves the current host transaction.
    ///
    /// # Errors
    /// Returns native or ownership failure without publishing partial data.
    pub fn synchronize(
        &mut self,
        sequence: &mut SequenceState,
        pool: &mut KvBlockPool,
    ) -> Result<Option<CopyOutcome>, CowTransferError> {
        let observation = self.copy.synchronize().map(|()| true);
        self.resolve(Some(sequence), pool, observation)
    }

    /// Drains an orphan/cancelled transaction whose target sequence was consumed.
    ///
    /// # Errors
    /// Unconfirmed device work retains page holds; completed errors discard pages.
    pub fn synchronize_orphan(
        &mut self,
        pool: &mut KvBlockPool,
    ) -> Result<Option<CopyOutcome>, CowTransferError> {
        self.cancelled = true;
        let observation = self.copy.synchronize().map(|()| true);
        self.resolve(None, pool, observation)
    }

    fn resolve(
        &mut self,
        sequence: Option<&mut SequenceState>,
        pool: &mut KvBlockPool,
        observation: Result<bool, CudaError>,
    ) -> Result<Option<CopyOutcome>, CowTransferError> {
        if !self.copy.is_quiescent() {
            return observation.map(|_| None).map_err(CowTransferError::Cuda);
        }
        let succeeded = observation.is_ok() && !self.cancelled;
        let outcome = match sequence {
            Some(sequence) => sequence.complete_copy_on_write(pool, &mut self.ticket, succeeded),
            None => pool.discard_completed_copy(&mut self.ticket),
        }
        .map_err(CowTransferError::Host)?;
        observation.map_err(CowTransferError::Cuda)?;
        Ok(Some(outcome))
    }
}
