//! Exact active whole-slab H2D audit, independent of retained aggregate admission.
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::{
    buffers::{BatchDeviceInput, BatchHostInput},
    error::{LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error},
    metadata::{PackedIterationLayout, pack_iteration_input},
};
use crate::llama::{ExecutionSite, LlamaBatchRow, LlamaOp};
use riley_cuda::{BorrowedH2DGraph, BorrowedH2DResources, CudaStream};
fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 H2D audit",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    pub(crate) fn audit_c07_h2d(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("H2D audit needs healthy completed iteration"));
        }
        if self.owner.forward.plan.sequence_length() != 1 {
            return Err(rejected("H2D audit requires M=1"));
        }
        let packed = self.owner.metadata.pack(rows)?;
        let layout = PackedIterationLayout::for_batch(&packed, 1)?;
        let (BatchDeviceInput::IterationBatch { slab }, BatchHostInput::IterationBatch(host)) =
            (&mut self.owner.device_input, &mut self.owner.host.input)
        else {
            return Err(rejected("packed owners required"));
        };
        if slab.byte_len() != host.pinned.byte_len()
            || usize::try_from(slab.byte_len()).ok() != Some(layout.total_bytes)
        {
            return Err(rejected(
                "active payload must exactly cover both actual slabs",
            ));
        }
        let mut expected = vec![0; layout.total_bytes];
        pack_iteration_input(&packed, 1, layout, &mut expected)?;
        let mut source = vec![0; expected.len()];
        let mut before = vec![0; expected.len()];
        let mut after = vec![0; expected.len()];
        let staging = &mut self.owner.forward.io_staging;
        let result = (|| -> Result<(), riley_cuda::CudaError> {
            host.pinned.read(0, &mut source)?;
            slab.download_to_slice(0, &mut before, staging, stream)?;
            Ok(())
        })();
        if let Err(source) = result {
            self.owner.poisoned = true;
            return Err(cuda_error(
                ExecutionSite::global(LlamaOp::OutputGather),
                source,
            ));
        }
        if source != expected || host.bytes[..layout.total_bytes] != expected || before != expected
        {
            self.owner.poisoned = true;
            return Err(rejected(
                "host/pinned/device slab differs from current packed request",
            ));
        }
        let result = (|| -> Result<(), riley_cuda::CudaError> {
            // A changed destination proves replay really performed the copy.
            slab.upload_from_slice(0, &vec![0xa5; expected.len()], staging, stream)?;
            let mut graph = BorrowedH2DGraph::prepare(BorrowedH2DResources {
                stream,
                input: &mut host.pinned,
                output: slab,
            })?;
            for _ in 0..16 {
                graph.replay(&expected)?;
            }
            graph.close()?;
            slab.download_to_slice(0, &mut after, staging, stream)?;
            host.pinned.read(0, &mut source)?;
            Ok(())
        })();
        if let Err(source) = result {
            self.owner.poisoned = true;
            return Err(cuda_error(
                ExecutionSite::global(LlamaOp::OutputGather),
                source,
            ));
        }
        if after != expected || source != expected || host.bytes[..layout.total_bytes] != expected {
            self.owner.poisoned = true;
            return Err(rejected("H2D replay or source preservation mismatch"));
        }
        Ok(())
    }
}
