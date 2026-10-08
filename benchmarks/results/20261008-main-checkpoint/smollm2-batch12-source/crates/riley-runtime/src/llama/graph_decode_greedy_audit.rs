//! Cold standalone argmax audit. D2H is completed after graph destruction.
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error,
};
use crate::llama::executor::output::decode_greedy_tokens;
use crate::llama::{ExecutionSite, LlamaOp};
use riley_cuda::{BorrowedArgmaxGraph, BorrowedArgmaxResources, CudaStream};
fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 greedy audit",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    pub(crate) fn audit_c07_greedy(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<u32> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("greedy audit needs a healthy completed iteration"));
        }
        if self.output_count != 1 || self.owner.forward.plan.sequence_length() != 1 {
            return Err(rejected("greedy audit requires one M=1 output"));
        }
        let logits = self
            .owner
            .gathered_logits
            .as_mut()
            .ok_or_else(|| rejected("gathered logits missing"))?;
        let results = self
            .owner
            .greedy_results
            .as_mut()
            .ok_or_else(|| rejected("greedy result owner missing"))?;
        let vocabulary = self.owner.forward.plan.dimensions().vocabulary_size();
        let staging = &mut self.owner.forward.io_staging;
        let mut original = vec![
            0;
            usize::try_from(results.byte_len())
                .map_err(|_| rejected("result host range overflow"))?
        ];
        let mut input = vec![
            0;
            usize::try_from(logits.byte_len())
                .map_err(|_| rejected("logits host range overflow"))?
        ];
        let mut records = [0; 8];
        let result = (|| -> Result<(), riley_cuda::CudaError> {
            results.download_to_slice(0, &mut original, staging, stream)?;
            logits.download_to_slice(0, &mut input, staging, stream)?;
            let mut graph = BorrowedArgmaxGraph::prepare(
                BorrowedArgmaxResources {
                    stream,
                    input: logits,
                    output: results,
                },
                1,
                vocabulary as u64,
            )?;
            for _ in 0..16 {
                graph.replay()?;
            }
            graph.close()?;
            results.download_to_slice(0, &mut records, staging, stream)?;
            results.upload_from_slice(0, &original, staging, stream)?;
            Ok(())
        })();
        if let Err(source) = result {
            self.owner.poisoned = true;
            return Err(cuda_error(
                ExecutionSite::global(LlamaOp::OutputGather),
                source,
            ));
        }
        let mut restored = vec![0; original.len()];
        let mut after = vec![0; input.len()];
        let verify = (|| -> Result<(), riley_cuda::CudaError> {
            results.download_to_slice(0, &mut restored, staging, stream)?;
            logits.download_to_slice(0, &mut after, staging, stream)?;
            Ok(())
        })();
        if let Err(source) = verify {
            self.owner.poisoned = true;
            return Err(cuda_error(
                ExecutionSite::global(LlamaOp::OutputGather),
                source,
            ));
        }
        if restored != original || after != input {
            self.owner.poisoned = true;
            return Err(rejected("greedy input or result restoration mismatch"));
        }
        let mut token = [0];
        if let Err(error) = decode_greedy_tokens(&records, vocabulary, &mut token) {
            if !matches!(error, LlamaBatchExecutorError::GreedyLogitsNonFinite { .. }) {
                self.owner.poisoned = true;
            }
            return Err(error);
        }
        Ok(token[0])
    }
}
