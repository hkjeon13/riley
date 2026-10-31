//! Tensor-parallel followers of a batch executor.
//!
//! Rank 0's [`PreparedLlamaBatchExecutor`] is the leader the scheduler drives.
//! Every other rank owns an executor of its own slice of the weights on its own
//! device, stream and host thread (CUDA step capture is per thread). Before the
//! leader dispatches an iteration it hands the same rows to every follower, so
//! all ranks pack, select shapes and record or replay steps identically, and
//! their peer all-reduces meet. Only the leader's outputs are read.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use riley_cuda::CudaStream;

use super::batch::{LlamaBatchBlockTable, LlamaBatchRow, LlamaBatchRowKind};
use super::batch_executor::{BatchOutputMode, PreparedLlamaBatchExecutor};
use super::executor::error::{LlamaBatchExecutorError, LlamaBatchExecutorResult};

/// Owned copy of one [`LlamaBatchRow`], sent to the follower threads.
struct OwnedRow {
    sequence_tag: u64,
    kind: LlamaBatchRowKind,
    input_token_ids: Vec<u32>,
    target_logical_length: u32,
    schema_version: u16,
    physical_block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
    logical_length: u32,
    output_slot: Option<u32>,
}

impl OwnedRow {
    fn new(row: LlamaBatchRow<'_>) -> Self {
        let table = row.block_table();
        Self {
            sequence_tag: row.sequence_tag(),
            kind: row.kind(),
            input_token_ids: row.input_token_ids().to_vec(),
            target_logical_length: row.target_logical_length(),
            schema_version: table.schema_version(),
            physical_block_ids: table.physical_block_ids().to_vec(),
            valid_tokens: table.valid_tokens().to_vec(),
            logical_length: table.logical_length(),
            output_slot: row.output_slot(),
        }
    }

    fn row(&self) -> LlamaBatchRow<'_> {
        LlamaBatchRow::new(
            self.sequence_tag,
            self.kind,
            &self.input_token_ids,
            self.target_logical_length,
            LlamaBatchBlockTable::new(
                self.schema_version,
                &self.physical_block_ids,
                &self.valid_tokens,
                self.logical_length,
            ),
            self.output_slot,
        )
    }
}

enum Command {
    Execute(Arc<Vec<OwnedRow>>, BatchOutputMode),
}

struct Follower {
    commands: Option<SyncSender<Command>>,
    replies: Receiver<Result<(), String>>,
    thread: Option<JoinHandle<LlamaBatchExecutorResult<()>>>,
}

/// The follower ranks 1.. of a tensor-parallel group.
pub struct TensorParallelFollowers {
    followers: Vec<Follower>,
    pending: usize,
}

impl std::fmt::Debug for TensorParallelFollowers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorParallelFollowers")
            .field("followers", &self.followers.len())
            .finish_non_exhaustive()
    }
}

impl TensorParallelFollowers {
    /// Starts one host thread per follower rank, each owning its prepared
    /// executor (with the group attached) and stream until the leader closes.
    #[must_use]
    pub fn spawn(ranks: Vec<(PreparedLlamaBatchExecutor, CudaStream)>) -> Self {
        let followers = ranks
            .into_iter()
            .enumerate()
            .map(|(index, (mut executor, mut stream))| {
                let (commands, inbox) = sync_channel::<Command>(1);
                let (outbox, replies) = sync_channel(1);
                let thread = std::thread::Builder::new()
                    .name(format!("riley-tp-rank{}", index + 1))
                    .spawn(move || {
                        while let Ok(Command::Execute(rows, mode)) = inbox.recv() {
                            let borrowed: Vec<LlamaBatchRow<'_>> =
                                rows.iter().map(OwnedRow::row).collect();
                            let result = executor
                                .execute_output(&borrowed, mode, &mut stream)
                                .map_err(|error| error.to_string());
                            if outbox.send(result).is_err() {
                                break;
                            }
                        }
                        stream.synchronize().ok();
                        executor.close()
                    })
                    .ok();
                Follower {
                    commands: Some(commands),
                    replies,
                    thread,
                }
            })
            .collect();
        Self {
            followers,
            pending: 0,
        }
    }

    /// Hands one iteration to every follower; [`Self::wait`] collects them.
    pub(super) fn dispatch(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        mode: BatchOutputMode,
    ) -> LlamaBatchExecutorResult<()> {
        let rows = Arc::new(rows.iter().copied().map(OwnedRow::new).collect::<Vec<_>>());
        for follower in &self.followers {
            let sent = follower
                .commands
                .as_ref()
                .is_some_and(|commands| commands.send(Command::Execute(Arc::clone(&rows), mode)).is_ok());
            if !sent {
                return Err(follower_failed("a follower thread has stopped"));
            }
            self.pending += 1;
        }
        Ok(())
    }

    /// Waits until every follower has enqueued the dispatched iteration.
    pub(super) fn wait(&mut self) -> LlamaBatchExecutorResult<()> {
        let mut first_error = None;
        for follower in &self.followers {
            if self.pending == 0 {
                break;
            }
            self.pending -= 1;
            match follower.replies.recv() {
                Ok(Ok(())) => {}
                Ok(Err(message)) => {
                    first_error.get_or_insert(follower_failed(message));
                }
                Err(_) => {
                    first_error.get_or_insert(follower_failed("a follower thread has stopped"));
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Stops every follower and closes its executor.
    pub(super) fn close(mut self) -> LlamaBatchExecutorResult<()> {
        let mut first_error = self.wait().err();
        for follower in &mut self.followers {
            follower.commands.take();
        }
        for follower in &mut self.followers {
            if let Some(thread) = follower.thread.take() {
                match thread.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        first_error.get_or_insert(error);
                    }
                    Err(_) => {
                        first_error.get_or_insert(follower_failed("a follower thread panicked"));
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl Drop for TensorParallelFollowers {
    fn drop(&mut self) {
        for follower in &mut self.followers {
            follower.commands.take();
        }
        for follower in &mut self.followers {
            if let Some(thread) = follower.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

fn follower_failed(reason: impl Into<String>) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::TensorParallelFollower {
        reason: reason.into(),
    }
}
