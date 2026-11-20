"""Replace only the matching layer-zero output at an observed C1 decode step."""
from pathlib import Path
import sys


def replace(source, anchor, new):
    if source.count(anchor) != 1:
        raise ValueError(f"unexpected source anchor: {anchor.splitlines()[0]}")
    return source.replace(anchor, new)


stage = Path(sys.argv[1])
file = stage / "src/crates/riley-runtime/src/llama/executor/dispatch.rs"
source = file.read_text()
source = replace(source, "struct LongTrace {", """const PROJECTION_REPLAY_BYTES: u64 = 3584 * 2;

struct LongTrace {""")
source = replace(
    source, "    entries: Vec<(&'static str, Option<usize>, u64, u64)>,",
    """    entries: Vec<(&'static str, Option<usize>, u64, u64)>,
    projection_replay: Option<(&'static str, u64)>,""",
)
source = replace(source, "            entries: Vec::with_capacity(10),", """            entries: Vec::with_capacity(10),
            projection_replay: None,""")
anchor = "    fn save(self, staging: &mut CudaPinnedHostBuffer) -> LlamaBatchExecutorResult<()> {"
source = replace(source, anchor, """    fn stage_projection_replay(
        &mut self,
        staging: &mut CudaPinnedHostBuffer,
    ) -> LlamaBatchExecutorResult<()> {
        for (operator, file) in [
            ("o_proj", "o-proj-override.bf16"),
            ("down_proj", "down-override.bf16"),
        ] {
            let bytes = match std::fs::read(self.directory.join(file)) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    eprintln!("W4 projection replay failed to read a row: {error}");
                    return Err(long_trace_error());
                }
            };
            if self.projection_replay.is_some()
                || self.dense_rows != 1
                || bytes.len() as u64 != PROJECTION_REPLAY_BYTES
            {
                return Err(long_trace_error());
            }
            let offset = staging.byte_len().checked_sub(PROJECTION_REPLAY_BYTES)
                .ok_or_else(long_trace_error)?;
            if offset % 2 != 0 {
                return Err(long_trace_error());
            }
            staging.write(offset, &bytes)
                .map_err(|source| batch_cuda(
                    ExecutionSite::global(LlamaOp::BatchMetadataUpload), source,
                ))?;
            self.projection_replay = Some((operator, offset));
        }
        Ok(())
    }

""" + anchor)
anchor = "fn trace_projection_plan("
source = replace(source, anchor, """fn replay_projection<S: CudaExecutionStream + ?Sized>(
    trace: &Option<(&mut LongTrace, &CudaPinnedHostBuffer)>,
    operator: &str,
    layer: usize,
    output: &mut CudaDeviceBuffer,
    hidden: u64,
    stream: &mut S,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    if layer != 0 {
        return Ok(());
    }
    if let Some((trace, staging)) = trace.as_ref() {
        if let Some((selected, offset)) = trace.projection_replay {
            if selected == operator {
                if hidden * 2 != PROJECTION_REPLAY_BYTES || trace.row != 0 {
                    return Err(long_trace_error());
                }
                output.copy_from_pinned_in_active_command_batch(
                    0, staging, offset, PROJECTION_REPLAY_BYTES, stream,
                ).map_err(|source| batch_cuda(site, source))?;
            }
        }
    }
    Ok(())
}

""" + anchor)
source = replace(source, "        if offset + bytes > staging.byte_len() {", """        if offset + bytes > trace.projection_replay.map_or(staging.byte_len(), |(_, offset)| offset) {""")
anchor = "    let mut execute_iteration_body = |batch: PackedBatchV1<'_>,"
source = replace(source, anchor, """    if let Some(trace) = &mut long_trace {
        trace.stage_projection_replay(&mut forward.io_staging)?;
    }
""" + anchor)
for point, statement in (
    ("attention_projection", '''                replay_projection(
                    &long_trace, "o_proj", layer_index, hidden_projection, hidden, stream,
                    ExecutionSite::layer(layer_index, LlamaOp::OutputProjection),
                )?;
'''),
    ("mlp_down", '''                replay_projection(
                    &long_trace, "down_proj", layer_index, hidden_current, hidden, stream,
                    ExecutionSite::layer(layer_index, LlamaOp::DownProjection),
                )?;
'''),
):
    anchor = f'                trace_row(\n                    &mut long_trace,\n                    "{point}",'
    source = replace(source, anchor, statement + anchor)
file.write_text(source)
