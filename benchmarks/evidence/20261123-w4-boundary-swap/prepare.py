"""Instrument only the first differing decode boundary, in an isolated build."""
from pathlib import Path
import sys


def replace(source, anchor, new):
    if source.count(anchor) != 1:
        raise ValueError(f"unexpected source anchor: {anchor.splitlines()[0]}")
    return source.replace(anchor, new)


file = Path(sys.argv[1]) / "src/crates/riley-runtime/src/llama/executor/dispatch.rs"
source = file.read_text()
source = replace(source, "struct LongTrace {", """const BOUNDARY_ROW_BYTES: u64 = 3584 * 2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    Attention,
    Down,
    Final,
}

struct LongTrace {
    replacement: Option<(Boundary, u64)>,""")
source = replace(source, "            entries: Vec::with_capacity(59),", """            entries: Vec::with_capacity(63),
            replacement: None,""")
anchor = "    fn save(self, staging: &mut CudaPinnedHostBuffer) -> LlamaBatchExecutorResult<()> {"
source = replace(source, anchor, """    fn stage_replacement(
        &mut self,
        staging: &mut CudaPinnedHostBuffer,
    ) -> LlamaBatchExecutorResult<()> {
        for (boundary, name, rows) in [
            (Boundary::Attention, "attention.bf16", 2),
            (Boundary::Down, "down.bf16", 2),
            (Boundary::Final, "final.bf16", 1),
        ] {
            let bytes = match std::fs::read(self.directory.join(name)) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    eprintln!("W4 boundary swap donor read failed: {error}");
                    return Err(long_trace_error());
                }
            };
            let extent = rows * BOUNDARY_ROW_BYTES;
            if self.replacement.is_some() || self.position != 7842 || self.dense_rows != 1
                || self.row != 0 || bytes.len() as u64 != extent
            {
                return Err(long_trace_error());
            }
            let offset = staging.byte_len().checked_sub(extent).ok_or_else(long_trace_error)?;
            staging.write(offset, &bytes)
                .map_err(|source| batch_cuda(ExecutionSite::global(LlamaOp::BatchMetadataUpload), source))?;
            self.replacement = Some((boundary, offset));
        }
        Ok(())
    }

""" + anchor)
source = replace(source, "        if offset + bytes > staging.byte_len() {",
                 "        if offset + bytes > trace.replacement.map_or(staging.byte_len(), |(_, start)| start) {")
anchor = "fn trace_row<S:"
source = replace(source, anchor, """#[allow(clippy::too_many_arguments)]
fn replace_boundary_row<S: CudaExecutionStream + ?Sized>(
    trace: &mut Option<(&mut LongTrace, &CudaPinnedHostBuffer)>,
    boundary: Boundary,
    label: &'static str,
    layer: Option<usize>,
    buffer: &mut CudaDeviceBuffer,
    normalized: bool,
    hidden: u64,
    stream: &mut S,
) -> LlamaBatchExecutorResult<()> {
    if let Some((state, staging)) = trace.as_ref() {
        if let Some((selected, offset)) = state.replacement {
            if selected == boundary {
                if hidden * 2 != BOUNDARY_ROW_BYTES || state.row != 0 {
                    return Err(long_trace_error());
                }
                let staging = *staging;
                trace_row(trace, label, layer, buffer, hidden, false, stream)?;
                let source_offset = offset + u64::from(normalized) * BOUNDARY_ROW_BYTES;
                buffer.copy_from_pinned_in_active_command_batch(
                    0, staging, source_offset, BOUNDARY_ROW_BYTES, stream,
                )
                .map_err(|source| batch_cuda(ExecutionSite::global(LlamaOp::OutputGather), source))?;
            }
        }
    }
    Ok(())
}

""" + anchor)
anchor = "    let mut execute_iteration_body = |batch: PackedBatchV1<'_>,"
source = replace(source, anchor, """    if let Some(trace) = &mut long_trace {
        trace.stage_replacement(&mut forward.io_staging)?;
    }
""" + anchor)

for residual, norm, variant, buffer, extra in [
    ("attention_residual", "post_attention_norm", "Attention", "&mut buffers.hidden_rotary", ""),
    ("layer_residual", "next_input_norm", "Down", "hidden_projection", " && norm_after_mlp"),
]:
    anchor = f'                trace_row(\n                    &mut long_trace,\n                    "{residual}",'
    source = replace(source, anchor, f"""                if layer_index == 0{extra} {{
                    replace_boundary_row(
                        &mut long_trace, Boundary::{variant}, "{residual}_before", Some(layer_index),
                        {buffer}, false, hidden, stream,
                    )?;
                    replace_boundary_row(
                        &mut long_trace, Boundary::{variant}, "{norm}_before", Some(layer_index),
                        &mut buffers.hidden_norm, true, hidden, stream,
                    )?;
                    trace_row(
                        &mut long_trace, "{norm}", Some(layer_index), &buffers.hidden_norm,
                        hidden, false, stream,
                    )?;
                }}
""" + anchor)

anchor = '        trace_row(\n            &mut long_trace,\n            "final_norm",'
source = replace(source, anchor, """        replace_boundary_row(
            &mut long_trace, Boundary::Final, "final_norm_before", None,
            &mut buffers.hidden_norm, false, hidden, stream,
        )?;
""" + anchor)
file.write_text(source)
