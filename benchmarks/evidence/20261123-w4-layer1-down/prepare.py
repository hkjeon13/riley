"""Add gated input capture and a layer-1 down-row intervention to the KV trace."""
from hashlib import sha256
from pathlib import Path
import sys


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected a single anchor: {old[:150]}")
    return source.replace(old, new, 1)


def prepare(root):
    path = root / "src/crates/riley-runtime/src/llama/executor/dispatch.rs"
    if sha256(path.read_bytes()).hexdigest() != "6aed150c2e2e449ea01989fc014429b9cabc08c467b0eb11cd96c34b0a07d88d":
        raise ValueError("wrong parent dispatch")
    source = path.read_text()
    source = replace_once(source, "    kv_donor: Option<u64>,",
                          "    kv_donor: Option<u64>,\n    layer1_down_donor: Option<u64>,")
    source = replace_once(source, "            kv_donor: None,",
                          "            kv_donor: None,\n            layer1_down_donor: None,")
    source = replace_once(source, """        Ok(())
    }

    fn trace_limit(""", """        if self.position == 7842 {
            match std::fs::read(self.directory.join("layer1-down.bf16")) {
                Ok(bytes) => {
                    if self.kv_donor.is_none() || bytes.len() as u64 != BOUNDARY_ROW_BYTES {
                        return Err(long_trace_error());
                    }
                    let offset = self.trace_limit(staging)
                        .checked_sub(BOUNDARY_ROW_BYTES).ok_or_else(long_trace_error)?;
                    staging.write(offset, &bytes).map_err(kv_trace_error)?;
                    self.layer1_down_donor = Some(offset);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    eprintln!("W4 layer-1 down donor read failed: {error}");
                    return Err(long_trace_error());
                }
            }
        }
        Ok(())
    }

    fn trace_limit(""")
    source = replace_once(source, """        self.kv_donor.or_else(|| self.replacement.map(|(_, start)| start))
            .unwrap_or_else(|| staging.byte_len())""",
                          """        self.layer1_down_donor
            .or(self.kv_donor)
            .or_else(|| self.replacement.map(|(_, start)| start))
            .unwrap_or_else(|| staging.byte_len())""")
    source = replace_once(source, """                if !mlp_done {
                    let gate_site = ExecutionSite::layer(layer_index, LlamaOp::GateProjection);""",
                          """                let trace_down = layer_index == 1 && long_trace.as_ref()
                    .is_some_and(|(trace, _)| trace.position == 7842);
                if trace_down {
                    if mlp_done || piecewise_rows != 0 || !norm_after_mlp
                        || dense_rows != 1 || intermediate_elements != 18_944
                    {
                        return Err(long_trace_error());
                    }
                    trace_row(&mut long_trace, "mlp_input", Some(layer_index),
                        &buffers.hidden_norm, hidden, false, stream)?;
                }
                if !mlp_done {
                    let gate_site = ExecutionSite::layer(layer_index, LlamaOp::GateProjection);""")
    source = replace_once(source, """                    let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
                    execute_gemm_with_row_groups_from(""", """                    if trace_down {
                        trace_row(&mut long_trace, "mlp_gated", Some(layer_index),
                            &buffers.gated_product, intermediate_elements, false, stream)?;
                    }
                    let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
                    execute_gemm_with_row_groups_from(""")
    source = replace_once(source, """                let mlp_residual_site = ExecutionSite::layer(layer_index, LlamaOp::MlpResidual);""",
                          """                if trace_down {
                    trace_row(&mut long_trace, "down_native", Some(layer_index),
                        hidden_current, hidden, false, stream)?;
                    if let Some((trace, staging)) = long_trace.as_ref() {
                        if let Some(offset) = trace.layer1_down_donor {
                            hidden_current.copy_from_pinned_in_active_command_batch(
                                0, staging, offset, BOUNDARY_ROW_BYTES, stream
                            ).map_err(kv_trace_error)?;
                        }
                    }
                    trace_row(&mut long_trace, "down_effective", Some(layer_index),
                        hidden_current, hidden, false, stream)?;
                }
                let mlp_residual_site = ExecutionSite::layer(layer_index, LlamaOp::MlpResidual);""")
    source = replace_once(source, """                    "layer_residual",
                    Some(layer_index),
                    hidden_projection,
                    hidden,
                    false,
                    stream,
                )?;
                Ok(())""", """                    "layer_residual",
                    Some(layer_index),
                    hidden_projection,
                    hidden,
                    false,
                    stream,
                )?;
                if trace_down {
                    trace_row(&mut long_trace, "layer2_input", Some(layer_index),
                        &buffers.hidden_norm, hidden, false, stream)?;
                }
                Ok(())""")
    path.write_text(source)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
