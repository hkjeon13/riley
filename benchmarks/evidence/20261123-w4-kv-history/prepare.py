"""Attach layer-1 history capture and replacement to the pinned W4 build."""
from hashlib import sha256
from pathlib import Path
import sys


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected a single anchor: {old[:130]}")
    return source.replace(old, new, 1)


def prepare(root):
    llama = root / "src/crates/riley-runtime/src/llama"
    dispatch = llama / "executor/dispatch.rs"
    forward = llama / "forward.rs"
    expected = {
        dispatch: "c79d1895452f96c3a8e3688cb2c14c1332c8f82c275a77419d24977226568172",
        forward: "0c3293d3e387c0b2a55bfa976514249184f24b74a107c70c3b4d4b59b4a046f1",
    }
    for path, checksum in expected.items():
        if sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"wrong parent source: {path}")
    source = dispatch.read_text()
    source = replace_once(source, "    replacement: Option<(Boundary, u64)>,",
                          "    replacement: Option<(Boundary, u64)>,\n    kv_donor: Option<u64>,")
    source = replace_once(source, "            replacement: None,",
                          "            replacement: None,\n            kv_donor: None,")
    source = replace_once(source, """        Ok(())
    }

    fn save(self,""", """        if self.position == 7842 {
            match std::fs::read(self.directory.join("past-kv.bf16")) {
                Ok(bytes) => {
                    if bytes.len() as u64 != 2 * KV_HISTORY_BYTES {
                        return Err(long_trace_error());
                    }
                    let offset = self.trace_limit(staging)
                        .checked_sub(2 * KV_HISTORY_BYTES).ok_or_else(long_trace_error)?;
                    staging.write(offset, &bytes).map_err(kv_trace_error)?;
                    self.kv_donor = Some(offset);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    eprintln!("W4 KV donor read failed: {error}");
                    return Err(long_trace_error());
                }
            }
        }
        Ok(())
    }

    fn trace_limit(&self, staging: &CudaPinnedHostBuffer) -> u64 {
        self.kv_donor.or_else(|| self.replacement.map(|(_, start)| start))
            .unwrap_or_else(|| staging.byte_len())
    }

    fn save(self,""")
    source = replace_once(source,
                          "trace.replacement.map_or(staging.byte_len(), |(_, start)| start)",
                          "trace.trace_limit(staging)")
    source = replace_once(source, "#[allow(\n    clippy::too_many_arguments,\n    clippy::too_many_lines,\n    clippy::cast_precision_loss,\n    clippy::large_types_passed_by_value\n)]\npub(in crate::llama) fn execute_packed(",
                          (root / "kv-trace.rs").read_text() + "\n#[allow(\n    clippy::too_many_arguments,\n    clippy::too_many_lines,\n    clippy::cast_precision_loss,\n    clippy::large_types_passed_by_value\n)]\npub(in crate::llama) fn execute_packed(")
    source = replace_once(source,
                          "            let attention_site = ExecutionSite::layer(layer_index, LlamaOp::RaggedPagedAttention);",
                          """            let mut history_restore = None;
            if layer_index == 1 && long_trace.as_ref().is_some_and(|(trace, _)| trace.position == 7842) {
                trace_row(&mut long_trace, "query", Some(1), &buffers.hidden_rotary,
                    query_heads * head_size, false, stream)?;
                let (trace, staging) = long_trace.as_mut().ok_or_else(long_trace_error)?;
                let history = TraceHistory::new(layout, layer_index, batch.host())?;
                let original_key = history.capture(trace, staging, key_cache, "past_key", false, stream)?;
                let original_value = history.capture(trace, staging, value_cache, "past_value", false, stream)?;
                history.capture(trace, staging, key_cache, "current_key", true, stream)?;
                history.capture(trace, staging, value_cache, "current_value", true, stream)?;
                if let Some(donor) = trace.kv_donor {
                    history.upload(staging, key_cache, donor, stream)?;
                    history.upload(staging, value_cache, donor + KV_HISTORY_BYTES, stream)?;
                    history.capture(trace, staging, key_cache, "injected_key", false, stream)?;
                    history.capture(trace, staging, value_cache, "injected_value", false, stream)?;
                    history_restore = Some((history, original_key, original_value));
                }
            }
            let attention_site = ExecutionSite::layer(layer_index, LlamaOp::RaggedPagedAttention);""")
    source = replace_once(source, "            let post_segment = owned_segments",
                          """            if layer_index == 1 && long_trace.as_ref().is_some_and(|(trace, _)| trace.position == 7842) {
                trace_row(&mut long_trace, "context", Some(1), &buffers.hidden_context,
                    query_heads * head_size, false, stream)?;
                if let Some((history, original_key, original_value)) = history_restore {
                    let (_, staging) = long_trace.as_ref().ok_or_else(long_trace_error)?;
                    history.upload(staging, key_cache, original_key, stream)?;
                    history.upload(staging, value_cache, original_value, stream)?;
                    let (trace, staging) = long_trace.as_mut().ok_or_else(long_trace_error)?;
                    history.capture(trace, staging, key_cache, "restored_key", false, stream)?;
                    history.capture(trace, staging, value_cache, "restored_value", false, stream)?;
                }
            }
            let post_segment = owned_segments""")
    dispatch.write_text(source)
    forward.write_text(replace_once(forward.read_text(),
                                   ".allocate_pinned_host_buffer(config.io_staging_bytes)",
                                   ".allocate_pinned_host_buffer(if std::env::var_os(\"RILEY_W4_TRACE_DIR\").is_some() {\n                config.io_staging_bytes.max(80 * 1024 * 1024)\n            } else {\n                config.io_staging_bytes\n            })"))


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
