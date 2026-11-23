const KV_HISTORY_POSITION: usize = 7842;
const KV_HISTORY_BYTES: u64 = KV_HISTORY_POSITION as u64 * 4 * 128 * 2;

fn kv_trace_error(source: riley_cuda::CudaError) -> LlamaBatchExecutorError {
    batch_cuda(ExecutionSite::layer(1, LlamaOp::RaggedPagedAttention), source)
}

struct TraceHistory {
    past: Vec<(u64, u64)>,
    current: Vec<(u64, u64)>,
}

impl TraceHistory {
    fn new(layout: KvLayout, layer: usize, host: PackedBatchHostV1<'_>)
        -> LlamaBatchExecutorResult<Self>
    {
        let blocks = host.block_ids();
        let slots = layout.block_size();
        let full = KV_HISTORY_POSITION / slots;
        let tail = KV_HISTORY_POSITION % slots;
        if layout.key_value_head_count() != 4 || layout.head_dimension() != 128
            || slots != 16 || host.row_positions() != [KV_HISTORY_POSITION as u32]
            || host.row_sequence_slots() != [0]
            || host.sequence_block_offsets() != [0, (full + 1) as u32]
            || blocks.len() != full + 1 || host.valid_tokens().len() != blocks.len()
            || host.valid_tokens()[..full].iter().any(|&count| count as usize != slots)
            || host.valid_tokens()[full] as usize != tail + 1
            || blocks.iter().copied().collect::<std::collections::BTreeSet<_>>().len() != blocks.len()
        {
            return Err(long_trace_error());
        }
        let offset = |block, head, token| {
            layout.element_byte_offset(layer, block, head, token).ok_or_else(long_trace_error)
        };
        let mut past = Vec::with_capacity(blocks.len());
        for &block in &blocks[..full] {
            let start = offset(block as usize, 0, 0)?;
            let bytes = layout.block_stride_bytes();
            // Coalesce only physically adjacent pages in logical order.
            match past.last_mut() {
                Some((previous, count)) if *previous + *count == start => *count += bytes,
                _ => past.push((start, bytes)),
            }
        }
        let block = blocks[full] as usize;
        let token_bytes = (layout.head_dimension() * 2) as u64;
        let mut current = Vec::new();
        for head in 0..layout.key_value_head_count() {
            past.push((offset(block, head, 0)?, tail as u64 * token_bytes));
            current.push((offset(block, head, tail)?, token_bytes));
        }
        if past.iter().map(|(_, count)| count).sum::<u64>() != KV_HISTORY_BYTES {
            return Err(long_trace_error());
        }
        Ok(Self { past, current })
    }

    fn capture<S: CudaExecutionStream + ?Sized>(
        &self,
        trace: &mut LongTrace,
        staging: &CudaPinnedHostBuffer,
        buffer: &CudaDeviceBuffer,
        label: &'static str,
        current: bool,
        stream: &mut S,
    ) -> LlamaBatchExecutorResult<u64> {
        let ranges = if current { &self.current } else { &self.past };
        let start = trace.end;
        for &(device, bytes) in ranges {
            if trace.end + bytes > trace.trace_limit(staging) {
                return Err(long_trace_error());
            }
            buffer.copy_to_pinned_in_active_command_batch(device, staging, trace.end, bytes, stream)
                .map_err(kv_trace_error)?;
            trace.end += bytes;
        }
        trace.entries.push((label, Some(1), start, trace.end - start));
        Ok(start)
    }

    fn upload<S: CudaExecutionStream + ?Sized>(
        &self,
        staging: &CudaPinnedHostBuffer,
        buffer: &mut CudaDeviceBuffer,
        mut offset: u64,
        stream: &mut S,
    ) -> LlamaBatchExecutorResult<()> {
        for &(device, bytes) in &self.past {
            buffer.copy_from_pinned_in_active_command_batch(device, staging, offset, bytes, stream)
                .map_err(kv_trace_error)?;
            offset += bytes;
        }
        Ok(())
    }
}
