//! Indexed `RoPE` borrowing actual buffers and a position span inside packed metadata.
use crate::{CudaDeviceBuffer, CudaError, CudaResult, CudaStream};

/// Exact dimensions and aligned position offset retained by native graph state.
#[derive(Clone, Copy, Debug)]
pub struct RopeGraphGeometry {
    /// Number of active input rows.
    pub rows: u64,
    /// Query or key head count.
    pub heads: u64,
    /// Full head width.
    pub head_size: u64,
    /// Even rotated prefix width.
    pub rotary_dimension: u64,
    /// Number of positions in each table.
    pub table_positions: u64,
    /// Byte offset of the U32 row positions within their parent allocation.
    pub positions_byte_offset: u64,
}

impl RopeGraphGeometry {
    /// Validates the aligned parent span without overflow.
    ///
    /// # Errors
    /// Rejects empty, unaligned or out-of-bounds position spans.
    pub fn validate_position_span(self, parent_bytes: u64) -> CudaResult<()> {
        if self.rows == 0
            || self.positions_byte_offset % 4 != 0
            || self.positions_byte_offset > parent_bytes
            || self.rows > (parent_bytes - self.positions_byte_offset) / 4
        {
            return Err(CudaError::out_of_range(
                "RopeGraphGeometry::validate_position_span",
                "position span is empty, unaligned or outside its parent",
            ));
        }
        Ok(())
    }
}

/// Five exclusive device owners and one stream, retained until close.
pub struct BorrowedRopeResources<'a> {
    /// Capture/replay stream.
    pub stream: &'a mut CudaStream,
    /// BF16 row-major input.
    pub input: &'a mut CudaDeviceBuffer,
    /// F32 cosine table.
    pub cos: &'a mut CudaDeviceBuffer,
    /// F32 sine table.
    pub sin: &'a mut CudaDeviceBuffer,
    /// Whole parent allocation containing row positions.
    pub positions: &'a mut CudaDeviceBuffer,
    /// Distinct BF16 output.
    pub output: &'a mut CudaDeviceBuffer,
}

/// One indexed `RoPE` node with immutable position span and whole-parent leases.
pub struct BorrowedRopeGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedRopeResources<'a>,
    terminal: bool,
}

#[cfg(feature = "cuda")]
struct CaptureGuard {
    native: crate::ffi::GraphCaptureHandle,
    active: bool,
}

#[cfg(feature = "cuda")]
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active && self.native.abort_with_transition().resource_release_known {
            super::finish_deferred_capture_contexts();
        }
    }
}

impl<'a> BorrowedRopeGraph<'a> {
    /// Captures the existing BF16 indexed `RoPE` primitive over borrowed owners.
    /// The host mirror validates bounds only; callers must verify device bytes.
    ///
    /// # Errors
    /// Rejects geometry, context, aliases, busy resources or CUDA failures.
    pub fn prepare(
        resources: BorrowedRopeResources<'a>,
        geometry: RopeGraphGeometry,
        positions_host: &[u32],
    ) -> CudaResult<Self> {
        const OP: &str = "BorrowedRopeGraph::prepare";
        geometry.validate_position_span(resources.positions.byte_len())?;
        #[cfg(feature = "cuda")]
        {
            super::validate_graph_indexed_rope_bf16_capture_preflight(
                resources.stream,
                resources.input,
                resources.cos,
                resources.sin,
                resources.positions,
                resources.output,
                positions_host,
                geometry.rows,
                geometry.heads,
                geometry.head_size,
                geometry.rotary_dimension,
                geometry.table_positions,
                OP,
            )?;
            let native = resources
                .stream
                .native
                .begin_graph_indexed_rope_bf16_span_capture(
                    resources.input.native_handle(),
                    resources.cos.native_handle(),
                    resources.sin.native_handle(),
                    resources.positions.native_handle(),
                    resources.output.native_handle(),
                    positions_host,
                    geometry.rows,
                    geometry.heads,
                    geometry.head_size,
                    geometry.rotary_dimension,
                    geometry.table_positions,
                    geometry.positions_byte_offset,
                    super::CudaGraphCaptureMode::ThreadLocal as u32,
                )?;
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            capture.native.enqueue_indexed_rope_bf16()?;
            let transition = capture.native.end();
            capture.active = !transition.owner_consumed;
            if transition.resource_release_known {
                super::finish_deferred_capture_contexts();
            }
            let mut graph = transition.result?;
            let native = graph.instantiate()?;
            Ok(Self {
                native,
                resources,
                terminal: false,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (resources, geometry, positions_host);
            Err(CudaError::unavailable(OP))
        }
    }

    /// Replays once and waits for known completion.
    ///
    /// # Errors
    /// Any CUDA failure permanently prevents further replay.
    pub fn replay(&mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedRopeGraph::replay";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        self.terminal = true;
        #[cfg(feature = "cuda")]
        {
            self.native
                .launch(&mut self.resources.stream.native)?
                .complete()?;
            self.terminal = false;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        Err(CudaError::unavailable(OP))
    }

    /// Releases graph leases before returning access to the executor allocations.
    ///
    /// # Errors
    /// Rejects terminal state and propagates native destruction failures.
    pub fn close(mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedRopeGraph::close";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self;
            Err(CudaError::unavailable(OP))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RopeGraphGeometry;
    #[test]
    fn packed_rope_positions_require_an_aligned_nonempty_bounded_span() {
        let mut shape = RopeGraphGeometry {
            rows: 1,
            heads: 9,
            head_size: 64,
            rotary_dimension: 64,
            table_positions: 2048,
            positions_byte_offset: 64,
        };
        assert!(shape.validate_position_span(68).is_ok());
        assert!(shape.validate_position_span(67).is_err());
        shape.positions_byte_offset = 1;
        assert!(shape.validate_position_span(68).is_err());
        shape.positions_byte_offset = u64::MAX - 3;
        assert!(shape.validate_position_span(u64::MAX).is_err());
        shape.positions_byte_offset = 0;
        shape.rows = u64::MAX;
        assert!(shape.validate_position_span(u64::MAX).is_err());
        shape.rows = 0;
        assert!(shape.validate_position_span(68).is_err());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod native_tests {
    #[test]
    #[ignore = "requires CUDA; direct native span rejection before capture"]
    fn native_rope_span_rejects_unaligned_and_overflow_offsets()
    -> Result<(), Box<dyn std::error::Error>> {
        let context = crate::CudaRuntime::initialize()?
            .device(0)?
            .create_context()?;
        let mut stream = context.create_stream()?;
        let input = context.allocate_device_buffer(128)?;
        let cos = context.allocate_device_buffer(1024)?;
        let sin = context.allocate_device_buffer(1024)?;
        let positions = context.allocate_device_buffer(128)?;
        let output = context.allocate_device_buffer(128)?;
        let stats = context.allocation_stats()?;
        for offset in [1, 128, u64::MAX, u64::MAX - 3] {
            assert!(
                stream
                    .native
                    .begin_graph_indexed_rope_bf16_span_capture(
                        input.native_handle(),
                        cos.native_handle(),
                        sin.native_handle(),
                        positions.native_handle(),
                        output.native_handle(),
                        &[0],
                        1,
                        1,
                        64,
                        64,
                        8,
                        offset,
                        super::super::CudaGraphCaptureMode::ThreadLocal as u32
                    )
                    .is_err()
            );
            assert_eq!(context.allocation_stats()?, stats);
        }
        input.close()?;
        cos.close()?;
        sin.close()?;
        positions.close()?;
        output.close()?;
        stream.close()?;
        assert!(context.allocation_stats()?.is_zero());
        context.close()?;
        Ok(())
    }
}
