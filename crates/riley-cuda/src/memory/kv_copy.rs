//! Same-pool asynchronous D2D page copies with native completion ownership.
use super::*;

/// Strided page geometry shared with the C ABI. Source and destination are
/// distinct pages in each of the separate K/V buffers. All units are bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct CudaPageCopySpec {
    pub layers: u64,
    pub heads: u64,
    pub layer_stride: u64,
    pub block_stride: u64,
    pub head_stride: u64,
    pub valid_bytes: u64,
    pub source_page: u32,
    pub destination_page: u32,
}

impl CudaPageCopySpec {
    /// Validates disjoint, initialized slices before any CUDA operation.
    ///
    /// # Errors
    /// Rejects zero dimensions, invalid strides, overlap and out-of-bounds pages.
    pub fn validate(self, key_bytes: u64, value_bytes: u64) -> CudaResult<()> {
        if self.layers == 0
            || self.heads == 0
            || self.layer_stride == 0
            || self.block_stride == 0
            || self.head_stride == 0
            || self.valid_bytes == 0
            || self.valid_bytes > self.head_stride
            || self.block_stride % self.head_stride != 0
            || self.block_stride / self.head_stride != self.heads
            || self.layer_stride % self.block_stride != 0
            || self.source_page == self.destination_page
            || u64::from(self.source_page) >= self.layer_stride / self.block_stride
            || u64::from(self.destination_page) >= self.layer_stride / self.block_stride
            || self.layers > key_bytes / self.layer_stride
            || self.layers > value_bytes / self.layer_stride
        {
            return Err(CudaError::out_of_range(
                "KV page copy",
                "invalid, overlapping or out-of-range layout",
            ));
        }
        Ok(())
    }
}

impl CudaDeviceBuffer {
    /// Enqueues all initialized slices of one K/V page and records one event.
    /// This cold local transport requires idle buffers and stream; it cannot
    /// bypass the ownership of buffers retained by a captured model graph.
    ///
    /// # Errors
    /// Preflight failures enqueue nothing. Errors after native ownership is
    /// acquired are reported by the returned token, never by dropping its holds.
    pub fn copy_kv_page_async<'a>(
        &'a mut self,
        values: &'a mut CudaDeviceBuffer,
        spec: CudaPageCopySpec,
        stream: &'a mut CudaStream,
    ) -> CudaResult<CudaPendingKvCopy<'a>> {
        const OPERATION: &str = "CudaDeviceBuffer::copy_kv_page_async";
        spec.validate(self.byte_len, values.byte_len)?;
        ensure_same_context(&self.context, &values.context, OPERATION)?;
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        self.use_state.begin(OPERATION, "key buffer")?;
        if let Err(error) = values.use_state.begin(OPERATION, "value buffer") {
            self.use_state.finish();
            return Err(error);
        }
        #[cfg(feature = "cuda")]
        {
            let native =
                match ffi::CopyHandle::kv_page(&self.native, &values.native, &spec, &stream.native)
                {
                    Ok(native) => native,
                    Err(error) => {
                        release_buffers(&self.use_state, &values.use_state);
                        return Err(error);
                    }
                };
            Ok(CudaPendingKvCopy {
                native: Some(native),
                keys: self,
                values,
                _stream: stream,
                quiescent: false,
                failure: None,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            release_buffers(&self.use_state, &values.use_state);
            Err(CudaError::unavailable(OPERATION))
        }
    }
}

/// Pending native page copy. Errors are sticky, so retrying a failed query
/// cannot accidentally turn a partially copied page into a successful import.
/// Forgetting this token retains native and Rust active-use guards. Drop waits;
/// unconfirmed completion leaks the native holds rather than exposing buffers.
pub struct CudaPendingKvCopy<'a> {
    #[cfg(feature = "cuda")]
    native: Option<ffi::CopyHandle>,
    #[cfg(feature = "cuda")]
    keys: &'a mut CudaDeviceBuffer,
    #[cfg(feature = "cuda")]
    values: &'a mut CudaDeviceBuffer,
    _stream: &'a mut CudaStream,
    quiescent: bool,
    failure: Option<CudaError>,
}

impl CudaPendingKvCopy<'_> {
    /// True only after native completion and context restoration are confirmed,
    /// including a completed but unsuccessful transfer.
    #[must_use]
    pub const fn is_quiescent(&self) -> bool {
        self.quiescent
    }

    /// Nonblocking event query. Not-ready is Ok(false).
    ///
    /// # Errors
    /// Returns the first observed copy/query/cleanup failure; it remains sticky.
    pub fn query(&mut self) -> CudaResult<bool> {
        self.observe(false)
    }

    /// Waits for the event, retaining ownership if completion is ambiguous.
    ///
    /// # Errors
    /// Reports sticky native failure. Check `is_quiescent` before releasing pages.
    pub fn synchronize(&mut self) -> CudaResult<()> {
        self.observe(true).map(|_| ())
    }

    fn observe(&mut self, wait: bool) -> CudaResult<bool> {
        if !self.quiescent {
            #[cfg(feature = "cuda")]
            {
                let native = self
                    .native
                    .as_mut()
                    .expect("live native copy until completion");
                let outcome = if wait {
                    native.synchronize()
                } else {
                    native.query()
                };
                if let Err(error) = outcome.result {
                    if self.failure.is_none() {
                        self.failure = Some(error);
                    }
                }
                if outcome.complete {
                    if let Err(error) = native.close() {
                        if self.failure.is_none() {
                            self.failure = Some(error);
                        }
                    }
                    release_buffers(&self.keys.use_state, &self.values.use_state);
                    self.quiescent = true;
                    self.native = None;
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = wait;
                return Err(CudaError::unavailable("query KV copy"));
            }
        }
        match &self.failure {
            Some(error) => Err(error.clone()),
            None => Ok(self.quiescent),
        }
    }
}

impl Drop for CudaPendingKvCopy<'_> {
    fn drop(&mut self) {
        if !self.quiescent {
            let _ = self.synchronize();
        }
        #[cfg(feature = "cuda")]
        if !self.quiescent {
            if let Some(native) = self.native.take() {
                std::mem::forget(native);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn page_copy_geometry_rejects_overlap_overflow_and_inconsistent_strides() {
        assert_eq!(std::mem::size_of::<CudaPageCopySpec>(), 56);
        let spec = CudaPageCopySpec {
            layers: 3,
            heads: 2,
            layer_stride: 4096,
            block_stride: 1024,
            head_stride: 512,
            valid_bytes: 480,
            source_page: 1,
            destination_page: 3,
        };
        spec.validate(12288, 12288).unwrap();
        assert!(spec.validate(12287, 12288).is_err());
        for bad in [
            CudaPageCopySpec {
                destination_page: 1,
                ..spec
            },
            CudaPageCopySpec {
                source_page: 4,
                ..spec
            },
            CudaPageCopySpec {
                layers: u64::MAX,
                ..spec
            },
            CudaPageCopySpec { heads: 3, ..spec },
            CudaPageCopySpec {
                valid_bytes: 513,
                ..spec
            },
            CudaPageCopySpec {
                layer_stride: 0,
                ..spec
            },
            CudaPageCopySpec {
                block_stride: 1023,
                ..spec
            },
            CudaPageCopySpec {
                head_stride: 0,
                ..spec
            },
        ] {
            assert!(bad.validate(u64::MAX, u64::MAX).is_err());
        }
    }
}
