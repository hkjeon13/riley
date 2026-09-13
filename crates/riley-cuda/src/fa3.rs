use crate::{CudaDeviceBuffer, CudaError, CudaResult, CudaStream};
#[cfg(not(all(feature = "cuda", riley_fa3)))]
use crate::{CudaErrorDomain, CudaErrorKind, CudaErrorStage};

/// Compile availability only, not numerical or device qualification.
pub const FA3_COMPILED: bool = cfg!(all(feature = "cuda", riley_fa3));

/// Immutable experimental BF16 Q9/KV3/head64/page16 metadata.
#[derive(Clone, Debug)]
#[cfg_attr(not(all(feature = "cuda", riley_fa3)), allow(dead_code))]
pub struct Fa3Batch {
    pub(crate) decode: bool,
    pub(crate) physical_pages: u32,
    pub(crate) q_indptr: Vec<i32>,
    pub(crate) kv_lengths: Vec<i32>,
    pub(crate) page_indptr: Vec<i32>,
    pub(crate) pages: Vec<i32>,
}
impl Fa3Batch {
    /// Prefill queries must be the suffix of each request's KV sequence.
    /// Decode requires exactly one query per request. Page IDs may be shared.
    pub fn new(
        decode: bool,
        physical_pages: u32,
        q_indptr: Vec<i32>,
        kv_lengths: Vec<i32>,
        page_indptr: Vec<i32>,
        pages: Vec<i32>,
    ) -> CudaResult<Self> {
        let bad = || {
            CudaError::invalid_argument("Fa3Batch::new", "invalid FA3 packed HND/page16 metadata")
        };
        let b = kv_lengths.len();
        if b == 0
            || b > 32
            || physical_pages == 0
            || physical_pages > 8192
            || q_indptr.len() != b + 1
            || page_indptr.len() != b + 1
            || pages.is_empty()
            || pages.len() > 8192
            || q_indptr[0] != 0
            || page_indptr[0] != 0
            || q_indptr[b] < 1
            || q_indptr[b] > 1024
            || page_indptr[b] != pages.len() as i32
        {
            return Err(bad());
        }
        for i in 0..b {
            let (q0, q1, p0, p1, k) = (
                q_indptr[i],
                q_indptr[i + 1],
                page_indptr[i],
                page_indptr[i + 1],
                kv_lengths[i],
            );
            if q0 < 0
                || q1 <= q0
                || q1 > q_indptr[b]
                || p0 < 0
                || p1 < p0
                || p1 > pages.len() as i32
                || !(1..=4096).contains(&k)
                || q1 - q0 > k
                || p1 - p0 != (k + 15) / 16
                || (decode && q1 - q0 != 1)
            {
                return Err(bad());
            }
        }
        if pages.iter().any(|&p| p < 0 || p >= physical_pages as i32) {
            return Err(bad());
        }
        Ok(Self {
            decode,
            physical_pages,
            q_indptr,
            kv_lengths,
            page_indptr,
            pages,
        })
    }
}

/// Exclusive borrowed owner for the experimental native primitive.
///
/// Buffers and stream remain borrowed through close/drop. Native active-use
/// leases additionally prevent premature free after an ambiguous close error.
/// This owner does not expose graph capture or dynamic metadata updates yet.
///
/// ```compile_fail
/// use riley_cuda::{Fa3Batch, PreparedFa3Attention, CudaStream, CudaDeviceBuffer};
/// fn cannot_free_bound_output(batch: &Fa3Batch, stream: &mut CudaStream,
///     mut q: CudaDeviceBuffer, mut k: CudaDeviceBuffer,
///     mut v: CudaDeviceBuffer, mut o: CudaDeviceBuffer) {
///     let mut plan = PreparedFa3Attention::new(batch, stream, &mut q, &mut k, &mut v, &mut o).unwrap();
///     o.close().unwrap();
///     plan.enqueue().unwrap();
/// }
/// ```
pub struct PreparedFa3Attention<'a> {
    #[cfg(all(feature = "cuda", riley_fa3))]
    native: crate::ffi::Fa3OwnerHandle,
    _stream: &'a mut CudaStream,
    _buffers: [&'a mut CudaDeviceBuffer; 4],
}
#[cfg(not(all(feature = "cuda", riley_fa3)))]
fn unavailable() -> CudaError {
    CudaError::new(
        CudaErrorKind::Unavailable,
        CudaErrorDomain::Rust,
        CudaErrorStage::Prepare,
        0,
        "FA3 attention",
        "build with cuda and RILEY_FA3_SOURCE pointing to pinned FA3/CUTLASS git objects",
    )
}
impl<'a> PreparedFa3Attention<'a> {
    /// Cold preparation. Q/O are packed BF16 [total_q,9,64]; K/V are HND
    /// [physical_pages,3,16,64]. No exact numerical equivalence is implied.
    pub fn new(
        batch: &Fa3Batch,
        stream: &'a mut CudaStream,
        q: &'a mut CudaDeviceBuffer,
        k: &'a mut CudaDeviceBuffer,
        v: &'a mut CudaDeviceBuffer,
        o: &'a mut CudaDeviceBuffer,
    ) -> CudaResult<Self> {
        for buffer in [&*q, &*k, &*v, &*o] {
            crate::runtime::ensure_same_context(
                &stream.context,
                buffer.context_owner(),
                "FA3 attention",
            )?;
            buffer.ensure_idle_for_operation("FA3 attention")?;
        }
        #[cfg(all(feature = "cuda", riley_fa3))]
        {
            let native = crate::ffi::Fa3OwnerHandle::create(
                batch,
                &stream.native,
                [
                    q.native_handle(),
                    k.native_handle(),
                    v.native_handle(),
                    o.native_handle(),
                ],
            )?;
            Ok(Self {
                native,
                _stream: stream,
                _buffers: [q, k, v, o],
            })
        }
        #[cfg(not(all(feature = "cuda", riley_fa3)))]
        {
            let _ = batch;
            Err(unavailable())
        }
    }
    /// Enqueue on the bound stream. Success does not mean completion.
    pub fn enqueue(&mut self) -> CudaResult<()> {
        #[cfg(all(feature = "cuda", riley_fa3))]
        {
            self.native.operation(0)
        }
        #[cfg(not(all(feature = "cuda", riley_fa3)))]
        {
            Err(unavailable())
        }
    }
    /// Wait for completion and propagate asynchronous errors.
    pub fn synchronize(&mut self) -> CudaResult<()> {
        #[cfg(all(feature = "cuda", riley_fa3))]
        {
            self.native.operation(1)
        }
        #[cfg(not(all(feature = "cuda", riley_fa3)))]
        {
            Err(unavailable())
        }
    }
    /// Drain the stream before releasing native leases and Rust borrows.
    pub fn close(mut self) -> CudaResult<()> {
        #[cfg(all(feature = "cuda", riley_fa3))]
        {
            self.native.close()
        }
        #[cfg(not(all(feature = "cuda", riley_fa3)))]
        {
            let _ = &mut self;
            Err(unavailable())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ragged_suffix_and_decode_contract() {
        assert!(
            Fa3Batch::new(
                false,
                8,
                vec![0, 1, 18],
                vec![31, 32],
                vec![0, 2, 4],
                vec![7, 1, 4, 2]
            )
            .is_ok()
        );
        assert!(
            Fa3Batch::new(
                true,
                8,
                vec![0, 1, 18],
                vec![31, 32],
                vec![0, 2, 4],
                vec![7, 1, 4, 2]
            )
            .is_err()
        );
        assert!(
            Fa3Batch::new(
                true,
                8,
                vec![0, 1, 2],
                vec![31, 32],
                vec![0, 2, 4],
                vec![7, 1, 4, 2]
            )
            .is_ok()
        );
    }
    #[test]
    fn malformed_offsets_and_physical_pages_rejected() {
        for q in [vec![], vec![0, i32::MAX], vec![0, i32::MIN], vec![0, 33]] {
            assert!(Fa3Batch::new(false, 2, q, vec![32], vec![0, 2], vec![0, 1]).is_err());
        }
        assert!(Fa3Batch::new(false, 2, vec![0, 2], vec![32], vec![0, 2], vec![0, 2]).is_err());
        assert!(
            Fa3Batch::new(
                false,
                2,
                vec![0, 2],
                vec![32],
                vec![0, i32::MIN],
                vec![0, 1]
            )
            .is_err()
        );
    }
}
