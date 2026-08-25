use std::fmt;
use std::ops::Range;

use crate::{DType, Layout, Shape, TensorError, TensorResult, TensorStorage};

/// An immutable strided tensor view anchored to a backing owner.
pub struct TensorView<'a, B: TensorStorage + ?Sized = [u8]> {
    storage: &'a B,
    dtype: DType,
    layout: Layout,
}

impl<'a, B: TensorStorage + ?Sized> TensorView<'a, B> {
    /// Validates a dtype and layout against borrowed storage.
    ///
    /// # Errors
    ///
    /// Returns an error if byte-range arithmetic overflows or the reachable
    /// layout range exceeds the backing capacity.
    pub fn new(storage: &'a B, dtype: DType, layout: Layout) -> TensorResult<Self> {
        validate_storage(storage, dtype, &layout)?;
        Ok(Self {
            storage,
            dtype,
            layout,
        })
    }

    /// Creates a canonical contiguous view at storage offset zero.
    ///
    /// # Errors
    ///
    /// Returns an error if shape/layout byte arithmetic overflows or the
    /// backing capacity is too small.
    pub fn from_contiguous(storage: &'a B, dtype: DType, shape: Shape) -> TensorResult<Self> {
        Self::new(storage, dtype, Layout::contiguous(shape)?)
    }

    /// Returns the dtype without changing it.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the full shape/strides/offset metadata.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the logical shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Returns the borrowed backing owner.
    #[must_use]
    pub const fn storage(&self) -> &'a B {
        self.storage
    }

    /// Returns logical bytes, excluding layout gaps and leading offset.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ByteLengthOverflow`] if the byte count overflows.
    pub fn logical_byte_len(&self) -> TensorResult<u64> {
        self.layout.logical_byte_len(self.dtype)
    }

    /// Returns the half-open reachable byte range in [`Self::storage`].
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ByteLengthOverflow`] if the range overflows.
    pub fn storage_byte_range(&self) -> TensorResult<Range<u64>> {
        self.layout.storage_byte_range(self.dtype)
    }

    /// Returns a half-open zero-copy slice.
    ///
    /// # Errors
    ///
    /// Returns an error for an absent axis, invalid range, offset overflow, or
    /// a resulting range beyond backing capacity.
    pub fn slice(&self, axis: usize, range: Range<usize>) -> TensorResult<Self> {
        Self::new(self.storage, self.dtype, self.layout.slice(axis, range)?)
    }

    /// Returns a zero-copy reshape or an explicit incompatibility error.
    ///
    /// # Errors
    ///
    /// Returns an error when counts differ, the source is non-contiguous, or
    /// resulting byte arithmetic/capacity validation fails.
    pub fn reshape(&self, requested: Shape) -> TensorResult<Self> {
        Self::new(self.storage, self.dtype, self.layout.reshape(requested)?)
    }

    /// Returns a zero-copy transposed view.
    ///
    /// # Errors
    ///
    /// Returns an error for an absent axis or failed resulting layout/capacity
    /// validation.
    pub fn transpose(&self, first: usize, second: usize) -> TensorResult<Self> {
        Self::new(
            self.storage,
            self.dtype,
            self.layout.transpose(first, second)?,
        )
    }

    /// Checks an operation's dtype capability without casting.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DTypeMismatch`] when `expected` differs.
    pub fn require_dtype(&self, expected: DType) -> TensorResult<()> {
        require_dtype(self.dtype, expected)
    }

    /// Checks an operation's layout capability without materializing a copy.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::NonContiguousLayout`] for non-contiguous views.
    pub fn require_contiguous(&self) -> TensorResult<()> {
        self.layout.require_contiguous()
    }
}

impl<B: TensorStorage + ?Sized> Clone for TensorView<'_, B> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage,
            dtype: self.dtype,
            layout: self.layout.clone(),
        }
    }
}

impl<B: TensorStorage + ?Sized> fmt::Debug for TensorView<'_, B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorView")
            .field("storage_byte_len", &self.storage.byte_len())
            .field("dtype", &self.dtype)
            .field("layout", &self.layout)
            .finish()
    }
}

/// An exclusive strided tensor view anchored to a mutable backing owner.
///
/// Construction conservatively rejects layouts that may address one storage
/// element through multiple logical indices. Transformations consume the view,
/// so the original exclusive handle cannot remain live alongside the result.
pub struct TensorViewMut<'a, B: TensorStorage + ?Sized = [u8]> {
    storage: &'a mut B,
    dtype: DType,
    layout: Layout,
}

impl<'a, B: TensorStorage + ?Sized> TensorViewMut<'a, B> {
    /// Validates exclusive storage, dtype, layout bounds, and non-aliasing.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range exceeds capacity or the layout is not
    /// conservatively proven non-overlapping.
    pub fn new(storage: &'a mut B, dtype: DType, layout: Layout) -> TensorResult<Self> {
        validate_storage(storage, dtype, &layout)?;
        if !layout.is_non_overlapping() {
            return Err(TensorError::MutableLayoutMayAlias);
        }
        Ok(Self {
            storage,
            dtype,
            layout,
        })
    }

    /// Creates an exclusive canonical contiguous view at offset zero.
    ///
    /// # Errors
    ///
    /// Returns an error if shape/layout byte arithmetic overflows or the
    /// backing capacity is too small.
    pub fn from_contiguous(storage: &'a mut B, dtype: DType, shape: Shape) -> TensorResult<Self> {
        Self::new(storage, dtype, Layout::contiguous(shape)?)
    }

    /// Returns the dtype without changing it.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the full shape/strides/offset metadata.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the logical shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Reborrows the view immutably.
    #[must_use]
    pub fn as_view(&self) -> TensorView<'_, B> {
        TensorView {
            storage: self.storage,
            dtype: self.dtype,
            layout: self.layout.clone(),
        }
    }

    /// Returns logical bytes, excluding layout gaps and leading offset.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ByteLengthOverflow`] if the byte count overflows.
    pub fn logical_byte_len(&self) -> TensorResult<u64> {
        self.layout.logical_byte_len(self.dtype)
    }

    /// Consumes this handle and returns a half-open zero-copy slice.
    ///
    /// # Errors
    ///
    /// Returns an error for an absent axis, invalid range, arithmetic overflow,
    /// capacity failure, or a resulting layout not proven non-overlapping.
    pub fn slice(self, axis: usize, range: Range<usize>) -> TensorResult<Self> {
        let layout = self.layout.slice(axis, range)?;
        Self::new(self.storage, self.dtype, layout)
    }

    /// Consumes this handle and returns a zero-copy reshape.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible metadata, capacity failure, or a
    /// resulting layout not proven non-overlapping.
    pub fn reshape(self, requested: Shape) -> TensorResult<Self> {
        let layout = self.layout.reshape(requested)?;
        Self::new(self.storage, self.dtype, layout)
    }

    /// Consumes this handle and returns a zero-copy transposed view.
    ///
    /// # Errors
    ///
    /// Returns an error for an absent axis, capacity failure, or a resulting
    /// layout not proven non-overlapping.
    pub fn transpose(self, first: usize, second: usize) -> TensorResult<Self> {
        let layout = self.layout.transpose(first, second)?;
        Self::new(self.storage, self.dtype, layout)
    }

    /// Checks an operation's dtype capability without casting.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DTypeMismatch`] when `expected` differs.
    pub fn require_dtype(&self, expected: DType) -> TensorResult<()> {
        require_dtype(self.dtype, expected)
    }

    /// Checks an operation's layout capability without materializing a copy.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::NonContiguousLayout`] for non-contiguous views.
    pub fn require_contiguous(&self) -> TensorResult<()> {
        self.layout.require_contiguous()
    }
}

impl<B: TensorStorage + ?Sized> fmt::Debug for TensorViewMut<'_, B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorViewMut")
            .field("storage_byte_len", &self.storage.byte_len())
            .field("dtype", &self.dtype)
            .field("layout", &self.layout)
            .finish()
    }
}

impl TensorViewMut<'_, [u8]> {
    /// Reborrows host bytes exclusively without exposing a resizable owner.
    #[must_use]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.storage
    }
}

fn validate_storage<B: TensorStorage + ?Sized>(
    storage: &B,
    dtype: DType,
    layout: &Layout,
) -> TensorResult<()> {
    let length = storage.byte_len();
    let required = layout.storage_byte_range(dtype)?.end;
    if required > length {
        return Err(TensorError::BufferTooSmall {
            required,
            actual: length,
        });
    }
    layout.logical_byte_len(dtype)?;
    Ok(())
}

fn require_dtype(actual: DType, expected: DType) -> TensorResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(TensorError::DTypeMismatch { expected, actual })
    }
}
