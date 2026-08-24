use std::ops::Range;

use crate::{DType, Shape, Strides, TensorError, TensorResult};

/// Shape, element strides, and element offset into backing storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layout {
    shape: Shape,
    strides: Strides,
    offset_elements: usize,
    span_end_elements: usize,
}

impl Layout {
    /// Creates and validates an explicit strided layout.
    ///
    /// # Errors
    ///
    /// Returns an error when ranks differ or reachable element-offset
    /// arithmetic overflows.
    pub fn new(shape: Shape, strides: Strides, offset_elements: usize) -> TensorResult<Self> {
        if shape.rank() != strides.rank() {
            return Err(TensorError::RankMismatch {
                shape_rank: shape.rank(),
                strides_rank: strides.rank(),
            });
        }

        let span_end_elements = storage_span_end(&shape, &strides, offset_elements)?;
        Ok(Self {
            shape,
            strides,
            offset_elements,
            span_end_elements,
        })
    }

    /// Creates a canonical row-major layout at element offset zero.
    ///
    /// # Errors
    ///
    /// Returns an error if canonical stride/span arithmetic overflows.
    pub fn contiguous(shape: Shape) -> TensorResult<Self> {
        let strides = Strides::contiguous(&shape)?;
        Self::new(shape, strides, 0)
    }

    /// Returns the logical shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns storage strides in elements.
    #[must_use]
    pub const fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Returns the first logical element's offset into backing storage.
    #[must_use]
    pub const fn offset_elements(&self) -> usize {
        self.offset_elements
    }

    /// Returns logical bytes, excluding gaps and the leading offset.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ByteLengthOverflow`] if the logical byte count
    /// cannot be represented by `u64`.
    pub fn logical_byte_len(&self, dtype: DType) -> TensorResult<u64> {
        let elements = u64::try_from(self.shape.element_count()).map_err(|_| {
            TensorError::ByteLengthOverflow {
                elements: self.shape.element_count(),
                element_size: dtype.size_bytes(),
            }
        })?;
        let element_size =
            u64::try_from(dtype.size_bytes()).map_err(|_| TensorError::ByteLengthOverflow {
                elements: self.shape.element_count(),
                element_size: dtype.size_bytes(),
            })?;
        elements
            .checked_mul(element_size)
            .ok_or(TensorError::ByteLengthOverflow {
                elements: self.shape.element_count(),
                element_size: dtype.size_bytes(),
            })
    }

    /// Returns the half-open byte range reachable in backing storage.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ByteLengthOverflow`] if applying the dtype width
    /// to the element offsets cannot be represented by `u64`.
    pub fn storage_byte_range(&self, dtype: DType) -> TensorResult<Range<u64>> {
        let element_size =
            u64::try_from(dtype.size_bytes()).map_err(|_| TensorError::ByteLengthOverflow {
                elements: self.offset_elements,
                element_size: dtype.size_bytes(),
            })?;
        let offset_elements =
            u64::try_from(self.offset_elements).map_err(|_| TensorError::ByteLengthOverflow {
                elements: self.offset_elements,
                element_size: dtype.size_bytes(),
            })?;
        let span_end_elements =
            u64::try_from(self.span_end_elements).map_err(|_| TensorError::ByteLengthOverflow {
                elements: self.span_end_elements,
                element_size: dtype.size_bytes(),
            })?;
        let start =
            offset_elements
                .checked_mul(element_size)
                .ok_or(TensorError::ByteLengthOverflow {
                    elements: self.offset_elements,
                    element_size: dtype.size_bytes(),
                })?;
        let end =
            span_end_elements
                .checked_mul(element_size)
                .ok_or(TensorError::ByteLengthOverflow {
                    elements: self.span_end_elements,
                    element_size: dtype.size_bytes(),
                })?;
        Ok(start..end)
    }

    /// Returns whether the layout is canonical contiguous row-major storage.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let mut expected = 1_usize;
        for (&extent, &stride) in self
            .shape
            .dimensions()
            .iter()
            .zip(self.strides.elements())
            .rev()
        {
            if extent > 1 && stride != expected {
                return false;
            }
            expected = match expected.checked_mul(extent) {
                Some(value) => value,
                None => return false,
            };
        }
        true
    }

    /// Conservatively proves that logical indices map to distinct elements.
    ///
    /// A `false` result means "not proven" and is rejected for mutable views.
    /// Contiguous, transposed-contiguous, and padded non-overlapping layouts are
    /// recognized. Empty layouts are non-overlapping by definition.
    #[must_use]
    pub fn is_non_overlapping(&self) -> bool {
        if self.shape.element_count() <= 1 {
            return true;
        }

        let mut dimensions: Vec<(usize, usize)> = self
            .shape
            .dimensions()
            .iter()
            .copied()
            .zip(self.strides.elements().iter().copied())
            .filter(|(extent, _)| *extent > 1)
            .map(|(extent, stride)| (stride, extent))
            .collect();
        dimensions.sort_unstable();

        let mut required_stride = 1_usize;
        for (stride, extent) in dimensions {
            if stride < required_stride {
                return false;
            }
            required_stride = match stride.checked_mul(extent) {
                Some(value) => value,
                None => return false,
            };
        }
        true
    }

    /// Returns success only for canonical contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::NonContiguousLayout`] for any other layout; no
    /// storage is copied.
    pub fn require_contiguous(&self) -> TensorResult<()> {
        if self.is_contiguous() {
            Ok(())
        } else {
            Err(TensorError::NonContiguousLayout)
        }
    }

    /// Creates a half-open zero-copy slice along one axis.
    ///
    /// # Errors
    ///
    /// Returns an error for an absent axis, an out-of-bounds range, or offset
    /// arithmetic overflow.
    pub fn slice(&self, axis: usize, range: Range<usize>) -> TensorResult<Self> {
        let Some(&extent) = self.shape.dimensions().get(axis) else {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                rank: self.shape.rank(),
            });
        };
        if range.start > range.end || range.end > extent {
            return Err(TensorError::SliceOutOfBounds {
                axis,
                start: range.start,
                end: range.end,
                extent,
            });
        }

        let offset_delta = range
            .start
            .checked_mul(self.strides.elements()[axis])
            .ok_or(TensorError::LayoutSpanOverflow)?;
        let offset_elements = self
            .offset_elements
            .checked_add(offset_delta)
            .ok_or(TensorError::LayoutSpanOverflow)?;
        let mut dimensions = self.shape.dimensions().to_vec();
        dimensions[axis] = range.end - range.start;
        Self::new(
            Shape::new(dimensions)?,
            self.strides.clone(),
            offset_elements,
        )
    }

    /// Reinterprets canonical contiguous storage with a new shape, without copy.
    ///
    /// # Errors
    ///
    /// Returns an error when element counts differ, the source is
    /// non-contiguous, or requested layout arithmetic overflows.
    pub fn reshape(&self, requested: Shape) -> TensorResult<Self> {
        if requested.element_count() != self.shape.element_count() {
            return Err(TensorError::ElementCountMismatch {
                source: self.shape.element_count(),
                requested: requested.element_count(),
            });
        }
        if requested == self.shape {
            return Ok(self.clone());
        }
        if !self.is_contiguous() {
            return Err(TensorError::NonContiguousReshape);
        }

        let strides = Strides::contiguous(&requested)?;
        Self::new(requested, strides, self.offset_elements)
    }

    /// Swaps two axes as a zero-copy metadata operation.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::AxisOutOfBounds`] if either axis is absent, or a
    /// layout arithmetic error if reconstruction overflows.
    pub fn transpose(&self, first: usize, second: usize) -> TensorResult<Self> {
        let rank = self.shape.rank();
        if first >= rank {
            return Err(TensorError::AxisOutOfBounds { axis: first, rank });
        }
        if second >= rank {
            return Err(TensorError::AxisOutOfBounds { axis: second, rank });
        }

        let mut dimensions = self.shape.dimensions().to_vec();
        let mut strides = self.strides.elements().to_vec();
        dimensions.swap(first, second);
        strides.swap(first, second);
        Self::new(
            Shape::new(dimensions)?,
            Strides::new(strides),
            self.offset_elements,
        )
    }
}

fn storage_span_end(
    shape: &Shape,
    strides: &Strides,
    offset_elements: usize,
) -> TensorResult<usize> {
    if shape.is_empty() {
        return Ok(offset_elements);
    }

    let last_offset = shape.dimensions().iter().zip(strides.elements()).try_fold(
        offset_elements,
        |offset, (&extent, &stride)| {
            let axis_offset = (extent - 1)
                .checked_mul(stride)
                .ok_or(TensorError::LayoutSpanOverflow)?;
            offset
                .checked_add(axis_offset)
                .ok_or(TensorError::LayoutSpanOverflow)
        },
    )?;
    last_offset
        .checked_add(1)
        .ok_or(TensorError::LayoutSpanOverflow)
}
