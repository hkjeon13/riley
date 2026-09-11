use crate::{TensorError, TensorResult};

/// Logical tensor extents in outermost-to-innermost order.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Shape {
    dimensions: Box<[usize]>,
    element_count: usize,
}

impl Shape {
    /// Validates dimensions and computes their logical element count.
    ///
    /// Rank-zero shapes represent one scalar. Any zero dimension makes the
    /// shape empty, even when multiplying the other dimensions would overflow.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ElementCountOverflow`] when the product of all
    /// non-zero dimensions cannot be represented by `usize`.
    pub fn new(dimensions: impl Into<Vec<usize>>) -> TensorResult<Self> {
        let dimensions = dimensions.into().into_boxed_slice();
        let element_count = if dimensions.contains(&0) {
            0
        } else {
            dimensions
                .iter()
                .try_fold(1_usize, |product, dimension| {
                    product.checked_mul(*dimension)
                })
                .ok_or(TensorError::ElementCountOverflow)?
        };

        Ok(Self {
            dimensions,
            element_count,
        })
    }

    /// Constructs the rank-zero shape of one scalar element.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            dimensions: Box::new([]),
            element_count: 1,
        }
    }

    /// Returns the dimensions.
    #[must_use]
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the validated number of logical elements.
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns whether at least one dimension is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.element_count == 0
    }
}

/// Storage strides measured in elements, not bytes.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Strides {
    elements: Box<[usize]>,
}

impl Strides {
    /// Constructs explicit element strides.
    #[must_use]
    pub fn new(elements: impl Into<Vec<usize>>) -> Self {
        Self {
            elements: elements.into().into_boxed_slice(),
        }
    }

    /// Computes canonical row-major strides for a validated shape.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ElementCountOverflow`] if a non-empty shape's
    /// canonical stride cannot be represented by `usize`.
    pub fn contiguous(shape: &Shape) -> TensorResult<Self> {
        if shape.is_empty() {
            return Ok(Self::new(vec![0; shape.rank()]));
        }

        let mut elements = vec![0; shape.rank()];
        let mut stride = 1_usize;
        for (axis, extent) in shape.dimensions().iter().enumerate().rev() {
            elements[axis] = stride;
            stride = stride
                .checked_mul(*extent)
                .ok_or(TensorError::ElementCountOverflow)?;
        }
        Ok(Self::new(elements))
    }

    /// Returns the element strides.
    #[must_use]
    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    /// Returns the number of strides.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.elements.len()
    }
}
