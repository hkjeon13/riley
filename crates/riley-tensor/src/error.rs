use std::error;
use std::fmt;

use crate::DType;

/// Result type for tensor metadata validation.
pub type TensorResult<T> = Result<T, TensorError>;

/// An explicit tensor metadata or view-contract failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TensorError {
    /// Multiplying non-zero dimensions exceeded `usize`.
    ElementCountOverflow,
    /// Converting a logical element count or storage span to bytes exceeded `u64`.
    ByteLengthOverflow {
        /// Logical element count or storage offset.
        elements: usize,
        /// Width of one element in bytes.
        element_size: usize,
    },
    /// Shape and strides have different ranks.
    RankMismatch {
        /// Number of shape dimensions.
        shape_rank: usize,
        /// Number of strides.
        strides_rank: usize,
    },
    /// Computing the greatest reachable storage offset exceeded `usize`.
    LayoutSpanOverflow,
    /// An axis does not exist in the layout.
    AxisOutOfBounds {
        /// Requested axis.
        axis: usize,
        /// Number of layout dimensions.
        rank: usize,
    },
    /// A half-open slice is not contained by its axis.
    SliceOutOfBounds {
        /// Sliced axis.
        axis: usize,
        /// Inclusive slice start.
        start: usize,
        /// Exclusive slice end.
        end: usize,
        /// Axis extent.
        extent: usize,
    },
    /// A reshape changed the number of logical elements.
    ElementCountMismatch {
        /// Original element count.
        source: usize,
        /// Requested element count.
        requested: usize,
    },
    /// A non-contiguous layout cannot be reinterpreted with the requested shape.
    NonContiguousReshape,
    /// An operation requires canonical contiguous row-major storage.
    NonContiguousLayout,
    /// A mutable view layout may map multiple logical indices to one element.
    MutableLayoutMayAlias,
    /// The backing storage does not contain the layout's reachable byte range.
    BufferTooSmall {
        /// Minimum byte length required from the start of the backing storage.
        required: u64,
        /// Actual backing byte length.
        actual: u64,
    },
    /// An operation requested a different dtype; no cast was attempted.
    DTypeMismatch {
        /// Dtype required by the operation.
        expected: DType,
        /// Dtype carried by the view.
        actual: DType,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElementCountOverflow => formatter.write_str("tensor element count exceeds usize"),
            Self::ByteLengthOverflow {
                elements,
                element_size,
            } => write!(
                formatter,
                "tensor byte length overflows: {elements} elements of {element_size} bytes"
            ),
            Self::RankMismatch {
                shape_rank,
                strides_rank,
            } => write!(
                formatter,
                "shape rank {shape_rank} does not match strides rank {strides_rank}"
            ),
            Self::LayoutSpanOverflow => {
                formatter.write_str("tensor layout storage span exceeds usize")
            }
            Self::AxisOutOfBounds { axis, rank } => {
                write!(formatter, "axis {axis} is out of bounds for rank {rank}")
            }
            Self::SliceOutOfBounds {
                axis,
                start,
                end,
                extent,
            } => write!(
                formatter,
                "slice {start}..{end} is out of bounds for axis {axis} with extent {extent}"
            ),
            Self::ElementCountMismatch { source, requested } => write!(
                formatter,
                "reshape changes element count from {source} to {requested}"
            ),
            Self::NonContiguousReshape => formatter.write_str(
                "zero-copy reshape requires canonical contiguous storage; no copy was made",
            ),
            Self::NonContiguousLayout => formatter
                .write_str("operation requires canonical contiguous storage; no copy was made"),
            Self::MutableLayoutMayAlias => formatter.write_str(
                "mutable tensor layout may alias one storage element from multiple indices",
            ),
            Self::BufferTooSmall { required, actual } => write!(
                formatter,
                "tensor layout requires {required} backing bytes, but only {actual} are available"
            ),
            Self::DTypeMismatch { expected, actual } => write!(
                formatter,
                "operation requires dtype {expected}, but view is {actual}; no cast was made"
            ),
        }
    }
}

impl error::Error for TensorError {}
