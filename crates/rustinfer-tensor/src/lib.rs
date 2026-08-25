//! Safe tensor metadata and borrowed storage views.
//!
//! Metadata never casts a dtype or materializes a contiguous copy. Callers must
//! check [`TensorView::require_dtype`] and [`TensorView::require_contiguous`]
//! before passing a view to an operation with narrower capabilities.
//!
//! Mutable views borrow their storage exclusively and reject layouts that may
//! map two logical elements to the same storage element. The borrow checker also
//! prevents constructing two mutable views over the same storage:
//!
//! ```compile_fail
//! use rustinfer_tensor::{DType, Shape, TensorViewMut};
//!
//! let mut storage = [0_u8; 16];
//! let shape = Shape::new([4]).unwrap();
//! let first = TensorViewMut::from_contiguous(&mut storage[..], DType::F32, shape.clone()).unwrap();
//! let second = TensorViewMut::from_contiguous(&mut storage[..], DType::F32, shape).unwrap();
//! drop((first, second));
//! ```
//!
//! The exclusive view itself is intentionally not cloneable:
//!
//! ```compile_fail
//! use rustinfer_tensor::{DType, Shape, TensorViewMut};
//!
//! let mut storage = [0_u8; 4];
//! let view = TensorViewMut::from_contiguous(&mut storage[..], DType::F32, Shape::scalar()).unwrap();
//! let duplicate = view.clone();
//! drop((view, duplicate));
//! ```
//!
//! A view cannot escape its actual backing owner:
//!
//! ```compile_fail
//! use rustinfer_tensor::{DType, Shape, TensorView};
//!
//! let view = {
//!     let storage = vec![0_u8; 4];
//!     TensorView::from_contiguous(&storage, DType::F32, Shape::scalar()).unwrap()
//! };
//! let _ = view.logical_byte_len();
//! ```

mod dtype;
mod error;
mod layout;
mod shape;
mod storage;
mod view;
mod workspace;

pub use dtype::DType;
pub use error::{TensorError, TensorResult};
pub use layout::Layout;
pub use shape::{Shape, Strides};
pub use storage::TensorStorage;
pub use view::{TensorView, TensorViewMut};
pub use workspace::Workspace;

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "tensor metadata and ownership boundary";

/// Whether tensor metadata is compiled with the CUDA backend available.
pub const CUDA_ENABLED: bool = cfg!(feature = "cuda");
