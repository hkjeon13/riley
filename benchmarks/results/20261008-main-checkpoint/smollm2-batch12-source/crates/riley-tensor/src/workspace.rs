use std::fmt;

use crate::{DType, Layout, TensorResult, TensorStorage, TensorView, TensorViewMut};

/// An explicitly supplied owning backing store for temporary tensor data.
///
/// `Workspace` never allocates, resizes, or clones its backing. Execution code
/// decides the required capacity and passes an existing owner into [`Self::new`].
/// It deliberately does not implement [`Clone`].
pub struct Workspace<B: TensorStorage> {
    storage: B,
}

impl<B: TensorStorage> Workspace<B> {
    /// Takes ownership of an explicitly allocated backing store.
    #[must_use]
    pub const fn new(storage: B) -> Self {
        Self { storage }
    }

    /// Returns the backing capacity in bytes.
    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.storage.byte_len()
    }

    /// Borrows the backing owner.
    #[must_use]
    pub const fn storage(&self) -> &B {
        &self.storage
    }

    /// Borrows the backing owner exclusively.
    #[must_use]
    pub fn storage_mut(&mut self) -> &mut B {
        &mut self.storage
    }

    /// Creates a validated immutable view anchored to this workspace.
    ///
    /// # Errors
    ///
    /// Returns an error when the layout's byte range overflows or exceeds the
    /// workspace capacity.
    pub fn view(&self, dtype: DType, layout: Layout) -> TensorResult<TensorView<'_, B>> {
        TensorView::new(&self.storage, dtype, layout)
    }

    /// Creates a validated exclusive view anchored to this workspace.
    ///
    /// # Errors
    ///
    /// Returns an error when bounds validation fails or the layout is not
    /// conservatively proven non-overlapping.
    pub fn view_mut(&mut self, dtype: DType, layout: Layout) -> TensorResult<TensorViewMut<'_, B>> {
        TensorViewMut::new(&mut self.storage, dtype, layout)
    }

    /// Returns the backing owner without copying it.
    #[must_use]
    pub fn into_inner(self) -> B {
        self.storage
    }
}

impl<B: TensorStorage> fmt::Debug for Workspace<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Workspace")
            .field("capacity_bytes", &self.capacity_bytes())
            .finish_non_exhaustive()
    }
}
