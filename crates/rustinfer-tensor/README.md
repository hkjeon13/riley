# rustinfer-tensor

`rustinfer-tensor` owns safe tensor metadata and backing-anchored views. PR 04
keeps allocation behind a separate boundary: this module has no allocator,
kernel, model, or CUDA execution behavior.

## Metadata contract

- `Shape` validates logical element-count overflow. Rank zero is one scalar; a
  shape with any zero dimension has zero elements.
- `Strides` and `Layout::offset_elements` are measured in elements. Byte ranges
  are computed only after applying an explicit `DType`.
- `Layout` records shape, strides, and an offset without copying storage.
  Slicing and transposition only transform metadata.
- `Layout::reshape` is zero-copy and accepts only equal element counts plus a
  canonical contiguous source (an unchanged shape is a metadata no-op). Empty
  layouts have no element order and may be reshaped to another empty shape.
- `TensorView<'a, B>` stores a real `&'a B`; `TensorViewMut<'a, B>` stores a real
  `&'a mut B` and is not cloneable. Views therefore cannot outlive their backing
  owner. Mutable construction also requires a conservatively proven
  non-overlapping layout, and mutable metadata transforms consume the view. A
  generic mutable view does not expose `&mut B`, which prevents resizing or
  closing a backing owner while validated metadata is live; host-slice views
  expose only non-resizable `&mut [u8]` bytes.
- `TensorStorage` is a sealed capacity contract. Host bytes are always
  supported. With the `cuda` feature, opaque `CudaDeviceBuffer` and
  `CudaPinnedHostBuffer` owners also implement it without exposing a raw
  pointer.
- `Workspace<B>` owns an explicitly supplied backing and never allocates,
  resizes, or clones it. Its views borrow that owner directly.

The Rust borrow anchors host-side ownership. CUDA copies additionally return a
pending token that exclusively borrows the originating stream and both buffers
until completion; native active-use state keeps forgotten tokens fail-closed.
Tensor metadata itself never launches work or claims that a host borrow proves
device completion.

## Explicit capability failures

This crate never casts dtype and never creates an implicit contiguous copy.
Operations declare requirements with `require_dtype` and
`require_contiguous`; mismatches return `TensorError::DTypeMismatch` or
`TensorError::NonContiguousLayout`. A planner or caller must schedule an
explicit conversion/materialization step in a later layer.

The initial canonical row-major layouts are:

```text
Hidden: [batch, sequence, hidden]
Q/K/V:  [batch, heads, sequence, head_dim]
```

Checkpoint weight layout and any later execution-packed weight layout remain
distinct metadata contracts; this crate does not silently translate between
them.
