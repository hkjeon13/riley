use riley_tensor::{
    DType, Layout, Shape, Strides, TensorError, TensorView, TensorViewMut, Workspace,
};

#[test]
fn dtype_widths_and_capability_checks_are_explicit() {
    assert_eq!(DType::U8.size_bytes(), 1);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);

    let storage = [0_u8; 8];
    let view = TensorView::from_contiguous(&storage[..], DType::F32, shape([2])).unwrap();
    assert_eq!(
        view.require_dtype(DType::F16),
        Err(TensorError::DTypeMismatch {
            expected: DType::F16,
            actual: DType::F32,
        })
    );
    assert_eq!(view.dtype(), DType::F32);
    assert_eq!(view.storage(), &storage);
}

#[test]
fn shape_detects_overflow_but_zero_size_short_circuits_it() {
    assert_eq!(Shape::scalar().element_count(), 1);
    assert_eq!(shape([2, 3, 4]).element_count(), 24);
    assert_eq!(
        Shape::new([usize::MAX, 2]),
        Err(TensorError::ElementCountOverflow)
    );

    let empty = shape([usize::MAX, usize::MAX, 0]);
    assert!(empty.is_empty());
    assert_eq!(empty.element_count(), 0);
    let layout = Layout::contiguous(empty).unwrap();
    assert_eq!(layout.strides().elements(), &[0, 0, 0]);
    assert!(layout.is_contiguous());
}

#[test]
fn logical_byte_length_and_layout_span_overflow_are_reported() {
    let layout = Layout::contiguous(shape([usize::MAX])).unwrap();
    assert_eq!(
        layout.logical_byte_len(DType::F64),
        Err(TensorError::ByteLengthOverflow {
            elements: usize::MAX,
            element_size: 8,
        })
    );

    assert_eq!(
        Layout::new(shape([2]), Strides::new([usize::MAX]), 1),
        Err(TensorError::LayoutSpanOverflow)
    );
    assert_eq!(
        Layout::new(shape([1]), Strides::new([0]), usize::MAX),
        Err(TensorError::LayoutSpanOverflow)
    );
}

#[test]
fn zero_sized_tensor_accepts_empty_and_one_past_storage_offsets() {
    let storage = [];
    let view = TensorView::from_contiguous(&storage[..], DType::F32, shape([3, 0, 7])).unwrap();
    assert_eq!(view.logical_byte_len().unwrap(), 0);
    assert_eq!(view.storage_byte_range().unwrap(), 0..0);
    assert!(view.layout().is_non_overlapping());
    let reshaped = view.reshape(shape([0, usize::MAX])).unwrap();
    assert_eq!(reshaped.shape().dimensions(), &[0, usize::MAX]);

    let storage = [0_u8; 16];
    let one_past = Layout::new(shape([0]), Strides::new([7]), 4).unwrap();
    let view = TensorView::new(&storage[..], DType::F32, one_past).unwrap();
    assert_eq!(view.storage_byte_range().unwrap(), 16..16);

    let beyond = Layout::new(shape([0]), Strides::new([7]), 5).unwrap();
    assert_eq!(
        TensorView::new(&storage[..], DType::F32, beyond).unwrap_err(),
        TensorError::BufferTooSmall {
            required: 20,
            actual: 16,
        }
    );
}

#[test]
fn view_slice_preserves_strides_and_computes_byte_offset() {
    let storage = [0_u8; 24];
    let view = TensorView::from_contiguous(&storage[..], DType::F32, shape([2, 3])).unwrap();
    let row = view.slice(0, 1..2).unwrap();

    assert_eq!(row.shape().dimensions(), &[1, 3]);
    assert_eq!(row.layout().strides().elements(), &[3, 1]);
    assert_eq!(row.layout().offset_elements(), 3);
    assert_eq!(row.storage_byte_range().unwrap(), 12..24);
    assert_eq!(row.logical_byte_len().unwrap(), 12);

    assert_eq!(
        view.slice(1, 2..4).unwrap_err(),
        TensorError::SliceOutOfBounds {
            axis: 1,
            start: 2,
            end: 4,
            extent: 3,
        }
    );
}

#[test]
fn contiguous_reshape_is_zero_copy_and_count_checked() {
    let storage = [0_u8; 24];
    let view = TensorView::from_contiguous(&storage[..], DType::F32, shape([2, 3])).unwrap();
    let reshaped = view.reshape(shape([3, 2])).unwrap();

    assert_eq!(reshaped.shape().dimensions(), &[3, 2]);
    assert_eq!(reshaped.layout().strides().elements(), &[2, 1]);
    assert!(std::ptr::eq(
        reshaped.storage().as_ptr(),
        view.storage().as_ptr()
    ));
    assert_eq!(
        view.reshape(shape([5])).unwrap_err(),
        TensorError::ElementCountMismatch {
            source: 6,
            requested: 5,
        }
    );
}

#[test]
fn transpose_is_non_contiguous_and_never_materializes() {
    let storage = [0_u8; 24];
    let view = TensorView::from_contiguous(&storage[..], DType::F32, shape([2, 3])).unwrap();
    let transposed = view.transpose(0, 1).unwrap();

    assert_eq!(transposed.shape().dimensions(), &[3, 2]);
    assert_eq!(transposed.layout().strides().elements(), &[1, 3]);
    assert!(!transposed.layout().is_contiguous());
    assert!(transposed.layout().is_non_overlapping());
    assert_eq!(
        transposed.require_contiguous(),
        Err(TensorError::NonContiguousLayout)
    );
    assert_eq!(
        transposed.reshape(shape([6])).unwrap_err(),
        TensorError::NonContiguousReshape
    );
    assert!(std::ptr::eq(
        transposed.storage().as_ptr(),
        storage.as_ptr()
    ));
}

#[test]
fn mutable_view_rejects_broadcast_and_overlapping_layouts() {
    let mut storage = [0_u8; 12];
    let overlapping = Layout::new(shape([2, 2]), Strides::new([1, 1]), 0).unwrap();
    assert_eq!(
        TensorViewMut::new(&mut storage[..], DType::F32, overlapping).unwrap_err(),
        TensorError::MutableLayoutMayAlias
    );

    let broadcast = Layout::new(shape([3]), Strides::new([0]), 0).unwrap();
    assert_eq!(
        TensorViewMut::new(&mut storage[..4], DType::F32, broadcast).unwrap_err(),
        TensorError::MutableLayoutMayAlias
    );

    let mut padded_storage = [0_u8; 20];
    let padded = Layout::new(shape([2, 2]), Strides::new([3, 1]), 0).unwrap();
    let view = TensorViewMut::new(&mut padded_storage[..], DType::F32, padded).unwrap();
    assert!(view.layout().is_non_overlapping());
}

#[test]
fn non_overlap_proof_has_no_false_positive_for_small_rank_three_layouts() {
    use std::collections::HashSet;

    for first_extent in 1..=3 {
        for second_extent in 1..=3 {
            for third_extent in 1..=3 {
                for first_stride in 0..=5 {
                    for second_stride in 0..=5 {
                        for third_stride in 0..=5 {
                            let layout = Layout::new(
                                shape([first_extent, second_extent, third_extent]),
                                Strides::new([first_stride, second_stride, third_stride]),
                                2,
                            )
                            .unwrap();
                            if !layout.is_non_overlapping() {
                                continue;
                            }

                            let mut addresses = HashSet::new();
                            for first in 0..first_extent {
                                for second in 0..second_extent {
                                    for third in 0..third_extent {
                                        let address = 2
                                            + first * first_stride
                                            + second * second_stride
                                            + third * third_stride;
                                        assert!(
                                            addresses.insert(address),
                                            "non-overlap proof accepted duplicate address {address} for {layout:?}"
                                        );
                                    }
                                }
                            }
                            let greatest = addresses.iter().copied().max().unwrap();
                            assert_eq!(
                                layout.storage_byte_range(DType::U8).unwrap().end,
                                u64::try_from(greatest + 1).unwrap()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn mutable_transforms_consume_the_exclusive_handle() {
    let mut storage = [0_u8; 24];
    let mut row = TensorViewMut::from_contiguous(&mut storage[..], DType::F32, shape([2, 3]))
        .unwrap()
        .slice(0, 1..2)
        .unwrap();
    row.bytes_mut()[12] = 7;
    assert_eq!(row.as_view().storage_byte_range().unwrap(), 12..24);
    drop(row);
    assert_eq!(storage[12], 7);
}

#[test]
fn backing_capacity_need_not_be_a_multiple_of_dtype_width() {
    let storage = [0_u8; 5];
    let view = TensorView::from_contiguous(&storage[..], DType::F32, shape([1])).unwrap();
    assert_eq!(view.storage_byte_range().unwrap(), 0..4);

    assert_eq!(
        TensorView::from_contiguous(&storage[..], DType::F32, shape([2])).unwrap_err(),
        TensorError::BufferTooSmall {
            required: 8,
            actual: 5,
        }
    );
}

#[test]
fn workspace_owns_explicit_capacity_without_hidden_resize() {
    let mut workspace = Workspace::new(vec![0_u8; 17]);
    assert_eq!(workspace.capacity_bytes(), 17);
    workspace.storage_mut()[15] = 9;
    let layout = Layout::contiguous(shape([4])).unwrap();
    {
        let view = workspace.view_mut(DType::F32, layout.clone()).unwrap();
        assert_eq!(view.as_view().storage().as_slice()[15], 9);
    }
    assert_eq!(workspace.view(DType::F32, layout).unwrap().storage()[15], 9);
    let storage = workspace.into_inner();
    assert_eq!(storage.len(), 17);
}

fn shape<const RANK: usize>(dimensions: [usize; RANK]) -> Shape {
    Shape::new(dimensions).unwrap()
}
