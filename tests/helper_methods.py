def assert_shape_dtype(arr, shape):
    assert arr.shape == shape
    assert arr.dtype is not None
