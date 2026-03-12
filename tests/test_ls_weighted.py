from unwrap_phase_gpu.algorithms import algo_ls_weighted

from tests.helper_methods import assert_shape_dtype


def test_ls_weighted_stack(fake_phase_data):
    phase_stack = fake_phase_data["stack"]
    out = algo_ls_weighted(
        phase_stack,
        border_thresh=1.0,
        n_iter=5,
    )
    assert_shape_dtype(out, phase_stack.shape)


def test_ls_weighted_2d(fake_phase_data):
    phase_single = fake_phase_data["single"]
    out = algo_ls_weighted(
        phase_single,
        border_thresh=2.0,
        n_iter=3,
    )
    assert_shape_dtype(out, phase_single.shape)
