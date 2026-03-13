from unwrap_phase_gpu.algorithms import algo_ls_poisson

from tests.helper_methods import assert_shape_dtype


def test_ls_poisson_stack(fake_phase_data):
    phase_stack = fake_phase_data["stack"]
    out = algo_ls_poisson(phase_stack)
    assert_shape_dtype(out, phase_stack.shape)


def test_ls_poisson_2d(fake_phase_data):
    phase_single = fake_phase_data["single"]
    out = algo_ls_poisson(phase_single)
    assert_shape_dtype(out, phase_single.shape)


def test_ls_poisson_stack_real_data(cell_phase_data):
    phase_stack = cell_phase_data
    out = algo_ls_poisson(phase_stack)
    assert_shape_dtype(out, phase_stack.shape)
