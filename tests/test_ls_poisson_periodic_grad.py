import unwrap_phase_gpu as upg
from unwrap_phase_gpu.algorithms import algo_ls_poisson_periodic_grad

from tests.helper_methods import assert_shape_dtype


def test_ls_poisson_periodic_grad_2d(fake_phase_data):
    phase_single = fake_phase_data["single"]
    out = algo_ls_poisson_periodic_grad(phase_single)
    assert_shape_dtype(out, phase_single.shape)


def test_ls_poisson_periodic_grad_3d(fake_phase_data):
    phase_stack = fake_phase_data["stack"]
    out = algo_ls_poisson_periodic_grad(phase_stack)
    assert_shape_dtype(out, phase_stack.shape)


def test_ls_poisson_periodic_grad_different_shape():
    xp = upg.get_ndarray_backend()
    rng = xp.random.default_rng(11)
    phase_stack = rng.uniform(
        -xp.pi, xp.pi, size=(2, 24, 20)).astype(xp.float32)
    out = algo_ls_poisson_periodic_grad(phase_stack)
    assert_shape_dtype(out, phase_stack.shape)
