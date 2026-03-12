from unwrap_phase_gpu.algorithms import algo_tvl1

from tests.helper_methods import assert_shape_dtype


def test_tvl1_stack(fake_phase_data):
    phase_stack = fake_phase_data["stack"]
    out = algo_tvl1(
        phase_stack,
        axis=0,
        n_iters=5,
        tau=0.2,
        sigma=0.2,
    )
    assert_shape_dtype(out, phase_stack.shape)


def test_tvl1_2d(fake_phase_data):
    phase_single = fake_phase_data["single"]
    out = algo_tvl1(
        phase_single,
        n_iters=4,
        tau=0.25,
        sigma=0.2,
    )
    assert_shape_dtype(out, phase_single.shape)
