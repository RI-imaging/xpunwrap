import pytest

import unwrap_phase_gpu as upg


@pytest.fixture(params=["numpy", "cupy"])
def backend(request):
    backend_name = request.param
    if backend_name == "cupy":
        try:
            __import__("cupy")
        except Exception:
            pytest.skip("cupy not installed")
    upg.set_ndarray_backend(backend_name)
    yield backend_name
    # always change back to numpy for cleanup
    upg.set_ndarray_backend("numpy")


@pytest.fixture
def fake_phase_data(backend):
    xp = upg.get_ndarray_backend()
    rng = xp.random.default_rng(7)
    phase_stack = rng.uniform(
        -xp.pi, xp.pi, size=(5, 32, 32)
    ).astype(xp.float32)
    phase_single = phase_stack[0]
    return {
        "stack": phase_stack,
        "single": phase_single,
    }
