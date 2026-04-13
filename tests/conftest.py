import pathlib
import pytest
import qpretrieve

import xpunwrap

data_path = pathlib.Path(__file__).parent / "data"


@pytest.fixture(params=["cupy", "numpy"])
def backend(request):
    backend_name = request.param
    if backend_name == "cupy":
        try:
            __import__("cupy")
        except Exception:
            pytest.skip("cupy not installed")
    xpunwrap.set_ndarray_backend(backend_name)
    yield backend_name
    # always change back to numpy for cleanup
    xpunwrap.set_ndarray_backend("numpy")


@pytest.fixture
def fake_phase_data(backend):
    xp = xpunwrap.get_ndarray_backend()
    rng = xp.random.default_rng(7)
    phase_stack = rng.uniform(
        -xp.pi, xp.pi, size=(5, 32, 32)
    ).astype(xp.float32)
    phase_single = phase_stack[0]
    return {
        "stack": phase_stack,
        "single": phase_single,
    }


@pytest.fixture
def cell_phase_data(backend):
    xp = xpunwrap.get_ndarray_backend()
    qpretrieve.set_ndarray_backend(backend)

    edata = xp.load(data_path / "hologram_cell.npz")
    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    bg = qpretrieve.OffAxisHologram(data=edata["bg_data"])
    holo.run_pipeline(filter_name="disk", filter_size=1 / 2,
                      scale_to_filter=True)
    bg.process_like(holo)
    phase_wrp = xp.asarray(holo.phase - bg.phase).astype(xp.float32)
    # make it a stack
    phase_wrp = xp.repeat(phase_wrp, repeats=10, axis=0)
    return phase_wrp
