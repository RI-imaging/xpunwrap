Backend (CPU or GPU) and FFT Engine
===================================

NDArray Backend Selection
-------------------------

You can switch between NumPy (CPU) and CuPy (GPU) at runtime. The active
array backend is exposed as ``xp`` for convenient use in your own code.

NumPy backend (CPU):

.. code-block:: python

   import xpunwrap

   xpunwrap.set_ndarray_backend("numpy")  # "numpy" is the default anyway
   xp = xpunwrap.get_ndarray_backend()
   xp.array(...)  # will create a numpy array that exists on the CPU

CuPy backend (GPU):

.. code-block:: python

   import xpunwrap

   xpunwrap.set_ndarray_backend("cupy")
   xp = xpunwrap.get_ndarray_backend()
   xp.array(...)  # will create a cupy array that exists on the GPU


Fourier Transform (FFT) Engine Selection
----------------------------------------

`xpUnwrap` also follows the `qpretrieve` and `nrefocus` packages by having
multiple packages available for performing Fourier transforms. These are
handled automatically in the background. See the example
:ref:`using_backends_and_fft-engines.py <example_backend_engine>` for more details.

  - For CPU NumPy and PyFFTW (optional dependency) are available.
    PyFFTW is preferred internally, but if not installed xpUnwrap will
    fall-back to NumPy.
  - For GPU CuPy is used. CuPy is an optional dependency and the
    installed CuPy version should match your CUDA version.

See the :ref:`installation section <installation>` for how to install the
optional dependencies.
