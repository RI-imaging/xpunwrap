Using CPU or GPU
================

Import And Backend Selection
----------------------------

You can switch between NumPy (CPU) and CuPy (GPU) at runtime. The active
array backend is exposed as ``xp`` for convenient use in your own code.

NumPy backend (CPU):

.. code-block:: python

   import unwrap_phase_gpu as upg

   upg.set_ndarray_backend("numpy")
   xp = upg.get_ndarray_backend()
   xp.array(...)  # will create a numpy array that exists on the CPU

CuPy backend (GPU):

.. code-block:: python

   import unwrap_phase_gpu as upg

   upg.set_ndarray_backend("cupy")
   xp = upg.get_ndarray_backend()
   xp.array(...)  # will create a cupy array that exists on the GPU
