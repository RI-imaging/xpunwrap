"""
Module that controls and exposes the active ndarray backend.
NumPy is used for CPU-only. CuPy is used for GPU.
"""

import importlib
import types
from typing import Any

_default_backend = "numpy"
_xp = importlib.import_module(_default_backend)


class NDArrayBackend:
    """Proxy object exposing the current ndarray backend.

    All attribute access is forwarded to the active backend module, so
    ``xp.array``, ``xp.zeros``, etc. resolve to the correct implementation
    without any conditional imports in calling code.
    """

    def __init__(self) -> None:
        self._xp = _xp

    def get(self) -> types.ModuleType:
        """Return the currently active backend module.

        Returns
        -------
        module
            The active ndarray backend (``numpy`` or ``cupy``).
        """
        return self._xp

    def set(self, backend_name: str = "numpy") -> None:
        """Switch the active ndarray backend.

        Parameters
        ----------
        backend_name : str
            Name of the backend to activate. ``"numpy"`` selects CPU,
            ``"cupy"`` selects GPU. Default ``"numpy"``.

        Raises
        ------
        ImportError
            If the requested backend is not installed.
        """
        global _xp
        try:
            self._xp = importlib.import_module(backend_name)
            _xp = self._xp  # keep global in sync
        except ModuleNotFoundError as err:
            raise ImportError(f"The backend '{backend_name}' is not "
                              f"installed. Either install it or use the "
                              f"default backend: 'numpy'.") from err

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookup to the active backend module.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            The corresponding attribute from the active backend.
        """
        return getattr(self._xp, name)

    def is_numpy(self) -> bool:
        """Return ``True`` if the active backend is NumPy (CPU).

        Returns
        -------
        bool
        """
        return self._xp.__name__.startswith("numpy")

    def is_cupy(self) -> bool:
        """Return ``True`` if the active backend is CuPy (GPU).

        Returns
        -------
        bool
        """
        return self._xp.__name__.startswith("cupy")


# Export a single global proxy instance
xp = NDArrayBackend()
# This is what is imported by the user
get_ndarray_backend = xp.get
set_ndarray_backend = xp.set
