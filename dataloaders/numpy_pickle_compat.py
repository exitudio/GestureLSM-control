import sys

import numpy as np


def install_numpy_core_aliases():
    """Allow NumPy 1.x to read pickles created by NumPy 2.x."""
    if not hasattr(np, "core"):
        return
    sys.modules.setdefault("numpy._core", np.core)
    for name in ("multiarray", "numeric", "umath", "fromnumeric"):
        if hasattr(np.core, name):
            sys.modules.setdefault(f"numpy._core.{name}", getattr(np.core, name))
