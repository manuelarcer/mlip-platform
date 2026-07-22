"""Shared utilities for core simulation modules."""
import numpy as np

# Conversion constant: 1 GPa = 0.006241509 eV/Ang^3
GPA_TO_EV_PER_ANG3 = 0.006241509


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if a CUDA device is present, else ``"cpu"``.

    A passed-through ``"cuda"`` or ``"cpu"`` is returned unchanged. Lives in
    core because the run record needs the resolved value and core must not
    import from the CLI layer.
    """
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def calc_fmax(forces) -> float:
    """Calculate maximum force magnitude from a forces array.

    Parameters
    ----------
    forces : array-like, shape (N, 3)
        Forces on each atom.

    Returns
    -------
    float
        Maximum force magnitude across all atoms.
    """
    forces = np.asarray(forces)
    return float(np.sqrt((forces**2).sum(axis=1).max()))
