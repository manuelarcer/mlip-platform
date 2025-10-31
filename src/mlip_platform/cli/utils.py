"""Shared utilities for CLI commands."""
import typer

# Lazy import: Check availability without importing heavy modules
def _check_fairchem():
    """Check if fairchem-core is available without importing it."""
    try:
        from importlib.metadata import distribution
        distribution('fairchem-core')
        return True
    except Exception:
        return False

def _check_sevenn():
    """Check if sevenn is available without importing it."""
    try:
        from importlib.metadata import distribution
        distribution('sevenn')
        return True
    except Exception:
        return False

def _check_mace():
    """Check if mace is available without importing it."""
    try:
        from importlib.metadata import distribution
        distribution('mace-torch')
        return True
    except Exception:
        return False

# Check availability (lightweight, no actual imports)
FAIRCHEM_AVAILABLE = _check_fairchem()
SEVENN_AVAILABLE = _check_sevenn()
MACE_AVAILABLE = _check_mace()

# Lazy-loaded module references (loaded only when needed)
_pretrained_mlip = None
_FAIRChemCalculator = None
_SevenNetCalculator = None
_mace_mp = None


def detect_mlip() -> str:
    """
    Detect available MLIP model in order of preference: UMA > SevenNet > MACE.

    Returns
    -------
    str
        Name of detected MLIP model

    Raises
    ------
    typer.Exit
        If no supported MLIP is found
    """
    if FAIRCHEM_AVAILABLE:
        return "uma-s-1p1"
    elif SEVENN_AVAILABLE:
        return "7net-mf-ompa"
    elif MACE_AVAILABLE:
        return "mace"
    else:
        raise typer.Exit(
            "❌ No supported MLIP found. Please install UMA (fairchem-core), SevenNet, or MACE."
        )


def validate_mlip(mlip: str) -> None:
    """
    Validate that the specified MLIP is available.

    Parameters
    ----------
    mlip : str
        MLIP model name to validate

    Raises
    ------
    typer.Exit
        If the specified MLIP is not available or unknown
    """
    if mlip == "auto":
        return  # Will be detected later

    if mlip == "mace" and not MACE_AVAILABLE:
        raise typer.Exit("❌ MACE not available. Install with: pip install mace-torch")
    elif mlip == "7net-mf-ompa" and not SEVENN_AVAILABLE:
        raise typer.Exit("❌ SevenNet not available. Install with: pip install sevenn")
    elif mlip.startswith("uma-") and not FAIRCHEM_AVAILABLE:
        raise typer.Exit("❌ UMA not available. Install with: pip install fairchem-core")
    elif not (mlip in ["mace", "7net-mf-ompa"] or mlip.startswith("uma-")):
        raise typer.Exit(
            f"❌ Unknown MLIP: {mlip}. Use 'uma-s-1p1', 'uma-m-1p1', 'mace', or '7net-mf-ompa'."
        )


def setup_calculator(atoms, mlip: str, uma_task: str = "omat"):
    """
    Attach calculator to atoms object based on MLIP choice.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to attach calculator to
    mlip : str
        MLIP model name
    uma_task : str, optional
        Task name for UMA models (default: "omat")
        Options: "omat", "oc20", "omol", "odac"

    Returns
    -------
    ase.Atoms
        Atoms object with calculator attached
    """
    global _mace_mp, _SevenNetCalculator, _pretrained_mlip, _FAIRChemCalculator

    if mlip == "mace":
        # Lazy import MACE only when needed
        if _mace_mp is None:
            from mace.calculators import mace_mp as _mace_mp_import
            _mace_mp = _mace_mp_import
        atoms.calc = _mace_mp(model="medium", device="cpu")

    elif mlip == "7net-mf-ompa":
        # Lazy import SevenNet only when needed
        if _SevenNetCalculator is None:
            from sevenn.calculator import SevenNetCalculator as _SevenNetCalculator_import
            _SevenNetCalculator = _SevenNetCalculator_import
        atoms.calc = _SevenNetCalculator("7net-mf-ompa", modal="mpa")

    elif mlip.startswith("uma-"):
        # Lazy import UMA only when needed
        if _pretrained_mlip is None or _FAIRChemCalculator is None:
            from fairchem.core import pretrained_mlip as _pretrained_mlip_import
            from fairchem.core import FAIRChemCalculator as _FAIRChemCalculator_import
            _pretrained_mlip = _pretrained_mlip_import
            _FAIRChemCalculator = _FAIRChemCalculator_import
        predictor = _pretrained_mlip.get_predict_unit(mlip, device="cpu")
        atoms.calc = _FAIRChemCalculator(predictor, task_name=uma_task)

    return atoms
