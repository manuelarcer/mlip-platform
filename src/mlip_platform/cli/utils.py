"""Shared utilities for CLI commands."""
import typer

# Import MLIPs with availability tracking
try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    FAIRCHEM_AVAILABLE = True
except ImportError:
    pretrained_mlip = None
    FAIRChemCalculator = None
    FAIRCHEM_AVAILABLE = False

try:
    from sevenn.calculator import SevenNetCalculator
    SEVENN_AVAILABLE = True
except ImportError:
    SevenNetCalculator = None
    SEVENN_AVAILABLE = False

try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    mace_mp = None
    MACE_AVAILABLE = False


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
    if mlip == "mace":
        atoms.calc = mace_mp(model="medium", device="cpu")
    elif mlip == "7net-mf-ompa":
        atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")
    elif mlip.startswith("uma-"):
        predictor = pretrained_mlip.get_predict_unit(mlip, device="cpu")
        atoms.calc = FAIRChemCalculator(predictor, task_name=uma_task)

    return atoms
