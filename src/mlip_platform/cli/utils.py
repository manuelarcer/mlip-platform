"""Shared utilities for CLI commands."""
import functools
from importlib.metadata import distribution, PackageNotFoundError

import typer


# ---------------------------------------------------------------------------
# CLI help strings (single source of truth — imported by every command)
# ---------------------------------------------------------------------------

MLIP_HELP = (
    "MLIP model. Use 'auto' (default) to auto-detect, or pass a tag explicitly: "
    "any 'uma-*' name (e.g. 'uma-s-1p2', the current default), 'mace' "
    "(MACE-MP-0 medium), 'mace-mh-1' (multi-head foundation; requires "
    "--mace-head), '7net-mf-ompa', or 'chgnet'. Any tag starting with 'uma-' "
    "is forwarded to FAIRChem unchanged."
)

MACE_HEAD_HELP = (
    "Head selection for multi-head MACE foundation models (mace-mh-*). One of "
    "'omat_pbe' (default; PBE bulk inorganic), 'oc20_usemppbe' (catalysis on "
    "surfaces), 'matpes_r2scan' (r2SCAN materials), 'mp_pbe_refit_add', "
    "'omol' (molecules), 'spice_wB97M' (organic). Ignored for non-MH MACE."
)

UMA_TASK_HELP = (
    "UMA task head: 'omat' (default; bulk inorganic materials), 'oc20' "
    "(catalysis on surfaces), 'omol' (molecules), or 'odac' (ODAC dataset). "
    "Ignored for non-UMA models."
)

DEVICE_HELP = (
    "Compute device: 'auto' (default; cuda if available, else cpu), "
    "'cuda' (force GPU), or 'cpu' (force CPU). For multi-GPU nodes, set "
    "CUDA_VISIBLE_DEVICES to pick which GPU."
)


# ---------------------------------------------------------------------------
# Package availability checks (lightweight, no heavy imports)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _check_fairchem() -> bool:
    """Check if fairchem-core is available without importing it."""
    try:
        distribution("fairchem-core")
        return True
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def _check_sevenn() -> bool:
    """Check if sevenn is available without importing it."""
    try:
        distribution("sevenn")
        return True
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def _check_mace() -> bool:
    """Check if mace is available without importing it."""
    try:
        distribution("mace-torch")
        return True
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def _check_chgnet() -> bool:
    """Check if chgnet is available without importing it."""
    try:
        distribution("chgnet")
        return True
    except PackageNotFoundError:
        return False


# Module-level convenience flags (still evaluated once at import)
FAIRCHEM_AVAILABLE = _check_fairchem()
SEVENN_AVAILABLE = _check_sevenn()
MACE_AVAILABLE = _check_mace()
CHGNET_AVAILABLE = _check_chgnet()


# ---------------------------------------------------------------------------
# MLIP detection / validation
# ---------------------------------------------------------------------------

def detect_mlip() -> str:
    """Detect available MLIP model in order of preference: UMA > SevenNet > MACE > CHGNet.

    Returns
    -------
    str
        Name of detected MLIP model.

    Raises
    ------
    typer.Exit
        If no supported MLIP is found.
    """
    if FAIRCHEM_AVAILABLE:
        return "uma-s-1p2"
    elif SEVENN_AVAILABLE:
        return "7net-mf-ompa"
    elif MACE_AVAILABLE:
        return "mace"
    elif CHGNET_AVAILABLE:
        return "chgnet"
    else:
        raise typer.Exit(
            "No supported MLIP found. Please install UMA (fairchem-core), SevenNet, MACE, or CHGNet."
        )


def validate_mlip(mlip: str) -> None:
    """Validate that the specified MLIP is available.

    Parameters
    ----------
    mlip : str
        MLIP model name to validate.

    Raises
    ------
    typer.Exit
        If the specified MLIP is not available or unknown.
    """
    if mlip == "auto":
        return

    if mlip == "mace" and not MACE_AVAILABLE:
        raise typer.Exit("MACE not available. Install with: pip install mace-torch")
    elif mlip == "7net-mf-ompa" and not SEVENN_AVAILABLE:
        raise typer.Exit("SevenNet not available. Install with: pip install sevenn")
    elif mlip.startswith("uma-") and not FAIRCHEM_AVAILABLE:
        raise typer.Exit("UMA not available. Install with: pip install fairchem-core")
    elif mlip == "chgnet" and not CHGNET_AVAILABLE:
        raise typer.Exit("CHGNet not available. Install with: pip install chgnet")
    elif mlip.startswith("mace-mh-") and not MACE_AVAILABLE:
        raise typer.Exit("MACE not available. Install with: pip install mace-torch")
    elif not (mlip in ["mace", "7net-mf-ompa", "chgnet"]
              or mlip.startswith("uma-")
              or mlip.startswith("mace-mh-")):
        raise typer.Exit(
            f"Unknown MLIP: {mlip}. Use any 'uma-*' tag (e.g. 'uma-s-1p2'), "
            f"'mace', '7net-mf-ompa', or 'chgnet'."
        )


def resolve_mlip(mlip: str) -> str:
    """Detect or validate MLIP and echo the result.

    Combines detect + validate + user echo into a single helper so every
    CLI command doesn't repeat the same pattern.

    Parameters
    ----------
    mlip : str
        MLIP model name or ``"auto"`` for auto-detection.

    Returns
    -------
    str
        Resolved MLIP model name.
    """
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"Auto-detected MLIP: {mlip}")
    else:
        validate_mlip(mlip)
        typer.echo(f"Using MLIP: {mlip}")
    return mlip


# ---------------------------------------------------------------------------
# Relax-atoms parsing
# ---------------------------------------------------------------------------

def parse_relax_atoms(relax_atoms_str: str, num_atoms: int) -> list[int]:
    """Parse a comma-separated atom index string and validate.

    Parameters
    ----------
    relax_atoms_str : str
        Comma-separated list of atom indices (e.g. ``"0,1,5"``).
    num_atoms : int
        Total number of atoms (for bounds checking).

    Returns
    -------
    list[int]
        Validated list of atom indices.

    Raises
    ------
    typer.Exit
        On invalid format or out-of-range indices.
    """
    try:
        indices = [int(i.strip()) for i in relax_atoms_str.split(",")]
    except ValueError:
        typer.echo("Error: --relax-atoms must be a comma-separated list of integers.")
        raise typer.Exit(code=1)

    invalid = [i for i in indices if i < 0 or i >= num_atoms]
    if invalid:
        typer.echo(f"Error: Invalid atom indices {invalid}. Must be between 0 and {num_atoms - 1}.")
        raise typer.Exit(code=1)

    return indices


# ---------------------------------------------------------------------------
# Calculator setup (lazy imports via lru_cache)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_mace_mp():
    from mace.calculators import mace_mp
    return mace_mp


@functools.lru_cache(maxsize=1)
def _load_sevenn_calculator():
    from sevenn.calculator import SevenNetCalculator
    return SevenNetCalculator


@functools.lru_cache(maxsize=1)
def _load_fairchem():
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    return pretrained_mlip, FAIRChemCalculator


@functools.lru_cache(maxsize=1)
def _load_chgnet_calculator():
    from chgnet.model.dynamics import CHGNetCalculator
    return CHGNetCalculator


# MACE foundation-model tag → release URL. Values are stable GitHub release
# assets owned by ACEsuit/mace-foundations.
_MACE_FOUNDATION_URLS = {
    "mace-mh-1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    "mace-mh-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
}


def _ensure_mace_foundation_checkpoint(tag: str) -> str:
    """Return a local path to the requested MACE foundation checkpoint,
    downloading it into ``~/.cache/mace/`` if it is not already present.
    """
    import os
    import urllib.request

    cache_dir = os.path.expanduser("~/.cache/mace")
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, f"{tag}.model")
    if os.path.exists(target):
        return target

    url = _MACE_FOUNDATION_URLS.get(tag)
    if url is None:
        raise typer.Exit(f"Unknown MACE foundation tag: {tag}")

    typer.echo(f"Downloading {tag} from {url} → {target}")
    urllib.request.urlretrieve(url, target)
    return target


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if a CUDA device is present, else ``"cpu"``.

    A passed-through ``"cuda"`` or ``"cpu"`` is returned unchanged.
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


def setup_calculator(atoms, mlip: str, uma_task: str = "omat",
                      device: str = "auto", mace_head: str = "omat_pbe"):
    """Attach calculator to atoms object based on MLIP choice.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to attach calculator to.
    mlip : str
        MLIP model name.
    uma_task : str, optional
        Task name for UMA models (default: ``"omat"``).
    device : str, optional
        Compute device: ``"auto"`` (default; cuda if available else cpu),
        ``"cuda"``, or ``"cpu"``.
    mace_head : str, optional
        Head name for multi-head MACE foundation models (mace-mh-*). Only
        used when ``mlip`` matches one of those tags. Default ``"omat_pbe"``
        (PBE bulk inorganic). Use ``"oc20_usemppbe"`` for catalysis on
        surfaces.

    Returns
    -------
    ase.Atoms
        Atoms object with calculator attached.
    """
    device = _resolve_device(device)

    if mlip == "mace":
        mace_mp = _load_mace_mp()
        atoms.calc = mace_mp(model="medium", device=device)

    elif mlip.startswith("mace-mh-"):
        from mace.calculators import MACECalculator
        ckpt = _ensure_mace_foundation_checkpoint(mlip)
        atoms.calc = MACECalculator(model_paths=ckpt, device=device,
                                     head=mace_head)

    elif mlip == "7net-mf-ompa":
        SevenNetCalculator = _load_sevenn_calculator()
        atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa", device=device)

    elif mlip.startswith("uma-"):
        pretrained_mlip, FAIRChemCalculator = _load_fairchem()
        predictor = pretrained_mlip.get_predict_unit(mlip, device=device)
        atoms.calc = FAIRChemCalculator(predictor, task_name=uma_task)

    elif mlip == "chgnet":
        CHGNetCalculator = _load_chgnet_calculator()
        atoms.calc = CHGNetCalculator(use_device=device)

    return atoms
