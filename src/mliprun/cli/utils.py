"""Shared utilities for CLI commands."""
import functools
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path

import typer
from click.core import ParameterSource

from mliprun.core.utils import resolve_device as _resolve_device


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

PLOT_HELP = (
    "Write PNG plots of the results. Off by default (plotting is opt-in): the "
    "figure/save is IO that dominates short runs, so pass --plot only when you "
    "want the figures. The CSV data is always written and can be plotted later."
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

# Base URL for the install recipes on GitHub. Update if the repo moves.
_INSTALL_DOCS_BASE = (
    "https://github.com/manuelarcer/mliprun/blob/main/docs/install"
)

# MLIP tag → recipe path (filename, optionally with #anchor). Unknown
# `uma-*` and `mace-mh-*` tags fall through to the package-level recipe
# without an anchor — see `_recipe_for_tag`.
_TAG_TO_RECIPE = {
    "mace": "mace.md#mace",
    "mace-mh-0": "mace.md#mace-mh-0",
    "mace-mh-1": "mace.md#mace-mh-1",
    "uma-s-1p2": "uma.md#uma-s-1p2",
    "7net-mf-ompa": "sevenn.md",
    "chgnet": "chgnet.md",
}


def _recipe_for_tag(mlip: str) -> str:
    """Return the recipe path for an MLIP tag (file, optionally with anchor).

    Returns an empty string for unrecognised tags.
    """
    if mlip in _TAG_TO_RECIPE:
        return _TAG_TO_RECIPE[mlip]
    if mlip.startswith("uma-"):
        return "uma.md"
    if mlip.startswith("mace-mh-"):
        return "mace.md"
    return ""


def _install_message(label: str, mlip: str) -> str:
    """Format a 'not available' error pointing at the install recipe.

    Always prints the absolute GitHub URL. Additionally prints the
    repo-relative local path if the recipe file is on disk (source
    checkouts / editable installs).
    """
    recipe = _recipe_for_tag(mlip)
    if not recipe:
        return (
            f"{label} not available, and no install recipe is registered for "
            f"tag '{mlip}'. Browse recipes: {_INSTALL_DOCS_BASE}/README.md"
        )

    recipe_file = recipe.split("#", 1)[0]
    url = f"{_INSTALL_DOCS_BASE}/{recipe}"
    repo_root = Path(__file__).resolve().parents[3]
    on_disk = (repo_root / "docs" / "install" / recipe_file).exists()

    msg = f"{label} not available. See install recipe:\n  Online: {url}"
    if on_disk:
        msg += f"\n  Local:  docs/install/{recipe}"
    return msg


def _no_mlip_message() -> str:
    """Message printed by `detect_mlip` when nothing is installed."""
    url = f"{_INSTALL_DOCS_BASE}/README.md"
    repo_root = Path(__file__).resolve().parents[3]
    on_disk = (repo_root / "docs" / "install" / "README.md").exists()

    msg = (
        "No supported MLIP installed. Pick one and follow its install recipe "
        "(each MLIP needs its own env — see ADR 0001):\n"
        f"  Online: {url}"
    )
    if on_disk:
        msg += "\n  Local:  docs/install/README.md"
    return msg


def detect_mlip() -> str:
    """Detect available MLIP model in order of preference: UMA > MACE > SevenNet > CHGNet.

    UMA is preferred when installed (the most accurate foundation model
    available here), but it is gated on Hugging Face and unusable without first
    requesting access. MACE is therefore placed ahead of SevenNet/CHGNet as the
    readily-usable fallback: a fresh environment with ``pip install mace-torch``
    lands on a working model without any access request.

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
    elif MACE_AVAILABLE:
        return "mace"
    elif SEVENN_AVAILABLE:
        return "7net-mf-ompa"
    elif CHGNET_AVAILABLE:
        return "chgnet"
    else:
        raise typer.Exit(_no_mlip_message())


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
        raise typer.Exit(_install_message("MACE", mlip))
    elif mlip == "7net-mf-ompa" and not SEVENN_AVAILABLE:
        raise typer.Exit(_install_message("SevenNet", mlip))
    elif mlip.startswith("uma-") and not FAIRCHEM_AVAILABLE:
        raise typer.Exit(_install_message("UMA", mlip))
    elif mlip == "chgnet" and not CHGNET_AVAILABLE:
        raise typer.Exit(_install_message("CHGNet", mlip))
    elif mlip.startswith("mace-mh-") and not MACE_AVAILABLE:
        raise typer.Exit(_install_message("MACE", mlip))
    elif not (mlip in ["mace", "7net-mf-ompa", "chgnet"]
              or mlip.startswith("uma-")
              or mlip.startswith("mace-mh-")):
        raise typer.Exit(
            f"Unknown MLIP: {mlip}. Use any 'uma-*' tag (e.g. 'uma-s-1p2'), "
            f"'mace', 'mace-mh-1', '7net-mf-ompa', or 'chgnet'. "
            f"See {_INSTALL_DOCS_BASE}/README.md for install recipes."
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


def build_calculator(mlip: str, uma_task: str = "omat",
                     device: str = "auto", mace_head: str = "omat_pbe"):
    """Build and return an ASE calculator for the given MLIP choice.

    This is the expensive step: it loads the model weights into memory (and,
    for GPU runs, onto the device). Building it once and reusing the returned
    object across many structures avoids paying the load cost repeatedly --
    see :func:`setup_calculator`, which builds and attaches in one call.

    Parameters
    ----------
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
    ase.calculators.calculator.Calculator
        The ready-to-use ASE calculator. Assign it to ``atoms.calc``.
    """
    device = _resolve_device(device)

    if mlip == "mace":
        mace_mp = _load_mace_mp()
        return mace_mp(model="medium", device=device)

    elif mlip.startswith("mace-mh-"):
        from mace.calculators import MACECalculator
        ckpt = _ensure_mace_foundation_checkpoint(mlip)
        return MACECalculator(model_paths=ckpt, device=device, head=mace_head)

    elif mlip == "7net-mf-ompa":
        SevenNetCalculator = _load_sevenn_calculator()
        return SevenNetCalculator("7net-mf-ompa", modal="mpa", device=device)

    elif mlip.startswith("uma-"):
        pretrained_mlip, FAIRChemCalculator = _load_fairchem()
        predictor = pretrained_mlip.get_predict_unit(mlip, device=device)
        return FAIRChemCalculator(predictor, task_name=uma_task)

    elif mlip == "chgnet":
        CHGNetCalculator = _load_chgnet_calculator()
        return CHGNetCalculator(use_device=device)

    return None


def setup_calculator(atoms, mlip: str, uma_task: str = "omat",
                     device: str = "auto", mace_head: str = "omat_pbe"):
    """Attach a freshly built calculator to ``atoms`` based on MLIP choice.

    Convenience wrapper that builds the calculator (see
    :func:`build_calculator`) and attaches it. For a series of structures
    sharing one model, call :func:`build_calculator` once and assign the
    returned object to each ``atoms.calc`` instead, to load the model only
    once.

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
        Head name for multi-head MACE foundation models (mace-mh-*).

    Returns
    -------
    ase.Atoms
        Atoms object with calculator attached.
    """
    atoms.calc = build_calculator(mlip, uma_task, device=device,
                                  mace_head=mace_head)
    return atoms


# ---------------------------------------------------------------------------
# Parameter provenance (for the run record)
# ---------------------------------------------------------------------------

#: Click's provenance vocabulary -> the record's. DEFAULT_MAP means a config
#: file supplied the value; we report it as "user" because a human wrote it.
_SOURCE_LABELS = {
    ParameterSource.COMMANDLINE: "user",
    ParameterSource.ENVIRONMENT: "env",
    ParameterSource.PROMPT: "prompt",
    ParameterSource.DEFAULT: "default",
    ParameterSource.DEFAULT_MAP: "user",
}


def param_sources_from_ctx(ctx) -> dict:
    """Map each CLI parameter name to where its value came from.

    Returns an empty dict when no context is available, which the record
    module then reports as ``unspecified`` rather than guessing.
    """
    if ctx is None:
        return {}
    sources = {}
    for name in getattr(ctx, "params", {}):
        try:
            src = ctx.get_parameter_source(name)
        except Exception:  # noqa: BLE001 -- provenance is best-effort
            continue
        if src is None:
            continue
        label = _SOURCE_LABELS.get(src)
        if label is None:
            continue
        sources[name] = label
    return sources
