"""Geometry optimization engine using ASE."""
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, GPMin, MDMin

from mlip_platform.core.utils import calc_fmax

logger = logging.getLogger(__name__)

OPTIMIZER_MAP = {
    "fire": FIRE,
    "bfgs": BFGS,
    "lbfgs": LBFGS,
    "bfgsls": BFGSLineSearch,
    "gpmin": GPMin,
    "mdmin": MDMin,
}


def run_optimization(
    atoms,
    optimizer: str = "fire",
    fmax: float = 0.05,
    max_steps: int = 200,
    trajectory: str = "opt.traj",
    logfile: str = "opt.log",
    output_dir: str | Path = ".",
    model_name: str = "mlip",
    verbose: bool = True,
) -> bool:
    """Run geometry optimization on an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached.
    optimizer : str
        Optimizer algorithm: ``'fire'``, ``'bfgs'``, ``'lbfgs'``,
        ``'bfgsls'``, ``'gpmin'``, ``'mdmin'``.
    fmax : float
        Force convergence criterion (eV/Ang).
    max_steps : int
        Maximum number of optimization steps.
    trajectory : str or Path
        Trajectory filename.
    logfile : str or Path
        Log filename.
    output_dir : str or Path
        Directory for output files.
    model_name : str
        Name of MLIP model for parameter file.
    verbose : bool
        If True, show optimization progress table.

    Returns
    -------
    bool
        Whether optimization converged.

    Raises
    ------
    ValueError
        If ``optimizer`` is not recognised.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    traj_file = output_path / trajectory
    log_file = output_path / logfile

    logfile_stem = Path(logfile).stem
    csv_file = output_path / f"{logfile_stem}_convergence.csv"
    convergence_plot = output_path / f"{logfile_stem}_convergence.png"
    final_structure = output_path / f"{logfile_stem}_final.vasp"

    optimizer_name = optimizer.lower()
    if optimizer_name not in OPTIMIZER_MAP:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Available: {list(OPTIMIZER_MAP.keys())}"
        )

    OptimizerClass = OPTIMIZER_MAP[optimizer_name]

    traj = Trajectory(str(traj_file), "w", atoms)

    log_data = {"step": [], "energy(eV)": [], "fmax(eV/A)": []}

    def log_convergence():
        step = opt.nsteps
        energy = atoms.get_potential_energy()
        fmax_val = calc_fmax(atoms.get_forces())
        log_data["step"].append(step)
        log_data["energy(eV)"].append(energy)
        log_data["fmax(eV/A)"].append(fmax_val)

    if verbose:
        opt = OptimizerClass(atoms, trajectory=str(traj_file), logfile=str(log_file))
        opt.attach(log_convergence, interval=1)
        logger.info("Starting optimization with %s (fmax=%.4f, max_steps=%d)", optimizer.upper(), fmax, max_steps)
        converged = opt.run(fmax=fmax, steps=max_steps)
    else:
        with open(log_file, "w") as lf:
            opt = OptimizerClass(atoms, trajectory=str(traj_file), logfile=lf)
            opt.attach(log_convergence, interval=1)
            converged = opt.run(fmax=fmax, steps=max_steps)

    final_energy = atoms.get_potential_energy()
    final_fmax = calc_fmax(atoms.get_forces())

    logger.info("Optimization complete (converged=%s, steps=%d, energy=%.6f eV, fmax=%.6f eV/Ang)",
                converged, opt.nsteps, final_energy, final_fmax)

    write(str(final_structure), atoms, format="vasp")

    df = pd.DataFrame(log_data)
    df.to_csv(csv_file, index=False)

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(df["step"], df["energy(eV)"], marker="o", markersize=4, linewidth=1.5)
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title(f"Energy Convergence ({optimizer.upper()})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["step"], df["fmax(eV/A)"], marker="o", markersize=4, linewidth=1.5, color="orange")
    ax2.axhline(y=fmax, color="r", linestyle="--", label=f"fmax target = {fmax}")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Max Force (eV/Ang)")
    ax2.set_title("Force Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(convergence_plot, dpi=150)
    plt.close()

    return converged
