import sys
from pathlib import Path
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE, BFGS, LBFGS, BFGSLineSearch, GPMin, MDMin
import pandas as pd
import matplotlib.pyplot as plt


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
    optimizer="fire",
    fmax=0.05,
    max_steps=200,
    trajectory="opt.traj",
    logfile="opt.log",
    output_dir=".",
    model_name="mlip",
    verbose=True
):
    """
    Run geometry optimization on an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached
    optimizer : str
        Optimizer algorithm: 'fire', 'bfgs', 'lbfgs', 'bfgsls', 'gpmin', 'mdmin'
    fmax : float
        Force convergence criterion (eV/Å)
    max_steps : int
        Maximum number of optimization steps
    trajectory : str or Path
        Trajectory filename
    logfile : str or Path
        Log filename
    output_dir : str or Path
        Directory for output files
    model_name : str
        Name of MLIP model for parameter file
    verbose : bool
        If True, show optimization progress table (default: True)

    Returns
    -------
    converged : bool
        Whether optimization converged
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    traj_file = output_path / trajectory
    log_file = output_path / logfile
    csv_file = output_path / "opt_convergence.csv"
    convergence_plot = output_path / "opt_convergence.png"
    final_structure = output_path / "opt_final.vasp"

    # Select optimizer
    optimizer_name = optimizer.lower()
    if optimizer_name not in OPTIMIZER_MAP:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Available: {list(OPTIMIZER_MAP.keys())}"
        )

    OptimizerClass = OPTIMIZER_MAP[optimizer_name]

    # Set up trajectory
    traj = Trajectory(str(traj_file), 'w', atoms)

    # Set up optimizer with controlled verbosity
    # When verbose=True, logfile is a string path which makes ASE print to both stdout and file
    # When verbose=False, logfile is opened as file handle which suppresses stdout
    if verbose:
        opt = OptimizerClass(atoms, trajectory=str(traj_file), logfile=str(log_file))
        logfile_handle = None
    else:
        logfile_handle = open(log_file, 'w')
        opt = OptimizerClass(atoms, trajectory=str(traj_file), logfile=logfile_handle)

    # Data collection for convergence analysis
    log_data = {
        "step": [],
        "energy(eV)": [],
        "fmax(eV/A)": [],
    }

    def log_convergence():
        """Callback to track convergence data"""
        step = opt.nsteps
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        fmax = (forces**2).sum(axis=1).max()**0.5

        log_data["step"].append(step)
        log_data["energy(eV)"].append(energy)
        log_data["fmax(eV/A)"].append(fmax)

    # Attach convergence logger
    opt.attach(log_convergence, interval=1)

    # Run optimization
    if verbose:
        print(f"\n🚀 Starting optimization with {optimizer.upper()}")
        print(f"   fmax = {fmax} eV/Å, max_steps = {max_steps}\n")

    converged = opt.run(fmax=fmax, steps=max_steps)

    # Close logfile handle if opened
    if logfile_handle:
        logfile_handle.close()

    # Final logging
    final_energy = atoms.get_potential_energy()
    final_forces = atoms.get_forces()
    final_fmax = (final_forces**2).sum(axis=1).max()**0.5

    print(f"\n✅ Optimization complete")
    print(f"   Converged: {converged}")
    print(f"   Steps: {opt.nsteps}")
    print(f"   Final energy: {final_energy:.6f} eV")
    print(f"   Final fmax: {final_fmax:.6f} eV/Å")

    # Save final structure
    write(str(final_structure), atoms, format="vasp")

    # Save convergence data to CSV
    df = pd.DataFrame(log_data)
    df.to_csv(csv_file, index=False)

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Energy convergence
    ax1.plot(df["step"], df["energy(eV)"], marker='o', markersize=4, linewidth=1.5)
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title(f"Energy Convergence ({optimizer.upper()})")
    ax1.grid(True, alpha=0.3)

    # Force convergence
    ax2.plot(df["step"], df["fmax(eV/A)"], marker='o', markersize=4, linewidth=1.5, color='orange')
    ax2.axhline(y=fmax, color='r', linestyle='--', label=f'fmax target = {fmax}')
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Max Force (eV/Å)")
    ax2.set_title("Force Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(convergence_plot, dpi=150)
    plt.close()

    return converged
