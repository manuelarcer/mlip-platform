import sys
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units

from .utils import load_calculator

def initialize_md(atoms, temperature_K=300, timestep_fs=2.0):
    """Initialize atoms with a velocity distribution and MD integrator."""
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    return dyn

def attach_logger(dyn, atoms, log_path=None, interval=5, stress=False):
    """Attach one or more loggers to the MD run."""
    # Always log to terminal
    stdout_logger = MDLogger(dyn, atoms, sys.stdout, header=True, stress=stress)
    dyn.attach(stdout_logger, interval=interval)

    # Optionally also log to file
    if log_path:
        file_logger = MDLogger(dyn, atoms, log_path, header=True, stress=stress)
        dyn.attach(file_logger, interval=interval)

def run_md(atoms, log_path=None, temperature_K=300, timestep_fs=2.0, steps=1000, model=None):
    """
    Run MD with automatic MLIP detection.

    Parameters
    ----------
    atoms : ASE Atoms
        The atomic system to simulate (must have a calculator assigned).
    log_path : str or None
        Path to write a log file (in addition to stdout).
    temperature_K : float
        Initial temperature in Kelvin.
    timestep_fs : float
        Time step in femtoseconds.
    steps : int
        Number of MD steps to run.
    model : str or None
        Optional name of the MLIP model to load; if None, defaults are used.
    """
    # Auto-detect MACE vs. SevenNet and attach the appropriate calculator
    atoms.calc = load_calculator(model)

    # Set up and run MD
    dyn = initialize_md(atoms, temperature_K, timestep_fs)
    attach_logger(dyn, atoms, log_path)
    dyn.run(steps)
