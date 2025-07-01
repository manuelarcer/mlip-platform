import sys
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units

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

def run_md(atoms, log_path=None, temperature_K=300, timestep_fs=2.0, steps=10, interval=5, stress=False):
    """Run an MD simulation with logging."""
    dyn = initialize_md(atoms, temperature_K=temperature_K, timestep_fs=timestep_fs)
    attach_logger(dyn, atoms, log_path=log_path, interval=interval, stress=stress)
    dyn.run(steps)
    return dyn
