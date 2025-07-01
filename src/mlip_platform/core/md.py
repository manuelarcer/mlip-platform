from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units

def initialize_md(atoms, temperature_K=300, timestep_fs=2.0):
    """Initialize atoms with a velocity distribution and MD integrator."""
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    return dyn

def attach_logger(dyn, atoms, log_path, interval=5, stress=False):
    """Attach a logger to the MD run."""
    logger = MDLogger(dyn, atoms, log_path, header=True, stress=stress)
    dyn.attach(logger, interval=interval)

def run_md(atoms, log_path, temperature_K=300, timestep_fs=2.0, steps=10, interval=5, stress=False):
    """Run an MD simulation with logging."""
    dyn = initialize_md(atoms, temperature_K=temperature_K, timestep_fs=timestep_fs)
    attach_logger(dyn, atoms, log_path, interval=interval, stress=stress)
    dyn.run(steps)
    return dyn
