# test_md_functionality_sevenn.py

import os
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units
from sevenn.calculator import SevenNetCalculator

def test_seven_md(tmp_path):
    # Path to your input structure
    structure_path = "test/fragment_initial.vasp"

    log_file = tmp_path / "seven_md.log"

    # Load structure
    atoms = read(structure_path)

    # Initialize SevenNet calculator
    calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")
    atoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Set up and run MD
    dyn = VelocityVerlet(atoms, timestep=2.0 * units.fs)
    dyn.attach(MDLogger(dyn, atoms, str(log_file), header=True, stress=False), interval=5)
    dyn.run(10)

    # Check that log file was created and has content
    assert os.path.exists(log_file)
    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0
