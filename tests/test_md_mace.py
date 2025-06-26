# test_md_functionality_mace.py

import os
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units
from mace.calculators import mace_mp

def test_mace_md(tmp_path):
    # Path to your structure 
    structure_path = "tests/fixtures/structures/fragment_initial.vasp"  

    log_file = tmp_path / "mace_md.log"

    # Load structure
    atoms = read(structure_path)

    # Load built-in MACE model
    calc = mace_mp(model="medium", device="cpu")
    atoms.calc = calc

    # Initialize velocities at 300 K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Set up and run MD
    dyn = VelocityVerlet(atoms, timestep=2.0 * units.fs)
    dyn.attach(MDLogger(dyn, atoms, str(log_file), header=True, stress=False), interval=5)
    dyn.run(10)

    # Assert output file is created and not empty
    assert os.path.exists(log_file)
    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0
