import os
import pytest
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units

# Try importing MACE
try:
    from mace.calculators import mace_mp
    mace_available = True
except ImportError:
    mace_available = False

@pytest.mark.skipif(not mace_available, reason="MACE not installed in this environment")
def test_mace_md(tmp_path):
    structure_path = "tests/fixtures/structures/fragment_initial.vasp"  
    log_file = tmp_path / "mace_md.log"

    atoms = read(structure_path)
    atoms.calc = mace_mp(model="medium", device="cpu")

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = VelocityVerlet(atoms, timestep=2.0 * units.fs)
    dyn.attach(MDLogger(dyn, atoms, str(log_file), header=True, stress=False), interval=5)
    dyn.run(10)

    assert os.path.exists(log_file)
    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0
