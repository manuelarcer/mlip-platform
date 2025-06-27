import os
import pytest
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units

# Try importing SevenNet
try:
    from sevenn.calculator import SevenNetCalculator
    sevenn_available = True
except ImportError:
    sevenn_available = False

@pytest.mark.skipif(not sevenn_available, reason="SevenNet not installed in this environment")
def test_seven_md(tmp_path):
    structure_path = "tests/fixtures/structures/fragment_initial.vasp"
    log_file = tmp_path / "seven_md.log"

    atoms = read(structure_path)
    atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = VelocityVerlet(atoms, timestep=2.0 * units.fs)
    dyn.attach(MDLogger(dyn, atoms, str(log_file), header=True, stress=False), interval=5)
    dyn.run(10)

    assert os.path.exists(log_file)
    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0 
