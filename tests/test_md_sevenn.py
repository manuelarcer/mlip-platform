import os
import pytest
from ase.io import read
from mlip_platform.core.md import run_md


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

    run_md(atoms, str(log_file), temperature_K=300, steps=10)

    assert os.path.exists(log_file)
    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0
