import os
import pytest
from ase.io import read
from mlip_platform.core.md import run_md


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

    run_md(atoms, str(log_file), temperature_K=300, steps=10)

    assert os.path.exists(log_file)
    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0
