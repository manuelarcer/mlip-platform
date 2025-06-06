from ase.io import read
from sevenn.calculator import SevenNetCalculator

def test_mlip_single_point():
    # Load a small POSCAR file for testing
    atoms = read("/Users/leeyuanzhang/Documents/mlip-platform-1/test/POSCAR", format="vasp")

    # Attach SevenNet calculator
    atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")

    # Run energy calculation
    energy = atoms.get_potential_energy()

    # Check that result is a float
    assert isinstance(energy, float)
