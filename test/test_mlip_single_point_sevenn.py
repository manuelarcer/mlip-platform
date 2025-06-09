import pytest
from ase.io import read

def test_mlip_single_point_sevenn():
    atoms = read("test/POSCAR", format="vasp")

    # Only import here, so pytest doesn't crash during collection
    from sevenn.calculator import SevenNetCalculator

    atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
