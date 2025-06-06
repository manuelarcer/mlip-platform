from ase.io import read
from mace.calculators import mace_mp

def test_mlip_single_point_mace():
    atoms = read("test/POSCAR", format="vasp")

    # Set up MACE calculator
    atoms.calc = mace_mp(model="medium", device='cpu')

    # Run energy calculation
    energy = atoms.get_potential_energy()

    # Assert output is a float
    assert isinstance(energy, float)
