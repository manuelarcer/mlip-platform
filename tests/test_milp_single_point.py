import os
import pytest
from ase.io import read

# Try importing MACE
try:
    from mace.calculators import mace_mp
    mace_available = True
except ImportError:
    mace_available = False

# Try importing SevenNet
try:
    from sevenn.calculator import SevenNetCalculator
    sevenn_available = True
except ImportError:
    sevenn_available = False

if not mace_available:
    print("[MACE] Skipped: MACE not available in this environment")

if not sevenn_available:
    print("[SEVENN] Skipped: SevenNet not available in this environment")


@pytest.mark.skipif(not mace_available, reason="MACE not installed")
def test_mlip_single_point_mace():
    print("\n[MACE] Running test_mlip_single_point_mace")
    poscar_path = os.path.join("test", "POSCAR")
    atoms = read(poscar_path, format="vasp")
    atoms.calc = mace_mp(model="medium", device="cpu")
    energy = atoms.get_potential_energy()
    print(f"[MACE] Energy: {energy}")
    assert isinstance(energy, float), "MACE: Energy is not a float"

@pytest.mark.skipif(not sevenn_available, reason="SevenNet not installed")
def test_mlip_single_point_sevenn():
    print("\n[SEVENN] Running test_mlip_single_point_sevenn")
    poscar_path = os.path.join("test", "POSCAR")
    atoms = read(poscar_path, format="vasp")
    atoms.calc = SevenNetCalculator(model="7net-mf-ompa", modal="mpa")
    energy = atoms.get_potential_energy()
    print(f"[SEVENN] Energy: {energy}")
    assert isinstance(energy, float), "SevenNet: Energy is not a float"
