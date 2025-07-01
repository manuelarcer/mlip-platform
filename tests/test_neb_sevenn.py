from ase.io import read
from mlip_platform.core.neb import CustomNEB
import pytest

try:
    from sevenn.calculator import SevenNetCalculator
    sevenn_available = True
except ImportError:
    sevenn_available = False

from mlip_platform.core.neb import CustomNEB

@pytest.mark.skipif(not sevenn_available, reason="SevenNet not installed")
def test_neb_run_sevenn():
    initial_path = "tests/fixtures/structures/fragment_initial.vasp"
    final_path = "tests/fixtures/structures/fragment_final.vasp"

    initial = read(initial_path, format="vasp")
    final = read(final_path, format="vasp")

    neb = CustomNEB(
        initial=initial,
        final=final,
        num_images=5,
        interp_fmax=0.1,
        interp_steps=100,
        fmax=0.05,
        mlip="7net-mf-ompa"
    )

    assert len(neb.images) == 5

    neb.interpolate_idpp()

    for img in neb.images[1:-1]:
        assert img.get_positions().shape == initial.get_positions().shape
        assert not any(p is None for p in img.get_positions().flatten())

    neb.run_neb()

    for img in neb.images:
        energy = img.get_potential_energy()
        assert isinstance(energy, float)
