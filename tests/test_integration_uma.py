"""Integration tests requiring UMA (fairchem-core).

All tests are skipped if fairchem-core is not installed.
"""
import pytest
import numpy as np

from ase.build import bulk
from ase.io import write

pytestmark = pytest.mark.uma


@pytest.fixture
def uma_atoms(tmp_workdir):
    """Create a Cu bulk structure file and return (atoms, vasp_path)."""
    from mlip_platform.cli.utils import setup_calculator

    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    vasp_path = tmp_workdir / "POSCAR"
    write(str(vasp_path), atoms, format="vasp")
    atoms = setup_calculator(atoms, "uma-s-1p2", "omat")
    return atoms, vasp_path


class TestSinglePointUMA:
    def test_energy_is_finite(self, uma_atoms):
        atoms, _ = uma_atoms
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

    def test_forces_shape(self, uma_atoms):
        atoms, _ = uma_atoms
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)


class TestOptimizationUMA:
    def test_short_optimization(self, uma_atoms, tmp_workdir):
        from mlip_platform.core.optimize import run_optimization

        atoms, _ = uma_atoms
        converged = run_optimization(
            atoms, optimizer="bfgs", fmax=0.5, max_steps=5,
            output_dir=tmp_workdir, verbose=False,
        )
        assert bool(converged) in (True, False)
        assert (tmp_workdir / "opt.traj").exists()


class TestNEBUMA:
    def test_3_images(self, tmp_workdir):
        from mlip_platform.core.neb import CustomNEB

        initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        final = initial.copy()
        pos = final.get_positions()
        pos[0] += np.array([0.3, 0.3, 0.0])
        final.set_positions(pos)

        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="uma-s-1p2", uma_task="omat",
            output_dir=tmp_workdir,
        )
        assert len(neb.images) == 5


class TestMDUMA:
    def test_nvt_10_steps(self, uma_atoms, tmp_workdir):
        from mlip_platform.core.md import run_md

        atoms, _ = uma_atoms
        run_md(
            atoms, ensemble="nvt", thermostat="langevin",
            temperature=300, steps=10, interval=5,
            output_dir=tmp_workdir,
        )
        assert (tmp_workdir / "md.traj").exists()
        assert (tmp_workdir / "md_energy.csv").exists()
