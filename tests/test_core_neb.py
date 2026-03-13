"""Tests for mlip_platform.core.neb using EMT calculator."""
import pytest
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT

from mlip_platform.core.neb import CustomNEB


def _make_neb_pair():
    """Create a simple initial/final pair for NEB testing."""
    initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    final = initial.copy()
    pos = final.get_positions()
    pos[0] += np.array([0.3, 0.3, 0.0])
    final.set_positions(pos)
    return initial, final


class TestCustomNEBInit:
    def test_image_count(self, tmp_workdir):
        initial, final = _make_neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_workdir,
        )
        assert len(neb.images) == 5  # 3 intermediate + 2 endpoints

    def test_default_images(self, tmp_workdir):
        initial, final = _make_neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=5,
            mlip="test", output_dir=tmp_workdir,
        )
        assert len(neb.images) == 7  # 5 + 2


class TestConstraints:
    def test_constraints_applied(self, tmp_workdir):
        initial, final = _make_neb_pair()
        relax = [0, 1]
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_workdir,
            relax_atoms=relax,
        )

        # Check that endpoints have FixAtoms constraint
        from ase.constraints import FixAtoms
        initial_constraints = [c for c in neb.initial.constraints if isinstance(c, FixAtoms)]
        assert len(initial_constraints) == 1

        # Fixed atoms should be all except relax atoms
        num_atoms = len(initial)
        fixed = initial_constraints[0].get_indices()
        expected_fixed = sorted(set(range(num_atoms)) - set(relax))
        assert sorted(fixed) == expected_fixed

    def test_idpp_skipped_with_constraints(self, tmp_workdir):
        initial, final = _make_neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_workdir,
            relax_atoms=[0],
        )
        # IDPP should not create files
        neb.interpolate_idpp()
        assert not (tmp_workdir / "idpp.traj").exists()


class TestProcessResults:
    def test_dataframe_columns(self, tmp_workdir):
        initial, final = _make_neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_workdir,
        )

        # Attach EMT calculators to all images
        for img in neb.images:
            img.calc = EMT()

        df = neb.process_results()
        assert "image_index" in df.columns
        assert "energy" in df.columns
        assert "relative_energy" in df.columns
        assert len(df) == 5

    def test_relative_energy_starts_at_zero(self, tmp_workdir):
        initial, final = _make_neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_workdir,
        )
        for img in neb.images:
            img.calc = EMT()

        df = neb.process_results()
        assert df["relative_energy"].iloc[0] == pytest.approx(0.0)


class TestExportPoscars:
    def test_creates_directories(self, tmp_workdir):
        initial, final = _make_neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp_workdir,
        )
        neb.export_poscars()

        for i in range(5):
            poscar = tmp_workdir / f"{i:02d}" / "POSCAR"
            assert poscar.exists(), f"POSCAR missing for image {i}"
