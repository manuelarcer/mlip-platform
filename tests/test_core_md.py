"""Tests for mlip_platform.core.md."""
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen

from mlip_platform.core.md import setup_dynamics, run_md


class TestSetupDynamics:
    def _make_atoms(self):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()
        return atoms

    def test_nve(self):
        atoms = self._make_atoms()
        dyn = setup_dynamics(atoms, ensemble="nve")
        assert isinstance(dyn, VelocityVerlet)

    def test_nvt_langevin(self):
        atoms = self._make_atoms()
        dyn = setup_dynamics(atoms, ensemble="nvt", thermostat="langevin", temperature=300)
        assert isinstance(dyn, Langevin)

    def test_nvt_berendsen(self):
        atoms = self._make_atoms()
        dyn = setup_dynamics(atoms, ensemble="nvt", thermostat="berendsen", temperature=300)
        assert isinstance(dyn, NVTBerendsen)

    def test_invalid_ensemble_raises(self):
        atoms = self._make_atoms()
        with pytest.raises(ValueError, match="Unknown ensemble"):
            setup_dynamics(atoms, ensemble="invalid")

    def test_invalid_thermostat_raises(self):
        atoms = self._make_atoms()
        with pytest.raises(ValueError, match="Unknown thermostat"):
            setup_dynamics(atoms, ensemble="nvt", thermostat="invalid")

    def test_invalid_barostat_raises(self):
        atoms = self._make_atoms()
        with pytest.raises(ValueError, match="Unknown barostat"):
            setup_dynamics(atoms, ensemble="npt", barostat="invalid")


class TestRunMd:
    def test_short_nve(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()

        run_md(
            atoms, ensemble="nve", steps=5, interval=1,
            output_dir=tmp_workdir,
        )

        assert (tmp_workdir / "md.traj").exists()
        assert (tmp_workdir / "md_energy.csv").exists()
        assert (tmp_workdir / "md_energy.png").exists()
        assert (tmp_workdir / "md_temperature.png").exists()

    def test_short_nvt_langevin(self, tmp_workdir):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.calc = EMT()

        run_md(
            atoms, ensemble="nvt", thermostat="langevin",
            temperature=300, steps=5, interval=1,
            output_dir=tmp_workdir,
        )

        import pandas as pd
        df = pd.read_csv(tmp_workdir / "md_energy.csv")
        assert len(df) > 0
        assert "temperature(K)" in df.columns
