"""Characterization (golden) regression tests.

Frozen baselines live in tests/goldens/. If any assertion here fails:
REPORT the numerical delta (quantity, magnitude, input) -- do NOT
regenerate the golden. Updating baselines is a human decision.
"""
import json

import numpy as np
import pandas as pd
import pytest
from ase.calculators.emt import EMT
from ase.io import read

from golden_inputs import (
    GOLDEN_DIR, make_md_atoms, make_neb_pair, make_opt_atoms,
)
from mlip_platform.core.md import run_md
from mlip_platform.core.neb import CustomNEB
from mlip_platform.core.optimize import run_optimization
from mlip_platform.core.utils import calc_fmax

# Per-quantity tolerances: energies and forces are not the same scale, and
# optimizer/integrator path-dependence amplifies float noise in positions.
E_RTOL, E_ATOL = 1e-9, 1e-8      # energies (eV)
POS_ATOL = 1e-6                   # positions (Angstrom)
F_RTOL, F_ATOL = 1e-9, 1e-6      # forces / fmax (eV/Angstrom)


def _load(name):
    return json.loads((GOLDEN_DIR / name).read_text())


class TestOptimizeGolden:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        golden = _load("optimize_cu_rattled.json")
        tmp = tmp_path_factory.mktemp("opt")
        atoms = make_opt_atoms()
        converged = run_optimization(
            atoms, optimizer="bfgs", fmax=0.02, max_steps=200,
            output_dir=tmp, verbose=False,
        )
        df = pd.read_csv(tmp / "opt_convergence.csv")
        final = read(tmp / "opt_final.vasp")
        return golden, converged, df, final

    def test_converged_flag(self, result):
        golden, converged, _, _ = result
        assert converged == golden["converged"]

    def test_final_energy(self, result):
        golden, _, df, _ = result
        np.testing.assert_allclose(
            df["energy(eV)"].iloc[-1], golden["final_energy_eV"],
            rtol=E_RTOL, atol=E_ATOL,
        )

    def test_final_fmax(self, result):
        golden, _, df, _ = result
        np.testing.assert_allclose(
            df["fmax(eV/A)"].iloc[-1], golden["final_fmax_eV_A"],
            rtol=F_RTOL, atol=F_ATOL,
        )

    def test_final_positions(self, result):
        golden, _, _, final = result
        np.testing.assert_allclose(
            final.get_positions(), np.array(golden["final_positions_A"]),
            rtol=0, atol=POS_ATOL,
        )

    def test_step_count(self, result):
        golden, _, df, _ = result
        assert len(df) == golden["n_csv_rows"]


class TestMDGolden:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        golden = _load("md_nve_cu.json")
        tmp = tmp_path_factory.mktemp("md")
        atoms = make_md_atoms()
        run_md(
            atoms, ensemble="nve", timestep=1.0, steps=200,
            log_interval=10, traj_interval=100, output_dir=tmp,
        )
        return golden, pd.read_csv(tmp / "md_energy.csv"), atoms

    def test_final_total_energy(self, result):
        golden, df, _ = result
        np.testing.assert_allclose(
            df["total_energy(eV)"].iloc[-1], golden["final_total_energy_eV"],
            rtol=E_RTOL, atol=E_ATOL,
        )

    def test_final_potential_energy(self, result):
        golden, df, _ = result
        np.testing.assert_allclose(
            df["potential_energy(eV)"].iloc[-1],
            golden["final_potential_energy_eV"],
            rtol=E_RTOL, atol=E_ATOL,
        )

    def test_final_kinetic_energy(self, result):
        golden, df, _ = result
        np.testing.assert_allclose(
            df["kinetic_energy(eV)"].iloc[-1],
            golden["final_kinetic_energy_eV"],
            rtol=E_RTOL, atol=E_ATOL,
        )

    def test_final_temperature(self, result):
        golden, df, _ = result
        np.testing.assert_allclose(
            df["temperature(K)"].iloc[-1], golden["final_temperature_K"],
            rtol=1e-9, atol=1e-6,
        )

    def test_final_positions(self, result):
        golden, _, atoms = result
        np.testing.assert_allclose(
            atoms.get_positions(), np.array(golden["final_positions_A"]),
            rtol=0, atol=POS_ATOL,
        )

    def test_row_count(self, result):
        golden, df, _ = result
        assert len(df) == golden["n_csv_rows"]


class TestNEBGolden:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        golden = _load("neb_idpp_profile.json")
        tmp = tmp_path_factory.mktemp("neb")
        initial, final = make_neb_pair()
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp,
        )
        neb.interpolate_idpp()
        for img in neb.images:
            img.calc = EMT()
        return golden, neb.process_results(), neb

    def test_energy_profile(self, result):
        golden, df, _ = result
        np.testing.assert_allclose(
            df["energy"].to_numpy(), np.array(golden["energies_eV"]),
            rtol=E_RTOL, atol=E_ATOL,
        )

    def test_relative_energy_profile(self, result):
        golden, df, _ = result
        np.testing.assert_allclose(
            df["relative_energy"].to_numpy(),
            np.array(golden["relative_energies_eV"]),
            rtol=E_RTOL, atol=E_ATOL,
        )

    def test_interpolated_positions(self, result):
        golden, _, neb = result
        for img, ref in zip(neb.images, golden["image_positions_A"]):
            np.testing.assert_allclose(
                img.get_positions(), np.array(ref), rtol=0, atol=POS_ATOL,
            )


class TestUtilsGolden:
    def test_calc_fmax(self):
        golden = _load("utils_calc_fmax.json")
        fmax = calc_fmax(np.array(golden["forces_input"]))
        np.testing.assert_allclose(
            fmax, golden["fmax"], rtol=F_RTOL, atol=F_ATOL,
        )
