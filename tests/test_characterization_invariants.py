"""Physics-invariant tests (property-based, no goldens needed).

These hold regardless of code version -- they catch unit-handling and
wiring bugs that golden tests can miss.
"""
import numpy as np
import pandas as pd
from ase.calculators.emt import EMT

from golden_inputs import make_cluster, make_md_atoms, make_opt_atoms
from mliprun.core.md import run_md
from mliprun.core.optimize import run_optimization
from mliprun.core.utils import calc_fmax


class TestTranslationInvariance:
    def test_energy_invariant_under_rigid_translation(self):
        atoms = make_opt_atoms()
        e0 = atoms.get_potential_energy()
        shifted = atoms.copy()
        shifted.translate([0.123, 0.234, 0.345])
        shifted.calc = EMT()
        np.testing.assert_allclose(
            shifted.get_potential_energy(), e0, rtol=0, atol=1e-8,
        )

    def test_optimization_final_energy_invariant_under_translation(self, tmp_path):
        results = []
        for i, shift in enumerate([(0, 0, 0), (0.5, 0.25, 0.75)]):
            atoms = make_opt_atoms()
            atoms.translate(shift)
            out = tmp_path / f"run{i}"
            run_optimization(
                atoms, optimizer="bfgs", fmax=0.02, max_steps=200,
                output_dir=out, verbose=False,
            )
            df = pd.read_csv(out / "opt_convergence.csv")
            results.append(df["energy(eV)"].iloc[-1])
        np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-6)


class TestRotationInvariance:
    def test_cluster_energy_invariant_under_rotation(self):
        # Isolated cluster: rotation is a true symmetry (no PBC coupling).
        atoms = make_cluster()
        e0 = atoms.get_potential_energy()
        rotated = atoms.copy()
        rotated.rotate(30, "z", center="COM")
        rotated.calc = EMT()
        np.testing.assert_allclose(
            rotated.get_potential_energy(), e0, rtol=0, atol=1e-8,
        )

    def test_fmax_invariant_under_rotation(self):
        atoms = make_cluster()
        f0 = calc_fmax(atoms.get_forces())
        rotated = atoms.copy()
        rotated.rotate(30, "z", center="COM")
        rotated.calc = EMT()
        np.testing.assert_allclose(
            calc_fmax(rotated.get_forces()), f0, rtol=1e-9, atol=1e-8,
        )


class TestForceEnergyConsistency:
    def test_forces_match_negative_energy_gradient(self):
        """Central finite differences: F_i ~= -dE/dx_i, h = 1e-4 Angstrom."""
        atoms = make_opt_atoms()
        analytic = atoms.get_forces()
        h = 1e-4
        # Three atoms x three directions is plenty to catch sign/unit bugs.
        for atom_idx in (0, 3, 7):
            for axis in range(3):
                energies = {}
                for sign in (+1, -1):
                    probe = atoms.copy()
                    pos = probe.get_positions()
                    pos[atom_idx, axis] += sign * h
                    probe.set_positions(pos)
                    probe.calc = EMT()
                    energies[sign] = probe.get_potential_energy()
                fd_force = -(energies[+1] - energies[-1]) / (2 * h)
                np.testing.assert_allclose(
                    analytic[atom_idx, axis], fd_force,
                    rtol=2e-3, atol=1e-6,
                    err_msg=f"atom {atom_idx} axis {axis}",
                )


class TestPermutationInvariance:
    def test_energy_invariant_under_identical_atom_swap(self):
        atoms = make_opt_atoms()  # all Cu -> any swap is a symmetry
        e0 = atoms.get_potential_energy()
        order = np.arange(len(atoms))
        order[[0, 5]] = order[[5, 0]]
        swapped = atoms[order]
        swapped.calc = EMT()
        np.testing.assert_allclose(
            swapped.get_potential_energy(), e0, rtol=0, atol=1e-10,
        )

    def test_fmax_invariant_under_identical_atom_swap(self):
        atoms = make_opt_atoms()
        f0 = calc_fmax(atoms.get_forces())
        order = np.arange(len(atoms))
        order[[0, 5]] = order[[5, 0]]
        swapped = atoms[order]
        swapped.calc = EMT()
        np.testing.assert_allclose(
            calc_fmax(swapped.get_forces()), f0, rtol=1e-12, atol=1e-12,
        )


class TestNVEEnergyConservation:
    def test_total_energy_drift_bounded(self, tmp_path):
        """NVE with dt=1 fs on EMT Cu: drift must stay < 0.5 meV/atom."""
        atoms = make_md_atoms()
        run_md(
            atoms, ensemble="nve", timestep=1.0, steps=200,
            log_interval=10, traj_interval=200, output_dir=tmp_path,
        )
        df = pd.read_csv(tmp_path / "md_energy.csv")
        etot = df["total_energy(eV)"].to_numpy()
        drift_per_atom = abs(etot - etot[0]).max() / len(atoms)
        assert drift_per_atom < 5e-4, (
            f"NVE energy drift {drift_per_atom:.2e} eV/atom exceeds 5e-4"
        )
