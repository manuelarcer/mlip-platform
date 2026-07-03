"""Deterministic input builders shared by golden generation and regression tests.

Every structure is fully seeded. If you change ANYTHING here, every golden
derived from it is invalidated -- and regenerating goldens is a human
decision (see tests/goldens/). Do not edit casually.
"""
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.cluster import Icosahedron
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

GOLDEN_DIR = Path(__file__).parent / "goldens"


def make_opt_atoms():
    """Cu fcc 2x2x2 supercell, rattled off equilibrium with a fixed seed."""
    atoms = bulk("Cu", "fcc", a=3.7) * (2, 2, 2)
    atoms.rattle(stdev=0.05, seed=42)
    atoms.calc = EMT()
    return atoms


def make_md_atoms():
    """Cu fcc 2x2x2 supercell with seeded 300 K velocities for NVE MD.

    NVE is required for determinism: run_md/setup_dynamics only reassign
    velocities for nvt/npt ensembles, so pre-seeded momenta survive.
    """
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.RandomState(42)
    )
    atoms.calc = EMT()
    return atoms


def make_neb_pair():
    """Initial/final pair for NEB: one atom displaced by (0.3, 0.3, 0) Angstrom.

    Mirrors tests/test_core_neb.py::_make_neb_pair. No calculators attached;
    callers attach EMT per image.
    """
    initial = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    final = initial.copy()
    pos = final.get_positions()
    pos[0] += np.array([0.3, 0.3, 0.0])
    final.set_positions(pos)
    return initial, final


def make_cluster():
    """Isolated Cu13 icosahedron in vacuum -- rotatable without PBC issues."""
    atoms = Icosahedron("Cu", noshells=2)
    atoms.center(vacuum=8.0)
    atoms.pbc = False
    atoms.calc = EMT()
    return atoms
