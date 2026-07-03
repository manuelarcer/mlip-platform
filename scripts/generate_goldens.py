"""One-shot generation of golden reference data for characterization tests.

STANDING RULE: goldens are frozen. This script REFUSES to overwrite an
existing golden -- updating a baseline is a human decision made by deleting
the file manually first.

Run once from the repo root:
    python scripts/generate_goldens.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from ase.calculators.emt import EMT
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from golden_inputs import (  # noqa: E402
    GOLDEN_DIR, make_md_atoms, make_neb_pair, make_opt_atoms,
)
from mlip_platform.core.md import run_md  # noqa: E402
from mlip_platform.core.neb import CustomNEB  # noqa: E402
from mlip_platform.core.optimize import run_optimization  # noqa: E402
from mlip_platform.core.utils import calc_fmax  # noqa: E402


def _write(name: str, payload: dict) -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    path = GOLDEN_DIR / name
    if path.exists():
        sys.exit(
            f"REFUSING to overwrite existing golden {path}. "
            "Regenerating baselines is a human decision."
        )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {path}")


def golden_optimize() -> None:
    atoms = make_opt_atoms()
    with tempfile.TemporaryDirectory() as tmp:
        converged = run_optimization(
            atoms, optimizer="bfgs", fmax=0.02, max_steps=200,
            output_dir=tmp, verbose=False,
        )
        df = pd.read_csv(Path(tmp) / "opt_convergence.csv")
        final = read(Path(tmp) / "opt_final.vasp")
    _write("optimize_cu_rattled.json", {
        "converged": bool(converged),
        "n_csv_rows": int(len(df)),
        "final_energy_eV": float(df["energy(eV)"].iloc[-1]),
        "final_fmax_eV_A": float(df["fmax(eV/A)"].iloc[-1]),
        "final_positions_A": final.get_positions().tolist(),
        "final_cell_A": final.get_cell()[:].tolist(),
    })


def golden_md() -> None:
    atoms = make_md_atoms()
    with tempfile.TemporaryDirectory() as tmp:
        run_md(
            atoms, ensemble="nve", timestep=1.0, steps=200,
            log_interval=10, traj_interval=100, output_dir=tmp,
        )
        df = pd.read_csv(Path(tmp) / "md_energy.csv")
    _write("md_nve_cu.json", {
        "n_csv_rows": int(len(df)),
        "final_total_energy_eV": float(df["total_energy(eV)"].iloc[-1]),
        "final_potential_energy_eV": float(df["potential_energy(eV)"].iloc[-1]),
        "final_kinetic_energy_eV": float(df["kinetic_energy(eV)"].iloc[-1]),
        "final_temperature_K": float(df["temperature(K)"].iloc[-1]),
        "final_positions_A": atoms.get_positions().tolist(),
    })


def golden_neb() -> None:
    initial, final = make_neb_pair()
    with tempfile.TemporaryDirectory() as tmp:
        neb = CustomNEB(
            initial=initial, final=final, num_images=3,
            mlip="test", output_dir=tmp,
        )
        neb.interpolate_idpp()
        for img in neb.images:
            img.calc = EMT()
        df = neb.process_results()
    _write("neb_idpp_profile.json", {
        "n_images": int(len(df)),
        "energies_eV": [float(e) for e in df["energy"]],
        "relative_energies_eV": [float(e) for e in df["relative_energy"]],
        "image_positions_A": [img.get_positions().tolist() for img in neb.images],
    })


def golden_utils() -> None:
    rng = np.random.RandomState(7)
    forces = rng.normal(scale=0.5, size=(12, 3))
    _write("utils_calc_fmax.json", {
        "forces_input": forces.tolist(),
        "fmax": float(calc_fmax(forces)),
    })


if __name__ == "__main__":
    golden_optimize()
    golden_md()
    golden_neb()
    golden_utils()
    print("All goldens written. Commit tests/goldens/ -- they are now frozen.")
