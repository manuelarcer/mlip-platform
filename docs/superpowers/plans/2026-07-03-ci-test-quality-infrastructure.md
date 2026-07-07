# CI + Test-Quality Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-push CI gate (tests + 90% diff coverage), a weekly audit/mutation workflow, and baseline characterization tests (golden references + physics invariants) for the four public entry points of `mlip_platform`.

**Architecture:** Two independent draft PRs off `main`. PR 1 (`ci/test-workflows`) adds dev extras to `setup.py` and both GitHub Actions workflows. PR 2 (`tests/characterization-baseline`) adds a shared deterministic-input module, a one-shot golden-generation script, committed golden JSON files, and two new test files (golden regressions, physics invariants). Everything runs on ASE's EMT calculator — deterministic, no GPU, no model downloads.

**Tech Stack:** GitHub Actions, pytest + pytest-cov, diff-cover, mutmut (<3.0), pip-audit, ASE/EMT, numpy.

## Decisions taken in the user's absence (flag for override)

The user was away when the handoff's mandated questions were asked. These are the recommended defaults; each is reversible:

1. **CI Python: 3.11** (matches the fairchem dev venv; single version).
2. **API surface to characterize:** all four — `core.optimize.run_optimization`, `core.md.setup_dynamics`/`run_md`, `core.neb.CustomNEB`, `core.utils.calc_fmax` + `core.params_io`.
3. **asetools:** PyPI's `asetools` is an unrelated pixel-art tool; CI installs the real one from `git+https://github.com/manuelarcer/asetools.git` (public, verified). No packaging change.
4. **Goldens on EMT** — they characterize *platform code paths* (optimizer wiring, file outputs, unit handling), not MLIP numerics. MLIP tests stay marker-gated and local.
5. **Default branch confirmed `main`**; fixtures already exist in `tests/fixtures/structures/` (no need to request structures).

## Global Constraints (from the standing rules — apply to every task)

- **Draft PRs only. Never merge, never push to main.** One concern per PR. Max 2 open agent PRs.
- **Never regenerate golden reference files.** The generation script must refuse to overwrite. On golden failure: report the numerical delta only.
- Every test must assert a numerical value or invariant. "Runs without raising" is not a test.
- Never use exact float equality; `np.testing.assert_allclose` with explicit per-quantity `rtol`/`atol`.
- No auto-upgrades of dependencies; pip-audit reports only.
- Full-repo mutation runs are not permitted — weekly mutmut is scoped to modules changed in the last 7 days.
- Repo facts: package import path `mlip_platform`, dist name `mlip-platform`, `src/` layout, **setup.py only (no pyproject.toml)**, `pytest.ini` sets `pythonpath = src` and `testpaths = tests`. Existing conftest fixture `tmp_workdir` (tests/conftest.py:115) gives a clean output dir.

---

## PR 1 — branch `ci/test-workflows`

### Task 1: Dev extras in setup.py

**Files:**
- Modify: `setup.py`

**Interfaces:**
- Produces: `pip install -e ".[dev]"` installs pytest, pytest-cov, diff-cover, mutmut, pip-audit. Tasks 2–3 rely on this.

- [ ] **Step 1: Create the branch from main**

```bash
git checkout main && git pull && git checkout -b ci/test-workflows
```

Expected: `Switched to a new branch 'ci/test-workflows'`. Do NOT branch from `feat/optimize-contcar-output` (it has uncommitted work — leave it alone).

- [ ] **Step 2: Add extras_require to setup.py**

In `setup.py`, after the `install_requires=[...]` block, add:

```python
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "diff-cover",
            # mutmut 3.x dropped the --paths-to-mutate CLI flag and requires
            # pyproject.toml config; this repo has no pyproject.toml.
            "mutmut>=2.4,<3.0",
            "pip-audit",
        ],
    },
```

- [ ] **Step 3: Verify the extras resolve**

```bash
/Users/juar/venv/fairchem/bin/python -c "
from setuptools import setup" 2>/dev/null; /Users/juar/venv/fairchem/bin/pip install -e ".[dev]" --quiet && /Users/juar/venv/fairchem/bin/python -c "import diff_cover, mutmut; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add setup.py
git commit -m "build: add dev extras (pytest, pytest-cov, diff-cover, mutmut, pip-audit)"
```

### Task 2: `.github/workflows/tests.yml`

**Files:**
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: `[dev]` extras from Task 1.
- Produces: a required-check-style job named `test` that fails on test failure OR diff coverage < 90%.

- [ ] **Step 1: Write the workflow**

```yaml
name: Tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      MPLBACKEND: Agg
    steps:
      - uses: actions/checkout@v4
        with:
          # Full history: diff-cover compares against origin/main.
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install package with dev deps
        run: |
          python -m pip install --upgrade pip
          # PyPI's "asetools" is an unrelated package; install the real one from GitHub.
          pip install git+https://github.com/manuelarcer/asetools.git
          pip install -e ".[dev]"

      - name: Run test suite with coverage
        run: pytest --cov=mlip_platform --cov-report=xml

      - name: Diff coverage gate (>= 90% on changed lines)
        run: |
          git fetch origin main
          diff-cover coverage.xml --compare-branch=origin/main --fail-under=90 --exclude setup.py --exclude "scripts/*"
```

Notes locked in:
- `fetch-depth: 0` is required; a shallow clone breaks `--compare-branch`.
- `--exclude setup.py --exclude "scripts/*"` — packaging metadata and one-shot scripts never execute under pytest; without the exclusion any PR touching them would fail the 90% gate spuriously. Only `src/` and `tests/` lines count.
- The editable install (`-e`) keeps coverage.xml paths as `src/mlip_platform/...`, matching git diff paths — diff-cover needs them to match.
- MLIP-gated tests (uma/mace/sevenn markers) auto-skip on the runner because those packages aren't installed — existing conftest logic handles this.

- [ ] **Step 2: Validate the YAML parses**

```bash
/Users/juar/venv/fairchem/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml')); print('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: add tests workflow with 90% diff-coverage gate"
```

### Task 3: `.github/workflows/weekly.yml`

**Files:**
- Create: `.github/workflows/weekly.yml`

**Interfaces:**
- Consumes: `[dev]` extras from Task 1.
- Produces: weekly artifact `weekly-quality-summary` (audit findings + mutation survivors).

- [ ] **Step 1: Write the workflow**

Cron: Mondays 06:00 SGT = Sundays 22:00 UTC → `0 22 * * 0`.

**Deviation from handoff, documented:** the handoff's literal command `git diff --name-only HEAD@{7.days.ago}` uses the reflog, which is empty in a fresh CI clone and would always error. Same semantics via commit history instead: diff against the last commit on main older than 7 days.

```yaml
name: Weekly quality checks

on:
  schedule:
    - cron: "0 22 * * 0"  # Mondays 06:00 SGT (UTC+8)
  workflow_dispatch: {}

jobs:
  weekly:
    runs-on: ubuntu-latest
    env:
      MPLBACKEND: Agg
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install package with dev deps
        run: |
          python -m pip install --upgrade pip
          pip install git+https://github.com/manuelarcer/asetools.git
          pip install -e ".[dev]"

      - name: Dependency vulnerability audit (report only, no upgrades)
        run: |
          echo "# Weekly quality summary ($(date -u +%F))" > summary.md
          echo "## pip-audit" >> summary.md
          pip-audit --format markdown >> summary.md 2>&1 || echo "pip-audit reported findings (see above)" >> summary.md

      - name: Determine modules changed in the last 7 days
        id: scope
        run: |
          BASE=$(git rev-list -1 --before="7 days ago" origin/main || true)
          if [ -z "$BASE" ]; then
            echo "No commit older than 7 days; skipping mutation testing." | tee -a summary.md
            echo "paths=" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          CHANGED=$(git diff --name-only "$BASE"..HEAD -- 'src/mlip_platform/**/*.py' | tr '\n' ',' | sed 's/,$//')
          echo "Changed source modules (7 days): ${CHANGED:-none}" | tee -a summary.md
          echo "paths=$CHANGED" >> "$GITHUB_OUTPUT"

      - name: Mutation testing (scoped — full-repo runs not permitted)
        if: steps.scope.outputs.paths != ''
        run: |
          echo "## mutmut survivors" >> summary.md
          mutmut run \
            --paths-to-mutate "${{ steps.scope.outputs.paths }}" \
            --tests-dir tests \
            --runner "python -m pytest -x -q -m 'not slow'" \
            || true
          mutmut results >> summary.md 2>&1 || true

      - name: Publish summary
        if: always()
        run: cat summary.md >> "$GITHUB_STEP_SUMMARY"

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: weekly-quality-summary
          path: summary.md
```

Notes locked in:
- `|| true` on mutmut run: surviving mutants are *reported*, not a red X — per handoff these are findings for a human, and standing rule 4 makes them actionable later.
- pip-audit likewise reports without failing the job (handoff: "Report findings; do not auto-upgrade").

- [ ] **Step 2: Validate the YAML parses**

```bash
/Users/juar/venv/fairchem/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/weekly.yml')); print('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/weekly.yml
git commit -m "ci: add weekly pip-audit + scoped mutation-testing workflow"
```

### Task 4: Open draft PR 1

- [ ] **Step 1: Push and open draft PR**

```bash
git push -u origin ci/test-workflows
gh pr create --draft --base main --title "ci: tests workflow with diff-coverage gate + weekly quality checks" --body "$(cat <<'EOF'
Adds CI infrastructure per the test-quality handoff:

- `.github/workflows/tests.yml` — on every push/PR: install (asetools from GitHub — the PyPI name is an unrelated package), `pytest --cov`, then `diff-cover --fail-under=90` against `origin/main`. Only changed lines count; legacy coverage is irrelevant.
- `.github/workflows/weekly.yml` — Mondays 06:00 SGT: `pip-audit` (report-only), `mutmut` scoped to modules changed in the last 7 days (skipped if none; full-repo runs not permitted), summary uploaded as artifact.
- `setup.py` — new `[dev]` extra: pytest, pytest-cov, diff-cover, mutmut<3.0, pip-audit.

Deviation from handoff: the 7-day scope uses `git rev-list --before="7 days ago"` instead of `HEAD@{7.days.ago}` — the reflog is empty in a fresh CI clone, so the literal command always fails there. Semantics are the same.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Verify the tests workflow ran on the PR**

```bash
gh run list --branch ci/test-workflows --limit 3
```

Expected: a `Tests` run appears (queued/running/completed). If it fails, read the log (`gh run view <id> --log-failed`); likely causes are the asetools install or a matplotlib backend issue. Fix forward on the branch — do not merge anything.

---

## PR 2 — branch `tests/characterization-baseline`

### Task 5: Shared deterministic golden-input builders

**Files:**
- Create: `tests/golden_inputs.py`

**Interfaces:**
- Produces (consumed by Tasks 6–8):
  - `make_opt_atoms() -> ase.Atoms` — rattled Cu bulk, EMT attached
  - `make_md_atoms() -> ase.Atoms` — Cu bulk, seeded Maxwell-Boltzmann velocities, EMT attached
  - `make_neb_pair() -> tuple[Atoms, Atoms]` — initial/final pair (no calculators)
  - `make_cluster() -> ase.Atoms` — isolated Cu13 cluster in vacuum, EMT attached
  - `GOLDEN_DIR: pathlib.Path` — `tests/goldens/`

The generation script and the regression tests must build *byte-identical* inputs, so the builders live in one importable module. Every stochastic element is explicitly seeded.

- [ ] **Step 1: Create the branch from main**

```bash
git checkout main && git checkout -b tests/characterization-baseline
```

- [ ] **Step 2: Write the module**

```python
"""Deterministic input builders shared by golden generation and regression tests.

Every structure is fully seeded. If you change ANYTHING here, every golden
derived from it is invalidated -- and regenerating goldens is a human
decision (standing rule 2). Do not edit casually.
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
    velocities for nvt/npt (core/md.py:105), so pre-seeded momenta survive.
    """
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.RandomState(42)
    )
    atoms.calc = EMT()
    return atoms


def make_neb_pair():
    """Initial/final pair for NEB: one atom displaced 0.3,0.3,0 Angstrom.

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
```

- [ ] **Step 3: Smoke-check it imports and is deterministic**

```bash
/Users/juar/venv/fairchem/bin/python -c "
from tests.golden_inputs import make_opt_atoms, make_md_atoms
import numpy as np
a, b = make_opt_atoms(), make_opt_atoms()
assert np.array_equal(a.positions, b.positions), 'rattle not deterministic'
m1, m2 = make_md_atoms(), make_md_atoms()
assert np.array_equal(m1.get_momenta(), m2.get_momenta()), 'velocities not deterministic'
print('deterministic ok')"
```

Expected: `deterministic ok`

- [ ] **Step 4: Commit**

```bash
git add tests/golden_inputs.py
git commit -m "test: add deterministic input builders for characterization goldens"
```

### Task 6: Golden generation script + frozen goldens

**Files:**
- Create: `scripts/generate_goldens.py`
- Create (generated, then committed): `tests/goldens/optimize_cu_rattled.json`, `tests/goldens/md_nve_cu.json`, `tests/goldens/neb_idpp_profile.json`, `tests/goldens/utils_calc_fmax.json`

**Interfaces:**
- Consumes: all builders from `tests/golden_inputs.py`; `run_optimization` (core/optimize.py:40), `run_md` (core/md.py:163), `CustomNEB` (core/neb.py:28), `calc_fmax` (core/utils.py:8).
- Produces: JSON goldens with keys documented per quantity below; Task 7 reads them.

- [ ] **Step 1: Write the script**

```python
"""One-shot generation of golden reference data for characterization tests.

STANDING RULE 2: goldens are frozen. This script REFUSES to overwrite an
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
            "Regenerating baselines is a human decision (standing rule 2)."
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
        "final_cell_A": final.get_cell().tolist(),
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
```

- [ ] **Step 2: Generate the goldens (the one permitted run)**

```bash
cd /Users/juar/github/work/mlip_platform && MPLBACKEND=Agg /Users/juar/venv/fairchem/bin/python scripts/generate_goldens.py
```

Expected: four `wrote tests/goldens/...` lines. Sanity-read the JSON: energies should be O(-10..10) eV, no NaN.

- [ ] **Step 3: Verify the refuse-to-overwrite guard**

```bash
MPLBACKEND=Agg /Users/juar/venv/fairchem/bin/python scripts/generate_goldens.py; echo "exit=$?"
```

Expected: `REFUSING to overwrite ...` and `exit=1`.

- [ ] **Step 4: Commit script + goldens together**

```bash
git add scripts/generate_goldens.py tests/goldens/
git commit -m "test: freeze golden reference data for optimize, MD, NEB, and utils (EMT)"
```

### Task 7: Golden regression tests

**Files:**
- Create: `tests/test_characterization_golden.py`

**Interfaces:**
- Consumes: goldens from Task 6, builders from Task 5.

**Tolerances (per quantity — never exact equality, never one-size-fits-all):**

| Quantity | rtol | atol | Rationale |
|---|---|---|---|
| energies (eV) | 1e-9 | 1e-8 | same code path; only BLAS/platform noise |
| positions (Å) | 0 | 1e-6 | optimizer/integrator path-dependence amplifies float noise |
| forces / fmax (eV/Å) | 1e-9 | 1e-6 | derivative of energy, one order looser |

If CI on Linux shows platform drift beyond these, loosening a tolerance is permitted (it is not a golden regeneration) — but record the observed delta in the PR.

- [ ] **Step 1: Write the failing tests**

```python
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
            df["potential_energy(eV)"].iloc[-1], golden["final_potential_energy_eV"],
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
```

- [ ] **Step 2: Run the golden tests**

```bash
MPLBACKEND=Agg /Users/juar/venv/fairchem/bin/python -m pytest tests/test_characterization_golden.py -v
```

Expected: ALL PASS (goldens were generated by the same code seconds ago; a failure here means non-determinism in the pipeline — investigate before proceeding, do not loosen tolerances to paper over it).

- [ ] **Step 3: Commit**

```bash
git add tests/test_characterization_golden.py
git commit -m "test: golden regression tests for optimize, MD, NEB, and calc_fmax"
```

### Task 8: Physics invariant tests

**Files:**
- Create: `tests/test_characterization_invariants.py`

**Interfaces:**
- Consumes: builders from Task 5 (`make_opt_atoms`, `make_md_atoms`, `make_cluster`).

- [ ] **Step 1: Write the tests**

```python
"""Physics-invariant tests (property-based, no goldens needed).

These hold regardless of code version -- they catch unit-handling and
wiring bugs that golden tests can miss.
"""
import numpy as np
import pandas as pd
import pytest
from ase.calculators.emt import EMT

from golden_inputs import make_cluster, make_md_atoms, make_opt_atoms
from mlip_platform.core.md import run_md
from mlip_platform.core.optimize import run_optimization
from mlip_platform.core.utils import calc_fmax


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
        """Central finite differences: F_i ~= -dE/dx_i, h = 1e-4 A."""
        atoms = make_opt_atoms()
        analytic = atoms.get_forces()
        h = 1e-4
        # Three atoms x three directions is plenty to catch sign/unit bugs.
        for atom_idx in (0, 3, 7):
            for axis in range(3):
                for sign, store in ((+1, "ep"), (-1, "em")):
                    probe = atoms.copy()
                    pos = probe.get_positions()
                    pos[atom_idx, axis] += sign * h
                    probe.set_positions(pos)
                    probe.calc = EMT()
                    if sign > 0:
                        ep = probe.get_potential_energy()
                    else:
                        em = probe.get_potential_energy()
                fd_force = -(ep - em) / (2 * h)
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
```

- [ ] **Step 2: Run the invariant tests**

```bash
MPLBACKEND=Agg /Users/juar/venv/fairchem/bin/python -m pytest tests/test_characterization_invariants.py -v
```

Expected: ALL PASS. If FD consistency fails at rtol=2e-3, first check h against EMT's own smoothness before touching the tolerance.

- [ ] **Step 3: Run the FULL suite to confirm nothing else broke**

```bash
MPLBACKEND=Agg /Users/juar/venv/fairchem/bin/python -m pytest -q
```

Expected: all pass or skip (uma/mace/sevenn markers auto-skip); zero new failures vs. `main`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_characterization_invariants.py
git commit -m "test: physics invariants (translation, rotation, F=-dE/dx, permutation, NVE drift)"
```

### Task 9: Open draft PR 2

- [ ] **Step 1: Push and open draft PR**

```bash
git push -u origin tests/characterization-baseline
gh pr create --draft --base main --title "test: baseline characterization tests (goldens + physics invariants)" --body "$(cat <<'EOF'
Baseline characterization per the test-quality handoff -- prerequisite for any future refactoring (tests-first ratchet).

- `tests/golden_inputs.py` -- fully seeded input builders shared by generation and tests (single source of truth for structures).
- `scripts/generate_goldens.py` -- one-shot generator; REFUSES to overwrite existing goldens (standing rule 2).
- `tests/goldens/*.json` -- frozen EMT baselines for `run_optimization`, `run_md` (NVE, seeded velocities), `CustomNEB` (IDPP profile), `calc_fmax`.
- `tests/test_characterization_golden.py` -- regression vs. goldens; per-quantity tolerances (energies rtol 1e-9/atol 1e-8 eV, positions atol 1e-6 A, forces atol 1e-6 eV/A).
- `tests/test_characterization_invariants.py` -- translation/rotation/permutation invariance, force-energy finite-difference consistency, NVE drift < 0.5 meV/atom.

Goldens are EMT: they characterize platform code paths (optimizer wiring, file outputs, unit handling), not MLIP numerics. MLIP-gated tests remain marker-based and local.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Verify CI passes on the PR**

```bash
gh run list --branch tests/characterization-baseline --limit 3
```

Expected: `Tests` run green (this PR adds only test files + one script; the diff-cover gate counts test lines, which execute — coverage of the diff should be ~100%). If Linux CI shows golden failures beyond tolerance: report the delta in a PR comment; loosening a tolerance is allowed with the observed delta recorded; regenerating goldens is NOT.

---

## Self-review notes

- **Spec coverage:** Task 1 ↔ handoff "add dev deps"; Task 2 ↔ Task 1 of handoff (all 5 steps, including fail conditions); Task 3 ↔ handoff Task 2 (cron SGT→UTC, audit report-only, scoped mutmut with skip-if-empty, summary artifact); Tasks 5–8 ↔ handoff Task 3 (fixtures exist already, goldens frozen, per-quantity tolerances, all four invariant families); Tasks 4 & 9 ↔ standing rule 1 (two draft PRs, the maximum allowed).
- **Known deviation (documented in workflow + PR body):** `HEAD@{7.days.ago}` replaced by `git rev-list -1 --before="7 days ago" origin/main` because the reflog is empty in CI clones.
- **Known risk:** cross-platform (macOS-generated goldens vs. Linux CI) float drift on optimizer trajectories. Mitigation: position atol 1e-6 with an explicit, recorded-loosening policy that does not count as regeneration.
- **mutmut pinned <3.0** because 3.x removed the `--paths-to-mutate` CLI needed for scoping without a pyproject.toml.
