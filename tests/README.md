# Tests

Pytest suite for `mliprun`. Most unit tests run on ASE's built-in EMT calculator and need no MLIP installed; integration tests against UMA, MACE, and SevenNet are gated behind markers and auto-skip when the corresponding package is missing.

## Layout

```
tests/
├── conftest.py                 # marker registration, MLIP-availability detection, shared fixtures
├── fixtures/structures/        # POSCARs and reference outputs used by integration tests
│
├── test_core_optimize.py       # core.optimize.run_optimization (EMT)
├── test_core_md.py             # core.md.setup_dynamics, run_md (EMT)
├── test_core_neb.py            # core.neb.CustomNEB end-to-end (EMT)
├── test_core_utils.py          # core.utils.calc_fmax, params_io
├── test_neb_restart.py         # NEB restart parsing + override logic (no MLIP)
├── test_neb_constrained.py     # highly-constrained NEB mode (relax_atoms)
│
├── test_cli.py                 # `--help` smoke tests for each command
├── test_cli_commands.py        # CLI argument parsing / help output
├── test_cli_utils.py           # detect_mlip / validate_mlip / resolve_mlip
│
├── test_integration_uma.py     # UMA end-to-end (marker: uma)
├── test_md_mace.py             # MACE MD smoke test
├── test_md_sevenn.py           # SevenNet MD smoke test
├── test_neb_mace.py            # MACE NEB smoke test
├── test_neb_sevenn.py          # SevenNet NEB smoke test
├── test_milp_single_point.py   # single-point energy on MACE / SevenNet
└── test_benchmark.py           # placeholder, currently empty
```

## Markers

Registered in `pytest.ini` and `conftest.py`:

| Marker  | Meaning |
|---------|---------|
| `uma`    | Requires `fairchem-core`. Auto-skipped if not installed. |
| `mace`   | Requires `mace-torch`. Auto-skipped if not installed. |
| `sevenn` | Requires `sevenn`. Auto-skipped if not installed. |
| `slow`   | Long-running tests. Deselect with `-m "not slow"`. |

`conftest.py:pytest_collection_modifyitems` walks every collected test and adds a `skip` marker if the package backing the test is missing. The MLIP-specific test files also use plain `@pytest.mark.skipif` guards as a belt-and-braces fallback.

## Running

```bash
# Default — runs everything that can run in the current environment
pytest

# Unit tests only (no MLIP installed or wanted)
pytest -m "not uma and not mace and not sevenn"

# Specific MLIP integration suite
pytest -m uma
pytest -m mace
pytest -m sevenn

# A single file or test
pytest tests/test_core_neb.py
pytest tests/test_neb_restart.py::TestParseParametersFile

# Exclude slow tests
pytest -m "not slow"
```

`pytest.ini` sets `pythonpath = src` and `testpaths = tests`, so `pytest` works from the repo root with no extra flags. `--maxfail=0 --continue-on-collection-errors` are on by default so a single broken import doesn't hide the rest of the run.

## Fixtures

Defined in `conftest.py`:

- `simple_atoms` / `simple_atoms_no_calc` — Cu FCC bulk with / without an EMT calculator attached.
- `initial_final_pair` — Two Cu (2×2×2) supercells with one displaced atom; used as a minimal NEB pathway.
- `mock_calculator` — bare `EMT()` instance.
- `tmp_workdir` — alias for pytest's `tmp_path`, returned as a `Path`.
- `fixtures_dir` — `tests/fixtures/structures/` as a `Path`, for tests that need pre-staged POSCARs.

## Adding a new test

1. Decide whether the test needs an MLIP. If not, use the EMT-based fixtures and put the file under `test_core_*` or `test_cli_*`.
2. If it does, mark the file or function with the matching marker (`uma`, `mace`, or `sevenn`) so it auto-skips elsewhere. Example:

   ```python
   import pytest
   pytestmark = pytest.mark.uma  # entire module gated on fairchem-core
   ```

3. Pre-staged structures go under `tests/fixtures/structures/<descriptive-name>/`. Avoid committing trajectory or PNG outputs unless a test needs them as reference data.
